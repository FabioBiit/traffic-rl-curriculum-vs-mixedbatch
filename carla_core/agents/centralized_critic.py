"""
CentralizedCriticModel — MAPPO Centralized Critic per RLlib 2.10.x
====================================================================
Paradigma CTDE (Centralized Training, Decentralized Execution):
  - Actor: vede solo obs locale (24D veicolo / 18D pedone)
  - Critic: vede global_obs = concat di TUTTE le obs degli agenti

Componenti:
  1. CentralizedCriticModel  — TorchModelV2 con actor/critic separati
  2. CentralizedCriticCallbacks — inietta global_obs nel SampleBatch
     via on_postprocess_trajectory(), ricalcola VF e GAE
  3. on_episode_start/step/end — Custom metrics per TensorBoard
     (success_rate, collision_rate, offroad_rate, stuck_rate, timeout_rate,
      route_completion, path_efficiency) per-policy

Struttura reti:
  Actor MLP:  obs_dim → 256 → 256 → action_logits
  Critic MLP: global_obs_dim → 256 → 256 → 1 (value)
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

logger = logging.getLogger(__name__)

# Key injected into SampleBatch for centralized critic
GLOBAL_OBS = "global_obs"

def _raise_on_nonfinite_np(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    if np.isfinite(arr).all():
        return
    bad = np.argwhere(~np.isfinite(arr))
    first = tuple(bad[0].tolist()) if bad.size else ()
    value = arr[first] if first else arr
    raise ValueError(f"{name} contains non-finite values at {first}: {value}")


def _raise_on_nonfinite_torch(name: str, tensor: torch.Tensor):
    if torch.isfinite(tensor).all():
        return
    bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
    first = tuple(bad[0].tolist()) if bad.numel() else ()
    value = tensor[first].detach().cpu().item() if first else float("nan")
    raise ValueError(f"{name} contains non-finite values at {first}: {value}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    MAPPO model: separate actor (local obs) + centralized critic (global obs).

    custom_model_config keys:
        hidden_size (int): hidden layer size (default 256)
        n_hidden_layers (int): number of hidden layers (default 2)
        global_obs_dim (int): dimension of concatenated global observation
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {})
        hidden = custom.get("hidden_size", 256)
        n_layers = custom.get("n_hidden_layers", 2)
        self._global_obs_dim = custom.get("global_obs_dim", 42)

        local_obs_dim = int(np.prod(obs_space.shape))

        # Actor: local obs → action logits
        actor_layers = []
        in_dim = local_obs_dim
        for _ in range(n_layers):
            actor_layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        actor_layers.append(nn.Linear(in_dim, num_outputs))
        self.actor = nn.Sequential(*actor_layers)

        # Critic: global obs → scalar value
        critic_layers = []
        in_dim = self._global_obs_dim
        for _ in range(n_layers):
            critic_layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self._cur_value = None

        logger.info(
            f"CentralizedCriticModel '{name}': "
            f"actor {local_obs_dim}->{num_outputs}, "
            f"critic {self._global_obs_dim}->1, "
            f"hidden={hidden}x{n_layers}"
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        _raise_on_nonfinite_torch("obs_flat", obs)
        action_logits = self.actor(obs)
        _raise_on_nonfinite_torch("action_logits", action_logits)

        # During training: global_obs injected by callbacks
        if GLOBAL_OBS in input_dict:
            global_obs = input_dict[GLOBAL_OBS].float()
        else:
            # Rollout: critic uses local obs zero-padded
            batch_sz = obs.shape[0]
            global_obs = torch.zeros(
                batch_sz, self._global_obs_dim, device=obs.device
            )
            global_obs[:, :obs.shape[-1]] = obs

        _raise_on_nonfinite_torch("global_obs", global_obs)
        self._cur_value = self.critic(global_obs).squeeze(-1)
        _raise_on_nonfinite_torch("value_function", self._cur_value)
        return action_logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "forward() must be called first"
        return self._cur_value


# ---------------------------------------------------------------------------
# Callbacks: inject global_obs + recompute GAE + custom metrics
# ---------------------------------------------------------------------------

class CentralizedCriticCallbacks(DefaultCallbacks):
    """
    RLlib callbacks for centralized critic MAPPO.

    on_episode_start():  init per-episode outcome tracking
    on_episode_step():   capture termination_reason when each agent terminates
    on_episode_end():    aggregate per-policy metrics → TensorBoard custom_metrics
    on_postprocess_trajectory():
      1. Concatena own_obs + opponent_obs → global_obs
      2. Ricalcola VF predictions con global_obs
      3. Ricalcola GAE advantages
    """

    # ------------------------------------------------------------------
    # Episode-level metrics (Block 1)
    # ------------------------------------------------------------------

    def on_episode_start(
        self, *, worker, base_env, policies, episode,
        env_index=None, **kwargs,
    ):
        """Initialize per-episode agent outcome tracking."""
        episode.user_data["agent_outcomes"] = {}

    def on_episode_step(
        self, *, worker, base_env, policies, episode,
        env_index=None, **kwargs,
    ):
        """Capture each agent's termination info at the step they terminate.

        Reads from two sources:
          1. episode.last_info_for() — for agents still in the obs dict
          2. env._terminated_agent_infos — side-channel for agents whose
             info can't be in the RLlib infos dict (keys must ⊆ obs keys)
        """
        # --- Source 1: standard RLlib info path ---
        for agent_id in episode.get_agents():
            if agent_id in episode.user_data["agent_outcomes"]:
                continue
            info = episode.last_info_for(agent_id)
            if not info:
                continue
            tr = info.get("termination_reason")
            if tr and tr != "alive":
                episode.user_data["agent_outcomes"][agent_id] = {
                    "termination_reason": tr,
                    "route_completion": info.get("route_completion", 0.0),
                    "path_efficiency": info.get("path_efficiency", 0.0),
                }

        # --- Source 2: side-channel for terminated agents ---
        try:
            raw_env = base_env.get_sub_environments()[0]
            # ParallelPettingZooEnv wraps our env in .par_env (or .env)
            inner = getattr(raw_env, "par_env", None) or getattr(raw_env, "env", None)
            if inner is None:
                inner = raw_env
            term_infos = getattr(inner, "_terminated_agent_infos", {})
            for agent_id, info in term_infos.items():
                if agent_id in episode.user_data["agent_outcomes"]:
                    continue
                tr = info.get("termination_reason")
                if tr and tr != "alive":
                    episode.user_data["agent_outcomes"][agent_id] = {
                        "termination_reason": tr,
                        "route_completion": info.get("route_completion", 0.0),
                        "path_efficiency": info.get("path_efficiency", 0.0),
                    }
        except (AttributeError, IndexError):
            pass  # Graceful fallback if env structure differs       

    def on_episode_end(
        self, *, worker, base_env, policies, episode,
        env_index=None, **kwargs,
    ):
        """Aggregate per-policy metrics → TensorBoard custom_metrics."""
        outcomes = episode.user_data.get("agent_outcomes", {})

        for policy_id in ("vehicle_policy", "pedestrian_policy"):
            prefix = "vehicle" if policy_id == "vehicle_policy" else "pedestrian"
            agent_data = {
                aid: out for aid, out in outcomes.items()
                if aid.startswith(prefix)
            }
            n = len(agent_data)
            if n == 0:
                continue

            reasons = [d["termination_reason"] for d in agent_data.values()]
            episode.custom_metrics[f"{policy_id}/success_rate"] = (
                reasons.count("route_complete") / n
            )
            episode.custom_metrics[f"{policy_id}/collision_rate"] = (
                reasons.count("collision") / n
            )
            episode.custom_metrics[f"{policy_id}/offroad_rate"] = (
                reasons.count("offroad") / n
            )
            episode.custom_metrics[f"{policy_id}/stuck_rate"] = (
                reasons.count("stuck") / n
            )
            episode.custom_metrics[f"{policy_id}/timeout_rate"] = (
                reasons.count("timeout") / n
            )
            episode.custom_metrics[f"{policy_id}/route_completion"] = float(
                np.mean([d["route_completion"] for d in agent_data.values()])
            )
            episode.custom_metrics[f"{policy_id}/path_efficiency"] = float(
                np.mean([d["path_efficiency"] for d in agent_data.values()])
            )

        # Debug: warn about agents with no captured outcome
        all_agents = set(episode.get_agents())
        missing = all_agents - set(outcomes.keys())
        if missing:
            logger.warning(
                f"Episode {episode.episode_id}: no termination_reason for: {missing}"
            )

    # ------------------------------------------------------------------
    # Trajectory postprocessing (centralized critic)
    # ------------------------------------------------------------------

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch: SampleBatch,
        original_batches: Dict,
        **kwargs,
    ):
        own_obs = postprocessed_batch[SampleBatch.CUR_OBS]
        _raise_on_nonfinite_np(f"{agent_id}.own_obs", own_obs)
        batch_size = own_obs.shape[0]

        # Collect other agents' obs
        opponent_obs_list = []
        for other_id in sorted(original_batches.keys()):
            other_data = original_batches[other_id]
            if other_id == agent_id:
                continue
            # RLlib 2.10.x: other_data = (policy_id, policy, SampleBatch)
            # Extract SampleBatch as last tuple element
            if isinstance(other_data, tuple):
                other_batch = other_data[-1]
            else:
                other_batch = other_data
            opp_obs = other_batch[SampleBatch.CUR_OBS]
            _raise_on_nonfinite_np(f"{agent_id}.opp_obs[{other_id}]", opp_obs)

            # Align batch sizes
            if len(opp_obs) > batch_size:
                opp_obs = opp_obs[:batch_size]
            elif len(opp_obs) < batch_size:
                pad_n = batch_size - len(opp_obs)
                opp_obs = np.concatenate(
                    [opp_obs, np.tile(opp_obs[-1:], (pad_n, 1))], axis=0
                )
            opponent_obs_list.append(opp_obs)

        policy = policies[policy_id]
        expected_dim = policy.model._global_obs_dim

        if opponent_obs_list:
            global_obs = np.concatenate([own_obs] + opponent_obs_list, axis=-1)
            # Pad if needed (e.g., some agents terminated early)
            if global_obs.shape[-1] < expected_dim:
                pad = np.zeros(
                    (batch_size, expected_dim - global_obs.shape[-1]),
                    dtype=np.float32,
                )
                global_obs = np.concatenate([global_obs, pad], axis=-1)
            elif global_obs.shape[-1] > expected_dim:
                global_obs = global_obs[:, :expected_dim]
        else:
            # No opponents — zero-pad
            pad_dim = expected_dim - own_obs.shape[-1]
            if pad_dim > 0:
                padding = np.zeros((batch_size, pad_dim), dtype=np.float32)
                global_obs = np.concatenate([own_obs, padding], axis=-1)
            else:
                global_obs = own_obs[:, :expected_dim]

        global_obs = global_obs.astype(np.float32)
        _raise_on_nonfinite_np(f"{agent_id}.global_obs", global_obs)
        postprocessed_batch[GLOBAL_OBS] = global_obs

        # Recompute VF predictions with global_obs
        device = next(policy.model.parameters()).device
        with torch.no_grad():
            global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=device)
            vf_preds = policy.model.critic(global_obs_t).squeeze(-1).cpu().numpy()

        _raise_on_nonfinite_np(f"{agent_id}.vf_preds", vf_preds)
        postprocessed_batch[SampleBatch.VF_PREDS] = vf_preds

        # Bootstrap value for incomplete trajectories
        last_r = 0.0
        if not postprocessed_batch[SampleBatch.TERMINATEDS][-1]:
            last_obs = own_obs[-1:]
            if opponent_obs_list:
                last_opp = [o[-1:] for o in opponent_obs_list]
                last_global = np.concatenate([last_obs] + last_opp, axis=-1)
            else:
                last_global = last_obs

            # Pad/truncate to expected_dim
            if last_global.shape[-1] < expected_dim:
                last_global = np.concatenate(
                    [last_global, np.zeros((1, expected_dim - last_global.shape[-1]),
                                           dtype=np.float32)],
                    axis=-1,
                )
            elif last_global.shape[-1] > expected_dim:
                last_global = last_global[:, :expected_dim]

            _raise_on_nonfinite_np(f"{agent_id}.last_global", last_global)
            with torch.no_grad():
                t = torch.as_tensor(last_global, dtype=torch.float32, device=device)
                last_r = policy.model.critic(t).squeeze(-1).item()
            if not np.isfinite(last_r):
                raise ValueError(f"{agent_id}.last_r is non-finite: {last_r}")

        # Recompute GAE
        train_batch = compute_advantages(
            postprocessed_batch,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=True,
            use_critic=True,
        )

        return train_batch
