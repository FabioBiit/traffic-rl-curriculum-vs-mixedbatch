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
        action_logits = self.actor(obs)

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

        self._cur_value = self.critic(global_obs).squeeze(-1)
        return action_logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "forward() must be called first"
        return self._cur_value


# ---------------------------------------------------------------------------
# Callbacks: inject global_obs + recompute GAE
# ---------------------------------------------------------------------------

class CentralizedCriticCallbacks(DefaultCallbacks):
    """
    RLlib callbacks for centralized critic MAPPO.

    on_postprocess_trajectory():
      1. Concatena own_obs + opponent_obs → global_obs
      2. Ricalcola VF predictions con global_obs
      3. Ricalcola GAE advantages
    """

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
        batch_size = own_obs.shape[0]

        # Collect other agents' obs
        opponent_obs_list = []
        for other_id, other_data in original_batches.items():
            if other_id == agent_id:
                continue
            # RLlib 2.10.x: other_data = (policy_id, policy, SampleBatch)
            # Extract SampleBatch as last tuple element
            if isinstance(other_data, tuple):
                other_batch = other_data[-1]
            else:
                other_batch = other_data
            opp_obs = other_batch[SampleBatch.CUR_OBS]

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
        postprocessed_batch[GLOBAL_OBS] = global_obs

        # Recompute VF predictions with global_obs
        device = next(policy.model.parameters()).device
        with torch.no_grad():
            global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=device)
            vf_preds = policy.model.critic(global_obs_t).squeeze(-1).cpu().numpy()

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

            with torch.no_grad():
                t = torch.as_tensor(last_global, dtype=torch.float32, device=device)
                last_r = policy.model.critic(t).squeeze(-1).item()

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