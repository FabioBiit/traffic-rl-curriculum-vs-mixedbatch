import json
import os
import subprocess
from collections import deque

import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from carla_core.agents.centralized_critic import (
    CentralizedCriticCallbacks,
    compute_global_obs_dim_with_mask,
)
from carla_core.envs.carla_multi_agent_env import (
    CarlaMultiAgentEnv,
    PEDESTRIAN_OBS_DIM,
    VEHICLE_OBS_DIM,
)


def _append_episode_json(path, record):
    """Append a JSON line to the episode log file."""
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)


def shutdown_carla_processes():
    """Best-effort shutdown of local CARLA server processes on Windows."""
    if os.name != "nt":
        return {
            "attempted": False,
            "killed_any": False,
            "not_found_only": True,
            "issues": [],
            "targets": [],
        }

    targets = []
    issues = []
    killed_any = False
    not_found_only = True

    for image_name in ("CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"):
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/IM", image_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            issues.append(f"{image_name}: {type(exc).__name__}: {exc}")
            not_found_only = False
            continue

        output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
        normalized = output.lower()
        targets.append({"image_name": image_name, "returncode": proc.returncode, "output": output})

        if proc.returncode == 0:
            killed_any = True
            continue

        if any(
            token in normalized
            for token in (
                "not found",
                "no running instance",
                "not running",
                "nessuna istanza",
                "non trovato",
                "non e' in esecuzione",
                "non è in esecuzione",
            )
        ):
            continue

        not_found_only = False
        issues.append(f"{image_name}: taskkill exited with code {proc.returncode}")

    return {
        "attempted": True,
        "killed_any": killed_any,
        "not_found_only": not_found_only,
        "issues": issues,
        "targets": targets,
    }


class MAPPOTrainingCallbacks(CentralizedCriticCallbacks):
    """Extends CentralizedCriticCallbacks with episode-level JSON logging."""

    def __init__(self):
        super().__init__()
        self._success_window = deque(maxlen=50)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )
        log_path = os.environ.get("MAPPO_EPISODE_LOG")
        outcomes = episode.user_data.get("agent_outcomes", {})
        if log_path:
            for agent_id, out in outcomes.items():
                policy_id = (
                    "vehicle_policy" if agent_id.startswith("vehicle") else "pedestrian_policy"
                )
                _append_episode_json(
                    log_path,
                    {
                        "episode_id": episode.episode_id,
                        "agent_id": agent_id,
                        "policy": policy_id,
                        "termination_reason": out["termination_reason"],
                        "route_completion": round(out["route_completion"], 4),
                        "path_efficiency": round(out["path_efficiency"], 4),
                        "step_count": episode.length,
                    },
                )

        all_reasons = [d["termination_reason"] for d in outcomes.values()]
        n_total = len(all_reasons)
        if n_total > 0:
            success_rate = all_reasons.count("route_complete") / n_total
            self._success_window.append(success_rate)
            episode.custom_metrics["success_rate"] = success_rate
            episode.custom_metrics["collision_rate"] = all_reasons.count("collision") / n_total
            episode.custom_metrics["route_completion"] = float(
                np.mean([d["route_completion"] for d in outcomes.values()])
            )
            episode.custom_metrics["window_success_rate"] = float(np.mean(self._success_window))


def rllib_env_creator(env_config):
    """Wrap CarlaMultiAgentEnv for RLlib via ParallelPettingZooEnv."""
    raw_env = CarlaMultiAgentEnv(config=env_config)
    return ParallelPettingZooEnv(raw_env)


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """vehicle_* -> vehicle_policy, pedestrian_* -> pedestrian_policy."""
    if agent_id.startswith("vehicle"):
        return "vehicle_policy"
    if agent_id.startswith("pedestrian"):
        return "pedestrian_policy"
    raise ValueError(f"Unknown agent_id: {agent_id}")


def _build_mappo_config(
    *,
    env_cfg,
    train_cfg,
    eval_cfg,
    n_gpus,
    n_workers,
    exp_seed,
    enable_periodic_evaluation,
):
    """Build the shared RLlib PPOConfig for training or final evaluation."""
    import gymnasium as gym

    roll = train_cfg.get("rollout", {})
    opt = train_cfg.get("optimization", {})
    model_cfg = train_cfg.get("model", {})
    eval_section = eval_cfg.get("evaluation", {})

    ag_cfg = env_cfg.get("agents", {})
    n_veh = ag_cfg.get("n_vehicles_rl", 1)
    n_ped = ag_cfg.get("n_pedestrians_rl", 1)
    global_obs_dim = compute_global_obs_dim_with_mask(n_veh, n_ped)

    veh_obs = gym.spaces.Box(-1, 1, (VEHICLE_OBS_DIM,), np.float32)
    veh_act = gym.spaces.Box(
        np.array([-1, -1], dtype=np.float32),
        np.array([1, 1], dtype=np.float32),
    )
    ped_obs = gym.spaces.Box(-1, 1, (PEDESTRIAN_OBS_DIM,), np.float32)
    ped_act = gym.spaces.Box(
        np.array([0, -1], dtype=np.float32),
        np.array([1, 1], dtype=np.float32),
    )

    hidden_size = model_cfg.get("hidden_size", 256)
    n_hidden = model_cfg.get("n_hidden_layers", 2)
    agent_order = [f"vehicle_{i}" for i in range(n_veh)] + [
        f"pedestrian_{i}" for i in range(n_ped)
    ]
    cc_config = {
        "hidden_size": hidden_size,
        "n_hidden_layers": n_hidden,
        "global_obs_dim": global_obs_dim,
        "agent_order": agent_order,
        "slot_obs_dims": {
            "vehicle": VEHICLE_OBS_DIM,
            "pedestrian": PEDESTRIAN_OBS_DIM,
        },
        "use_popart": model_cfg.get("use_popart", False),
        "popart_beta": model_cfg.get("popart_beta", 3e-4),
        "use_attention": model_cfg.get("use_attention", False),
        "attention_embed_dim": model_cfg.get("attention_embed_dim", 64),
        "attention_heads": model_cfg.get("attention_heads", 4)
    }
    if enable_periodic_evaluation:
        raise NotImplementedError(
            "Periodic RLlib evaluation is disabled in the CARLA multi-agent trainer. "
            "Use the explicit final multi-scenario evaluation instead."
        )
    evaluation_interval = None

    return (
        PPOConfig()
        .environment(env="CarlaMultiAgent-v0", env_config=env_cfg)
        .debugging(seed=exp_seed)
        .multi_agent(
            policies={
                "vehicle_policy": (
                    None,
                    veh_obs,
                    veh_act,
                    {"model": {"custom_model": "cc_model", "custom_model_config": cc_config}},
                ),
                "pedestrian_policy": (
                    None,
                    ped_obs,
                    ped_act,
                    {"model": {"custom_model": "cc_model", "custom_model_config": cc_config}},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["vehicle_policy", "pedestrian_policy"],
        )
        .callbacks(MAPPOTrainingCallbacks)
        .resources(num_gpus=n_gpus)
        .rollouts(
            num_rollout_workers=n_workers,
            rollout_fragment_length=roll.get("rollout_fragment_length", 200),
            batch_mode="complete_episodes",
        )
        .training(
            train_batch_size=roll.get("train_batch_size", 4000),
            sgd_minibatch_size=roll.get("sgd_minibatch_size", 256),
            num_sgd_iter=roll.get("num_sgd_iter", 10),
            lr=opt.get("lr", 3e-4),
            gamma=opt.get("gamma", 0.99),
            lambda_=opt.get("gae_lambda", 0.95),
            clip_param=opt.get("clip_param", 0.2),
            entropy_coeff=opt.get("entropy_coeff", 0.01),
            vf_loss_coeff=opt.get("vf_loss_coeff", 0.5),
            grad_clip=opt.get("grad_clip", 0.5),
            vf_clip_param=opt.get("vf_clip_param", 10.0),
            use_kl_loss=opt.get("use_kl_loss", True),
            kl_target=opt.get("kl_target", 0.02),
            kl_coeff=opt.get("kl_coeff", 0.3),
        )
        .evaluation(
            evaluation_interval=evaluation_interval,
            evaluation_duration=eval_section.get("episodes_per_map", 5),
            evaluation_duration_unit="episodes",
            evaluation_num_workers=0,
            evaluation_config={"explore": not eval_section.get("deterministic_policy", True)},
        )
        .framework("torch")
    )
