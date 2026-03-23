"""
Train MAPPO on CarlaMultiAgentEnv — Multi-Agent via RLlib (v0.1)
================================================================
MAPPO = PPO + centralized critic (CTDE paradigm).

Architecture:
  - vehicle_policy:    actor sees 24D local obs, critic sees global_obs
  - pedestrian_policy: actor sees 18D local obs, critic sees global_obs
  - global_obs = concat(all agent obs) → centralized value function

Components:
  - CarlaMultiAgentEnv (PettingZoo ParallelEnv) → ParallelPettingZooEnv
  - CentralizedCriticModel (custom TorchModelV2)
  - CentralizedCriticCallbacks (injects global_obs, recomputes GAE)

Uso:
    python carla_core/training/train_carla_mappo.py
    python carla_core/training/train_carla_mappo.py --timesteps 50000 --workers 0
    python carla_core/training/train_carla_mappo.py --no-gpu --timesteps 10000
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch
import yaml

# Suppress CARLA SIGABRT on Windows
if sys.platform == "win32":
    signal.signal(signal.SIGABRT, signal.SIG_IGN)

import numpy as np
import random
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from carla_core.envs.carla_multi_agent_env import (
    CarlaMultiAgentEnv,
    VEHICLE_OBS_DIM,
    PEDESTRIAN_OBS_DIM,
)
from carla_core.agents.centralized_critic import (
    CentralizedCriticModel,
    CentralizedCriticCallbacks,
)

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    p = Path(path)
    return yaml.safe_load(open(p)) if p.exists() else {}


def rllib_env_creator(env_config):
    """Wrap CarlaMultiAgentEnv for RLlib via ParallelPettingZooEnv."""
    raw_env = CarlaMultiAgentEnv(config=env_config)
    return ParallelPettingZooEnv(raw_env)


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """vehicle_* → vehicle_policy, pedestrian_* → pedestrian_policy."""
    if agent_id.startswith("vehicle"):
        return "vehicle_policy"
    elif agent_id.startswith("pedestrian"):
        return "pedestrian_policy"
    raise ValueError(f"Unknown agent_id: {agent_id}")


def compute_global_obs_dim(n_vehicles, n_pedestrians):
    """global_obs = concat all agent obs."""
    return n_vehicles * VEHICLE_OBS_DIM + n_pedestrians * PEDESTRIAN_OBS_DIM


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAPPO Training on CARLA")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--env-config", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    # --- Load configs ---
    train_cfg = load_yaml(args.train_config or base / "configs" / "train_mappo.yaml")
    env_cfg = load_yaml(args.env_config or base / "configs" / "multi_agent.yaml")

    sched = train_cfg.get("schedule", {})
    res = train_cfg.get("resources", {})
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})
    model_cfg = train_cfg.get("model", {})

    total_ts = args.timesteps or sched.get("total_timesteps", 50_000)
    n_workers = args.workers if args.workers is not None else res.get("num_workers", 0)
    
    if n_workers > 0:
        print("[WARNING] num_workers > 0 requires separate CARLA instances per worker.")
        print("          Forcing num_workers = 0 (single CARLA instance).")
        n_workers = 0

    exp_seed = train_cfg.get("experiment", {}).get("seed", 42)
    env_cfg.setdefault("traffic", {})
    env_cfg["traffic"]["seed"] = exp_seed

    n_gpus = 0 if args.no_gpu else res.get("num_gpus", 1)
    batch_size = roll.get("train_batch_size", 4000)
    ckpt_freq = sched.get("checkpoint_freq", 10_000)

    # Agent counts
    ag_cfg = env_cfg.get("agents", {})
    n_veh = ag_cfg.get("n_vehicles_rl", 1)
    n_ped = ag_cfg.get("n_pedestrians_rl", 1)
    global_obs_dim = compute_global_obs_dim(n_veh, n_ped)

    # Wire remaining config fields
    exp_cfg = train_cfg.get("experiment", {})
    out_base = exp_cfg.get("output_dir", str(base / "experiments"))

    # Output dir
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    name = exp_cfg.get("name", "carla_mappo")
    out_dir = args.checkpoint_dir or str(Path(out_base) / f"{name}_{ts_str}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"CARLA MAPPO Training — Centralized Critic (CTDE)")
    print(f"{'=' * 60}")
    print(f"  Agents: {n_veh}V + {n_ped}P | global_obs: {global_obs_dim}D")
    print(f"  Budget: {total_ts:,} steps | Workers: {n_workers} | GPU: {n_gpus}")
    print(f"  Policies: vehicle({VEHICLE_OBS_DIM}D), pedestrian({PEDESTRIAN_OBS_DIM}D)")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}\n")

    # --- Ray init ---
    ray.init(num_cpus=max(n_workers + 2, 2), num_gpus=n_gpus, log_to_driver=False)

    torch.manual_seed(exp_seed)
    np.random.seed(exp_seed)
    random.seed(exp_seed)

    # --- Register ---
    register_env("CarlaMultiAgent-v0", rllib_env_creator)
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    # --- Spaces (per-policy, single-agent) ---
    import gymnasium as gym
    veh_obs = gym.spaces.Box(-1, 1, (VEHICLE_OBS_DIM,), np.float32)
    veh_act = gym.spaces.Box(np.array([-1, -1], dtype=np.float32),
                              np.array([1, 1], dtype=np.float32))
    ped_obs = gym.spaces.Box(-1, 1, (PEDESTRIAN_OBS_DIM,), np.float32)
    ped_act = gym.spaces.Box(np.array([0, -1], dtype=np.float32),
                              np.array([1, 1], dtype=np.float32))

    hidden_size = model_cfg.get("hidden_size", 256)
    n_hidden = model_cfg.get("n_hidden_layers", 2)
    cc_config = {
        "hidden_size": hidden_size,
        "n_hidden_layers": n_hidden,
        "global_obs_dim": global_obs_dim,
    }

    # --- PPO Config ---
    config = (
        PPOConfig()
        .environment(env="CarlaMultiAgent-v0", env_config=env_cfg)
        .debugging(seed=exp_seed)
        .multi_agent(
            policies={
                "vehicle_policy": (
                    None, veh_obs, veh_act,
                    {"model": {"custom_model": "cc_model",
                               "custom_model_config": cc_config}},
                ),
                "pedestrian_policy": (
                    None, ped_obs, ped_act,
                    {"model": {"custom_model": "cc_model",
                               "custom_model_config": cc_config}},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["vehicle_policy", "pedestrian_policy"],
        )
        .callbacks(CentralizedCriticCallbacks)
        .resources(num_gpus=n_gpus)
        .rollouts(
            num_rollout_workers=n_workers,
            rollout_fragment_length=roll.get("rollout_fragment_length", 200),
        )
        .training(
            train_batch_size=batch_size,
            sgd_minibatch_size=roll.get("sgd_minibatch_size", 256),
            num_sgd_iter=roll.get("num_sgd_iter", 10),
            lr=opt.get("lr", 3e-4),
            gamma=opt.get("gamma", 0.99),
            lambda_=opt.get("gae_lambda", 0.95),
            clip_param=opt.get("clip_param", 0.2),
            entropy_coeff=opt.get("entropy_coeff", 0.01),
            vf_loss_coeff=opt.get("vf_loss_coeff", 0.5),
            grad_clip=opt.get("grad_clip", 0.5),
            use_kl_loss=opt.get("use_kl_loss", True)
        )
        .evaluation(
            evaluation_interval=max(1, int(sched.get("eval_freq", 10_000) / batch_size)),
            evaluation_duration=5,
            evaluation_duration_unit="episodes",
            evaluation_num_workers=0,
        )
        .framework("torch")
    )

    # --- Build & Train ---
    algo = config.build()
    print("MAPPO training avviato.\n")

    ts_done = 0
    iteration = 0
    t0 = time.time()
    best_reward = float("-inf")

    try:
        while ts_done < total_ts:
            result = algo.train()
            iteration += 1
            ts_done = result.get("timesteps_total", 0)

            # Per-policy rewards
            pol_rew = result.get("policy_reward_mean", {})
            veh_r = pol_rew.get("vehicle_policy", 0)
            ped_r = pol_rew.get("pedestrian_policy", 0)
            tot_r = result.get("episode_reward_mean", 0)
            ep_len = result.get("episode_len_mean", 0)
            eps = result.get("episodes_total", 0)

            elapsed = time.time() - t0
            pct = ts_done / total_ts * 100
            eta = (elapsed / max(ts_done, 1)) * (total_ts - ts_done)

            bar_len = 30
            filled = int(bar_len * ts_done / total_ts)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

            print(
                f"  [{bar}] {pct:5.1f}% | "
                f"{ts_done:,}/{total_ts:,} | "
                f"R:{tot_r:+.1f} (V:{veh_r:+.1f} P:{ped_r:+.1f}) | "
                f"Len:{ep_len:.0f} | Eps:{eps} | "
                f"ST: {int(elapsed//3600)}h{int((elapsed%3600)//60):02d}m / ETA: {int(eta//3600)}h{int((eta%3600)//60):02d}m"
            )

            if tot_r > best_reward:
                best_reward = tot_r

            if ts_done % ckpt_freq < batch_size:
                ckpt = algo.save(out_dir)
                print(f"    -> Checkpoint: {ckpt}")

    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        try:
            final = algo.save(out_dir)
            print(f"\nCheckpoint finale: {final}")
        except Exception as e:
            print(f"\nCheckpoint fallito: {e}")

        print(f"\n{'=' * 60}")
        print(f"MAPPO Training Completato")
        print(f"{'=' * 60}")
        print(f"  Steps: {ts_done:,} | Best reward: {best_reward:.1f}")
        print(f"  Tempo: {(time.time() - t0) / 60:.1f} min")
        print(f"  Output: {out_dir}")

        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
