"""
Train PPO on CarlaEnv — Single Agent via RLlib (v0.2)
=====================================================
Changelog v0.2:
  - Progress bar (tqdm)
  - Ridotto NPC default (10V+10P) per RTX 3080 16GB
  - SIGABRT handler per Windows/CARLA cleanup crash
  - persist_traffic=True (NPC non ri-spawnati ogni episodio)

Uso:
    python carla_core/training/train_carla_ppo.py --timesteps 50000 --workers 0
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch
import numpy as np
import random
import yaml

# Suppress CARLA SIGABRT on Windows before any carla import
if sys.platform == "win32":
    signal.signal(signal.SIGABRT, signal.SIG_IGN)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from carla_core.envs.carla_env import CarlaEnv

# Suppress Ray deprecation warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


def load_yaml(path):
    p = Path(path)
    return yaml.safe_load(open(p)) if p.exists() else {}


def env_creator(env_config):
    return CarlaEnv(config=env_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    train_cfg = load_yaml(base / "configs" / "train.yaml")
    env_cfg = load_yaml(base / "configs" / "env.yaml")

    # NPC count from env.yaml, persist across episodes
    env_cfg.setdefault("traffic", {})
    env_cfg["traffic"]["persist_traffic"] = True

    sched = train_cfg.get("schedule", {})
    res = train_cfg.get("resources", {})
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})
    exp_cfg = train_cfg.get("experiment", {})
    exp_seed = exp_cfg.get("seed", 42)
    env_cfg.setdefault("traffic", {})
    env_cfg["traffic"]["seed"] = exp_seed

    total_ts = args.timesteps or sched.get("total_timesteps", 200_000)
    n_workers = args.workers if args.workers is not None else res.get("num_workers", 0)
    if n_workers > 0:
        print("[WARNING] num_workers > 0 requires separate CARLA instances per worker.")
        print("          Forcing num_workers = 0 (single CARLA instance).")
        n_workers = 0
    n_gpus = 0 if args.no_gpu else res.get("num_gpus", 0)
    batch_size = roll.get("train_batch_size", 4000)
    ckpt_freq = sched.get("checkpoint_freq", 20_000)

    out_base = exp_cfg.get("output_dir", str(base / "experiments"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = exp_cfg.get("name", "carla_ppo")
    out_dir = args.checkpoint_dir or str(Path(out_base) / f"{name}_{ts}")

    print(f"{'='*50}")
    print(f"CARLA PPO Training — Single Agent")
    print(f"{'='*50}")
    print(f"  Budget: {total_ts:,} steps | Workers: {n_workers} | GPU: {n_gpus}")
    print(f"  NPC: {env_cfg['traffic']['n_vehicles']}V + {env_cfg['traffic']['n_pedestrians']}P")
    print(f"  Output: {out_dir}")
    print(f"{'='*50}\n")

    ray.init(num_cpus=max(n_workers + 1, 2), num_gpus=n_gpus, log_to_driver=False)
    register_env("CarlaEnv-v0", env_creator)

    torch.manual_seed(exp_seed)
    np.random.seed(exp_seed)
    random.seed(exp_seed)

    config = (
        PPOConfig()
        .environment(env="CarlaEnv-v0", env_config=env_cfg)
        .debugging(seed=exp_seed)
        .resources(num_gpus=n_gpus)
        .rollouts(num_rollout_workers=n_workers,
                  rollout_fragment_length=roll.get("rollout_fragment_length", 200))
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
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=5,
            evaluation_duration_unit="episodes",
            evaluation_num_workers=0,
        )
        .framework("torch")
        .debugging(seed=exp_seed)
    )

    algo = config.build()
    print("Training avviato.\n")

    ts_done = 0
    iteration = 0
    t0 = time.time()
    best_reward = float("-inf")

    try:
        while ts_done < total_ts:
            result = algo.train()
            iteration += 1
            ts_done = result.get("timesteps_total", 0)
            rew = result.get("episode_reward_mean", 0)
            ep_len = result.get("episode_len_mean", 0)
            eps = result.get("episodes_total", 0)

            elapsed = time.time() - t0
            pct = ts_done / total_ts * 100
            eta = (elapsed / max(ts_done, 1)) * (total_ts - ts_done)

            # Progress bar style output
            bar_len = 30
            filled = int(bar_len * ts_done / total_ts)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(f"  [{bar}] {pct:5.1f}% | "
                  f"{ts_done:,}/{total_ts:,} | "
                  f"R:{rew:+.1f} | Len:{ep_len:.0f} | "
                  f"Eps:{eps} | "
                  f"{elapsed/60:.1f}m / ETA {eta/60:.1f}m")

            if rew > best_reward:
                best_reward = rew

            if ts_done % ckpt_freq < batch_size:
                algo.save(out_dir)

    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        try:
            final = algo.save(out_dir)
            print(f"\nCheckpoint: {final}")
        except Exception:
            pass
        print(f"Best reward: {best_reward:.1f} | Steps: {ts_done:,}")
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
