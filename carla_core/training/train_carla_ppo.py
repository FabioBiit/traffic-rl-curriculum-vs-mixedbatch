"""
Train PPO on CarlaEnv — Single Agent via RLlib
===============================================
Primo training single-agent. Legge config da train.yaml + env.yaml.

Prerequisiti:
    - Server CARLA in esecuzione (CarlaUE4.exe -quality-level=Low)
    - pip install carla==0.9.16 "ray[rllib]==2.10.0" gymnasium pyyaml

Uso:
    python carla_core/training/train_carla_ppo.py
    python carla_core/training/train_carla_ppo.py --timesteps 50000 --workers 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from carla_core.envs.carla_env import CarlaEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def env_creator(env_config: dict):
    """Factory for RLlib. Passes env_config as override to CarlaEnv."""
    return CarlaEnv(config=env_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps from train.yaml")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override num_workers (0=local only)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    # Load configs
    base = Path(__file__).resolve().parent.parent
    train_cfg = load_yaml(base / "configs" / "train.yaml")
    env_cfg = load_yaml(base / "configs" / "env.yaml")

    # Extract params with overrides
    sched = train_cfg.get("schedule", {})
    res = train_cfg.get("resources", {})
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})

    total_timesteps = args.timesteps or sched.get("total_timesteps", 200_000)
    num_workers = args.workers if args.workers is not None else res.get("num_workers", 1)
    num_gpus = 0 if args.no_gpu else res.get("num_gpus", 0)
    checkpoint_freq = sched.get("checkpoint_freq", 20_000)
    eval_freq = sched.get("eval_freq", 10_000)

    # Output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = train_cfg.get("experiment", {}).get("name", "carla_ppo")
    output_dir = args.checkpoint_dir or str(
        base / "experiments" / f"{exp_name}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*50}")
    print(f"CARLA PPO Training — Single Agent")
    print(f"{'='*50}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Workers: {num_workers} | GPUs: {num_gpus}")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}\n")

    # Init Ray
    ray.init(
        num_cpus=max(num_workers + 1, 2),
        num_gpus=num_gpus,
        log_to_driver=True,
    )

    # Register env
    register_env("CarlaEnv-v0", env_creator)

    # Build config
    # IMPORTANT: con CARLA, num_workers=0 è più sicuro per il primo test
    # perché ogni worker apre una connessione al server.
    # Con workers>0, serve un server CARLA per worker OPPURE port multiplexing.
    config = (
        PPOConfig()
        .environment(
            env="CarlaEnv-v0",
            env_config=env_cfg,  # Passa env.yaml come override
        )
        .resources(
            num_gpus=num_gpus,
        )
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=roll.get("rollout_fragment_length", 200),
            # Se workers=0, il driver fa il rollout
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
        )
        .evaluation(
            evaluation_interval=max(1, eval_freq // roll.get("train_batch_size", 4000)),
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_num_workers=0,
        )
        .framework("torch")
    )

    algo = config.build()
    print("Algoritmo costruito. Inizio training...\n")

    # Training loop
    timesteps_done = 0
    iteration = 0
    t_start = time.time()
    best_reward = float("-inf")

    try:
        while timesteps_done < total_timesteps:
            result = algo.train()
            iteration += 1
            timesteps_done = result.get("timesteps_total", 0)
            ep_reward = result.get("episode_reward_mean", 0)
            ep_len = result.get("episode_len_mean", 0)
            eps = result.get("episodes_total", 0)

            elapsed = time.time() - t_start
            eta = (elapsed / max(timesteps_done, 1)) * (total_timesteps - timesteps_done)

            print(f"[Iter {iteration}] "
                  f"Steps: {timesteps_done:,}/{total_timesteps:,} | "
                  f"Reward: {ep_reward:.1f} | Len: {ep_len:.0f} | "
                  f"Eps: {eps} | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

            # Checkpoint
            if timesteps_done % checkpoint_freq < roll.get("train_batch_size", 4000):
                ckpt_path = algo.save(output_dir)
                print(f"  Checkpoint: {ckpt_path}")

            # Track best
            if ep_reward > best_reward:
                best_reward = ep_reward

    except KeyboardInterrupt:
        print("\nTraining interrotto.")
    finally:
        # Final save
        final_path = algo.save(output_dir)
        print(f"\nCheckpoint finale: {final_path}")
        print(f"Best reward: {best_reward:.1f}")
        print(f"Timesteps totali: {timesteps_done:,}")

        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
