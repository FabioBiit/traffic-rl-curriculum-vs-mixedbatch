"""
Test CarlaEnv v0.1 — Verifica wrapper Gymnasium
================================================
Lancia 3 episodi con azioni random, stampa stats per episodio.

Prerequisiti:
    - Server CARLA in esecuzione
    - pip install carla==0.9.16 gymnasium pyyaml

Uso:
    python carla_core/scripts/test_carla_env.py
    python carla_core/scripts/test_carla_env.py --episodes 5 --max-steps 200
"""

import argparse
import sys
import time

try:
    import gymnasium as gym
    import numpy as np
except ImportError as e:
    print(f"[ERRORE] {e}")
    sys.exit(1)

# Import wrapper
sys.path.insert(0, ".")
from carla_core.envs.carla_env import CarlaEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    override = {"episode": {"max_steps": args.max_steps}}
    env = CarlaEnv(config=override, config_path=args.config)

    print(f"Obs space: {env.observation_space}")
    print(f"Act space: {env.action_space}")
    print(f"Max steps: {args.max_steps}\n")

    total_steps = 0
    t_start = time.time()

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

            total_steps += ep_steps
            reason = "collision" if info.get("collision") else \
                     "route_done" if info.get("route_completion", 0) >= 1.0 else \
                     "truncated"

            print(f"Ep {ep+1}/{args.episodes} | "
                  f"Steps: {ep_steps} | Reward: {ep_reward:.1f} | "
                  f"Route: {info.get('route_completion',0):.0%} | "
                  f"End: {reason}")

    finally:
        env.close()

    elapsed = time.time() - t_start
    fps = total_steps / elapsed if elapsed > 0 else 0
    print(f"\nTotale: {total_steps} steps in {elapsed:.1f}s ({fps:.1f} steps/s)")
    print("[OK] CarlaEnv v0.1 funziona.")


if __name__ == "__main__":
    main()
