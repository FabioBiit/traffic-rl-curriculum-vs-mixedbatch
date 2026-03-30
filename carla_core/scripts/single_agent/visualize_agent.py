"""
Visualize Trained Agent in CARLA
================================
Carica checkpoint RLlib e fa girare l'agente con rendering visivo.

Uso:
    python carla_core/scripts/visualize_agent.py --checkpoint <path>
    python carla_core/scripts/visualize_agent.py --checkpoint carla_core/experiments/carla_ppo_20260319/checkpoint_000010
    
    NOTE: --checkpoint expects the checkpoint subdirectory path, not the experiment root.

IMPORTANTE: Avvia CARLA CON rendering (senza -RenderOffScreen):
    C:/CARLA_0.9.16/CarlaUE4.exe -quality-level=Medium -windowed -ResX=1280 -ResY=720
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import carla
from carla_core.envs.carla_env import CarlaEnv


def env_creator(env_config):
    return CarlaEnv(config=env_config)


def update_spectator(world, ego, distance=8.0, height=5.0):
    """Move spectator camera behind and above the ego vehicle."""
    ego_transform = ego.get_transform()
    fwd = ego_transform.get_forward_vector()
    loc = ego_transform.location

    # Camera position: behind and above
    cam_loc = carla.Location(
        x=loc.x - fwd.x * distance,
        y=loc.y - fwd.y * distance,
        z=loc.z + height,
    )
    # Look down toward the vehicle
    pitch = -math.degrees(math.atan2(height, distance))
    cam_rot = carla.Rotation(pitch=pitch, yaw=ego_transform.rotation.yaw)

    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    # Env config: rendering ON, no_rendering OFF
    env_cfg = {
        "world": {"no_rendering": False},
        "traffic": {"n_vehicles": 15, 
                    "n_pedestrians": 20, 
                    "persist_traffic": True},
        "episode": {"max_steps": 1500},
    }

    ray.init(num_cpus=2, num_gpus=0 if args.no_gpu else 1, log_to_driver=False)
    register_env("CarlaEnv-v0", env_creator)

    config = (
        PPOConfig()
        .environment(env="CarlaEnv-v0", env_config=env_cfg)
        .resources(num_gpus=0 if args.no_gpu else 1)
        .rollouts(num_rollout_workers=0)
        .framework("torch")
    )

    algo = config.build()
    algo.restore(args.checkpoint)
    print(f"Checkpoint caricato: {args.checkpoint}\n")

    # Crea env locale per visualizzazione
    env = CarlaEnv(config=env_cfg)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            print(f"--- Ep {ep+1}/{args.episodes} avviato ---")

            while not done:
                action = algo.compute_single_action(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

                # Follow camera: spectator segue l'ego
                if env._ego and env._world:
                    update_spectator(env._world, env._ego)

                time.sleep(0.03)  # ~30 FPS visivo

            reason = "collision" if info.get("collision") else \
                     "route_done" if info.get("route_completion", 0) >= 1.0 else \
                     "truncated"
            print(f"Ep {ep+1}/{args.episodes} | "
                  f"Steps: {ep_steps} | Reward: {ep_reward:.1f} | "
                  f"Route: {info.get('route_completion',0):.0%} | "
                  f"End: {reason}")

    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        env.close()
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()