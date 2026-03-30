"""
Visualize MAPPO Multi-Agent in CARLA
=====================================
Carica checkpoint MAPPO RLlib e fa girare veicoli + pedoni RL
con rendering visivo. Spectator segue vehicle_0.

Uso:
    python carla_core/scripts/visualize_mappo_agent.py --checkpoint <experiment_dir_or_checkpoint_path>

    NOTE: --checkpoint is passed directly to algo.restore().
          RLlib 2.10 accepts both experiment directories and checkpoint subdirectory paths.
    
IMPORTANTE: Avvia CARLA CON rendering (senza -RenderOffScreen):
    C:/CARLA_0.9.16/CarlaUE4.exe -quality-level=Medium -windowed -ResX=1280 -ResY=720
"""

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    signal.signal(signal.SIGABRT, signal.SIG_IGN)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import yaml
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import carla
from carla_core.envs.carla_multi_agent_env import (
    CarlaMultiAgentEnv,
    VEHICLE_OBS_DIM,
    PEDESTRIAN_OBS_DIM,
)
from carla_core.agents.centralized_critic import (
    CentralizedCriticModel,
    CentralizedCriticCallbacks,
    compute_global_obs_dim_with_mask
)


os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    p = Path(path)
    return yaml.safe_load(open(p)) if p.exists() else {}


def rllib_env_creator(env_config):
    return ParallelPettingZooEnv(CarlaMultiAgentEnv(config=env_config))


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id.startswith("vehicle"):
        return "vehicle_policy"
    return "pedestrian_policy"


def compute_global_obs_dim(n_veh, n_ped):
    return compute_global_obs_dim_with_mask(n_veh, n_ped)


def update_spectator(world, actor, distance=10.0, height=6.0):    
    """Follow camera: spectator dietro e sopra l'attore."""
    t = actor.get_transform()
    fwd = t.get_forward_vector()
    loc = t.location
    cam_loc = carla.Location(
        x=loc.x - fwd.x * distance,
        y=loc.y - fwd.y * distance,
        z=loc.z + height,
    )
    pitch = -math.degrees(math.atan2(height, distance))
    world.get_spectator().set_transform(
        carla.Transform(cam_loc, carla.Rotation(pitch=pitch, yaw=t.rotation.yaw))
    )


def draw_waypoints(world, agent_data):
    """Draw route waypoints for all agents in CARLA debug view."""
    for ad in agent_data.values():
        for i, wp in enumerate(ad.route_waypoints):
            loc = wp.transform.location
            if i < ad.current_wp_idx:
                color = carla.Color(0, 255, 0)       # green — reached
            elif i == ad.current_wp_idx:
                color = carla.Color(255, 0, 0)        # red — current target
            else:
                color = carla.Color(255, 255, 0)      # yellow — future

            world.debug.draw_point(
                loc + carla.Location(z=0.5),
                size=0.15,
                color=color,
                life_time=0.1,
            )
            world.debug.draw_string(
                loc + carla.Location(z=1.0),
                f"{i}",
                draw_shadow=False,
                color=color,
                life_time=0.1,
            )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize MAPPO agents in CARLA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--follow", type=str, default="vehicle_0",
                        help="Agent da seguire con la camera (es: vehicle_0, pedestrian_1)")
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    # --- Configs ---
    train_cfg = load_yaml(args.train_config or base / "configs" / "train_mappo.yaml")
    env_cfg = load_yaml(args.env_config or base / "configs" / "multi_agent.yaml")

    # Override: rendering ON + riduzione carico per stabilità streaming
    env_cfg.setdefault("world", {})
    env_cfg["world"]["no_rendering"] = False
    env_cfg.setdefault("simulator", {})
    env_cfg["simulator"]["timeout_seconds"] = 30.0
    env_cfg.setdefault("traffic", {})
    env_cfg["traffic"]["n_vehicles_npc"] = 3
    env_cfg["traffic"]["n_pedestrians_npc"] = 3

    ag_cfg = env_cfg.get("agents", {})
    n_veh = ag_cfg.get("n_vehicles_rl", 1)
    n_ped = ag_cfg.get("n_pedestrians_rl", 1)
    global_obs_dim = compute_global_obs_dim(n_veh, n_ped)

    model_cfg = train_cfg.get("model", {})
    hidden_size = model_cfg.get("hidden_size", 256)
    n_hidden = model_cfg.get("n_hidden_layers", 2)
    cc_config = {
        "hidden_size": hidden_size,
        "n_hidden_layers": n_hidden,
        "global_obs_dim": global_obs_dim,
    }

    n_gpus = 0 if args.no_gpu else 1
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})

    print(f"{'=' * 60}")
    print(f"MAPPO Visualization — {n_veh}V + {n_ped}P")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Following: {args.follow}")
    print(f"  global_obs_dim: {global_obs_dim}")
    print(f"{'=' * 60}\n")

    # --- Ray + registrations ---
    ray.init(num_cpus=2, num_gpus=n_gpus, log_to_driver=False)
    register_env("CarlaMultiAgent-v0", rllib_env_creator)
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    # --- Spaces ---
    import gymnasium as gym
    veh_obs = gym.spaces.Box(-1, 1, (VEHICLE_OBS_DIM,), np.float32)
    veh_act = gym.spaces.Box(np.array([-1, -1], dtype=np.float32),
                              np.array([1, 1], dtype=np.float32))
    ped_obs = gym.spaces.Box(-1, 1, (PEDESTRIAN_OBS_DIM,), np.float32)
    ped_act = gym.spaces.Box(np.array([0, -1], dtype=np.float32),
                              np.array([1, 1], dtype=np.float32))

    # --- PPO Config (must match training config) ---
    config = (
        PPOConfig()
        .environment(env="CarlaMultiAgent-v0", env_config=env_cfg)
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
            num_rollout_workers=0,
            rollout_fragment_length=roll.get("rollout_fragment_length", 200),
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
        .framework("torch")
    )

    # --- Build + Restore ---
    algo = config.build()
    algo.restore(args.checkpoint)
    print("Checkpoint caricato.\n")

    # --- Env locale per visualizzazione ---
    env = CarlaMultiAgentEnv(config=env_cfg)

    try:
        for ep in range(args.episodes):
            obs, infos = env.reset()
            time.sleep(2.0)  # Stabilizzazione server dopo spawn
            ep_rewards = {a: 0.0 for a in env.possible_agents}
            ep_max_route = {a: 0.0 for a in env.possible_agents}
            ep_final_info = {}
            step = 0

            print(f"--- Ep {ep + 1}/{args.episodes} ---")

            while env.agents:
                actions = {}
                for agent_id in env.agents:
                    pid = policy_mapping_fn(agent_id)
                    actions[agent_id] = algo.compute_single_action(
                        obs[agent_id], policy_id=pid, explore=False
                    )

                obs, rewards, terms, truncs, infos = env.step(actions)
                step += 1

                for a in rewards:
                    ep_rewards[a] += rewards[a]

                for a in infos:
                    rc = infos[a].get("route_completion", 0)
                    ep_max_route[a] = max(ep_max_route[a], rc)
                    ep_final_info[a] = infos[a]

                for a, info in env._terminated_agent_infos.items():
                    rc = info.get("route_completion", 0)
                    ep_max_route[a] = max(ep_max_route[a], rc)
                    ep_final_info[a] = info

                # Spectator segue l'agente scelto con --follow
                followed = env._agent_data.get(args.follow)
                if followed and followed.actor and followed.actor.is_alive and env._world:
                    # Pedoni: camera più vicina e bassa
                    if args.follow.startswith("pedestrian"):
                        update_spectator(env._world, followed.actor, distance=5.0, height=3.0)
                    else:
                        update_spectator(env._world, followed.actor)

                # Draw WP for all agents
                draw_waypoints(env._world, env._agent_data)

                time.sleep(0.03)  # ~30 FPS

            print(f"  Steps: {step}")
            for a in env.possible_agents:
                info = ep_final_info.get(a, {})
                collision = info.get("collision", False)
                route = ep_max_route.get(a, 0)
                print(f"  {a}: R={ep_rewards[a]:+.1f} | "
                      f"route={route:.0%} | "
                      f"{'COLLISION' if collision else 'ok'}")

    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        env.close()
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
