"""
Visualize MAPPO Multi-Agent in CARLA
====================================
Load a MAPPO RLlib checkpoint and run RL vehicles + pedestrians with
visual rendering. The spectator follows the selected agent.

Usage:
    python carla_core/scripts/visualize_mappo_agent.py \
        --checkpoint <experiment_dir_or_checkpoint_path> \
        --difficulty path|traffic|mixed \
        [--level easy|medium|hard]

IMPORTANT: Start CARLA with rendering enabled:
    C:/CARLA_0.9.16/CarlaUE4.exe -quality-level=Medium -windowed -ResX=1280 -ResY=720
"""

import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    signal.signal(signal.SIGABRT, signal.SIG_IGN)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import ray
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import carla
from carla_core.agents.centralized_critic import (
    CentralizedCriticCallbacks,
    CentralizedCriticModel,
    compute_global_obs_dim_with_mask,
)
from carla_core.envs.carla_multi_agent_env import (
    PEDESTRIAN_OBS_DIM,
    VEHICLE_OBS_DIM,
    CarlaMultiAgentEnv,
    apply_level_config,
)


os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


def load_yaml(path):
    p = Path(path)
    return yaml.safe_load(open(p)) if p.exists() else {}


def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _resolve_visualization_artifact_dir(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = _project_root() / path
    path = path.resolve(strict=False)
    if path.exists() and path.is_dir():
        return path
    return path.parent


def _update_visualization_status(status_path, **fields):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    _write_json_atomic(status_path, payload)


def rllib_env_creator(env_config):
    return ParallelPettingZooEnv(CarlaMultiAgentEnv(config=env_config))


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id.startswith("vehicle"):
        return "vehicle_policy"
    if agent_id.startswith("pedestrian"):
        return "pedestrian_policy"
    raise ValueError(f"Unknown agent_id: {agent_id}")


def update_spectator(world, actor, distance=10.0, height=6.0):
    """Follow camera: spectator behind and above the actor."""
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
                color = carla.Color(0, 255, 0)
            elif i == ad.current_wp_idx:
                color = carla.Color(255, 0, 0)
            else:
                color = carla.Color(255, 255, 0)

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


def _set_algo_env_close_mode_for_teardown(algo, mode):
    """Switch existing RLlib env instances to the requested close mode."""
    workers = getattr(algo, "workers", None)
    if workers is None:
        return 0

    def _switch_env_mode(env):
        changed = 0
        seen = set()
        pending = [env]

        while pending:
            current = pending.pop()
            if current is None:
                continue

            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)

            setter = getattr(current, "set_close_mode", None)
            if callable(setter):
                setter(mode)
                changed += 1

            for attr_name in ("par_env", "env", "unwrapped"):
                try:
                    nested = getattr(current, attr_name)
                except Exception:
                    continue
                if nested is None or nested is current:
                    continue
                pending.append(nested)

        return changed

    results = workers.foreach_env(_switch_env_mode)
    return int(sum(sum(worker_results) for worker_results in results if worker_results))


def _new_episode_diag_stats(agent_ids):
    return {
        agent_id: {
            "loop_seen": False,
            "speed_sum_kmh": 0.0,
            "speed_samples": 0,
            "speed_max_kmh": 0.0,
            "steer_abs_sum": 0.0,
            "steer_samples": 0,
            "steer_delta_sum": 0.0,
            "steer_delta_samples": 0,
            "last_steer_raw": None,
        }
        for agent_id in agent_ids
    }


def _update_episode_diag_stats(env, diag_stats):
    for agent_id, ad in env._agent_data.items():
        stats = diag_stats.get(agent_id)
        if stats is None or not ad.actor or not ad.actor.is_alive:
            continue

        vel = ad.actor.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        stats["speed_sum_kmh"] += speed_kmh
        stats["speed_samples"] += 1
        stats["speed_max_kmh"] = max(stats["speed_max_kmh"], speed_kmh)

        ctrl = ad.actor.get_control()
        steer_raw = float(getattr(ctrl, "steer", 0.0) or 0.0)
        stats["steer_abs_sum"] += abs(steer_raw)
        stats["steer_samples"] += 1
        if stats["last_steer_raw"] is not None:
            stats["steer_delta_sum"] += abs(steer_raw - stats["last_steer_raw"])
            stats["steer_delta_samples"] += 1
        stats["last_steer_raw"] = steer_raw

        if bool(getattr(ad, "loop_penalty_active", False)):
            stats["loop_seen"] = True


def _is_loop_like(diag):
    speed_samples = diag.get("speed_samples", 0)
    steer_samples = diag.get("steer_samples", 0)
    steer_delta_samples = diag.get("steer_delta_samples", 0)

    avg_speed_kmh = (
        diag["speed_sum_kmh"] / speed_samples if speed_samples > 0 else 0.0
    )
    avg_abs_steer = (
        diag["steer_abs_sum"] / steer_samples if steer_samples > 0 else 0.0
    )
    avg_steer_delta = (
        diag["steer_delta_sum"] / steer_delta_samples
        if steer_delta_samples > 0
        else 0.0
    )

    return bool(
        diag.get("loop_seen", False)
        and (avg_speed_kmh >= 3.0 or diag.get("speed_max_kmh", 0.0) >= 8.0)
        and (avg_abs_steer >= 0.15 or avg_steer_delta >= 0.08)
    )


def _classify_agent_outcome(final_info, diag):
    termination_reason = str(final_info.get("termination_reason", "") or "").strip().lower()
    if final_info.get("collision") or termination_reason == "collision":
        return "collision"
    if termination_reason == "offroad":
        return "offroad"
    if termination_reason == "route_complete":
        return "destination"
    if _is_loop_like(diag):
        return "loop"
    if termination_reason == "stuck":
        return "stuck"
    if termination_reason == "timeout":
        return "timeout"
    if termination_reason and termination_reason != "alive":
        return termination_reason
    return "warning-ok"


def _build_worker_payload(args):
    artifact_dir = _resolve_visualization_artifact_dir(args.checkpoint)
    runtime_dir = artifact_dir / "_visualize_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    status_path = artifact_dir / "visualization_status.json"
    return {
        "checkpoint": args.checkpoint,
        "episodes": int(args.episodes),
        "no_gpu": bool(args.no_gpu),
        "follow": args.follow,
        "env_config": args.env_config,
        "train_config": args.train_config,
        "level": args.level,
        "difficulty": args.difficulty,
        "map": args.map,
        "npc_vehicles": args.npc_vehicles,
        "npc_pedestrians": args.npc_pedestrians,
        "keep_training_load": bool(args.keep_training_load),
        "artifact_dir": str(artifact_dir),
        "runtime_dir": str(runtime_dir),
        "status_path": str(status_path),
        "session_id": f"viz_{artifact_dir.name}",
    }


def _run_visualization_worker(payload):
    status_path = Path(payload["status_path"])
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    session_id = str(payload.get("session_id") or Path(payload["artifact_dir"]).name)
    total_episodes = int(payload["episodes"])

    _update_visualization_status(
        status_path,
        session_id=session_id,
        pid=os.getpid(),
        completed=False,
        reason="visualization worker running",
        exit_code=None,
        started_at=started_at,
        finished_at=None,
        current_phase="starting",
        completed_episodes=0,
        total_episodes=total_episodes,
        current_episode=0,
        episodes_finished_cleanly=False,
    )

    base = Path(__file__).resolve().parent.parent
    train_cfg = load_yaml(payload.get("train_config") or base / "configs" / "train_mappo.yaml")
    env_cfg = load_yaml(payload.get("env_config") or base / "configs" / "multi_agent.yaml")

    env_cfg.setdefault("world", {})
    if payload.get("difficulty"):
        lv = load_yaml(base / "configs" / "levels.yaml")
        env_cfg["levels"] = lv[f"levels_{payload['difficulty']}"]
    if payload.get("level"):
        apply_level_config(env_cfg, str(payload["level"]).strip().lower())
    env_cfg["world"]["no_rendering"] = False
    if payload.get("map"):
        env_cfg["world"]["map"] = payload["map"]
    env_cfg.setdefault("simulator", {})
    env_cfg["simulator"]["timeout_seconds"] = 30.0
    env_cfg.setdefault("traffic", {})
    if not payload.get("keep_training_load", False):
        env_cfg["traffic"]["n_vehicles_npc"] = 3
        env_cfg["traffic"]["n_pedestrians_npc"] = 3
    if payload.get("npc_vehicles") is not None:
        env_cfg["traffic"]["n_vehicles_npc"] = int(payload["npc_vehicles"])
    if payload.get("npc_pedestrians") is not None:
        env_cfg["traffic"]["n_pedestrians_npc"] = int(payload["npc_pedestrians"])

    ag_cfg = env_cfg.get("agents", {})
    n_veh = ag_cfg.get("n_vehicles_rl", 1)
    n_ped = ag_cfg.get("n_pedestrians_rl", 1)
    global_obs_dim = compute_global_obs_dim_with_mask(n_veh, n_ped)

    model_cfg = train_cfg.get("model", {})
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
        "attention_heads": model_cfg.get("attention_heads", 4),
        "use_gnn": model_cfg.get("use_gnn", False),
        "gnn_embed_dim": model_cfg.get("gnn_embed_dim", 64),
        "gnn_heads": model_cfg.get("gnn_heads", 4),
        "gnn_layers": model_cfg.get("gnn_layers", 2),
    }

    n_gpus = 0 if payload.get("no_gpu", False) else 1
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})

    print(f"{'=' * 60}")
    print(f"MAPPO Visualization - {n_veh}V + {n_ped}P")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {payload['checkpoint']}")
    print(f"  Level: {payload.get('level') or 'base-env'}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Following: {payload['follow']}")
    print(f"  Map: {env_cfg['world'].get('map')}")
    print(
        f"  NPC traffic: {env_cfg['traffic'].get('n_vehicles_npc', 0)}V + "
        f"{env_cfg['traffic'].get('n_pedestrians_npc', 0)}P"
    )
    print(f"  Rendering: {'ON' if not env_cfg['world'].get('no_rendering', True) else 'OFF'}")
    print(f"  global_obs_dim: {global_obs_dim}")
    print(f"{'=' * 60}\n")

    algo = None
    env = None
    ray_started = False
    episodes_completed = 0
    episodes_finished_cleanly = False

    try:
        ray.init(num_cpus=2, num_gpus=n_gpus, log_to_driver=False)
        ray_started = True
        register_env("CarlaMultiAgent-v0", rllib_env_creator)
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

        import gymnasium as gym

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

        config = (
            PPOConfig()
            .environment(env="CarlaMultiAgent-v0", env_config=env_cfg)
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
                vf_clip_param=opt.get("vf_clip_param", 10.0),
                use_kl_loss=opt.get("use_kl_loss", True),
                kl_target=opt.get("kl_target", 0.02),
                kl_coeff=opt.get("kl_coeff", 0.3),
            )
            .framework("torch")
        )

        algo = config.build()
        algo.restore(payload["checkpoint"])
        print("Checkpoint caricato.\n")

        env = CarlaMultiAgentEnv(config=env_cfg)

        for ep in range(total_episodes):
            _update_visualization_status(
                status_path,
                session_id=session_id,
                pid=os.getpid(),
                completed=False,
                reason="visualization worker running",
                exit_code=None,
                started_at=started_at,
                finished_at=None,
                current_phase="episode_running",
                completed_episodes=episodes_completed,
                total_episodes=total_episodes,
                current_episode=ep + 1,
                episodes_finished_cleanly=False,
            )

            obs, infos = env.reset()
            time.sleep(2.0)
            ep_rewards = {a: 0.0 for a in env.possible_agents}
            ep_max_route = {a: 0.0 for a in env.possible_agents}
            ep_final_info = {}
            ep_diag_stats = _new_episode_diag_stats(env.possible_agents)
            step = 0

            print(f"--- Ep {ep + 1}/{total_episodes} ---")

            while env.agents:
                actions = {}
                for agent_id in env.agents:
                    policy_id = policy_mapping_fn(agent_id)
                    actions[agent_id] = algo.compute_single_action(
                        obs[agent_id],
                        policy_id=policy_id,
                        explore=False,
                    )

                obs, rewards, terms, truncs, infos = env.step(actions)
                step += 1
                _update_episode_diag_stats(env, ep_diag_stats)

                for agent_id in rewards:
                    ep_rewards[agent_id] += rewards[agent_id]

                for agent_id in infos:
                    route_completion = infos[agent_id].get("route_completion", 0)
                    ep_max_route[agent_id] = max(ep_max_route[agent_id], route_completion)
                    ep_final_info[agent_id] = infos[agent_id]

                for agent_id, info in env._terminated_agent_infos.items():
                    route_completion = info.get("route_completion", 0)
                    ep_max_route[agent_id] = max(ep_max_route[agent_id], route_completion)
                    ep_final_info[agent_id] = info

                followed = env._agent_data.get(payload["follow"])
                if followed and followed.actor and followed.actor.is_alive and env._world:
                    if str(payload["follow"]).startswith("pedestrian"):
                        update_spectator(env._world, followed.actor, distance=5.0, height=3.0)
                    else:
                        update_spectator(env._world, followed.actor)

                draw_waypoints(env._world, env._agent_data)
                time.sleep(0.03)

            print(f"  Steps: {step}")
            for agent_id in env.possible_agents:
                info = ep_final_info.get(agent_id, {})
                route = ep_max_route.get(agent_id, 0)
                outcome = _classify_agent_outcome(info, ep_diag_stats.get(agent_id, {}))
                print(
                    f"  {agent_id}: R={ep_rewards[agent_id]:+.1f} | "
                    f"route={route:.0%} | {outcome}"
                )

            episodes_completed = ep + 1
            _update_visualization_status(
                status_path,
                session_id=session_id,
                pid=os.getpid(),
                completed=False,
                reason="visualization worker running",
                exit_code=None,
                started_at=started_at,
                finished_at=None,
                current_phase="episode_completed",
                completed_episodes=episodes_completed,
                total_episodes=total_episodes,
                current_episode=episodes_completed,
                episodes_finished_cleanly=False,
            )

        episodes_finished_cleanly = True
        _update_visualization_status(
            status_path,
            session_id=session_id,
            pid=os.getpid(),
            completed=False,
            reason="episodes completed, entering teardown",
            exit_code=None,
            started_at=started_at,
            finished_at=None,
            current_phase="episodes_completed",
            completed_episodes=episodes_completed,
            total_episodes=total_episodes,
            current_episode=episodes_completed,
            episodes_finished_cleanly=True,
        )
    except KeyboardInterrupt:
        print("\nInterrotto.")
        _update_visualization_status(
            status_path,
            session_id=session_id,
            pid=os.getpid(),
            completed=False,
            reason="manual interrupt during visualization",
            exit_code=130,
            started_at=started_at,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            current_phase="interrupted",
            completed_episodes=episodes_completed,
            total_episodes=total_episodes,
            current_episode=episodes_completed,
            episodes_finished_cleanly=episodes_finished_cleanly,
        )
        return 130
    except Exception as exc:
        _update_visualization_status(
            status_path,
            session_id=session_id,
            pid=os.getpid(),
            completed=False,
            reason=f"visualization worker error: {type(exc).__name__}: {exc}",
            exit_code=1,
            started_at=started_at,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            current_phase="failed",
            completed_episodes=episodes_completed,
            total_episodes=total_episodes,
            current_episode=episodes_completed,
            episodes_finished_cleanly=episodes_finished_cleanly,
        )
        raise
    finally:
        _update_visualization_status(
            status_path,
            session_id=session_id,
            pid=os.getpid(),
            completed=False,
            reason="worker teardown in progress",
            exit_code=None,
            started_at=started_at,
            finished_at=None,
            current_phase="teardown",
            completed_episodes=episodes_completed,
            total_episodes=total_episodes,
            current_episode=episodes_completed,
            episodes_finished_cleanly=episodes_finished_cleanly,
        )

        if env is not None:
            try:
                env.set_close_mode("robust")
            except Exception:
                pass

        if algo is not None:
            try:
                _set_algo_env_close_mode_for_teardown(algo, "robust")
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"[WARN] env.close() failed during visualization teardown: {exc}")

        if algo is not None:
            try:
                algo.stop()
            except Exception as exc:
                print(f"[WARN] algo.stop() failed during visualization teardown: {exc}")

        if ray_started:
            try:
                ray.shutdown()
            except Exception as exc:
                print(f"[WARN] ray.shutdown() failed during visualization teardown: {exc}")
        gc.collect()

    _update_visualization_status(
        status_path,
        session_id=session_id,
        pid=os.getpid(),
        completed=True,
        reason=None,
        exit_code=0,
        started_at=started_at,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        current_phase="completed",
        completed_episodes=episodes_completed,
        total_episodes=total_episodes,
        current_episode=episodes_completed,
        episodes_finished_cleanly=True,
    )
    return 0


def _launch_visualization_worker(args):
    payload = _build_worker_payload(args)
    runtime_dir = Path(payload["runtime_dir"])
    job_path = runtime_dir / "visualization_job.json"
    status_path = Path(payload["status_path"])
    _write_json_atomic(
        status_path,
        {
            "session_id": payload["session_id"],
            "pid": os.getpid(),
            "completed": False,
            "reason": "visualization supervisor launching worker",
            "exit_code": None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "current_phase": "launching",
            "completed_episodes": 0,
            "total_episodes": int(payload["episodes"]),
            "current_episode": 0,
            "episodes_finished_cleanly": False,
        },
    )
    _write_json_atomic(job_path, payload)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-job",
        str(job_path),
    ]

    proc = subprocess.run(
        command,
        cwd=str(_project_root()),
        check=False,
    )

    status = _load_json(status_path) if status_path.exists() else {}
    completed_episodes = int(status.get("completed_episodes", 0) or 0)
    total_episodes = int(status.get("total_episodes", 0) or 0)
    phase = str(status.get("current_phase", "") or "")

    if proc.returncode == 0:
        return 0

    if status.get("completed") or (
        completed_episodes >= total_episodes > 0
        and phase in {"episodes_completed", "teardown", "completed"}
    ):
        print(
            "[WARN] Visualization worker terminated during final teardown after "
            "all episodes completed. Session considered successful."
        )
        return 0

    return int(status.get("exit_code", proc.returncode or 1) or 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize MAPPO agents in CARLA")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument(
        "--follow",
        type=str,
        default="vehicle_0",
        help="Agent to follow with the camera (e.g. vehicle_0, pedestrian_1)",
    )
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument(
        "--level",
        type=str,
        choices=("easy", "medium", "hard"),
        default="easy",
        help="Level to visualize within the chosen difficulty set (default: easy)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["path", "traffic", "mixed"],
        required=True,
        help="Level set to use: path (route varies), traffic (NPC varies), mixed (both vary)",
    )
    parser.add_argument("--map", type=str, default=None, help="Override CARLA map (e.g. Town05)")
    parser.add_argument("--npc-vehicles", type=int, default=None, help="Override NPC vehicles")
    parser.add_argument(
        "--npc-pedestrians",
        type=int,
        default=None,
        help="Override NPC pedestrians",
    )
    parser.add_argument(
        "--keep-training-load",
        action="store_true",
        help="Use the NPC load from env-config without automatic reduction",
    )
    parser.add_argument("--worker-job", type=str, default=None)
    args = parser.parse_args()

    if args.worker_job:
        return _run_visualization_worker(_load_json(args.worker_job))

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --worker-job is used")

    return _launch_visualization_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
