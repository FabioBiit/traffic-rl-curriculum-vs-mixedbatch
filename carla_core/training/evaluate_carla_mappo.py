"""
Evaluate CARLA MAPPO agents - Final evaluation via RLlib (v0.1)
===============================================================
Restores a trained MAPPO checkpoint and runs deterministic evaluation on
CarlaMultiAgentEnv across multiple CARLA maps and traffic profiles.

Architecture:
  - vehicle_policy:    actor sees vehicle local obs, critic sees fixed-slot global_obs
  - pedestrian_policy: actor sees pedestrian local obs, critic sees fixed-slot global_obs
  - global_obs = concat of all current per-agent obs slots + alive_mask

Evaluation flow:
  - parent process supervises scenarios/episodes, progress, retries, and artifacts
  - child process runs one evaluation episode:
      build -> restore -> evaluate -> save result -> stop
  - subprocess isolation contains libcarla crashes without aborting the whole eval

Components:
  - CarlaMultiAgentEnv -> ParallelPettingZooEnv
  - CentralizedCriticModel
  - shared MAPPO runtime builder
  - final_eval_job.json handoff from training

Usage:
    python -m carla_core.training.evaluate_carla_mappo --job "carla_core/experiments/<run>/final_eval_job.json"
"""

import argparse
import csv
import gc
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path

import carla
import numpy as np
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from carla_core.agents.centralized_critic import CentralizedCriticModel
from carla_core.envs.carla_multi_agent_env import apply_level_config, get_level_configs
from carla_core.training.mappo_runtime import (
    _build_mappo_config,
    rllib_env_creator,
    shutdown_carla_processes,
)

logger = logging.getLogger(__name__)


class FinalEvaluationInterrupted(KeyboardInterrupt):
    """KeyboardInterrupt carrying partial final-evaluation artifacts."""

    def __init__(self, *, raw, summary, traces, metric_keys, reason):
        super().__init__(reason)
        self.raw = raw
        self.summary = summary
        self.traces = traces
        self.metric_keys = metric_keys
        self.reason = reason


class FinalEvaluationFailed(RuntimeError):
    """Exception carrying partial final-evaluation artifacts on runtime failure."""

    def __init__(self, *, raw, summary, traces, metric_keys, reason):
        super().__init__(reason)
        self.raw = raw
        self.summary = summary
        self.traces = traces
        self.metric_keys = metric_keys
        self.reason = reason


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, type):
        return str(obj)
    if callable(obj):
        return f"<function {getattr(obj, '__name__', str(obj))}>"
    return str(obj)


def _get_nested(mapping, *keys):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _coerce_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _profile_name_to_level(profile_name):
    name = str(profile_name).strip().lower()
    if name in {"low", "easy"}:
        return "easy"
    if name == "medium":
        return "medium"
    if name in {"high", "hard"}:
        return "hard"
    return None


def _mean_metric_dicts(rows, keys):
    if not rows:
        return {}
    merged = {}
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        merged[key] = float(np.mean(vals)) if vals else None
    return merged


def _expand_eval_entries(scenario_cfg):
    entries = []
    for idx, entry in enumerate(scenario_cfg.get("entries", []) or []):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid eval scenario entry at index {idx}: expected dict")

        entry_name = str(entry.get("name") or f"scenario_{idx + 1}")
        level_name = entry.get("level")
        summary_level = str(entry.get("summary_level") or level_name or entry_name)
        maps = []
        if entry.get("map") is not None:
            maps.append(str(entry["map"]))
        for map_name in entry.get("maps", []) or []:
            maps.append(str(map_name))

        if not maps:
            raise ValueError(
                f"Eval scenario '{entry_name}' must define 'map' or 'maps'."
            )

        for map_name in maps:
            scenario_name = entry_name if len(maps) == 1 else f"{entry_name}_{map_name}"
            entries.append(
                {
                    "scenario_name": scenario_name,
                    "map_name": map_name,
                    "level_name": level_name,
                    "summary_level": summary_level,
                    "overrides": {
                        "n_vehicles_npc": entry.get("n_vehicles_npc"),
                        "n_pedestrians_npc": entry.get("n_pedestrians_npc"),
                        "route_distance_m": entry.get("route_distance_m"),
                        "route_distance_m_pedestrian": entry.get(
                            "route_distance_m_pedestrian"
                        ),
                    },
                }
            )
    return entries


def _build_scenario_env_cfg(
    *,
    base_env_cfg,
    map_name,
    level_name,
    level_configs,
    limits,
    seed_base,
    reset_count,
    overrides=None,
):
    scenario_env_cfg = deepcopy(base_env_cfg)
    scenario_env_cfg.setdefault("traffic", {})
    scenario_env_cfg["traffic"]["seed"] = int(seed_base)
    scenario_env_cfg.setdefault("runtime", {})
    scenario_env_cfg["runtime"]["close_mode"] = "robust"

    if level_name:
        apply_level_config(
            scenario_env_cfg,
            level_name,
            level_configs=level_configs,
            reset_count=reset_count,
        )

    scenario_env_cfg.setdefault("world", {})
    scenario_env_cfg["world"]["map"] = str(map_name)
    scenario_env_cfg.setdefault("episode", {})

    overrides = overrides or {}
    if overrides.get("n_vehicles_npc") is not None:
        scenario_env_cfg["traffic"]["n_vehicles_npc"] = int(overrides["n_vehicles_npc"])
    if overrides.get("n_pedestrians_npc") is not None:
        scenario_env_cfg["traffic"]["n_pedestrians_npc"] = int(overrides["n_pedestrians_npc"])
    if overrides.get("route_distance_m") is not None:
        scenario_env_cfg["episode"]["route_distance_m"] = int(overrides["route_distance_m"])
    if overrides.get("route_distance_m_pedestrian") is not None:
        scenario_env_cfg["episode"]["route_distance_m_pedestrian"] = int(
            overrides["route_distance_m_pedestrian"]
        )

    if "max_steps_per_episode" in limits:
        scenario_env_cfg["episode"]["max_steps"] = int(limits["max_steps_per_episode"])

    return scenario_env_cfg


def _build_evaluation_summary(raw, train_map, scenario_entries=None):
    if scenario_entries:
        summary = {}
        grouped_rows = {}
        for entry in scenario_entries:
            metrics = raw.get(entry["map_name"], {}).get(entry["scenario_name"])
            if metrics is None:
                continue
            grouped_rows.setdefault(entry["summary_level"], []).append(metrics)

        for level_name in ("easy", "medium", "hard", "test"):
            rows = grouped_rows.get(level_name, [])
            if not rows:
                continue
            metrics = _mean_metric_dicts(rows, ["success_rate", "collision_rate"])
            summary[level_name] = {
                "success_rate": metrics.get("success_rate"),
                "collision_rate": metrics.get("collision_rate"),
            }
        return summary

    summary = {}
    if train_map in raw:
        for profile_name, metrics in raw[train_map].items():
            level = _profile_name_to_level(profile_name)
            if level and level not in summary:
                summary[level] = {
                    "success_rate": metrics.get("success_rate"),
                    "collision_rate": metrics.get("collision_rate"),
                }

    test_rows = []
    for map_name, profiles in raw.items():
        if train_map is not None and map_name == train_map:
            continue
        test_rows.extend(profiles.values())
    if not test_rows:
        for profiles in raw.values():
            test_rows.extend(profiles.values())
    if test_rows:
        test_metrics = _mean_metric_dicts(test_rows, ["success_rate", "collision_rate"])
        summary["test"] = {
            "success_rate": test_metrics.get("success_rate"),
            "collision_rate": test_metrics.get("collision_rate"),
        }
    return summary


def _render_progress_bar(progress, width=24):
    progress = max(0.0, min(float(progress), 1.0))
    filled = int(progress * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_duration_hm(total_seconds):
    if total_seconds is None:
        return None
    total_seconds = max(0, int(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours}h{minutes:02d}m"


def _compute_eval_timing(*, started_epoch_s, completed_episodes, total_episodes, now_epoch_s=None):
    if started_epoch_s is None:
        return None, None

    try:
        start_value = float(started_epoch_s)
    except (TypeError, ValueError):
        return None, None

    now_value = float(now_epoch_s if now_epoch_s is not None else time.time())
    elapsed_s = max(0.0, now_value - start_value)

    try:
        completed = int(completed_episodes or 0)
    except (TypeError, ValueError):
        completed = 0
    try:
        total = int(total_episodes or 0)
    except (TypeError, ValueError):
        total = 0

    eta_s = None
    if total > 0:
        remaining = max(total - completed, 0)
        if completed > 0:
            eta_s = 0.0 if remaining == 0 else (elapsed_s / completed) * remaining

    return elapsed_s, eta_s


def _attach_status_timing(payload):
    payload = dict(payload)
    elapsed_s, eta_s = _compute_eval_timing(
        started_epoch_s=payload.get("started_epoch_s"),
        completed_episodes=payload.get("completed_episodes"),
        total_episodes=payload.get("total_episodes"),
    )
    payload["elapsed_seconds"] = elapsed_s
    payload["elapsed_hm"] = _format_duration_hm(elapsed_s)
    payload["eta_seconds"] = eta_s
    payload["eta_hm"] = _format_duration_hm(eta_s)
    return payload


def _aggregate_eval_metric_rows(rows, keys):
    if not rows:
        return {key: None for key in keys}

    sum_keys = {"wall_clock_seconds", "infraction_count"}
    any_true_keys = {"wall_clock_timeout"}
    merged = {}
    for key in keys:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        if not vals:
            merged[key] = None
        elif key in sum_keys:
            merged[key] = float(np.sum(vals))
        elif key in any_true_keys:
            merged[key] = any(bool(v) for v in vals)
        else:
            merged[key] = float(np.mean(vals))
    return merged


def _offset_eval_episode_traces(traces, *, start_episode_index):
    offset = int(start_episode_index)
    rows = []
    for row in traces:
        updated = dict(row)
        updated["episode_index"] = offset + int(updated.get("episode_index", 0) or 0)
        rows.append(updated)
    return rows


def _resolve_eval_metric_keys(eval_cfg):
    metric_cfg = eval_cfg.get("metrics", {}) or {}
    metric_aliases = [
        ("success_rate", "success_rate", True),
        ("collision_rate", "collision_rate", True),
        ("stuck_rate", "stuck_rate", True),
        ("offroad_rate", "offroad_rate", True),
        ("route_completion", "route_completion", True),
        ("avg_episode_length", "episode_length_mean", True),
        ("avg_reward", "reward_mean", True),
        ("timeout_rate", "timeout_rate", True),
        ("path_efficiency", "path_efficiency", True),
        ("infraction_count", "infraction_count", False),
    ]

    selected = []
    unsupported = []
    for cfg_key, metric_key, supported in metric_aliases:
        enabled = bool(metric_cfg.get(cfg_key, cfg_key == "path_efficiency"))
        if not enabled:
            continue
        if supported:
            selected.append(metric_key)
        else:
            unsupported.append(metric_key)

    for metric_key in ("success_rate", "collision_rate"):
        if metric_key not in selected:
            selected.append(metric_key)

    return selected, unsupported


def _extract_eval_metrics_from_result(eval_result, metric_keys):
    def _custom_metric_containers():
        return [
            eval_result.get("custom_metrics", {}) or {},
            _get_nested(eval_result, "sampler_results", "custom_metrics") or {},
            _get_nested(eval_result, "env_runner_results", "custom_metrics") or {},
            _get_nested(eval_result, "evaluation", "custom_metrics") or {},
            _get_nested(eval_result, "evaluation", "sampler_results", "custom_metrics") or {},
            _get_nested(eval_result, "evaluation", "env_runner_results", "custom_metrics") or {},
        ]

    def _metric(*names):
        containers = _custom_metric_containers()
        for name in names:
            for container in containers:
                if name in container:
                    return _coerce_float(container[name])

        # Fallback: average per-policy metrics when the aggregate key is absent.
        for name in names:
            prefixed_values = []
            suffix = f"/{name}"
            for container in containers:
                for key, value in container.items():
                    if key.endswith(suffix):
                        coerced = _coerce_float(value)
                        if coerced is not None:
                            prefixed_values.append(coerced)
            if prefixed_values:
                return float(np.mean(prefixed_values))
        return None

    def _scalar_from_result(*names):
        containers = [
            eval_result,
            eval_result.get("sampler_results", {}) or {},
            eval_result.get("env_runner_results", {}) or {},
            eval_result.get("evaluation", {}) or {},
            _get_nested(eval_result, "evaluation", "sampler_results") or {},
            _get_nested(eval_result, "evaluation", "env_runner_results") or {},
        ]
        for name in names:
            for container in containers:
                if isinstance(container, dict) and name in container:
                    coerced = _coerce_float(container[name])
                    if coerced is not None:
                        return coerced
        return None

    evaluation_time_ms = _scalar_from_result("evaluation_time_ms")
    wall_clock_seconds = (
        evaluation_time_ms / 1000.0
        if evaluation_time_ms is not None
        else _scalar_from_result("time_this_iter_s")
    )

    rows = {
        "success_rate": _metric("success_rate_mean", "success_rate"),
        "collision_rate": _metric("collision_rate_mean", "collision_rate"),
        "stuck_rate": _metric("stuck_rate_mean", "stuck_rate"),
        "offroad_rate": _metric("offroad_rate_mean", "offroad_rate"),
        "timeout_rate": _metric("timeout_rate_mean", "timeout_rate"),
        "route_completion": _metric("route_completion_mean", "route_completion"),
        "path_efficiency": _metric("path_efficiency_mean", "path_efficiency"),
        "reward_mean": _scalar_from_result(
            "episode_reward_mean",
            "episode_return_mean",
        ),
        "episode_length_mean": _scalar_from_result(
            "episode_len_mean",
            "episode_length_mean",
        ),
        "infraction_count": None,
        "wall_clock_timeout": None,
        "wall_clock_seconds": wall_clock_seconds,
    }
    return {key: rows.get(key) for key in metric_keys}


def _extract_eval_episode_traces(eval_result, map_name, profile_name):
    hist_stats = (
        _get_nested(eval_result, "sampler_results", "hist_stats")
        or _get_nested(eval_result, "env_runner_results", "hist_stats")
        or _get_nested(eval_result, "evaluation", "sampler_results", "hist_stats")
        or _get_nested(eval_result, "evaluation", "env_runner_results", "hist_stats")
        or eval_result.get("hist_stats")
        or {}
    )
    rewards = hist_stats.get("episode_reward", []) or []
    lengths = hist_stats.get("episode_lengths", []) or []
    n_rows = max(len(rewards), len(lengths))
    traces = []
    for idx in range(n_rows):
        traces.append(
            {
                "map": map_name,
                "profile": profile_name,
                "episode_index": idx,
                "episode_reward": _coerce_float(rewards[idx]) if idx < len(rewards) else None,
                "episode_length": _coerce_float(lengths[idx]) if idx < len(lengths) else None,
            }
        )
    return traces


def _save_evaluation_plots(out_path, run_name, evaluation_raw, metric_keys):
    flat_rows = []
    for map_name, profiles in evaluation_raw.items():
        for profile_name, metrics in profiles.items():
            flat_rows.append((f"{map_name}/{profile_name}", metrics))

    if not flat_rows or not metric_keys:
        return

    available_keys = []
    for key in metric_keys:
        if any(row.get(key) is not None for _, row in flat_rows):
            available_keys.append(key)

    if not available_keys:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Evaluation plots skipped: matplotlib unavailable (%s)", exc)
        return

    labels = [label for label, _ in flat_rows]
    cols = 2 if len(available_keys) > 1 else 1
    rows = (len(available_keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, available_keys):
        values = [metrics.get(key) for _, metrics in flat_rows]
        plot_values = [np.nan if value is None else value for value in values]
        x = np.arange(len(labels))
        ax.bar(x, plot_values, color="#1f77b4")
        ax.set_title(key.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        finite_vals = [v for v in values if v is not None and np.isfinite(v)]
        if finite_vals and all(0.0 <= v <= 1.0 for v in finite_vals):
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    for ax in axes[len(available_keys) :]:
        ax.remove()

    fig.tight_layout()
    plot_path = out_path / f"{run_name}_evaluation_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tail_text_block(text, *, lines=30):
    if not text:
        return ""
    parts = str(text).splitlines()
    if len(parts) <= lines:
        return "\n".join(parts)
    return "\n".join(parts[-lines:])


def _normalize_command_args(command_value):
    if not command_value:
        return None
    if isinstance(command_value, (list, tuple)):
        args = [str(part).strip() for part in command_value if str(part).strip()]
        return args or None
    text = str(command_value).strip()
    if not text:
        return None
    return shlex.split(text, posix=(os.name != "nt"))


def _is_local_carla_host(host):
    return str(host).strip().lower() in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _resolve_carla_server_command(env_cfg, runtime_cfg):
    runtime_cfg = runtime_cfg or {}
    simulator_cfg = env_cfg.get("simulator", {}) or {}
    world_cfg = env_cfg.get("world", {}) or {}
    port = int(simulator_cfg.get("port", 2000))

    candidates = []
    for key in ("carla_server_command", "server_launch_command"):
        if runtime_cfg.get(key):
            candidates.append(runtime_cfg.get(key))
    for env_name in ("CARLA_SERVER_CMD", "CARLA_SERVER_COMMAND"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(env_value)

    if os.name == "nt":
        default_exe = Path("C:/CARLA_0.9.16/CarlaUE4.exe")
        if default_exe.exists():
            default_cmd = [
                str(default_exe),
                f"-carla-rpc-port={port}",
                "-quality-level=Medium",
            ]
            if bool(world_cfg.get("no_rendering", False)):
                default_cmd.append("-RenderOffScreen")
            candidates.append(default_cmd)

    for candidate in candidates:
        args = _normalize_command_args(candidate)
        if args:
            return args
    return None


def _launch_carla_server(command_args):
    args = _normalize_command_args(command_args)
    if not args:
        return {"ok": False, "reason": "missing CARLA server launch command"}

    exe_path = Path(args[0])
    cwd = str(exe_path.parent) if exe_path.is_absolute() and exe_path.exists() else None
    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        popen_kwargs["close_fds"] = True

    try:
        proc = subprocess.Popen(args, cwd=cwd, **popen_kwargs)
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "command": args}

    return {"ok": True, "pid": proc.pid, "command": args}


def _wait_for_carla_server(env_cfg, *, timeout_seconds, poll_seconds):
    simulator_cfg = env_cfg.get("simulator", {}) or {}
    host = simulator_cfg.get("host", "127.0.0.1")
    port = int(simulator_cfg.get("port", 2000))
    client_timeout_s = min(5.0, max(1.0, float(simulator_cfg.get("timeout_seconds", 20.0))))
    deadline = time.time() + max(5.0, float(timeout_seconds))
    poll_s = max(0.5, float(poll_seconds))
    last_error = None
    attempts = 0

    while time.time() < deadline:
        attempts += 1
        try:
            client = carla.Client(host, port)
            client.set_timeout(client_timeout_s)
            world = client.get_world()
            _ = world.get_map().name
            return {"ok": True, "attempts": attempts}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(poll_s)

    return {"ok": False, "attempts": attempts, "last_error": last_error}


def _looks_like_carla_server_failure(failure):
    if not failure:
        return False

    returncode = failure.get("returncode")
    if returncode in {None, 3221225477, 3221226505, -1073741819, -1073740791}:
        return True

    haystack = "\n".join(
        str(part or "")
        for part in (
            failure.get("reason"),
            failure.get("stdout"),
            failure.get("stderr"),
        )
    ).lower()

    server_tokens = (
        "access violation",
        "sigabrt",
        "pyinit_libcarla",
        "waiting for the simulator",
        "simulator is ready and connected",
        "time-out of 20000ms",
        "failed to destroy actor",
        "load_world",
        "unable to destroy actor",
        "episode worker timeout",
    )
    return any(token in haystack for token in server_tokens)


def _looks_like_ray_startup_failure(failure):
    if not failure:
        return False

    haystack = "\n".join(
        str(part or "")
        for part in (
            failure.get("reason"),
            failure.get("stdout"),
            failure.get("stderr"),
        )
    ).lower()

    ray_tokens = (
        "the current node timed out during startup",
        "current node timed out during startup",
        "some of the ray processes failed to startup",
        "ray processes failed to startup",
        "auto_init_ray",
        "ray.init()",
        "ray.init(",
    )
    return any(token in haystack for token in ray_tokens)


def _recover_carla_server(env_cfg, runtime_cfg, *, reason):
    runtime_cfg = runtime_cfg or {}
    simulator_cfg = env_cfg.get("simulator", {}) or {}
    host = simulator_cfg.get("host", "127.0.0.1")
    recovery_timeout_s = float(
        runtime_cfg.get(
            "carla_server_recovery_timeout_seconds",
            max(60.0, float(simulator_cfg.get("timeout_seconds", 20.0)) * 6.0),
        )
    )
    poll_s = float(runtime_cfg.get("carla_server_poll_seconds", 2.0))
    post_kill_sleep_s = float(runtime_cfg.get("carla_server_post_kill_sleep_seconds", 3.0))
    startup_grace_s = float(runtime_cfg.get("carla_server_startup_grace_seconds", 10.0))

    shutdown_result = shutdown_carla_processes()
    if post_kill_sleep_s > 0:
        time.sleep(post_kill_sleep_s)

    launch_result = {
        "ok": False,
        "reason": "restart skipped",
        "command": None,
    }
    if _is_local_carla_host(host):
        launch_command = _resolve_carla_server_command(env_cfg, runtime_cfg)
        if launch_command:
            launch_result = _launch_carla_server(launch_command)
            if launch_result["ok"] and startup_grace_s > 0:
                time.sleep(startup_grace_s)
        else:
            launch_result = {
                "ok": False,
                "reason": "no CARLA server command configured or discovered",
                "command": None,
            }

    wait_result = _wait_for_carla_server(
        env_cfg,
        timeout_seconds=recovery_timeout_s,
        poll_seconds=poll_s,
    )

    parts = [f"recovery after {reason}"]
    if shutdown_result.get("killed_any"):
        parts.append("stale CARLA process killed")
    elif shutdown_result.get("issues"):
        parts.append("shutdown issues: " + "; ".join(shutdown_result["issues"]))
    elif shutdown_result.get("attempted"):
        parts.append("no running CARLA process found")

    if launch_result.get("command"):
        parts.append(
            "restart "
            + ("ok" if launch_result.get("ok") else f"failed ({launch_result.get('reason')})")
        )
    elif _is_local_carla_host(host):
        parts.append(launch_result.get("reason", "restart unavailable"))

    if wait_result["ok"]:
        parts.append("server reachable")
    else:
        parts.append(f"server still unavailable ({wait_result.get('last_error')})")

    return {
        "ok": bool(wait_result["ok"]),
        "message": "; ".join(parts),
        "shutdown": shutdown_result,
        "launch": launch_result,
        "wait": wait_result,
    }


def _run_episode_job(payload):
    _carla_prepare_world(
        payload["env_cfg"],
        force_reload=bool(payload.get("force_reload", False)),
    )
    _register_eval_runtime()

    eval_config = _build_mappo_config(
        env_cfg=payload["env_cfg"],
        train_cfg=payload["train_cfg"],
        eval_cfg=payload["eval_cfg"],
        n_gpus=int(payload["n_gpus"]),
        n_workers=0,
        exp_seed=int(payload["seed_base"]),
        enable_periodic_evaluation=False,
    )

    eval_algo = None
    try:
        eval_algo = eval_config.build()
        eval_algo.restore(payload["checkpoint_path"])
        eval_result = eval_algo.evaluate()
        result_payload = {
            "metrics": _extract_eval_metrics_from_result(
                eval_result,
                payload["metric_keys"],
            ),
            "traces": _extract_eval_episode_traces(
                eval_result,
                map_name=payload["map_name"],
                profile_name=payload["profile_name"],
            )
            if payload.get("save_traces", False)
            else [],
        }
        _write_json_atomic(payload["result_path"], result_payload)
        return 0
    finally:
        if eval_algo is not None:
            try:
                eval_algo.stop()
            except Exception as stop_exc:
                print(
                    "    [WARN] eval child stop failed during "
                    f"{payload['map_name']}/{payload['profile_name']} episodio "
                    f"{payload['episode_number']}/{payload['episodes_per_scenario']}: "
                    f"{type(stop_exc).__name__}: {stop_exc}"
                )
        try:
            ray.shutdown()
        except Exception:
            pass
        gc.collect()


def _run_episode_eval_subprocess(
    *,
    out_dir,
    checkpoint_path,
    env_cfg,
    train_cfg,
    eval_cfg,
    seed_base,
    n_gpus,
    map_name,
    profile_name,
    episode_idx,
    episodes_per_scenario,
    metric_keys,
    save_traces,
    force_reload,
    timeout_seconds,
):
    runtime_dir = Path(out_dir) / "_eval_episode_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    job_path = runtime_dir / "current_episode_job.json"
    result_path = runtime_dir / "current_episode_result.json"
    if result_path.exists():
        result_path.unlink()

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "env_cfg": deepcopy(env_cfg),
        "train_cfg": deepcopy(train_cfg),
        "eval_cfg": deepcopy(eval_cfg),
        "seed_base": int(seed_base),
        "n_gpus": int(n_gpus),
        "map_name": map_name,
        "profile_name": profile_name,
        "episode_idx": int(episode_idx),
        "episode_number": int(episode_idx) + 1,
        "episodes_per_scenario": int(episodes_per_scenario),
        "metric_keys": list(metric_keys),
        "save_traces": bool(save_traces),
        "force_reload": bool(force_reload),
        "result_path": str(result_path),
    }
    _write_json_atomic(job_path, payload)

    command = [
        sys.executable,
        "-m",
        "carla_core.training.evaluate_carla_mappo",
        "--episode-job",
        str(job_path),
    ]

    try:
        proc = subprocess.run(
            command,
            cwd=str(_project_root()),
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "reason": (
                "episode worker timeout: "
                f"{map_name}/{profile_name} episodio {episode_idx + 1}/{episodes_per_scenario}"
            ),
            "stdout": _tail_text_block(exc.stdout),
            "stderr": _tail_text_block(exc.stderr),
            "returncode": None,
        }

    if proc.returncode != 0:
        return {
            "ok": False,
            "reason": (
                "episode worker failed: "
                f"{map_name}/{profile_name} episodio {episode_idx + 1}/{episodes_per_scenario} "
                f"(exit {proc.returncode})"
            ),
            "stdout": _tail_text_block(proc.stdout),
            "stderr": _tail_text_block(proc.stderr),
            "returncode": proc.returncode,
        }

    if not result_path.exists():
        return {
            "ok": False,
            "reason": (
                "episode worker missing result artifact: "
                f"{map_name}/{profile_name} episodio {episode_idx + 1}/{episodes_per_scenario}"
            ),
            "stdout": _tail_text_block(proc.stdout),
            "stderr": _tail_text_block(proc.stderr),
            "returncode": proc.returncode,
        }

    with open(result_path, "r", encoding="utf-8") as f:
        result_payload = json.load(f)

    return {
        "ok": True,
        "metrics": result_payload.get("metrics", {}),
        "traces": result_payload.get("traces", []) or [],
        "stdout": _tail_text_block(proc.stdout),
        "stderr": _tail_text_block(proc.stderr),
        "returncode": proc.returncode,
    }


def _run_evaluation_scenarios(
    *,
    out_dir,
    checkpoint_path,
    base_env_cfg,
    train_cfg,
    eval_cfg,
    seed_base,
    n_gpus,
    status_callback=None,
):
    def _publish_status(**fields):
        if status_callback is None:
            return
        try:
            status_callback(**fields)
        except Exception:
            pass

    eval_section = eval_cfg.get("evaluation", {})
    if not eval_section.get("enabled", True):
        return {}, {}, [], []

    scenario_cfg = eval_cfg.get("scenarios", {})
    limits = eval_cfg.get("limits", {})
    entries = _expand_eval_entries(scenario_cfg)
    legacy_maps = scenario_cfg.get("maps", [])
    legacy_profiles = scenario_cfg.get("traffic_profiles", [])
    use_entry_scenarios = bool(entries)

    if not use_entry_scenarios and (not legacy_maps or not legacy_profiles):
        return {}, {}, [], []

    parallel_envs = int(eval_section.get("parallel_envs", 1))
    if parallel_envs != 1:
        raise ValueError(
            "eval.parallel_envs=%s is not supported in CARLA multi-agent evaluation "
            "with a single simulator instance. Use parallel_envs=1." % parallel_envs
        )
    metric_keys, unsupported_metric_keys = _resolve_eval_metric_keys(eval_cfg)
    if unsupported_metric_keys:
        logger.warning(
            "Evaluation metrics requested but not available in current pipeline: %s",
            ", ".join(sorted(unsupported_metric_keys)),
        )

    raw = {}
    traces = []
    train_map = base_env_cfg.get("world", {}).get("map")
    level_configs = get_level_configs(base_env_cfg)
    scenario_groups = {}
    summary_entries = entries if use_entry_scenarios else None
    if use_entry_scenarios:
        for entry in entries:
            scenario_groups.setdefault(entry["map_name"], []).append(entry)
    else:
        for map_name in legacy_maps:
            scenario_groups[map_name] = []
            for profile in legacy_profiles:
                scenario_groups[map_name].append(
                    {
                        "scenario_name": profile.get("name", "unnamed"),
                        "map_name": map_name,
                        "level_name": None,
                        "summary_level": _profile_name_to_level(profile.get("name", "")),
                        "overrides": {
                            "n_vehicles_npc": profile.get("n_vehicles"),
                            "n_pedestrians_npc": profile.get("n_pedestrians"),
                            "route_distance_m": profile.get("route_distance_m"),
                            "route_distance_m_pedestrian": profile.get(
                                "route_distance_m_pedestrian"
                            ),
                        },
                    }
                )
    aggregate_keys = list(dict.fromkeys(metric_keys + unsupported_metric_keys))
    save_traces = bool(eval_cfg.get("outputs", {}).get("save_per_episode_trace", False))
    episodes_per_map = int(eval_section.get("episodes_per_map", 1))
    runtime_cfg = eval_cfg.get("runtime", {}) or {}
    episode_worker_timeout_s = int(runtime_cfg.get("subprocess_timeout_seconds", 14400))
    max_retries_per_episode = int(runtime_cfg.get("max_retries_per_episode", 1))
    reload_world_between_episodes = bool(
        runtime_cfg.get(
            "reload_world_between_episodes",
            runtime_cfg.get("reload_world_between_scenarios", False),
        )
    )
    max_steps_per_episode = (
        int(limits.get("max_steps_per_episode", 0))
        if "max_steps_per_episode" in limits
        else None
    )
    total_scenarios = sum(len(profiles) for profiles in scenario_groups.values())
    total_episodes = total_scenarios * episodes_per_map
    total_env_steps_budget = (
        total_episodes * max_steps_per_episode if max_steps_per_episode is not None else None
    )
    scenario_idx = 0
    completed_episodes_total = 0
    eval_t0 = time.time()
    episode_stall_warning_s = 600.0
    try:
        heartbeat_interval_s = max(
            5.0,
            float(((eval_cfg.get("runtime") or {}).get("heartbeat_interval_seconds", 30.0))),
        )
    except (TypeError, ValueError):
        heartbeat_interval_s = 30.0

    print()
    print(f"{'=' * 19}EVAL{'=' * 20}")
    if use_entry_scenarios:
        print(f"  Eval scenarios: {total_scenarios} (explicit entries)")
    else:
        print(
            f"  Eval scenarios: {total_scenarios} "
            f"({len(legacy_maps)} maps x {len(legacy_profiles)} profiles)"
        )
    print(f"  Eval episodes: {total_episodes}")
    if total_env_steps_budget is not None:
        print(f"  Eval max env-steps: {total_env_steps_budget:,}\n")

    _publish_status(
        reason="evaluation subprocess running",
        progress=0.0,
        completed_scenarios=0,
        total_scenarios=total_scenarios,
        total_episodes=total_episodes,
        completed_episodes=0,
        current_phase="starting",
        current_scenario_idx=0,
        current_map=None,
        current_profile=None,
        progress_mode="exact",
        started_epoch_s=eval_t0,
    )

    for map_name, profiles in scenario_groups.items():
        map_rows = raw.setdefault(map_name, {})
        for profile in profiles:
            scenario_idx += 1
            profile_name = profile.get("scenario_name", profile.get("name", "unnamed"))
            scenario_t0 = time.time()
            completed_scenarios = scenario_idx - 1
            scenario_episode_rows = []
            _publish_status(
                reason="evaluation subprocess running",
                progress=(completed_episodes_total / total_episodes if total_episodes else 0.0),
                completed_scenarios=completed_scenarios,
                total_scenarios=total_scenarios,
                total_episodes=total_episodes,
                completed_episodes=completed_episodes_total,
                current_phase="scenario_pending",
                current_scenario_idx=scenario_idx,
                current_map=map_name,
                current_profile=profile_name,
                progress_mode="exact",
                started_epoch_s=eval_t0,
            )
            print(
                f"  [Eval {scenario_idx}/{total_scenarios}] {map_name}/{profile_name} | "
                f"episodes={episodes_per_map}"
            )
            print(f"{'=' * 43}")
            for episode_idx in range(episodes_per_map):
                scenario_env_cfg = _build_scenario_env_cfg(
                    base_env_cfg=base_env_cfg,
                    map_name=map_name,
                    level_name=profile.get("level_name"),
                    level_configs=level_configs,
                    limits=limits,
                    seed_base=seed_base,
                    reset_count=episode_idx,
                    overrides=profile.get("overrides"),
                )

                current_episode_number = episode_idx + 1
                exact_progress = (
                    completed_episodes_total / total_episodes if total_episodes else 0.0
                )
                heartbeat_stop = threading.Event()
                episode_t0 = time.time()
                episode_stall_warned = [False]

                def _heartbeat():
                    while not heartbeat_stop.wait(heartbeat_interval_s):
                        progress = (
                            completed_episodes_total / total_episodes if total_episodes else 0.0
                        )
                        total_elapsed_s, eta_s = _compute_eval_timing(
                            started_epoch_s=eval_t0,
                            completed_episodes=completed_episodes_total,
                            total_episodes=total_episodes,
                        )
                        eta_label = _format_duration_hm(eta_s) or "n/a"
                        print(
                            f"Eval progress {_render_progress_bar(progress)} "
                            f"{progress * 100:5.1f}% exact "
                            f"({completed_episodes_total}/{total_episodes} episodi completati, "
                            f"scenario {scenario_idx}/{total_scenarios}, "
                            f"episodio {current_episode_number}/{episodes_per_map}) | "
                            f"Elapsed {_format_duration_hm(total_elapsed_s)} | ETA {eta_label}"
                        )
                        _publish_status(
                            reason="evaluation subprocess running",
                            progress=progress,
                            completed_scenarios=completed_scenarios,
                            total_scenarios=total_scenarios,
                            total_episodes=total_episodes,
                            completed_episodes=completed_episodes_total,
                            episodes_per_scenario=episodes_per_map,
                            current_episode_idx=current_episode_number,
                            current_phase="episode_running",
                            current_scenario_idx=scenario_idx,
                            current_map=map_name,
                            current_profile=profile_name,
                            progress_mode="exact",
                            started_epoch_s=eval_t0,
                        )
                        current_elapsed_s = time.time() - episode_t0
                        if not episode_stall_warned[0] and current_elapsed_s >= episode_stall_warning_s:
                            print(
                                f"[WARN] Episodio {current_episode_number}/{episodes_per_map} "
                                f"di {map_name}/{profile_name} ancora in corso dopo "
                                f"{current_elapsed_s / 60:.1f}m."
                            )
                            episode_stall_warned[0] = True

                heartbeat_thread = threading.Thread(
                    target=_heartbeat,
                    name=f"eval-heartbeat-{scenario_idx}-{current_episode_number}",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    single_episode_eval_cfg = deepcopy(eval_cfg)
                    single_episode_eval_cfg.setdefault("evaluation", {})
                    single_episode_eval_cfg["evaluation"]["episodes_per_map"] = 1
                    last_failure = None
                    episode_result = None

                    for attempt in range(max_retries_per_episode + 1):
                        episode_result = _run_episode_eval_subprocess(
                            out_dir=_project_root() / out_dir if not Path(out_dir).is_absolute() else out_dir,
                            checkpoint_path=checkpoint_path,
                            env_cfg=scenario_env_cfg,
                            train_cfg=train_cfg,
                            eval_cfg=single_episode_eval_cfg,
                            seed_base=seed_base,
                            n_gpus=n_gpus,
                            map_name=map_name,
                            profile_name=profile_name,
                            episode_idx=episode_idx,
                            episodes_per_scenario=episodes_per_map,
                            metric_keys=aggregate_keys,
                            save_traces=save_traces,
                            force_reload=reload_world_between_episodes,
                            timeout_seconds=episode_worker_timeout_s,
                        )
                        if episode_result["ok"]:
                            break

                        last_failure = episode_result
                        print(
                            f"    [WARN] {episode_result['reason']}"
                            + (
                                f" | retry {attempt + 1}/{max_retries_per_episode}"
                                if attempt < max_retries_per_episode
                                else ""
                            )
                        )
                        if episode_result.get("stdout"):
                            print(_tail_text_block(episode_result["stdout"], lines=20))
                        if episode_result.get("stderr"):
                            print(_tail_text_block(episode_result["stderr"], lines=20))

                        if attempt < max_retries_per_episode:
                            if _looks_like_carla_server_failure(episode_result):
                                recovery = _recover_carla_server(
                                    scenario_env_cfg,
                                    runtime_cfg,
                                    reason=episode_result["reason"],
                                )
                                print(f"    [WARN] {recovery['message']}")
                            else:
                                if _looks_like_ray_startup_failure(episode_result):
                                    print(
                                        "    [WARN] retrying after Ray startup failure; "
                                        "preserving CARLA server"
                                    )
                                gc.collect()
                                time.sleep(2.0)

                    if not episode_result or not episode_result["ok"]:
                        failure = last_failure or {"reason": "unknown episode worker failure"}
                        if scenario_episode_rows:
                            map_rows[profile_name] = _aggregate_eval_metric_rows(
                                scenario_episode_rows, aggregate_keys
                            )
                        raw[map_name] = map_rows
                        raise FinalEvaluationFailed(
                            raw=deepcopy(raw),
                            summary=_build_evaluation_summary(raw, train_map, summary_entries),
                            traces=list(traces),
                            metric_keys=list(aggregate_keys),
                            reason=failure["reason"],
                        )

                    scenario_episode_rows.append(episode_result["metrics"])
                    if save_traces and episode_result["traces"]:
                        traces.extend(
                            _offset_eval_episode_traces(
                                episode_result["traces"],
                                start_episode_index=episode_idx,
                            )
                        )
                except KeyboardInterrupt:
                    if scenario_episode_rows:
                        map_rows[profile_name] = _aggregate_eval_metric_rows(
                            scenario_episode_rows, aggregate_keys
                        )
                    raw[map_name] = map_rows
                    print(
                        f"    interrotto dall'utente durante {map_name}/{profile_name} "
                        f"episodio {current_episode_number}/{episodes_per_map}"
                    )
                    raise FinalEvaluationInterrupted(
                        raw=deepcopy(raw),
                        summary=_build_evaluation_summary(raw, train_map, summary_entries),
                        traces=list(traces),
                        metric_keys=list(aggregate_keys),
                        reason=(
                            "manual interrupt during final evaluation "
                            f"({map_name}/{profile_name}, episode "
                            f"{current_episode_number}/{episodes_per_map})"
                        ),
                    ) from None
                except Exception as exc:
                    if scenario_episode_rows:
                        map_rows[profile_name] = _aggregate_eval_metric_rows(
                            scenario_episode_rows, aggregate_keys
                        )
                    raw[map_name] = map_rows
                    raise FinalEvaluationFailed(
                        raw=deepcopy(raw),
                        summary=_build_evaluation_summary(raw, train_map, summary_entries),
                        traces=list(traces),
                        metric_keys=list(aggregate_keys),
                        reason=(
                            "final evaluation error: "
                            f"{type(exc).__name__} ({map_name}/{profile_name}, "
                            f"episode {current_episode_number}/{episodes_per_map})"
                        ),
                    ) from exc
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1.0)
                    gc.collect()

                completed_episodes_total += 1
                exact_progress = (
                    completed_episodes_total / total_episodes if total_episodes else 0.0
                )
                _publish_status(
                    reason="evaluation subprocess running",
                    progress=exact_progress,
                    completed_scenarios=completed_scenarios,
                    total_scenarios=total_scenarios,
                    total_episodes=total_episodes,
                    completed_episodes=completed_episodes_total,
                    episodes_per_scenario=episodes_per_map,
                    current_episode_idx=current_episode_number,
                    current_phase="episode_completed",
                    current_scenario_idx=scenario_idx,
                    current_map=map_name,
                    current_profile=profile_name,
                    progress_mode="exact",
                    started_epoch_s=eval_t0,
                )
                print(
                    f"    episodio {current_episode_number}/{episodes_per_map} completato | "
                    f"progress {_render_progress_bar(exact_progress)} {exact_progress * 100:5.1f}%"
                )

            map_rows[profile_name] = _aggregate_eval_metric_rows(
                scenario_episode_rows, aggregate_keys
            )
            scenario_elapsed = time.time() - scenario_t0
            total_elapsed = time.time() - eval_t0
            remaining = total_scenarios - scenario_idx
            eta = (total_elapsed / scenario_idx) * remaining if scenario_idx > 0 else 0.0
            exact_progress = (
                completed_episodes_total / total_episodes if total_episodes else 0.0
            )
            _publish_status(
                reason="evaluation subprocess running",
                progress=exact_progress,
                completed_scenarios=scenario_idx,
                total_scenarios=total_scenarios,
                total_episodes=total_episodes,
                completed_episodes=completed_episodes_total,
                episodes_per_scenario=episodes_per_map,
                current_episode_idx=episodes_per_map,
                current_phase="scenario_completed",
                current_scenario_idx=scenario_idx,
                current_map=map_name,
                current_profile=profile_name,
                progress_mode="exact",
                started_epoch_s=eval_t0,
            )
            print(
                f"Eval progress {_render_progress_bar(exact_progress)} {exact_progress * 100:5.1f}% "
                f"exact ({completed_episodes_total}/{total_episodes} episodi, "
                f"{scenario_idx}/{total_scenarios} scenari completati) | "
                f"Elapsed {_format_duration_hm(total_elapsed)} | ETA {_format_duration_hm(eta)}"
            )
            print(
                f"Done in {int(scenario_elapsed//3600)}h{int((scenario_elapsed%3600)//60):02d}m | "
                f"Elapsed {int(total_elapsed//3600)}h{int((total_elapsed%3600)//60):02d}m | "
                f"ETA {int(eta//3600)}h{int((eta%3600)//60):02d}m"
            )
        raw[map_name] = map_rows

    summary = _build_evaluation_summary(raw, train_map, summary_entries)
    return raw, summary, traces, aggregate_keys


def _save_evaluation_artifacts(
    base_dir,
    run_name,
    eval_cfg,
    evaluation_raw,
    evaluation_summary,
    evaluation_traces=None,
    evaluation_metric_keys=None,
):
    outputs = eval_cfg.get("outputs", {})
    if not outputs:
        return

    output_dir = outputs.get("output_dir")
    if output_dir:
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            out_path = Path(base_dir) / out_path
    else:
        out_path = Path(base_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "evaluation": evaluation_summary,
        "evaluation_raw": evaluation_raw,
    }
    if outputs.get("save_per_episode_trace", False):
        payload["evaluation_traces"] = evaluation_traces or []

    if outputs.get("save_json", False):
        json_path = out_path / f"{run_name}_evaluation.json"
        _write_json_atomic(json_path, payload)

    if outputs.get("save_csv", False):
        csv_path = out_path / f"{run_name}_evaluation.csv"
        csv_metric_keys = evaluation_metric_keys or []
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["map", "profile", *csv_metric_keys])
            writer.writeheader()
            for map_name, profiles in evaluation_raw.items():
                for profile_name, metrics in profiles.items():
                    row = {"map": map_name, "profile": profile_name}
                    for key in csv_metric_keys:
                        row[key] = metrics.get(key)
                    writer.writerow(row)

    if outputs.get("save_per_episode_trace", False):
        trace_path = out_path / f"{run_name}_evaluation_traces.json"
        _write_json_atomic(trace_path, evaluation_traces or [])

    if outputs.get("save_plots", False):
        _save_evaluation_plots(
            out_path=out_path,
            run_name=run_name,
            evaluation_raw=evaluation_raw,
            metric_keys=evaluation_metric_keys or [],
        )


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(payload), f, indent=2)
    os.replace(tmp_path, path)


def _write_status(status_path, payload):
    _write_json_atomic(status_path, payload)


def _load_job(job_path):
    with open(job_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def _now_status_fields():
    return {
        "heartbeat_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "heartbeat_epoch_s": time.time(),
    }


def _carla_prepare_world(env_cfg, *, force_reload):
    """Prepare the target world without taking ownership of runtime settings.

    The env remains the single owner of CARLA world settings, weather, and
    traffic-manager configuration during evaluation episodes.
    """
    sim = env_cfg["simulator"]
    client = carla.Client(sim["host"], sim["port"])
    client.set_timeout(sim["timeout_seconds"])

    target_map = env_cfg["world"]["map"]
    try:
        available_maps = [m.split("/")[-1] for m in client.get_available_maps()]
    except Exception:
        available_maps = []

    if available_maps and target_map not in available_maps:
        raise ValueError(
            f"Target map '{target_map}' not available in CARLA server. "
            f"Available maps: {sorted(available_maps)}"
        )

    current_map = None
    try:
        current_map = client.get_world().get_map().name.split("/")[-1]
    except Exception:
        current_map = None

    if current_map != target_map:
        client.load_world(target_map)
        return

    if force_reload:
        world = client.reload_world(False)
        if world is None:
            world = client.get_world()
        _ = world.get_map().name


def _register_eval_runtime():
    # RLlib requires explicit registration in each subprocess that builds an Algorithm.
    register_env("CarlaMultiAgent-v0", rllib_env_creator)
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)


def _maybe_save_artifacts(
    *,
    out_dir,
    eval_cfg,
    evaluation_raw,
    evaluation_summary,
    evaluation_traces,
    evaluation_metric_keys,
):
    if not (evaluation_summary or evaluation_raw):
        return False
    _save_evaluation_artifacts(
        base_dir=_project_root(),
        run_name=Path(out_dir).name,
        eval_cfg=eval_cfg,
        evaluation_raw=evaluation_raw,
        evaluation_summary=evaluation_summary,
        evaluation_traces=evaluation_traces,
        evaluation_metric_keys=evaluation_metric_keys,
    )
    return True


def _update_results_json(
    *,
    results_path,
    completed,
    reason=None,
    evaluation_summary=None,
    evaluation_raw=None,
    require_exists=False,
):
    if not results_path:
        if require_exists:
            raise RuntimeError("results_path missing from final_eval_job.json")
        return False

    results_path = Path(results_path)
    if not results_path.exists():
        if require_exists:
            raise FileNotFoundError(f"results.json not found: {results_path}")
        return False

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload.setdefault("meta", {})
    payload["meta"]["final_evaluation_completed"] = bool(completed)
    if completed:
        payload["meta"].pop("final_evaluation_skip_reason", None)
    else:
        payload["meta"]["final_evaluation_skip_reason"] = (
            str(reason) if reason is not None else "final evaluation failed"
        )

    if evaluation_summary is not None:
        payload["evaluation"] = evaluation_summary
    if evaluation_raw is not None:
        payload["evaluation_raw"] = evaluation_raw

    _write_json_atomic(results_path, payload)
    return True


def _update_running_status(status_path, state, **fields):
    state.update(fields)
    state.update(_now_status_fields())
    payload = _attach_status_timing(state)
    progress = float(payload.get("progress", 0.0) or 0.0)
    progress = min(max(progress, 0.0), 1.0)
    payload["progress"] = progress
    payload["progress_bar"] = _render_progress_bar(progress)
    _write_status(status_path, payload)


def main():
    parser = argparse.ArgumentParser(description="Isolated CARLA MAPPO final evaluation")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--job")
    mode_group.add_argument("--episode-job")
    args = parser.parse_args()

    if args.episode_job:
        episode_payload = _load_job(args.episode_job)
        return _run_episode_job(episode_payload)

    job = _load_job(args.job)
    out_dir = job["out_dir"]
    status_path = Path(out_dir) / "evaluation_status.json"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    session_id = str(job.get("session_id") or Path(out_dir).name)
    status_state = {
        "session_id": session_id,
        "pid": os.getpid(),
        "completed": False,
        "reason": "evaluation subprocess running",
        "exit_code": None,
        "artifacts_written": False,
        "started_at": started_at,
        "started_epoch_s": time.time(),
        "finished_at": None,
        "evaluation_metric_keys": [],
        "progress": 0.0,
        "completed_scenarios": 0,
        "total_scenarios": 0,
        "total_episodes": 0,
        "completed_episodes": 0,
        "episodes_per_scenario": 0,
        "current_episode_idx": 0,
        "current_phase": "starting",
        "current_scenario_idx": 0,
        "current_map": None,
        "current_profile": None,
        "progress_mode": "exact",
    }

    _update_running_status(status_path, status_state)

    try:
        def _status_callback(**fields):
            _update_running_status(status_path, status_state, **fields)

        evaluation_raw, evaluation_summary, evaluation_traces, evaluation_metric_keys = (
            _run_evaluation_scenarios(
                out_dir=out_dir,
                checkpoint_path=job["checkpoint_path"],
                base_env_cfg=deepcopy(job["env_cfg"]),
                train_cfg=deepcopy(job["train_cfg"]),
                eval_cfg=deepcopy(job["eval_cfg"]),
                seed_base=int(job["seed_base"]),
                n_gpus=int(job["n_gpus"]),
                status_callback=_status_callback,
            )
        )
        artifacts_written = _maybe_save_artifacts(
            out_dir=out_dir,
            eval_cfg=job["eval_cfg"],
            evaluation_raw=evaluation_raw,
            evaluation_summary=evaluation_summary,
            evaluation_traces=evaluation_traces,
            evaluation_metric_keys=evaluation_metric_keys,
        )
        try:
            _update_results_json(
                results_path=job.get("results_path"),
                completed=True,
                evaluation_summary=evaluation_summary,
                evaluation_raw=evaluation_raw,
                require_exists=True,
            )
        except Exception as exc:
            raise FinalEvaluationFailed(
                raw=deepcopy(evaluation_raw),
                summary=deepcopy(evaluation_summary),
                traces=list(evaluation_traces),
                metric_keys=list(evaluation_metric_keys),
                reason=f"results.json update failed: {type(exc).__name__}: {exc}",
            ) from exc
        _write_status(
            status_path,
            _attach_status_timing(
                {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": True,
                "reason": None,
                "exit_code": 0,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
                "started_epoch_s": status_state.get("started_epoch_s"),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": evaluation_metric_keys,
                "progress": 1.0,
                "progress_bar": _render_progress_bar(1.0),
                "current_phase": "completed",
                "completed_scenarios": status_state.get("total_scenarios", 0),
                "total_scenarios": status_state.get("total_scenarios", 0),
                "total_episodes": status_state.get("total_episodes", 0),
                "completed_episodes": status_state.get("total_episodes", 0),
                "episodes_per_scenario": status_state.get("episodes_per_scenario", 0),
                "current_episode_idx": status_state.get("episodes_per_scenario", 0),
                "current_scenario_idx": status_state.get("total_scenarios", 0),
                "current_map": status_state.get("current_map"),
                "current_profile": status_state.get("current_profile"),
                "progress_mode": "exact",
                **_now_status_fields(),
                }
            ),
        )
        return 0
    except FinalEvaluationInterrupted as exc:
        artifacts_written = _maybe_save_artifacts(
            out_dir=out_dir,
            eval_cfg=job["eval_cfg"],
            evaluation_raw=exc.raw,
            evaluation_summary=exc.summary,
            evaluation_traces=exc.traces,
            evaluation_metric_keys=exc.metric_keys,
        )
        _update_results_json(
            results_path=job.get("results_path"),
            completed=False,
            reason=exc.reason,
            evaluation_summary=exc.summary,
            evaluation_raw=exc.raw,
        )
        _write_status(
            status_path,
            _attach_status_timing(
                {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": exc.reason,
                "exit_code": 130,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
                "started_epoch_s": status_state.get("started_epoch_s"),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": exc.metric_keys,
                "progress": status_state.get("progress", 0.0),
                "progress_bar": _render_progress_bar(status_state.get("progress", 0.0)),
                "current_phase": "interrupted",
                "completed_scenarios": status_state.get("completed_scenarios", 0),
                "total_scenarios": status_state.get("total_scenarios", 0),
                "total_episodes": status_state.get("total_episodes", 0),
                "completed_episodes": status_state.get("completed_episodes", 0),
                "episodes_per_scenario": status_state.get("episodes_per_scenario", 0),
                "current_episode_idx": status_state.get("current_episode_idx", 0),
                "current_scenario_idx": status_state.get("current_scenario_idx", 0),
                "current_map": status_state.get("current_map"),
                "current_profile": status_state.get("current_profile"),
                "progress_mode": status_state.get("progress_mode", "exact"),
                **_now_status_fields(),
                }
            ),
        )
        return 130
    except FinalEvaluationFailed as exc:
        artifacts_written = _maybe_save_artifacts(
            out_dir=out_dir,
            eval_cfg=job["eval_cfg"],
            evaluation_raw=exc.raw,
            evaluation_summary=exc.summary,
            evaluation_traces=exc.traces,
            evaluation_metric_keys=exc.metric_keys,
        )
        _update_results_json(
            results_path=job.get("results_path"),
            completed=False,
            reason=exc.reason,
            evaluation_summary=exc.summary,
            evaluation_raw=exc.raw,
        )
        _write_status(
            status_path,
            _attach_status_timing(
                {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": exc.reason,
                "exit_code": 1,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
                "started_epoch_s": status_state.get("started_epoch_s"),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": exc.metric_keys,
                "progress": status_state.get("progress", 0.0),
                "progress_bar": _render_progress_bar(status_state.get("progress", 0.0)),
                "current_phase": "failed",
                "completed_scenarios": status_state.get("completed_scenarios", 0),
                "total_scenarios": status_state.get("total_scenarios", 0),
                "total_episodes": status_state.get("total_episodes", 0),
                "completed_episodes": status_state.get("completed_episodes", 0),
                "episodes_per_scenario": status_state.get("episodes_per_scenario", 0),
                "current_episode_idx": status_state.get("current_episode_idx", 0),
                "current_scenario_idx": status_state.get("current_scenario_idx", 0),
                "current_map": status_state.get("current_map"),
                "current_profile": status_state.get("current_profile"),
                "progress_mode": status_state.get("progress_mode", "exact"),
                **_now_status_fields(),
                }
            ),
        )
        print(f"[WARN] Final evaluation failed: {exc.reason}")
        return 1
    except KeyboardInterrupt:
        _update_results_json(
            results_path=job.get("results_path"),
            completed=False,
            reason="manual interrupt during evaluation subprocess",
        )
        _write_status(
            status_path,
            _attach_status_timing(
                {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": "manual interrupt during evaluation subprocess",
                "exit_code": 130,
                "artifacts_written": False,
                "started_at": started_at,
                "started_epoch_s": status_state.get("started_epoch_s"),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": [],
                "progress": status_state.get("progress", 0.0),
                "progress_bar": _render_progress_bar(status_state.get("progress", 0.0)),
                "current_phase": "interrupted",
                "completed_scenarios": status_state.get("completed_scenarios", 0),
                "total_scenarios": status_state.get("total_scenarios", 0),
                "total_episodes": status_state.get("total_episodes", 0),
                "completed_episodes": status_state.get("completed_episodes", 0),
                "episodes_per_scenario": status_state.get("episodes_per_scenario", 0),
                "current_episode_idx": status_state.get("current_episode_idx", 0),
                "current_scenario_idx": status_state.get("current_scenario_idx", 0),
                "current_map": status_state.get("current_map"),
                "current_profile": status_state.get("current_profile"),
                "progress_mode": status_state.get("progress_mode", "exact"),
                **_now_status_fields(),
                }
            ),
        )
        print("\nValutazione finale interrotta manualmente.")
        return 130
    except Exception as exc:
        _update_results_json(
            results_path=job.get("results_path"),
            completed=False,
            reason=f"evaluation subprocess error: {type(exc).__name__}",
        )
        _write_status(
            status_path,
            _attach_status_timing(
                {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": f"evaluation subprocess error: {type(exc).__name__}",
                "exit_code": 1,
                "artifacts_written": False,
                "started_at": started_at,
                "started_epoch_s": status_state.get("started_epoch_s"),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": [],
                "progress": status_state.get("progress", 0.0),
                "progress_bar": _render_progress_bar(status_state.get("progress", 0.0)),
                "current_phase": "failed",
                "completed_scenarios": status_state.get("completed_scenarios", 0),
                "total_scenarios": status_state.get("total_scenarios", 0),
                "total_episodes": status_state.get("total_episodes", 0),
                "completed_episodes": status_state.get("completed_episodes", 0),
                "episodes_per_scenario": status_state.get("episodes_per_scenario", 0),
                "current_episode_idx": status_state.get("current_episode_idx", 0),
                "current_scenario_idx": status_state.get("current_scenario_idx", 0),
                "current_map": status_state.get("current_map"),
                "current_profile": status_state.get("current_profile"),
                "progress_mode": status_state.get("progress_mode", "exact"),
                **_now_status_fields(),
                }
            ),
        )
        traceback.print_exc()
        return 1
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
        gc.collect()
        shutdown_result = shutdown_carla_processes()
        if shutdown_result["killed_any"]:
            print("\nServer CARLA arrestato al termine della valutazione finale.")
        elif shutdown_result["issues"]:
            joined_issues = "; ".join(shutdown_result["issues"])
            print(f"\n[WARN] Chiusura CARLA a fine eval non pulita: {joined_issues}")
        else:
            print("\nNessun processo CARLA trovato da arrestare al termine della valutazione finale.")


if __name__ == "__main__":
    raise SystemExit(main())
