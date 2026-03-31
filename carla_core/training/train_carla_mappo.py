"""
Train MAPPO on CarlaMultiAgentEnv — Multi-Agent via RLlib (v0.1)
================================================================
MAPPO = PPO + centralized critic (CTDE paradigm).

Architecture:
  - vehicle_policy:    actor sees 25D local obs, critic sees global_obs (138D)
  - pedestrian_policy: actor sees 19D local obs, critic sees global_obs (138D)
  - global_obs = fixed-slot concat [v0_25D|v1|v2|p0_19D|p1|p2|alive_mask_6D] = 138D
  
Components:
  - CarlaMultiAgentEnv (PettingZoo ParallelEnv) → ParallelPettingZooEnv
  - CentralizedCriticModel (custom TorchModelV2)
  - CentralizedCriticCallbacks (injects global_obs, recomputes GAE)

Uso:
    python carla_core/training/train_carla_mappo.py
    python carla_core/training/train_carla_mappo.py --timesteps 50000 --workers 0
    python carla_core/training/train_carla_mappo.py --no-gpu --timesteps 10000
"""

import json
import argparse
import csv
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from copy import deepcopy
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
    compute_global_obs_dim_with_mask,
)

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
logger = logging.getLogger(__name__)

def _append_episode_json(path, record):
    """Append a JSON line to the episode log file.
    With num_workers=0 (single CARLA instance), no lock is needed.
    The file is opened/closed per-write for crash safety.
    """
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)

class MAPPOTrainingCallbacks(CentralizedCriticCallbacks):
    """Extends CentralizedCriticCallbacks with episode-level JSON logging."""

    def __init__(self):
        super().__init__()
        self._success_window = deque(maxlen=50)

    def on_episode_end(self, *, worker, base_env, policies, episode,
                       env_index=None, **kwargs):
        # Let parent compute custom_metrics first
        super().on_episode_end(
            worker=worker, base_env=base_env, policies=policies,
            episode=episode, env_index=env_index, **kwargs,
        )
        # Episode JSON logging
        log_path = os.environ.get("MAPPO_EPISODE_LOG")
        outcomes = episode.user_data.get("agent_outcomes", {})
        if log_path:
            for agent_id, out in outcomes.items():
                policy_id = "vehicle_policy" if agent_id.startswith("vehicle") else "pedestrian_policy"
                _append_episode_json(log_path, {
                    "episode_id": episode.episode_id,
                    "agent_id": agent_id,
                    "policy": policy_id,
                    "termination_reason": out["termination_reason"],
                    "route_completion": round(out["route_completion"], 4),
                    "path_efficiency": round(out["path_efficiency"], 4),
                    "step_count": episode.length,
                })

        # Aggregate metrics (all agents, both policies)
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
            episode.custom_metrics["window_success_rate"] = float(
                np.mean(self._success_window)
            )


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


def _sanitize_for_json(obj):
    """Recursively convert non-serializable objects for JSON export."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, type):
        return str(obj)
    elif callable(obj):
        return f"<function {getattr(obj, '__name__', str(obj))}>"
    else:
        return str(obj)


def _find_nonfinite(obj, path="result"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = _find_nonfinite(v, f"{path}.{k}")
            if found:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found = _find_nonfinite(v, f"{path}[{i}]")
            if found:
                return found
        return None

    if isinstance(obj, np.ndarray):
        if not np.isfinite(obj).all():
            bad = np.argwhere(~np.isfinite(obj))
            first = tuple(bad[0].tolist()) if bad.size else ()
            value = obj[first] if first else obj
            return f"{path} contains non-finite values at {first}: {value}"
        return None

    if isinstance(obj, torch.Tensor):
        if not torch.isfinite(obj).all():
            bad = (~torch.isfinite(obj)).nonzero(as_tuple=False)
            first = tuple(bad[0].tolist()) if bad.numel() else ()
            value = obj[first].detach().cpu().item() if first else float('nan')
            return f"{path} contains non-finite values at {first}: {value}"
        return None

    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(float(obj)):
            return f"{path}={obj}"

    return None


def _raise_on_nonfinite_result(result):
    checks = {
        "result.episode_reward_mean": result.get("episode_reward_mean"),
        "result.policy_reward_mean": result.get("policy_reward_mean"),
        "result.sampler_results": result.get("sampler_results"),
        "result.info.learner": result.get("info", {}).get("learner", {}),
    }

    for path, value in checks.items():
        found = _find_nonfinite(value, path)
        if found:
            raise ValueError(f"NaN/Inf detected in training result: {found}")


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


def _safe_rate(numerator, denominator):
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _read_episode_log_delta(log_path, start_offset):
    stats = {"total": 0, "success": 0, "collision": 0}
    if not log_path or not os.path.exists(log_path):
        return 0, stats

    file_size = os.path.getsize(log_path)
    offset = min(max(int(start_offset), 0), file_size)

    with open(log_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                break
            offset = f.tell()
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed episodes.jsonl line at offset %s", offset)
                continue

            termination_reason = record.get("termination_reason")
            stats["total"] += 1
            if termination_reason == "route_complete":
                stats["success"] += 1
            if termination_reason == "collision":
                stats["collision"] += 1

    return offset, stats


def _extract_custom_metric(result, *names):
    containers = [
        result.get("custom_metrics", {}),
        _get_nested(result, "sampler_results", "custom_metrics") or {},
        _get_nested(result, "evaluation", "custom_metrics") or {},
        _get_nested(result, "evaluation", "sampler_results", "custom_metrics") or {},
    ]
    for name in names:
        for container in containers:
            if name in container:
                return _coerce_float(container[name])
    return None


def _extract_reward_std(result):
    reward_lists = [
        _get_nested(result, "hist_stats", "episode_reward"),
        _get_nested(result, "sampler_results", "hist_stats", "episode_reward"),
    ]
    for rewards in reward_lists:
        if rewards:
            arr = np.asarray(rewards, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(np.std(arr))
    return None


def _extract_eval_metric(result, *names):
    containers = [
        _get_nested(result, "evaluation", "custom_metrics") or {},
        _get_nested(result, "evaluation", "sampler_results", "custom_metrics") or {},
    ]
    for name in names:
        for container in containers:
            if name in container:
                return _coerce_float(container[name])
    return None


def _build_evaluation_payload(result):
    sr = _extract_eval_metric(result, "success_rate_mean", "success_rate")
    cr = _extract_eval_metric(
        result,
        "collision_rate_mean",
        "collision_rate",
        "vehicle_collision_rate_mean",
    )

    if sr is None and cr is None:
        return {}

    return {
        "test": {
            "success_rate": sr,
            "collision_rate": cr,
        }
    }


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


def _build_evaluation_summary(raw, train_map):
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
    return "[" + ("\u2588" * filled) + ("\u2591" * (width - filled)) + "]"


def _estimate_eval_progress(
    *,
    total_scenarios,
    completed_scenarios,
    current_elapsed_s,
    completed_scenario_durations,
):
    if total_scenarios <= 0:
        return 0.0, False
    if not completed_scenario_durations:
        return completed_scenarios / total_scenarios, False

    avg_duration = max(float(np.mean(completed_scenario_durations)), 1e-6)
    current_fraction = min(max(current_elapsed_s / avg_duration, 0.0), 0.99)
    progress = (completed_scenarios + current_fraction) / total_scenarios
    return min(max(progress, 0.0), 0.999), True


def _resolve_eval_metric_keys(eval_cfg):
    metric_cfg = eval_cfg.get("metrics", {}) or {}
    metric_aliases = [
        ("success_rate", "success_rate", True),
        ("collision_rate", "collision_rate", True),
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

    # Keep compatibility metrics even if they are toggled off in config.
    for metric_key in ("success_rate", "collision_rate"):
        if metric_key not in selected:
            selected.append(metric_key)

    return selected, unsupported


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
                    {"model": {"custom_model": "cc_model",
                               "custom_model_config": cc_config}},
                ),
                "pedestrian_policy": (
                    None,
                    ped_obs,
                    ped_act,
                    {"model": {"custom_model": "cc_model",
                               "custom_model_config": cc_config}},
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


def _extract_eval_metrics_from_result(eval_result, metric_keys):
    """Extract aggregated metrics from a single RLlib evaluate() result."""
    def _metric(*names):
        containers = [
            eval_result.get("custom_metrics", {}),
            _get_nested(eval_result, "sampler_results", "custom_metrics") or {},
        ]
        for name in names:
            for container in containers:
                if name in container:
                    return _coerce_float(container[name])
        return None

    evaluation_time_ms = _coerce_float(eval_result.get("evaluation_time_ms"))
    wall_clock_seconds = (
        evaluation_time_ms / 1000.0
        if evaluation_time_ms is not None
        else _coerce_float(eval_result.get("time_this_iter_s"))
    )

    rows = {
        "success_rate": _metric("success_rate_mean", "success_rate"),
        "collision_rate": _metric("collision_rate_mean", "collision_rate"),
        "timeout_rate": _metric("timeout_rate_mean", "timeout_rate"),
        "route_completion": _metric("route_completion_mean", "route_completion"),
        "path_efficiency": _metric("path_efficiency_mean", "path_efficiency"),
        "reward_mean": _coerce_float(
            eval_result.get("episode_reward_mean"),
            _coerce_float(_get_nested(eval_result, "sampler_results", "episode_reward_mean")),
        ),
        "episode_length_mean": _coerce_float(
            eval_result.get("episode_len_mean"),
            _coerce_float(_get_nested(eval_result, "sampler_results", "episode_len_mean")),
        ),
        "infraction_count": None,
        "wall_clock_timeout": None,
        "wall_clock_seconds": wall_clock_seconds,
    }
    return {key: rows.get(key) for key in metric_keys}


def _extract_eval_episode_traces(eval_result, map_name, profile_name):
    """Build per-episode trace rows from RLlib hist_stats when available."""
    hist_stats = (
        _get_nested(eval_result, "sampler_results", "hist_stats")
        or eval_result.get("hist_stats")
        or {}
    )
    rewards = hist_stats.get("episode_reward", []) or []
    lengths = hist_stats.get("episode_lengths", []) or []
    n_rows = max(len(rewards), len(lengths))
    traces = []
    for idx in range(n_rows):
        traces.append({
            "map": map_name,
            "profile": profile_name,
            "episode_index": idx,
            "episode_reward": _coerce_float(rewards[idx]) if idx < len(rewards) else None,
            "episode_length": _coerce_float(lengths[idx]) if idx < len(lengths) else None,
        })
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

    for ax in axes[len(available_keys):]:
        ax.remove()

    fig.tight_layout()
    plot_path = out_path / f"{run_name}_evaluation_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_evaluation_scenarios(
    *,
    checkpoint_path,
    base_env_cfg,
    train_cfg,
    eval_cfg,
    seed_base,
    n_gpus,
):
    eval_section = eval_cfg.get("evaluation", {})
    if not eval_section.get("enabled", True):
        return {}, {}, [], []

    scenario_cfg = eval_cfg.get("scenarios", {})
    maps = scenario_cfg.get("maps", [])
    traffic_profiles = scenario_cfg.get("traffic_profiles", [])
    limits = eval_cfg.get("limits", {})

    if not maps or not traffic_profiles:
        return {}, {}, [], []

    parallel_envs = int(eval_section.get("parallel_envs", 1))
    if parallel_envs != 1:
        raise ValueError(
            "eval.parallel_envs=%s is not supported in CARLA multi-agent evaluation "
            "with a single simulator instance. Use parallel_envs=1."
            % parallel_envs
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
    aggregate_keys = list(dict.fromkeys(metric_keys + unsupported_metric_keys))
    save_traces = bool(eval_cfg.get("outputs", {}).get("save_per_episode_trace", False))
    episodes_per_map = int(eval_section.get("episodes_per_map", 1))
    max_steps_per_episode = int(limits.get("max_steps_per_episode", 0)) if "max_steps_per_episode" in limits else None
    total_scenarios = len(maps) * len(traffic_profiles)
    total_episodes = total_scenarios * episodes_per_map
    total_env_steps_budget = (
        total_episodes * max_steps_per_episode
        if max_steps_per_episode is not None
        else None
    )
    scenario_idx = 0
    completed_scenario_durations = []
    eval_t0 = time.time()
    scenario_stall_warning_s = 600.0

    print()
    print(f"{'=' * 19}EVAL{'=' * 20}")
    print(
        f"  Eval scenarios: {total_scenarios} "
        f"({len(maps)} maps x {len(traffic_profiles)} profiles)"
    )
    print(f"  Eval episodes: {total_episodes}")
    if total_env_steps_budget is not None:
        print(f"  Eval max env-steps: {total_env_steps_budget:,}")

    for map_name in maps:
        map_rows = {}
        for profile in traffic_profiles:
            scenario_idx += 1
            profile_name = profile.get("name", "unnamed")
            scenario_t0 = time.time()
            completed_scenarios = scenario_idx - 1
            print(
                f"  [Eval {scenario_idx}/{total_scenarios}] "
                f"{map_name}/{profile_name} | episodes={episodes_per_map}"
            )
            print(f"{'=' * 43}")
            initial_progress = completed_scenarios / total_scenarios
            print(
                f"    Eval progress {_render_progress_bar(initial_progress)} "
                f"{initial_progress * 100:5.1f}% estimated "
                f"({completed_scenarios}/{total_scenarios} scenari completati, "
                f"{scenario_idx}/{total_scenarios} in corso)"
            )
            scenario_env_cfg = deepcopy(base_env_cfg)
            scenario_env_cfg.setdefault("world", {})
            scenario_env_cfg["world"]["map"] = map_name
            scenario_env_cfg.setdefault("traffic", {})
            scenario_env_cfg["traffic"]["n_vehicles_npc"] = int(profile.get("n_vehicles", 0))
            scenario_env_cfg["traffic"]["n_pedestrians_npc"] = int(profile.get("n_pedestrians", 0))
            scenario_env_cfg.setdefault("episode", {})
            if "max_steps_per_episode" in limits:
                scenario_env_cfg["episode"]["max_steps"] = int(limits["max_steps_per_episode"])
            scenario_env_cfg.setdefault("traffic", {})
            scenario_env_cfg["traffic"]["seed"] = int(seed_base)
            eval_config = _build_mappo_config(
                env_cfg=scenario_env_cfg,
                train_cfg=train_cfg,
                eval_cfg=eval_cfg,
                n_gpus=n_gpus,
                n_workers=0,
                exp_seed=seed_base,
                enable_periodic_evaluation=False,
            )
            eval_algo = eval_config.build()
            heartbeat_stop = threading.Event()
            scenario_stall_warned = [False]

            def _heartbeat():
                while not heartbeat_stop.wait(60.0):
                    current_elapsed_s = time.time() - scenario_t0
                    progress, has_estimate = _estimate_eval_progress(
                        total_scenarios=total_scenarios,
                        completed_scenarios=completed_scenarios,
                        current_elapsed_s=current_elapsed_s,
                        completed_scenario_durations=completed_scenario_durations,
                    )
                    estimate_label = "estimated" if has_estimate else "estimated, baseline pending"
                    print(
                        f"    Eval progress {_render_progress_bar(progress)} "
                        f"{progress * 100:5.1f}% {estimate_label} "
                        f"({completed_scenarios}/{total_scenarios} scenari completati, "
                        f"{scenario_idx}/{total_scenarios} in corso)"
                    )
                    if (
                        not scenario_stall_warned[0]
                        and current_elapsed_s >= scenario_stall_warning_s
                    ):
                        print(
                            f"    [WARN] Scenario {map_name}/{profile_name} ancora in corso "
                            f"dopo {current_elapsed_s / 60:.1f}m senza completamento."
                        )
                        scenario_stall_warned[0] = True

            heartbeat_thread = threading.Thread(
                target=_heartbeat,
                name=f"eval-heartbeat-{scenario_idx}",
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                eval_algo.restore(checkpoint_path)
                eval_result = eval_algo.evaluate()
                metrics = _extract_eval_metrics_from_result(eval_result, aggregate_keys)
                map_rows[profile_name] = metrics
                if save_traces:
                    traces.extend(
                        _extract_eval_episode_traces(
                            eval_result,
                            map_name=map_name,
                            profile_name=profile_name,
                        )
                    )
            except KeyboardInterrupt:
                raw[map_name] = map_rows
                print(f"    interrotto dall'utente durante {map_name}/{profile_name}")
                raise FinalEvaluationInterrupted(
                    raw=deepcopy(raw),
                    summary=_build_evaluation_summary(raw, train_map),
                    traces=list(traces),
                    metric_keys=list(aggregate_keys),
                    reason=f"manual interrupt during final evaluation ({map_name}/{profile_name})",
                ) from None
            except Exception as exc:
                raw[map_name] = map_rows
                raise FinalEvaluationFailed(
                    raw=deepcopy(raw),
                    summary=_build_evaluation_summary(raw, train_map),
                    traces=list(traces),
                    metric_keys=list(aggregate_keys),
                    reason=(
                        "final evaluation error: "
                        f"{type(exc).__name__} ({map_name}/{profile_name})"
                    ),
                ) from exc
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
                eval_algo.stop()
            scenario_elapsed = time.time() - scenario_t0
            completed_scenario_durations.append(scenario_elapsed)
            total_elapsed = time.time() - eval_t0
            remaining = total_scenarios - scenario_idx
            eta = (total_elapsed / scenario_idx) * remaining if scenario_idx > 0 else 0.0
            exact_progress = scenario_idx / total_scenarios
            print(
                f"    Eval progress {_render_progress_bar(exact_progress)} "
                f"{exact_progress * 100:5.1f}% exact "
                f"({scenario_idx}/{total_scenarios} scenari completati)"
            )
            print(
                f"    done in {scenario_elapsed/60:.1f}m | "
                f"elapsed {total_elapsed/60:.1f}m | ETA {eta/60:.1f}m"
            )
        raw[map_name] = map_rows

    summary = _build_evaluation_summary(raw, train_map)
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
        with open(json_path, "w") as f:
            json.dump(_sanitize_for_json(payload), f, indent=2)

    if outputs.get("save_csv", False):
        csv_path = out_path / f"{run_name}_evaluation.csv"
        csv_metric_keys = evaluation_metric_keys or []
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["map", "profile", *csv_metric_keys],
            )
            writer.writeheader()
            for map_name, profiles in evaluation_raw.items():
                for profile_name, metrics in profiles.items():
                    row = {"map": map_name, "profile": profile_name}
                    for key in csv_metric_keys:
                        row[key] = metrics.get(key)
                    writer.writerow(row)

    if outputs.get("save_per_episode_trace", False):
        trace_path = out_path / f"{run_name}_evaluation_traces.json"
        with open(trace_path, "w") as f:
            json.dump(_sanitize_for_json(evaluation_traces or []), f, indent=2)

    if outputs.get("save_plots", False):
        _save_evaluation_plots(
            out_path=out_path,
            run_name=run_name,
            evaluation_raw=evaluation_raw,
            metric_keys=evaluation_metric_keys or [],
        )


def _derive_mode(exp_cfg, name):
    mode = exp_cfg.get("mode")
    if mode in {"batch", "curriculum"}:
        return str(mode)

    candidates = [
        str(name),
        str(exp_cfg.get("name", "")),
        str(exp_cfg.get("output_dir", "")),
    ]
    for candidate in candidates:
        lower_name = candidate.lower()
        if "curriculum" in lower_name:
            return "curriculum"
        if "batch" in lower_name:
            return "batch"
    return "unknown"


def _build_results_payload(
    *,
    exp_cfg,
    name,
    total_ts,
    ts_done,
    elapsed_s,
    result,
    timeseries,
    evaluation=None,
    evaluation_raw=None,
    cumulative_success_rate=None,
    cumulative_collision_rate=None,
    final_evaluation_skip_reason=None,
):
    result = result or {}
    status = "COMPLETATO" if ts_done >= total_ts else "STOP_EARLY_COLLASSO"
    mode = _derive_mode(exp_cfg, name)
    total_episodes = int(result.get("episodes_total", 0) or 0)

    success_rate = cumulative_success_rate
    if success_rate is None:
        success_rate = _extract_custom_metric(result, "success_rate_mean", "success_rate")

    collision_rate = cumulative_collision_rate
    if collision_rate is None:
        collision_rate = _extract_custom_metric(
            result,
            "collision_rate_mean",
            "collision_rate",
            "vehicle_collision_rate_mean",
        )

    evaluation = evaluation if evaluation is not None else _build_evaluation_payload(result)
    evaluation_raw = evaluation_raw or {}

    payload = {
        "meta": {
            "mode": mode,
            "status": status,
            "simulator": "CARLA",
            "algorithm": "MAPPO",
            "name": name,
            "seed": exp_cfg.get("seed", 42),
            "total_timesteps_budget": int(total_ts),
            "total_timesteps_actual": int(ts_done),
            "total_episodes": total_episodes,
            "wall_clock_seconds": float(elapsed_s),
            "final_evaluation_completed": final_evaluation_skip_reason is None,
        },
        "timeseries": timeseries,
        "evaluation": evaluation,
        "evaluation_raw": evaluation_raw,
        "training_summary": {
            "cumulative_success_rate": float("nan") if success_rate is None else float(success_rate),
            "cumulative_collision_rate": float("nan") if collision_rate is None else float(collision_rate),
            "best_reward_mean": _coerce_float(max((p["reward_mean"] for p in timeseries if p["reward_mean"] is not None), default=None), float("nan")),
            "final_reward_mean": _coerce_float(result.get("episode_reward_mean"), float("nan")),
            "final_episode_length_mean": _coerce_float(result.get("episode_len_mean"), float("nan")),
        },
        "curriculum_history": [],
    }
    if final_evaluation_skip_reason is not None:
        payload["meta"]["final_evaluation_skip_reason"] = str(final_evaluation_skip_reason)
    return payload

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAPPO Training on CARLA")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--mode", type=str, choices=["batch", "curriculum"], default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--env-config", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    # --- Load configs ---
    train_cfg = load_yaml(args.train_config or base / "configs" / "train_mappo.yaml")
    env_cfg = load_yaml(args.env_config or base / "configs" / "multi_agent.yaml")
    eval_cfg = load_yaml(base / "configs" / "eval.yaml")

    sched = train_cfg.get("schedule", {})
    res = train_cfg.get("resources", {})
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})
    model_cfg = train_cfg.get("model", {})
    stop_on_nan = sched.get("stop_on_nan", True)
    eval_section = eval_cfg.get("evaluation", {})

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
    global_obs_dim = compute_global_obs_dim_with_mask(n_veh, n_ped)

    # Wire remaining config fields
    exp_cfg = train_cfg.get("experiment", {})
    if args.mode:
        exp_cfg["mode"] = args.mode
    out_base = exp_cfg.get("output_dir", str(base / "experiments"))

    # Output dir
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    name = exp_cfg.get("name", "carla_mappo")
    out_dir = args.checkpoint_dir or str(Path(out_base) / f"{name}_{ts_str}")
    resolved_mode = _derive_mode(
        {**exp_cfg, "output_dir": out_base},
        out_dir,
    )
    if resolved_mode == "unknown":
        raise ValueError(
            "Unable to resolve experiment mode. Set experiment.mode to "
            "'batch' or 'curriculum', or pass --mode explicitly."
        )
    exp_cfg["mode"] = resolved_mode
    os.makedirs(out_dir, exist_ok=True)

    # Episode-level JSON log path (read by MAPPOTrainingCallbacks)
    os.environ["MAPPO_EPISODE_LOG"] = os.path.join(out_dir, "episodes.jsonl")
    episode_log_path = os.environ["MAPPO_EPISODE_LOG"]
    episode_log_offset = os.path.getsize(episode_log_path) if os.path.exists(episode_log_path) else 0
    cumulative_agent_outcomes = {"total": 0, "success": 0, "collision": 0}

    print(f"{'=' * 60}")
    print(f"CARLA MAPPO Training — Centralized Critic (CTDE)")
    print(f"{'=' * 60}")
    print(f"  Agents: {n_veh}V + {n_ped}P | global_obs: {global_obs_dim}D")
    print(f"  Budget: {total_ts:,} steps | Workers: {n_workers} | GPU: {n_gpus}")
    print(f"  Policies: vehicle({VEHICLE_OBS_DIM}D), pedestrian({PEDESTRIAN_OBS_DIM}D)")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}\n")

    run_meta = {
        "timestamp": ts_str,
        "name": name,
        "mode": _derive_mode(exp_cfg, name),
        "total_timesteps": total_ts,
        "seed": exp_seed,
        "n_vehicles_rl": n_veh,
        "n_pedestrians_rl": n_ped,
        "global_obs_dim": global_obs_dim,
        "optimization": opt,
        "rollout": roll,
        "model": model_cfg,
        "env_config": env_cfg,
        "eval_config": eval_cfg,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    # --- Ray init ---
    ray.init(num_cpus=max(n_workers + 2, 2), num_gpus=n_gpus, log_to_driver=False)

    torch.manual_seed(exp_seed)
    np.random.seed(exp_seed)
    random.seed(exp_seed)

    # --- Register ---
    register_env("CarlaMultiAgent-v0", rllib_env_creator)
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    # --- PPO Config ---
    config = _build_mappo_config(
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_gpus=n_gpus,
        n_workers=n_workers,
        exp_seed=exp_seed,
        enable_periodic_evaluation=False,
    )

    # --- Build & Train ---
    algo = config.build()
    print("MAPPO training avviato.\n")

    ts_done = 0
    iteration = 0
    t0 = time.time()
    best_reward = float("-inf")
    timeseries = []
    result = {}
    evaluation = {}
    evaluation_raw = {}
    evaluation_traces = []
    evaluation_metric_keys = []
    cumulative_success_rate = None
    cumulative_collision_rate = None
    should_run_final_evaluation = True
    final_evaluation_skip_reason = None
    final_checkpoint = None

    try:
        while ts_done < total_ts:
            result = algo.train()
            iteration += 1
            ts_done = result.get("timesteps_total", 0)

            if stop_on_nan:
                _raise_on_nonfinite_result(result)

            # Per-policy rewards
            pol_rew = result.get("policy_reward_mean", {})
            veh_r = pol_rew.get("vehicle_policy", 0)
            ped_r = pol_rew.get("pedestrian_policy", 0)
            tot_r = result.get("episode_reward_mean", 0)
            ep_len = result.get("episode_len_mean", 0)
            eps = result.get("episodes_total", 0)

            episode_log_offset, delta_stats = _read_episode_log_delta(
                episode_log_path, episode_log_offset
            )
            for key, value in delta_stats.items():
                cumulative_agent_outcomes[key] += value
            cumulative_success_rate = _safe_rate(
                cumulative_agent_outcomes["success"],
                cumulative_agent_outcomes["total"],
            )
            cumulative_collision_rate = _safe_rate(
                cumulative_agent_outcomes["collision"],
                cumulative_agent_outcomes["total"],
            )

            elapsed = time.time() - t0
            pct = ts_done / total_ts * 100
            eta = (elapsed / max(ts_done, 1)) * (total_ts - ts_done)

            timeseries.append({
                "timestep": int(ts_done),
                "success_rate": cumulative_success_rate,
                "collision_rate": cumulative_collision_rate,
                "window_success_rate": _extract_custom_metric(
                    result,
                    "window_success_rate_mean",
                    "window_success_rate",
                ),
                "reward_mean": _coerce_float(tot_r),
                "reward_std": _extract_reward_std(result),
                "episode_length_mean": _coerce_float(ep_len),
            })

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
        should_run_final_evaluation = False
        final_evaluation_skip_reason = "manual interrupt"
        print("\nInterrotto.")
    except Exception as e:
        should_run_final_evaluation = False
        final_evaluation_skip_reason = f"training error: {type(e).__name__}"
        print(f"\n[ERROR] Training crash at step {ts_done}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Salva su file
        log_path = os.path.join(out_dir, "crash_log.txt")
        with open(log_path, "w") as f:
            f.write(f"Step: {ts_done}\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {type(e).__name__}: {e}\n\n")
            traceback.print_exc(file=f)
    finally:
        try:
            final_checkpoint = algo.save(out_dir)
            print(f"\nCheckpoint finale: {final_checkpoint}")
        except Exception as e:
            print(f"\nCheckpoint fallito: {e}")

        # Save last training result (full RLlib dict) before stopping the trainer.
        try:
            if iteration > 0:
                result_path = os.path.join(out_dir, "last_result.json")
                with open(result_path, "w") as f:
                    json.dump(_sanitize_for_json(result), f, indent=2)
                print(f"  Last result: {result_path}")
        except Exception as e:
            print(f"  [WARN] Could not save last_result.json: {e}")

        try:
            algo.stop()
        except Exception as e:
            print(f"  [WARN] algo.stop() failed: {e}")
        finally:
            algo = None

        try:
            if (
                eval_section.get("enabled", True)
                and iteration > 0
                and should_run_final_evaluation
                and final_checkpoint is not None
            ):
                print("\nValutazione finale multi-scenario in corso...")
                evaluation_raw, evaluation, evaluation_traces, evaluation_metric_keys = _run_evaluation_scenarios(
                    checkpoint_path=final_checkpoint,
                    base_env_cfg=env_cfg,
                    train_cfg=train_cfg,
                    eval_cfg=eval_cfg,
                    seed_base=exp_seed,
                    n_gpus=n_gpus,
                )
                if evaluation or evaluation_raw:
                    _save_evaluation_artifacts(
                        base_dir=base.parent,
                        run_name=Path(out_dir).name,
                        eval_cfg=eval_cfg,
                        evaluation_raw=evaluation_raw,
                        evaluation_summary=evaluation,
                        evaluation_traces=evaluation_traces,
                        evaluation_metric_keys=evaluation_metric_keys,
                    )
            else:
                reason = final_evaluation_skip_reason or (
                    "disabled, no completed iterations, or missing final checkpoint"
                )
                print(f"\nValutazione finale multi-scenario saltata ({reason}).")
        except FinalEvaluationInterrupted as exc:
            final_evaluation_skip_reason = exc.reason
            evaluation_raw = exc.raw
            evaluation = exc.summary
            evaluation_traces = exc.traces
            evaluation_metric_keys = exc.metric_keys
            if evaluation or evaluation_raw:
                _save_evaluation_artifacts(
                    base_dir=base.parent,
                    run_name=Path(out_dir).name,
                    eval_cfg=eval_cfg,
                    evaluation_raw=evaluation_raw,
                    evaluation_summary=evaluation,
                    evaluation_traces=evaluation_traces,
                    evaluation_metric_keys=evaluation_metric_keys,
                )
            print("\nValutazione finale interrotta manualmente.")
        except FinalEvaluationFailed as exc:
            final_evaluation_skip_reason = exc.reason
            evaluation_raw = exc.raw
            evaluation = exc.summary
            evaluation_traces = exc.traces
            evaluation_metric_keys = exc.metric_keys
            if evaluation or evaluation_raw:
                _save_evaluation_artifacts(
                    base_dir=base.parent,
                    run_name=Path(out_dir).name,
                    eval_cfg=eval_cfg,
                    evaluation_raw=evaluation_raw,
                    evaluation_summary=evaluation,
                    evaluation_traces=evaluation_traces,
                    evaluation_metric_keys=evaluation_metric_keys,
                )
            print(f"  [WARN] Final evaluation failed: {exc.reason}")
        except KeyboardInterrupt:
            final_evaluation_skip_reason = "manual interrupt during final evaluation"
            evaluation_raw = evaluation_raw or {}
            evaluation = evaluation or {}
            evaluation_traces = evaluation_traces or []
            evaluation_metric_keys = evaluation_metric_keys or []
            print("\nValutazione finale interrotta manualmente.")
        except Exception as e:
            final_evaluation_skip_reason = f"final evaluation error: {type(e).__name__}"
            print(f"  [WARN] Final evaluation failed: {e}")

        print(f"\n{'=' * 60}")
        print(f"MAPPO Training Completato")
        print(f"{'=' * 60}")
        print(f"  Steps: {ts_done:,} | Best reward: {best_reward:.1f}")
        print(f"  Tempo: {(time.time() - t0) / 60:.1f} min")
        print(f"  Output: {out_dir}")

        try:
            results_payload = _build_results_payload(
                exp_cfg=exp_cfg,
                name=name,
                total_ts=total_ts,
                ts_done=ts_done,
                elapsed_s=time.time() - t0,
                result=result,
                timeseries=timeseries,
                evaluation=evaluation,
                evaluation_raw=evaluation_raw,
                cumulative_success_rate=cumulative_success_rate,
                cumulative_collision_rate=cumulative_collision_rate,
                final_evaluation_skip_reason=final_evaluation_skip_reason,
            )
            results_path = os.path.join(out_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(_sanitize_for_json(results_payload), f, indent=2)
            print(f"  Results schema: {results_path}")
        except Exception as e:
            print(f"  [WARN] Could not save results.json: {e}")

        ray.shutdown()


if __name__ == "__main__":
    main()
