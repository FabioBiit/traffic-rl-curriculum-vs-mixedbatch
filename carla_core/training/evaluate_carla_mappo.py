import argparse
import csv
import gc
import json
import logging
import os
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
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


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


def _run_evaluation_scenarios(
    *,
    checkpoint_path,
    base_env_cfg,
    train_cfg,
    eval_cfg,
    seed_base,
    n_gpus,
    status_callback=None,
    scenario_setup_callback=None,
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
    maps = scenario_cfg.get("maps", [])
    traffic_profiles = scenario_cfg.get("traffic_profiles", [])
    limits = eval_cfg.get("limits", {})

    if not maps or not traffic_profiles:
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
    aggregate_keys = list(dict.fromkeys(metric_keys + unsupported_metric_keys))
    save_traces = bool(eval_cfg.get("outputs", {}).get("save_per_episode_trace", False))
    episodes_per_map = int(eval_section.get("episodes_per_map", 1))
    max_steps_per_episode = (
        int(limits.get("max_steps_per_episode", 0))
        if "max_steps_per_episode" in limits
        else None
    )
    total_scenarios = len(maps) * len(traffic_profiles)
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
    print(
        f"  Eval scenarios: {total_scenarios} ({len(maps)} maps x {len(traffic_profiles)} profiles)"
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
        current_phase="starting",
        current_scenario_idx=0,
        current_map=None,
        current_profile=None,
        progress_mode="exact",
    )

    for map_name in maps:
        map_rows = {}
        for profile in traffic_profiles:
            scenario_idx += 1
            profile_name = profile.get("name", "unnamed")
            scenario_t0 = time.time()
            completed_scenarios = scenario_idx - 1
            scenario_episode_rows = []
            _publish_status(
                reason="evaluation subprocess running",
                progress=(completed_episodes_total / total_episodes if total_episodes else 0.0),
                completed_scenarios=completed_scenarios,
                total_scenarios=total_scenarios,
                total_episodes=total_episodes,
                current_phase="scenario_pending",
                current_scenario_idx=scenario_idx,
                current_map=map_name,
                current_profile=profile_name,
                progress_mode="exact",
            )
            print(
                f"  [Eval {scenario_idx}/{total_scenarios}] {map_name}/{profile_name} | "
                f"episodes={episodes_per_map}"
            )
            print(f"{'=' * 43}")
            for episode_idx in range(episodes_per_map):
                scenario_env_cfg = deepcopy(base_env_cfg)
                scenario_env_cfg.setdefault("world", {})
                scenario_env_cfg["world"]["map"] = map_name
                scenario_env_cfg.setdefault("traffic", {})
                scenario_env_cfg["traffic"]["n_vehicles_npc"] = int(profile.get("n_vehicles", 0))
                scenario_env_cfg["traffic"]["n_pedestrians_npc"] = int(
                    profile.get("n_pedestrians", 0)
                )
                scenario_env_cfg.setdefault("episode", {})
                if "max_steps_per_episode" in limits:
                    scenario_env_cfg["episode"]["max_steps"] = int(limits["max_steps_per_episode"])
                scenario_env_cfg.setdefault("traffic", {})
                scenario_env_cfg["traffic"]["seed"] = int(seed_base)
                scenario_env_cfg.setdefault("runtime", {})
                scenario_env_cfg["runtime"]["close_mode"] = "robust"

                current_episode_number = episode_idx + 1
                exact_progress = (
                    completed_episodes_total / total_episodes if total_episodes else 0.0
                )
                if scenario_setup_callback is not None:
                    _publish_status(
                        reason="evaluation subprocess running",
                        progress=exact_progress,
                        completed_scenarios=completed_scenarios,
                        total_scenarios=total_scenarios,
                        total_episodes=total_episodes,
                        completed_episodes=completed_episodes_total,
                        episodes_per_scenario=episodes_per_map,
                        current_episode_idx=current_episode_number,
                        current_phase="episode_setup",
                        current_scenario_idx=scenario_idx,
                        current_map=map_name,
                        current_profile=profile_name,
                        progress_mode="exact",
                    )
                    try:
                        scenario_setup_callback(
                            map_name=map_name,
                            profile_name=profile_name,
                            scenario_idx=scenario_idx,
                            total_scenarios=total_scenarios,
                            env_cfg=deepcopy(scenario_env_cfg),
                            episode_idx=current_episode_number,
                            episodes_per_scenario=episodes_per_map,
                            completed_episodes=completed_episodes_total,
                            total_episodes=total_episodes,
                        )
                    except Exception as exc:
                        if scenario_episode_rows:
                            map_rows[profile_name] = _aggregate_eval_metric_rows(
                                scenario_episode_rows, aggregate_keys
                            )
                        raw[map_name] = map_rows
                        raise FinalEvaluationFailed(
                            raw=deepcopy(raw),
                            summary=_build_evaluation_summary(raw, train_map),
                            traces=list(traces),
                            metric_keys=list(aggregate_keys),
                            reason=(
                                "final evaluation episode setup error: "
                                f"{type(exc).__name__} ({map_name}/{profile_name}, "
                                f"episode {current_episode_number}/{episodes_per_map})"
                            ),
                        ) from exc

                single_episode_eval_cfg = deepcopy(eval_cfg)
                single_episode_eval_cfg.setdefault("evaluation", {})
                single_episode_eval_cfg["evaluation"]["episodes_per_map"] = 1
                eval_config = _build_mappo_config(
                    env_cfg=scenario_env_cfg,
                    train_cfg=train_cfg,
                    eval_cfg=single_episode_eval_cfg,
                    n_gpus=n_gpus,
                    n_workers=0,
                    exp_seed=seed_base,
                    enable_periodic_evaluation=False,
                )
                eval_algo = None
                heartbeat_stop = threading.Event()
                episode_t0 = time.time()
                episode_stall_warned = [False]

                def _heartbeat():
                    while not heartbeat_stop.wait(heartbeat_interval_s):
                        progress = (
                            completed_episodes_total / total_episodes if total_episodes else 0.0
                        )
                        print(
                            f"Eval progress {_render_progress_bar(progress)} "
                            f"{progress * 100:5.1f}% exact "
                            f"({completed_episodes_total}/{total_episodes} episodi completati, "
                            f"scenario {scenario_idx}/{total_scenarios}, "
                            f"episodio {current_episode_number}/{episodes_per_map})"
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
                    eval_algo = eval_config.build()
                    eval_algo.restore(checkpoint_path)
                    eval_result = eval_algo.evaluate()
                    episode_metrics = _extract_eval_metrics_from_result(eval_result, aggregate_keys)
                    scenario_episode_rows.append(episode_metrics)
                    if save_traces:
                        traces.extend(
                            _offset_eval_episode_traces(
                                _extract_eval_episode_traces(
                                    eval_result,
                                    map_name=map_name,
                                    profile_name=profile_name,
                                ),
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
                        summary=_build_evaluation_summary(raw, train_map),
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
                        summary=_build_evaluation_summary(raw, train_map),
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
                    if eval_algo is not None:
                        try:
                            eval_algo.stop()
                        except Exception as stop_exc:
                            print(
                                "    [WARN] eval_algo.stop() failed during "
                                f"{map_name}/{profile_name} episodio "
                                f"{current_episode_number}/{episodes_per_map}: "
                                f"{type(stop_exc).__name__}: {stop_exc}"
                            )
                        finally:
                            del eval_algo
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
            )
            print(
                f"Eval progress {_render_progress_bar(exact_progress)} {exact_progress * 100:5.1f}% "
                f"exact ({completed_episodes_total}/{total_episodes} episodi, "
                f"{scenario_idx}/{total_scenarios} scenari completati)"
            )
            print(
                f"Done in {int(scenario_elapsed//3600)}h{int((scenario_elapsed%3600)//60):02d}m | "
                f"Elapsed {int(total_elapsed//3600)}h{int((total_elapsed%3600)//60):02d}m | "
                f"ETA {int(eta//3600)}h{int((eta%3600)//60):02d}m"
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
        world = client.load_world(target_map)
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = bool(sim.get("sync_mode", True))
    settings.fixed_delta_seconds = sim["fixed_delta_seconds"]
    if env_cfg["world"].get("no_rendering", False):
        settings.no_rendering_mode = True
    world.apply_settings(settings)

    # CARLA determinism guidance recommends reloading the world for each new
    # repetition while keeping the synchronous settings already applied.
    if force_reload:
        world = client.reload_world(False)
        if world is None:
            world = client.get_world()
        weather = env_cfg["world"].get("weather_preset", "ClearNoon")
        if hasattr(carla.WeatherParameters, weather):
            world.set_weather(getattr(carla.WeatherParameters, weather))

    tm = client.get_trafficmanager(int(sim.get("traffic_manager_port", 8000)))
    tm.set_synchronous_mode(bool(sim.get("sync_mode", True)))
    tm.set_random_device_seed(int(env_cfg["traffic"].get("seed", 42)))
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
    payload = dict(state)
    progress = float(payload.get("progress", 0.0) or 0.0)
    progress = min(max(progress, 0.0), 1.0)
    payload["progress"] = progress
    payload["progress_bar"] = _render_progress_bar(progress)
    _write_status(status_path, payload)


def main():
    parser = argparse.ArgumentParser(description="Isolated CARLA MAPPO final evaluation")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

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
        runtime_cfg = job.get("eval_cfg", {}).get("runtime", {})
        reload_world_between_episodes = bool(
            runtime_cfg.get(
                "reload_world_between_episodes",
                runtime_cfg.get("reload_world_between_scenarios", True),
            )
        )

        _carla_prepare_world(job["env_cfg"], force_reload=False)
        _register_eval_runtime()

        def _status_callback(**fields):
            _update_running_status(status_path, status_state, **fields)

        def _scenario_setup_callback(**fields):
            if not reload_world_between_episodes:
                return
            gc.collect()
            _carla_prepare_world(fields["env_cfg"], force_reload=True)

        evaluation_raw, evaluation_summary, evaluation_traces, evaluation_metric_keys = (
            _run_evaluation_scenarios(
                checkpoint_path=job["checkpoint_path"],
                base_env_cfg=deepcopy(job["env_cfg"]),
                train_cfg=deepcopy(job["train_cfg"]),
                eval_cfg=deepcopy(job["eval_cfg"]),
                seed_base=int(job["seed_base"]),
                n_gpus=int(job["n_gpus"]),
                status_callback=_status_callback,
                scenario_setup_callback=_scenario_setup_callback,
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
            {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": True,
                "reason": None,
                "exit_code": 0,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
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
            },
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
            {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": exc.reason,
                "exit_code": 130,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
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
            },
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
            {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": exc.reason,
                "exit_code": 1,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
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
            },
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
            {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": "manual interrupt during evaluation subprocess",
                "exit_code": 130,
                "artifacts_written": False,
                "started_at": started_at,
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
            },
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
            {
                "session_id": session_id,
                "pid": os.getpid(),
                "completed": False,
                "reason": f"evaluation subprocess error: {type(exc).__name__}",
                "exit_code": 1,
                "artifacts_written": False,
                "started_at": started_at,
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
            },
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
