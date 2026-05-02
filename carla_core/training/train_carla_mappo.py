"""
Train MAPPO on CarlaMultiAgentEnv - Multi-Agent via RLlib (v0.1)
================================================================
MAPPO = PPO + centralized critic (CTDE paradigm).

Architecture:
  - vehicle_policy:    actor sees 25D local obs, critic sees global_obs (138D)
  - pedestrian_policy: actor sees 19D local obs, critic sees global_obs (138D)
  - global_obs = fixed-slot concat [v0_25D|v1|v2|p0_19D|p1|p2|alive_mask_6D] = 138D
  
Components:
  - CarlaMultiAgentEnv (PettingZoo ParallelEnv) -> ParallelPettingZooEnv
  - CentralizedCriticModel (custom TorchModelV2)
  - CentralizedCriticCallbacks (injects global_obs, recomputes GAE)

Uso:
    python carla_core/training/train_carla_mappo.py
    python carla_core/training/train_carla_mappo.py --timesteps 50000 --workers 0
    python carla_core/training/train_carla_mappo.py --no-gpu --timesteps 10000
"""

import json
import argparse
import gc
import logging
import os
import signal
import subprocess
import sys
import time
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
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from carla_core.envs.carla_multi_agent_env import (
    apply_level_config,
    CarlaMultiAgentEnv,
    VEHICLE_OBS_DIM,
    PEDESTRIAN_OBS_DIM,
)
from carla_core.agents.centralized_critic import (
    CentralizedCriticModel,
    compute_global_obs_dim_with_mask,
)
from carla_core.training.mappo_runtime import _build_mappo_config
from carla_core.training.curriculum_batch_manager import (
    EpisodeTracker,
    CurriculumManager,
    BatchLevelSampler,
)

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
logger = logging.getLogger(__name__)


def _shutdown_carla_server():
    if sys.platform != "win32":
        return False, "CARLA shutdown skipped (non-Windows platform)."

    try:
        proc = subprocess.run(
            ["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"CARLA shutdown command failed: {type(exc).__name__}: {exc}"

    output = " ".join(
        part.strip() for part in [proc.stdout or "", proc.stderr or ""] if part and part.strip()
    ).strip()
    if proc.returncode == 0:
        return True, output or "CARLA server terminated."
    return False, output or f"taskkill exited with code {proc.returncode}"


def _resolve_project_path(path_value, *, project_root):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve(strict=False)


def _coerce_path_candidate(value):
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        try:
            candidate = Path(value)
        except (TypeError, ValueError, OSError):
            return None
        return candidate if str(candidate).strip() else None
    return None


def _resolve_checkpoint_path(checkpoint_result, *, project_root):
    pending = [checkpoint_result]
    seen = set()

    while pending:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        candidate = _coerce_path_candidate(current)
        if candidate is not None:
            return _resolve_project_path(candidate, project_root=project_root)

        for attr_name in ("path", "checkpoint", "checkpoint_dir"):
            try:
                nested = getattr(current, attr_name)
            except Exception:
                continue
            if nested is None or nested is current:
                continue
            pending.append(nested)

    return None


def _validate_final_eval_artifacts(*, checkpoint_path, out_dir):
    issues = []
    if checkpoint_path is None:
        issues.append("checkpoint path could not be resolved from algo.save() result")
    elif not Path(checkpoint_path).exists():
        issues.append(f"checkpoint missing: {checkpoint_path}")

    last_result_path = Path(out_dir) / "last_result.json"
    if not last_result_path.exists():
        issues.append(f"last_result.json missing: {last_result_path}")

    return issues


def load_yaml(path):
    p = Path(path)
    return yaml.safe_load(open(p)) if p.exists() else {}


def rllib_env_creator(env_config):
    """Wrap CarlaMultiAgentEnv for RLlib via ParallelPettingZooEnv."""
    raw_env = CarlaMultiAgentEnv(config=env_config)
    return ParallelPettingZooEnv(raw_env)


def _unwrap_carla_env(algo):
    """Unwrap CarlaMultiAgentEnv from RLlib worker for set_level() access.

    With num_workers=0, the env lives in the local worker.
    Returns None if unwrapping fails (e.g. remote workers).
    """
    try:
        worker = algo.workers.local_worker()
        env = worker.env
        # ParallelPettingZooEnv → par_env → CarlaMultiAgentEnv
        inner = getattr(env, "par_env", None) or getattr(env, "env", None)
        if inner is None:
            inner = env
        if hasattr(inner, "set_level"):
            return inner
        # One more level (some RLlib versions)
        deeper = getattr(inner, "env", None)
        if deeper is not None and hasattr(deeper, "set_level"):
            return deeper
    except Exception as e:
        logger.warning("Could not unwrap CARLA env: %s", e)
    return None


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


def _apply_delta_stats_to_tracker(tracker, delta_stats):
    if tracker is None:
        return
    tracker.record_counts(
        successes=delta_stats.get("success", 0),
        collisions=delta_stats.get("collision", 0),
        total=delta_stats.get("total", 0),
    )


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

def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(payload), f, indent=2)
    os.replace(tmp_path, path)


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


def _write_final_eval_job(
    *,
    out_dir,
    checkpoint_path,
    env_cfg,
    train_cfg,
    eval_cfg,
    seed_base,
    n_gpus,
    results_path,
):
    job_path = Path(out_dir) / "final_eval_job.json"
    session_id = Path(out_dir).name
    payload = {
        "session_id": session_id,
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(out_dir),
        "run_name": session_id,
        "seed_base": int(seed_base),
        "n_gpus": int(n_gpus),
        "results_path": str(results_path),
        "env_cfg": deepcopy(env_cfg),
        "train_cfg": deepcopy(train_cfg),
        "eval_cfg": deepcopy(eval_cfg),
    }
    _write_json_atomic(job_path, payload)
    return str(job_path)



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


def _scope_output_base_by_mode(out_base_path, mode):
    """Nest experiment outputs under experiments/<mode> when applicable."""
    out_base_path = Path(out_base_path)
    if mode not in {"batch", "curriculum"}:
        return out_base_path
    if out_base_path.name.lower() == mode:
        return out_base_path
    return out_base_path / mode


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
    parser.add_argument("--use-popart", action="store_true",
                        help="Override model.use_popart=true (Block 4.2)")
    parser.add_argument("--use-attention", action="store_true",
                        help="Override model.use_attention=true (Block 4.4)")
    parser.add_argument("--use-gnn", action="store_true",
                        help="Override model.use_gnn=true (Block 4.5). "
                             "Combine with --use-attention -> GAT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, choices=["path", "traffic", "mixed"],
                        required=True)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    # --- Load configs ---
    train_cfg = load_yaml(args.train_config or base / "configs" / "train_mappo.yaml")
    env_cfg = load_yaml(args.env_config or base / "configs" / "multi_agent.yaml")
    eval_cfg = load_yaml(base / "configs" / "eval.yaml")

    if args.use_popart:
        train_cfg.setdefault("model", {})["use_popart"] = True
        logger.info("CLI override: use_popart=True (Block 4.2)")
    if args.use_attention:
        train_cfg.setdefault("model", {})["use_attention"] = True
        logger.info("CLI override: use_attention=True (Block 4.4)")
    if args.use_gnn:
        train_cfg.setdefault("model", {})["use_gnn"] = True
        logger.info("CLI override: use_gnn=True (Block 4.5)")

    sched = train_cfg.get("schedule", {})
    res = train_cfg.get("resources", {})
    opt = train_cfg.get("optimization", {})
    roll = train_cfg.get("rollout", {})
    model_cfg = train_cfg.get("model", {})
    runtime_cfg = train_cfg.get("runtime", {})
    stop_on_nan = sched.get("stop_on_nan", True)
    eval_section = eval_cfg.get("evaluation", {})

    total_ts = args.timesteps or sched.get("total_timesteps", 50_000)
    n_workers = args.workers if args.workers is not None else res.get("num_workers", 0)
    
    if n_workers > 0:
        print("[WARNING] num_workers > 0 requires separate CARLA instances per worker.")
        print("          Forcing num_workers = 0 (single CARLA instance).")
        n_workers = 0

    exp_seed = args.seed
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
    project_root = base.parent
    out_base = exp_cfg.get("output_dir", str(base / "experiments"))
    out_base_path = _resolve_project_path(out_base, project_root=project_root)

    # Output dir
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    name = exp_cfg.get("name", "carla_mappo")
    if args.checkpoint_dir:
        tentative_out_dir = str(
            _resolve_project_path(args.checkpoint_dir, project_root=project_root)
        )
    else:
        tentative_out_dir = str((out_base_path / f"{name}_{ts_str}").resolve(strict=False))
    resolved_mode = _derive_mode(
        {**exp_cfg, "output_dir": str(out_base_path)},
        tentative_out_dir,
    )
    if resolved_mode == "unknown":
        raise ValueError(
            "Unable to resolve experiment mode. Set experiment.mode to "
            "'batch' or 'curriculum', or pass --mode explicitly."
        )
    exp_cfg["mode"] = resolved_mode

    if args.checkpoint_dir:
        out_dir = tentative_out_dir
    else:
        out_base_path = _scope_output_base_by_mode(out_base_path, resolved_mode)
        exp_cfg["output_dir"] = str(out_base_path)
        out_dir = str((out_base_path / f"{name}_{ts_str}").resolve(strict=False))
    os.makedirs(out_dir, exist_ok=True)

    # Episode-level JSON log path (read by MAPPOTrainingCallbacks)
    os.environ["MAPPO_EPISODE_LOG"] = os.path.join(out_dir, "episodes.jsonl")
    episode_log_path = os.environ["MAPPO_EPISODE_LOG"]
    episode_log_offset = os.path.getsize(episode_log_path) if os.path.exists(episode_log_path) else 0
    cumulative_agent_outcomes = {"total": 0, "success": 0, "collision": 0}
    cb_cfg = load_yaml(base / "configs" / "curriculum_batch.yaml")

    # Apply conservative optimizer overrides for multi-level runs
    if resolved_mode in ("curriculum", "batch"):
        opt_overrides = cb_cfg.get("optimization_overrides", {})
        if opt_overrides:
            for k, v in opt_overrides.items():
                if k in opt or k in roll:
                    target = roll if k in roll else opt
                    old_val = target.get(k)
                    target[k] = v
                    logger.info("Override %s: %s -> %s (multi-level stabilization)", k, old_val, v)

    lv = load_yaml(base / "configs" / "levels.yaml")
    env_cfg["levels"] = lv[f"levels_{args.difficulty}"]

    build_env_cfg = deepcopy(env_cfg)
    level_manager = None
    level_tracker = None
    executed_level_trackers = None
    current_training_level = None
    initial_level = None
    initial_map = None
    teacher_diagnostics = None

    if resolved_mode == "curriculum":
        cc = cb_cfg.get("curriculum", {}) # Finetuning Run3
        window_size = cc.get("window_size", 50)
        level_manager = CurriculumManager(
            levels=cc.get("levels", ["easy", "medium", "hard"]),
            total_budget_timesteps=total_ts,
            default_success_rate_threshold=cc.get("success_rate_threshold", 0.45),
            default_collision_rate_threshold=cc.get(
                "collision_rate_threshold",
                cc.get("collision_threshold", 0.30),
            ),
            default_min_episodes=cc.get("min_episodes", 50),
            unlock_criteria=cc.get("unlock_criteria", {}),
            budget_constraints=cc.get("budget_constraints", {}),
            base_sampling_weights=cc.get("base_sampling_weights", {}),
            probation_sampling_weights=cc.get("probation_sampling_weights", {}),
            probation_blocks_after_unlock=cc.get("probation_blocks_after_unlock", 2),
            probation_blocks_after_cap_pressure=cc.get("probation_blocks_after_cap_pressure", 1),
            teacher_seed=cc.get("teacher_seed", exp_seed),
            window_size=window_size,
        )
        executed_level_trackers = {
            level_name: EpisodeTracker(window_size=window_size)
            for level_name in level_manager.levels
        }
        level_tracker = None
        initial_level = level_manager.current_level
        initial_map = apply_level_config(build_env_cfg, initial_level)
    elif resolved_mode == "batch":
        bc = cb_cfg.get("batch", {})
        level_manager = BatchLevelSampler(
            levels=bc.get("levels", ["easy", "medium", "hard"]),
            seed=bc.get("seed", exp_seed),
        )
        level_tracker = EpisodeTracker(window_size=50)
        initial_level = level_manager.sample()
        initial_map = apply_level_config(build_env_cfg, initial_level)

    print(f"{'=' * 60}")
    print(f"CARLA MAPPO Training - Centralized Critic (CTDE)")
    print(f"{'=' * 60}")
    print(f"Agents: {n_veh}V + {n_ped}P | global_obs: {global_obs_dim}D")
    print(f"Budget: {total_ts:,} steps | Workers: {n_workers} | GPU: {n_gpus}")
    print(f"Policies: vehicle({VEHICLE_OBS_DIM}D), pedestrian({PEDESTRIAN_OBS_DIM}D)")
    print(f"Output: {out_dir}")
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
        "runtime": runtime_cfg,
        "env_config": build_env_cfg,
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
        env_cfg=build_env_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_gpus=n_gpus,
        n_workers=n_workers,
        exp_seed=exp_seed,
        enable_periodic_evaluation=False,
    )

    # --- Build & Train ---
    algo = config.build()
    print("\nMAPPO training avviato.\n")

    # --- Block 5.4: Level manager setup ---
    if level_manager is not None and _unwrap_carla_env(algo) is None:
        logger.error("Cannot unwrap env for %s — falling back to baseline", resolved_mode)
        level_manager = None
        level_tracker = None
        executed_level_trackers = None
        current_training_level = None
        initial_level = None
        initial_map = None
        teacher_diagnostics = None

    if isinstance(level_manager, CurriculumManager):
        current_training_level = initial_level
        print(f"  Curriculum mode: starting at '{initial_level}' (map={initial_map})")
    elif isinstance(level_manager, BatchLevelSampler):
        current_training_level = initial_level
        print(f"  Batch mode: starting at '{initial_level}' (map={initial_map})")
    else:
        print("  Baseline mode: no level switching")

    if level_manager is not None and current_training_level is not None:
        raw_env = _unwrap_carla_env(algo)
        if raw_env is not None:
            raw_env.set_level(current_training_level)

    ts_done = 0
    iteration = 0
    t0 = time.time()
    best_reward = float("-inf")
    timeseries = []
    result = {}
    evaluation = {}
    evaluation_raw = {}
    cumulative_success_rate = None
    cumulative_collision_rate = None
    should_run_final_evaluation = True
    final_evaluation_skip_reason = None
    final_checkpoint = None
    training_failed = False
    results_path = str((Path(out_dir) / "results.json").resolve(strict=False))
    finalization_issues = []

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

            # --- Block 5.4: Level management per iteration ---
            if level_manager is not None:
                iter_ts = result.get(
                    "num_env_steps_sampled_this_iter",
                    result.get("timesteps_this_iter", 0),
                )
                raw_env = _unwrap_carla_env(algo)

                if isinstance(level_manager, CurriculumManager):
                    executed_level = current_training_level or level_manager.current_level
                    executed_tracker = None
                    if executed_level_trackers is not None:
                        executed_tracker = executed_level_trackers.get(executed_level)

                    if executed_tracker is not None:
                        executed_tracker.add_timesteps(iter_ts)
                        _apply_delta_stats_to_tracker(executed_tracker, delta_stats)

                    level_manager.record_execution(executed_level, iter_ts)

                    if raw_env:
                        next_level, teacher_diagnostics = level_manager.get_episode_level(
                            trackers=executed_level_trackers,
                            global_timestep=ts_done,
                        )
                        for teacher_event in teacher_diagnostics.get("events", []):
                            print(f"    [teacher] {teacher_event}")
                        if next_level != current_training_level:
                            raw_env.set_level(next_level)
                            current_training_level = next_level
                            teacher_probs = teacher_diagnostics.get("probabilities", {})
                            probs_repr = ", ".join(
                                f"{level_name}={teacher_probs.get(level_name, 0.0):.2f}"
                                for level_name in level_manager.levels
                                if level_name in teacher_probs
                            )
                            print(f"    [teacher] next_level='{next_level}' probs[{probs_repr}]")

                elif isinstance(level_manager, BatchLevelSampler):
                    if level_tracker is not None:
                        level_tracker.add_timesteps(iter_ts)
                    # Batch: sample next level (no tracker dependency)
                    if raw_env:
                        next_level = level_manager.sample()
                        if next_level != current_training_level:
                            raw_env.set_level(next_level)
                            current_training_level = next_level

            elapsed = time.time() - t0
            pct = ts_done / total_ts * 100
            if ts_done >= total_ts:
                eta = 0.0
            else:
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
                "level": current_training_level  # Block 5.4
            })

            bar_len = 30
            filled = int(bar_len * ts_done / total_ts)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

            level_tag = f" | Lv:{current_training_level}" if current_training_level else ""
            print(
                f"[{bar}] {pct:5.1f}% | "
                f"{ts_done:,}/{total_ts:,} | "
                f"R:{tot_r:+.1f} (V:{veh_r:+.1f} P:{ped_r:+.1f}) | "
                f"Eps:{eps}{level_tag} | Len:{ep_len:.0f} |"
                f"ETS: {int(elapsed//3600)}h{int((elapsed%3600)//60):02d}m / ETA: {int(eta//3600)}h{int((eta%3600)//60):02d}m"
            )

            if tot_r > best_reward:
                best_reward = tot_r

            if ts_done % ckpt_freq < batch_size:
                ckpt = algo.save(out_dir)
                ckpt_path = _resolve_checkpoint_path(ckpt, project_root=project_root)
                print(f"    -> Checkpoint: {ckpt_path or ckpt}")

    except KeyboardInterrupt:
        should_run_final_evaluation = False
        final_evaluation_skip_reason = "manual interrupt"
        print("\nInterrotto.")
    except Exception as e:
        should_run_final_evaluation = False
        training_failed = True
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
        if algo is not None and not training_failed and iteration > 0:
            try:
                final_checkpoint_result = algo.save(out_dir)
                final_checkpoint = _resolve_checkpoint_path(
                    final_checkpoint_result,
                    project_root=project_root,
                )
                if final_checkpoint is None:
                    raise ValueError(
                        "Could not resolve checkpoint filesystem path from algo.save() result"
                    )
                print(f"\nCheckpoint finale: {final_checkpoint}")
            except Exception as e:
                print(f"\nCheckpoint fallito: {e}")
                finalization_issues.append(
                    f"final checkpoint save failed: {type(e).__name__}: {e}"
                )
        elif training_failed:
            print("\nCheckpoint finale saltato (training crash).")
        else:
            print("\nCheckpoint finale saltato (nessuna iterazione completata).")

        # Save last training result (full RLlib dict) before stopping the trainer.
        try:
            if iteration > 0:
                result_path = os.path.join(out_dir, "last_result.json")
                with open(result_path, "w") as f:
                    json.dump(_sanitize_for_json(result), f, indent=2)
                print(f"  Last result: {result_path}")
        except Exception as e:
            print(f"  [WARN] Could not save last_result.json: {e}")
            finalization_issues.append(
                f"last_result.json save failed: {type(e).__name__}: {e}"
            )

        try:
            if (
                eval_section.get("enabled", True)
                and iteration > 0
                and should_run_final_evaluation
                and final_checkpoint is not None
            ):
                artifact_issues = _validate_final_eval_artifacts(
                    checkpoint_path=final_checkpoint,
                    out_dir=out_dir,
                )
                if artifact_issues:
                    raise RuntimeError("; ".join(artifact_issues))
                job_path = _write_final_eval_job(
                    out_dir=out_dir,
                    checkpoint_path=final_checkpoint,
                    env_cfg=env_cfg,
                    train_cfg=train_cfg,
                    eval_cfg=eval_cfg,
                    seed_base=exp_seed,
                    n_gpus=n_gpus,
                    results_path=results_path,
                )
                eval_command = (
                    "python -m carla_core.training.evaluate_carla_mappo "
                    f"--job \"{job_path}\""
                )
                final_evaluation_skip_reason = (
                    "final evaluation pending: launch manually"
                )
                print(f"\nJob evaluation: {job_path}")
                print(f"Per lanciarla: {eval_command}")
            else:
                reason = final_evaluation_skip_reason or (
                    "disabled, no completed iterations, or missing final checkpoint"
                )
                final_evaluation_skip_reason = reason
                print(f"\nValutazione finale multi-scenario saltata ({reason}).")
        except Exception as e:
            final_evaluation_skip_reason = f"failed to prepare final eval job: {type(e).__name__}"
            print(f"  [WARN] Final evaluation job creation failed: {e}")
            finalization_issues.append(
                f"final eval job creation failed: {type(e).__name__}: {e}"
            )

        print(f"\n{'=' * 60}")
        print(f"        MAPPO Training Completato")
        print(f"{'=' * 60}")
        print(f"Steps: {ts_done:,} | Best reward: {best_reward:.1f}")
        print(f"Tempo: {(time.time() - t0) / 60:.1f} min")
        print(f"Output: {out_dir}")

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
            # Block 5.4: inject level management summary
            if level_manager is not None:
                if isinstance(level_manager, CurriculumManager):
                    results_payload["curriculum"] = level_manager.summary()
                    results_payload["curriculum"]["current_training_level"] = current_training_level
                    if teacher_diagnostics is not None:
                        results_payload["curriculum"]["last_decision"] = deepcopy(teacher_diagnostics)
                    results_payload["curriculum_history"] = deepcopy(
                        results_payload["curriculum"].get("unlock_history", [])
                    )
                    if current_training_level is not None and executed_level_trackers is not None:
                        current_tracker = executed_level_trackers.get(current_training_level)
                        if current_tracker is not None:
                            current_summary = current_tracker.summary()
                            current_summary["scope"] = "current_training_level"
                            current_summary["tracked_level"] = current_training_level
                            results_payload["level_tracker"] = current_summary
                    if executed_level_trackers:
                        results_payload["level_trackers"] = {
                            level_name: tracker.summary()
                            for level_name, tracker in executed_level_trackers.items()
                        }
                elif isinstance(level_manager, BatchLevelSampler):
                    results_payload["batch_sampling"] = level_manager.summary()
            if level_tracker is not None and "level_tracker" not in results_payload:
                results_payload["level_tracker"] = level_tracker.summary()
            _write_json_atomic(results_path, results_payload)
            print(f"\nResults schema: {results_path}")
        except Exception as e:
            print(f"[WARN] Could not save results.json: {e}")
            finalization_issues.append(
                f"results.json save failed: {type(e).__name__}: {e}"
            )

        try:
            try:
                switched_envs = _set_algo_env_close_mode_for_teardown(algo, "robust")
                print(
                    f"Switch close mode for final train teardown: robust ({switched_envs} envs)"
                )
            except Exception as e:
                print(f"[WARN] Could not enable robust close mode before algo.stop(): {e}")
                finalization_issues.append(
                    "enable robust close mode before algo.stop() failed: "
                    f"{type(e).__name__}: {e}"
                )
            algo.stop()
        except Exception as e:
            print(f"[WARN] algo.stop() failed: {e}")
            finalization_issues.append(f"algo.stop() failed: {type(e).__name__}: {e}")
        finally:
            algo = None
            gc.collect()
            try:
                ray.shutdown()
            except Exception as e:
                print(f"[WARN] ray.shutdown() failed after training teardown: {e}")
                finalization_issues.append(
                    f"ray.shutdown() failed after training teardown: {type(e).__name__}: {e}"
                )
            carla_shutdown_ok, carla_shutdown_msg = _shutdown_carla_server()
            if carla_shutdown_ok:
                print(f"\nCARLA shutdown: {carla_shutdown_msg}")
            else:
                print(f"[WARN] CARLA shutdown: {carla_shutdown_msg}")
                finalization_issues.append(f"CARLA shutdown warning: {carla_shutdown_msg}")
            if not training_failed and not finalization_issues:
                print("Train chiuso correttamente senza errori.\n")
            elif not training_failed:
                print("Train completato, ma la chiusura non e' stata pulita.\n")
                for issue in finalization_issues:
                    print(f"    - {issue}")


if __name__ == "__main__":
    main()
