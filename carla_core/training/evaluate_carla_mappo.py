import argparse
import gc
import json
import os
import time
import traceback
from copy import deepcopy
from pathlib import Path

import carla
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from carla_core.training.train_carla_mappo import (
    FinalEvaluationFailed,
    FinalEvaluationInterrupted,
    CentralizedCriticModel,
    _render_progress_bar,
    _run_evaluation_scenarios,
    _sanitize_for_json,
    _save_evaluation_artifacts,
    rllib_env_creator,
)


def _write_status(status_path, payload):
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(payload), f, indent=2)
    os.replace(tmp_path, status_path)


def _load_job(job_path):
    with open(job_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def _now_status_fields():
    return {
        "heartbeat_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "heartbeat_epoch_s": time.time(),
    }


def _carla_preflight(env_cfg):
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

    # CARLA recommends reloading the world between repetitions to guarantee a
    # clean simulator state. The final eval runs in its own subprocess, so we
    # can safely re-take ownership of the target map before building RLlib.
    world = client.load_world(target_map)
    _ = world.get_map().name
    _ = client.get_trafficmanager(int(sim.get("traffic_manager_port", 8000)))


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
        "current_phase": "starting",
        "current_scenario_idx": 0,
        "current_map": None,
        "current_profile": None,
        "progress_mode": "exact",
    }

    _update_running_status(status_path, status_state)

    try:
        _carla_preflight(job["env_cfg"])
        _register_eval_runtime()

        def _status_callback(**fields):
            _update_running_status(status_path, status_state, **fields)

        evaluation_raw, evaluation_summary, evaluation_traces, evaluation_metric_keys = (
            _run_evaluation_scenarios(
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


if __name__ == "__main__":
    raise SystemExit(main())
