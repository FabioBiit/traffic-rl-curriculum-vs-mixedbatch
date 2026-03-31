import argparse
import gc
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import carla
import ray

from carla_core.training.train_carla_mappo import (
    FinalEvaluationFailed,
    FinalEvaluationInterrupted,
    _run_evaluation_scenarios,
    _sanitize_for_json,
    _save_evaluation_artifacts,
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


def _carla_preflight(env_cfg):
    sim = env_cfg["simulator"]
    client = carla.Client(sim["host"], sim["port"])
    client.set_timeout(sim["timeout_seconds"])

    target_map = env_cfg["world"]["map"]
    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    if current_map != target_map:
        world = client.load_world(target_map)

    settings = world.get_settings()
    settings.synchronous_mode = bool(sim.get("sync_mode", True))
    settings.fixed_delta_seconds = sim["fixed_delta_seconds"]
    if env_cfg.get("world", {}).get("no_rendering", False):
        settings.no_rendering_mode = True
    world.apply_settings(settings)

    tm = client.get_trafficmanager(int(sim.get("traffic_manager_port", 8000)))
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(int(env_cfg.get("traffic", {}).get("seed", 42)))


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


def main():
    parser = argparse.ArgumentParser(description="Isolated CARLA MAPPO final evaluation")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    job = _load_job(args.job)
    out_dir = job["out_dir"]
    status_path = Path(out_dir) / "evaluation_status.json"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    _write_status(
        status_path,
        {
            "completed": False,
            "reason": "evaluation subprocess running",
            "exit_code": None,
            "artifacts_written": False,
            "started_at": started_at,
            "finished_at": None,
            "evaluation_metric_keys": [],
        },
    )

    try:
        _carla_preflight(job["env_cfg"])
        evaluation_raw, evaluation_summary, evaluation_traces, evaluation_metric_keys = (
            _run_evaluation_scenarios(
                checkpoint_path=job["checkpoint_path"],
                base_env_cfg=deepcopy(job["env_cfg"]),
                train_cfg=deepcopy(job["train_cfg"]),
                eval_cfg=deepcopy(job["eval_cfg"]),
                seed_base=int(job["seed_base"]),
                n_gpus=int(job["n_gpus"]),
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
                "completed": True,
                "reason": None,
                "exit_code": 0,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": evaluation_metric_keys,
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
                "completed": False,
                "reason": exc.reason,
                "exit_code": 130,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": exc.metric_keys,
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
                "completed": False,
                "reason": exc.reason,
                "exit_code": 1,
                "artifacts_written": artifacts_written,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": exc.metric_keys,
            },
        )
        print(f"[WARN] Final evaluation failed: {exc.reason}")
        return 1
    except KeyboardInterrupt:
        _write_status(
            status_path,
            {
                "completed": False,
                "reason": "manual interrupt during evaluation subprocess",
                "exit_code": 130,
                "artifacts_written": False,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": [],
            },
        )
        print("\nValutazione finale interrotta manualmente.")
        return 130
    except Exception as exc:
        _write_status(
            status_path,
            {
                "completed": False,
                "reason": f"evaluation subprocess error: {type(exc).__name__}",
                "exit_code": 1,
                "artifacts_written": False,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_metric_keys": [],
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
