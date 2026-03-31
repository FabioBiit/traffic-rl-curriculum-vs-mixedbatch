import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


PARENT_POLL_INTERVAL_S = 1.0
PARENT_EXIT_TIMEOUT_S = 120.0
POST_PARENT_COOLDOWN_S = 5.0
RUNTIME_QUIESCENCE_TIMEOUT_S = 30.0


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_launcher_status(status_path, **fields):
    status_path = Path(status_path)
    payload = _load_json(status_path, {})
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    payload.setdefault("started_at", now)
    payload.update(fields)
    payload["updated_at"] = now
    _write_json_atomic(status_path, payload)
    return payload


def _is_process_alive(pid):
    pid = int(pid)
    if pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102

        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _load_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _snapshot_descendant_processes(parent_pid):
    parent_pid = int(parent_pid)
    if parent_pid <= 0:
        return {}

    psutil = _load_psutil()
    if psutil is None:
        return {}

    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        return {}

    tracked = {}
    for child in children:
        try:
            tracked[int(child.pid)] = float(child.create_time())
        except (psutil.Error, OSError, ValueError):
            continue
    return tracked


def _normalize_tracked_processes(payload):
    if not isinstance(payload, dict):
        return {}

    tracked = {}
    for pid, created_at in payload.items():
        try:
            pid_int = int(pid)
            created_at_float = float(created_at)
        except (TypeError, ValueError):
            continue
        if pid_int > 0:
            tracked[pid_int] = created_at_float
    return tracked


def _wait_for_parent_exit(parent_pid):
    tracked_processes = {}
    deadline = time.time() + PARENT_EXIT_TIMEOUT_S
    while _is_process_alive(parent_pid):
        tracked_processes.update(_snapshot_descendant_processes(parent_pid))
        if time.time() >= deadline:
            return tracked_processes, True
        time.sleep(PARENT_POLL_INTERVAL_S)
    return tracked_processes, False


def _is_tracked_process_alive(pid, created_at=None):
    pid = int(pid)
    if pid <= 0:
        return False

    psutil = _load_psutil()
    if psutil is not None and created_at is not None:
        try:
            proc = psutil.Process(pid)
            return abs(float(proc.create_time()) - float(created_at)) < 1e-3
        except psutil.Error:
            return False

    return _is_process_alive(pid)


def _alive_tracked_processes(tracked_processes):
    alive = {}
    for pid, created_at in dict(tracked_processes).items():
        if _is_tracked_process_alive(pid, created_at):
            alive[int(pid)] = float(created_at)
    return alive


def _wait_for_runtime_quiescence(tracked_processes):
    deadline = time.time() + RUNTIME_QUIESCENCE_TIMEOUT_S
    alive = _alive_tracked_processes(tracked_processes)
    while alive and time.time() < deadline:
        time.sleep(PARENT_POLL_INTERVAL_S)
        alive = _alive_tracked_processes(tracked_processes)
    return alive


def _artifacts_ready(job):
    checkpoint_path = Path(job["checkpoint_path"])
    out_dir = Path(job["out_dir"])
    last_result_path = out_dir / "last_result.json"
    return checkpoint_path.exists() and last_result_path.exists()


def _acquire_lock(lock_path):
    lock_path = Path(lock_path)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    return True


def _release_stale_lock_if_needed(lock_path):
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return False

    try:
        payload = _load_json(lock_path, {})
    except Exception:
        payload = {}

    owner_pid = int(payload.get("pid", 0) or 0)
    if owner_pid > 0 and _is_process_alive(owner_pid):
        return False

    try:
        lock_path.unlink()
        return True
    except OSError:
        return False


def _resolve_evaluation_output_dir(*, base_dir, eval_cfg):
    outputs = eval_cfg.get("outputs", {})
    output_dir = outputs.get("output_dir")
    if output_dir:
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            out_path = Path(base_dir) / out_path
    else:
        out_path = Path(base_dir)
    return out_path


def _resolve_evaluation_json_path(*, base_dir, run_name, eval_cfg):
    out_path = _resolve_evaluation_output_dir(base_dir=base_dir, eval_cfg=eval_cfg)
    return out_path / f"{run_name}_evaluation.json"


def _load_evaluation_payload(job):
    eval_json_path = _resolve_evaluation_json_path(
        base_dir=job["project_root"],
        run_name=job["run_name"],
        eval_cfg=job["eval_cfg"],
    )
    payload = _load_json(eval_json_path, {})
    return (
        payload.get("evaluation_raw", {}) or {},
        payload.get("evaluation", {}) or {},
    )


def _load_evaluation_status(out_dir):
    status_path = Path(out_dir) / "evaluation_status.json"
    return _load_json(
        status_path,
        {
            "completed": False,
            "reason": "evaluation status file missing",
            "exit_code": None,
            "artifacts_written": False,
        },
    )


def _resolve_evaluation_failure_reason(*, return_code, status):
    reason = status.get("reason")
    exit_code = status.get("exit_code")
    finished_at = status.get("finished_at")
    completed = bool(status.get("completed", False))

    if return_code == 0 and completed:
        return None

    if return_code != 0 and (
        reason in {None, "", "evaluation subprocess started", "evaluation subprocess running"}
        or exit_code is None
        or finished_at in {None, ""}
    ):
        return (
            "evaluation subprocess crashed before writing final status "
            f"(exit code {return_code})"
        )

    if return_code == 0 and not completed:
        return reason or "evaluation subprocess exited without completion status"

    return reason or f"evaluation subprocess failed (exit code {return_code})"


def _update_results_payload(*, job, completed, reason=None):
    results_path_value = job.get("results_path")
    if not results_path_value:
        return False

    results_path = Path(results_path_value)
    if not results_path.exists():
        return False

    payload = _load_json(results_path, {})
    payload.setdefault("meta", {})
    payload["meta"]["final_evaluation_completed"] = bool(completed)

    if completed:
        payload["meta"].pop("final_evaluation_skip_reason", None)
    else:
        payload["meta"]["final_evaluation_skip_reason"] = (
            str(reason) if reason is not None else "final evaluation failed"
        )

    evaluation_raw, evaluation = _load_evaluation_payload(job)
    if evaluation or evaluation_raw:
        payload["evaluation"] = evaluation
        payload["evaluation_raw"] = evaluation_raw

    _write_json_atomic(results_path, payload)
    return True


def _safe_update_results_payload(*, job, completed, reason=None):
    try:
        return _update_results_payload(job=job, completed=completed, reason=reason)
    except Exception:
        traceback.print_exc()
        return False


def _launch_evaluation(job):
    cmd = [
        sys.executable,
        "-m",
        "carla_core.training.evaluate_carla_mappo",
        "--job",
        str(Path(job["out_dir"]) / "final_eval_job.json"),
    ]
    log_path = Path(job["out_dir"]) / "final_evaluation.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(job["project_root"]),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="CARLA MAPPO final evaluation launcher")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    job_path = Path(args.job)
    out_dir = job_path.parent
    status_path = out_dir / "final_eval_launcher_status.json"
    lock_path = out_dir / "final_eval.lock"
    job = {}
    parent_pid = 0

    try:
        job = _load_json(job_path, {})
    except Exception as exc:
        reason = f"failed to parse final eval job: {type(exc).__name__}"
        _update_launcher_status(
            status_path,
            state="evaluation_failed",
            reason=reason,
            parent_pid=0,
            launcher_pid=os.getpid(),
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=1,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        traceback.print_exc()
        return 1

    out_dir = Path(job.get("out_dir", job_path.parent))
    status_path = out_dir / "final_eval_launcher_status.json"
    lock_path = out_dir / "final_eval.lock"
    parent_pid = int(job.get("parent_pid", 0) or 0)

    lock_acquired = False
    try:
        _update_launcher_status(
            status_path,
            state="pending_parent_shutdown",
            reason=None,
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=None,
            finished_at=None,
        )

        required_fields = [
            "checkpoint_path",
            "out_dir",
            "run_name",
            "results_path",
            "project_root",
            "parent_pid",
        ]
        missing_fields = [field for field in required_fields if not job.get(field)]
        if missing_fields:
            reason = (
                "final evaluation launcher job missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
            _update_launcher_status(
                status_path,
                state="evaluation_failed",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=1,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1

        tracked_runtime_processes = _normalize_tracked_processes(
            job.get("tracked_runtime_processes", {})
        )
        observed_runtime_processes, parent_exit_timed_out = _wait_for_parent_exit(
            parent_pid
        )
        tracked_runtime_processes.update(observed_runtime_processes)
        if parent_exit_timed_out:
            reason = (
                "parent process did not exit before launcher timeout "
                f"({int(PARENT_EXIT_TIMEOUT_S)}s)"
            )
            _safe_update_results_payload(job=job, completed=False, reason=reason)
            _update_launcher_status(
                status_path,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                tracked_runtime_pids=sorted(
                    int(pid) for pid in tracked_runtime_processes
                ),
            )
            return 1

        _update_launcher_status(
            status_path,
            state="waiting_for_runtime_quiescence",
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=None,
            finished_at=None,
            tracked_runtime_pids=sorted(int(pid) for pid in tracked_runtime_processes),
        )

        time.sleep(POST_PARENT_COOLDOWN_S)

        alive_runtime_processes = _wait_for_runtime_quiescence(
            tracked_runtime_processes
        )
        if alive_runtime_processes:
            alive_runtime_pids = sorted(int(pid) for pid in alive_runtime_processes)
            reason = (
                "training runtime still active after parent exit: "
                + ", ".join(str(pid) for pid in alive_runtime_pids)
            )
            _safe_update_results_payload(job=job, completed=False, reason=reason)
            _update_launcher_status(
                status_path,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                alive_runtime_pids=alive_runtime_pids,
            )
            return 1

        _update_launcher_status(
            status_path,
            state="waiting_for_artifacts",
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=None,
            finished_at=None,
            tracked_runtime_pids=sorted(int(pid) for pid in tracked_runtime_processes),
            alive_runtime_pids=[],
        )

        eval_status = _load_evaluation_status(out_dir)
        if eval_status.get("completed", False):
            _safe_update_results_payload(job=job, completed=True, reason=None)
            _update_launcher_status(
                status_path,
                state="evaluation_completed",
                reason=None,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=True,
                evaluation_completed=True,
                evaluation_exit_code=0,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 0

        if not _artifacts_ready(job):
            reason = "final evaluation artifacts missing after parent shutdown"
            _safe_update_results_payload(job=job, completed=False, reason=reason)
            _update_launcher_status(
                status_path,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1

        if lock_path.exists():
            _release_stale_lock_if_needed(lock_path)

        lock_acquired = _acquire_lock(lock_path)
        if not lock_acquired:
            eval_status = _load_evaluation_status(out_dir)
            if eval_status.get("completed", False):
                _safe_update_results_payload(job=job, completed=True, reason=None)
                _update_launcher_status(
                    status_path,
                    state="evaluation_completed",
                    reason=None,
                    parent_pid=parent_pid,
                    launcher_pid=os.getpid(),
                    evaluation_started=True,
                    evaluation_completed=True,
                    evaluation_exit_code=0,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                return 0

            reason = "evaluation already in progress or completed"
            _update_launcher_status(
                status_path,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1

        _update_launcher_status(
            status_path,
            state="launching_evaluation",
            reason=None,
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=True,
            evaluation_completed=False,
            evaluation_exit_code=None,
            finished_at=None,
        )
        return_code = _launch_evaluation(job)
        eval_status = _load_evaluation_status(out_dir)

        if return_code == 0 and eval_status.get("completed", False):
            _safe_update_results_payload(job=job, completed=True, reason=None)
            _update_launcher_status(
                status_path,
                state="evaluation_completed",
                reason=None,
                parent_pid=parent_pid,
                launcher_pid=os.getpid(),
                evaluation_started=True,
                evaluation_completed=True,
                evaluation_exit_code=0,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 0

        reason = _resolve_evaluation_failure_reason(
            return_code=return_code,
            status=eval_status,
        )
        _safe_update_results_payload(job=job, completed=False, reason=reason)
        _update_launcher_status(
            status_path,
            state="evaluation_failed",
            reason=reason,
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=True,
            evaluation_completed=False,
            evaluation_exit_code=return_code,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return 1

    except KeyboardInterrupt:
        reason = "final evaluation launcher interrupted"
        _safe_update_results_payload(job=job, completed=False, reason=reason)
        _update_launcher_status(
            status_path,
            state="evaluation_failed",
            reason=reason,
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=130,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return 130
    except Exception as exc:
        reason = f"final evaluation launcher error: {type(exc).__name__}"
        _safe_update_results_payload(job=job, completed=False, reason=reason)
        _update_launcher_status(
            status_path,
            state="evaluation_failed",
            reason=reason,
            parent_pid=parent_pid,
            launcher_pid=os.getpid(),
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=1,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        traceback.print_exc()
        return 1
    finally:
        if lock_acquired and lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
