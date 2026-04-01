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
EVALUATION_SUBPROCESS_TIMEOUT_S = 14400.0
EVALUATION_STATUS_POLL_INTERVAL_S = 5.0
EVALUATION_HEARTBEAT_TIMEOUT_S = 180.0


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


def _render_progress_bar(progress, width=24):
    progress = max(0.0, min(float(progress), 1.0))
    filled = int(progress * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


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


def _windows_subprocess_creationflags():
    flags = 0
    if sys.platform == "win32":
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        flags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        flags |= getattr(subprocess, "CREATE_DEFAULT_ERROR_MODE", 0)
    return flags


def _resolve_subprocess_executable(*, windowed=False):
    executable = sys.executable
    if sys.platform == "win32":
        current = Path(sys.executable)
        if windowed:
            pythonw = current.with_name("pythonw.exe")
            if pythonw.exists():
                executable = str(pythonw)
        else:
            python = current.with_name("python.exe")
            if python.exists():
                executable = str(python)
    return executable


def _load_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _normalize_pid_exclusions(exclude_pids=None):
    excluded = set()
    for pid in exclude_pids or ():
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        if pid_int > 0:
            excluded.add(pid_int)
    return excluded


def _snapshot_descendant_processes(parent_pid, exclude_pids=None):
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

    excluded = _normalize_pid_exclusions(exclude_pids)
    tracked = {}
    for child in children:
        try:
            child_pid = int(child.pid)
            if child_pid in excluded:
                continue
            tracked[child_pid] = float(child.create_time())
        except (psutil.Error, OSError, ValueError):
            continue
    return tracked


def _normalize_tracked_processes(payload, exclude_pids=None):
    if not isinstance(payload, dict):
        return {}

    excluded = _normalize_pid_exclusions(exclude_pids)
    tracked = {}
    for pid, created_at in payload.items():
        try:
            pid_int = int(pid)
            created_at_float = float(created_at)
        except (TypeError, ValueError):
            continue
        if pid_int > 0 and pid_int not in excluded:
            tracked[pid_int] = created_at_float
    return tracked


def _wait_for_parent_exit(parent_pid, exclude_pids=None):
    tracked_processes = {}
    deadline = time.time() + PARENT_EXIT_TIMEOUT_S
    while _is_process_alive(parent_pid):
        tracked_processes.update(
            _snapshot_descendant_processes(parent_pid, exclude_pids=exclude_pids)
        )
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


def _alive_tracked_processes(tracked_processes, exclude_pids=None):
    excluded = _normalize_pid_exclusions(exclude_pids)
    alive = {}
    for pid, created_at in dict(tracked_processes).items():
        if int(pid) in excluded:
            continue
        if _is_tracked_process_alive(pid, created_at):
            alive[int(pid)] = float(created_at)
    return alive


def _wait_for_runtime_quiescence(tracked_processes, exclude_pids=None):
    deadline = time.time() + RUNTIME_QUIESCENCE_TIMEOUT_S
    alive = _alive_tracked_processes(tracked_processes, exclude_pids=exclude_pids)
    while alive and time.time() < deadline:
        time.sleep(PARENT_POLL_INTERVAL_S)
        alive = _alive_tracked_processes(
            tracked_processes, exclude_pids=exclude_pids
        )
    return alive


def _artifacts_ready(job):
    return not _missing_artifacts(job)


def _missing_artifacts(job):
    checkpoint_path = Path(job["checkpoint_path"])
    out_dir = Path(job["out_dir"])
    last_result_path = out_dir / "last_result.json"

    missing = []
    if not checkpoint_path.exists():
        missing.append(f"checkpoint missing: {checkpoint_path}")
    if not last_result_path.exists():
        missing.append(f"last_result.json missing: {last_result_path}")
    return missing


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


def _load_launcher_status(status_path):
    return _load_json(status_path, {})


def _evaluation_status_matches_session(status, session_id):
    return bool(session_id) and str(status.get("session_id") or "") == str(session_id)


def _preexisting_stale_session_reason(*, job, launcher_status, evaluation_status):
    session_id = str(job.get("session_id") or "")
    if not session_id:
        return None

    eval_started = bool(evaluation_status.get("started_at"))
    eval_finished = bool(evaluation_status.get("finished_at"))
    eval_completed = bool(evaluation_status.get("completed", False))
    if _evaluation_status_matches_session(evaluation_status, session_id) and eval_started:
        if eval_completed:
            return "evaluation already completed for this session"
        if eval_finished:
            return "stale evaluation record already exists for this session"
        reason = str(evaluation_status.get("reason") or "")
        if reason in {"evaluation subprocess starting", "evaluation subprocess running"}:
            return "stale in-progress evaluation record already exists for this session"

    if str(launcher_status.get("session_id") or "") == session_id:
        launcher_state = str(launcher_status.get("state") or "")
        launcher_finished = bool(launcher_status.get("finished_at"))
        if launcher_state == "evaluation_completed":
            return "evaluation already completed for this session"
        if launcher_finished and launcher_state in {"evaluation_failed", "evaluation_skipped"}:
            return "stale launcher record already exists for this session"

    return None


def _mark_evaluation_status_terminal(out_dir, *, reason, exit_code):
    status_path = Path(out_dir) / "evaluation_status.json"
    payload = _load_evaluation_status(out_dir)
    payload["completed"] = False
    payload["reason"] = str(reason)
    payload["exit_code"] = int(exit_code)
    payload["artifacts_written"] = bool(payload.get("artifacts_written", False))
    payload["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload["heartbeat_at"] = payload["finished_at"]
    payload["heartbeat_epoch_s"] = time.time()
    progress = float(payload.get("progress", 0.0) or 0.0)
    payload["progress"] = max(0.0, min(progress, 1.0))
    payload["progress_bar"] = _render_progress_bar(payload["progress"])
    _write_json_atomic(status_path, payload)
    return payload


def _resolve_heartbeat_epoch(status):
    try:
        heartbeat_epoch_s = status.get("heartbeat_epoch_s")
        if heartbeat_epoch_s is not None:
            return float(heartbeat_epoch_s)
    except (TypeError, ValueError):
        pass
    return None


def _resolve_eval_runtime_float(job, key, default_value):
    runtime_cfg = ((job.get("eval_cfg") or {}).get("runtime") or {})
    raw_value = runtime_cfg.get(key)
    try:
        if raw_value is not None:
            return max(1.0, float(raw_value))
    except (TypeError, ValueError):
        pass
    return float(default_value)


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


def _resolve_evaluation_timeout_seconds(job):
    return _resolve_eval_runtime_float(
        job, "subprocess_timeout_seconds", EVALUATION_SUBPROCESS_TIMEOUT_S
    )


def _resolve_evaluation_heartbeat_timeout_seconds(job):
    return _resolve_eval_runtime_float(
        job, "heartbeat_timeout_seconds", EVALUATION_HEARTBEAT_TIMEOUT_S
    )


def _resolve_evaluation_status_poll_seconds(job):
    return _resolve_eval_runtime_float(
        job, "status_poll_interval_seconds", EVALUATION_STATUS_POLL_INTERVAL_S
    )


def _launch_evaluation(job):
    cmd = [
        _resolve_subprocess_executable(windowed=False),
        "-m",
        "carla_core.training.evaluate_carla_mappo",
        "--job",
        str(Path(job["out_dir"]) / "final_eval_job.json"),
    ]
    log_path = Path(job["out_dir"]) / "final_evaluation.log"
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env.setdefault("PYTHONUTF8", "1")
    creationflags = _windows_subprocess_creationflags()
    launcher_status_path = Path(job["out_dir"]) / "final_eval_launcher_status.json"
    deadline = time.time() + _resolve_evaluation_timeout_seconds(job)
    heartbeat_timeout_s = _resolve_evaluation_heartbeat_timeout_seconds(job)
    poll_interval_s = _resolve_evaluation_status_poll_seconds(job)
    last_logged_signature = None
    last_heartbeat_epoch = time.time()
    last_status = _load_evaluation_status(job["out_dir"])

    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(job["project_root"]),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
            creationflags=creationflags,
            close_fds=True,
        )
        while True:
            return_code = proc.poll()
            current_status = _load_evaluation_status(job["out_dir"])
            if current_status:
                last_status = current_status

            heartbeat_epoch = _resolve_heartbeat_epoch(last_status)
            if heartbeat_epoch is not None:
                last_heartbeat_epoch = heartbeat_epoch

            progress = float(last_status.get("progress", 0.0) or 0.0)
            progress = max(0.0, min(progress, 1.0))
            progress_bar = _render_progress_bar(progress)
            current_signature = (
                last_status.get("heartbeat_at"),
                last_status.get("current_scenario_idx"),
                last_status.get("progress"),
                last_status.get("current_map"),
                last_status.get("current_profile"),
            )
            if current_signature != last_logged_signature:
                scenario_idx = int(last_status.get("current_scenario_idx", 0) or 0)
                total_scenarios = int(last_status.get("total_scenarios", 0) or 0)
                current_map = last_status.get("current_map") or "-"
                current_profile = last_status.get("current_profile") or "-"
                print(
                    "Eval heartbeat "
                    f"{progress_bar} {progress * 100:5.1f}% "
                    f"({scenario_idx}/{total_scenarios}) "
                    f"{current_map}/{current_profile}",
                    file=log_file,
                    flush=True,
                )
                last_logged_signature = current_signature

            _update_launcher_status(
                launcher_status_path,
                state="launching_evaluation",
                reason=None,
                parent_pid=int(job["parent_pid"]),
                launcher_pid=os.getpid(),
                evaluation_started=True,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=None,
                evaluation_pid=proc.pid,
                evaluation_progress=progress,
                evaluation_progress_bar=progress_bar,
                evaluation_heartbeat_at=last_status.get("heartbeat_at"),
                current_scenario_idx=int(last_status.get("current_scenario_idx", 0) or 0),
                total_scenarios=int(last_status.get("total_scenarios", 0) or 0),
                current_map=last_status.get("current_map"),
                current_profile=last_status.get("current_profile"),
                session_id=job.get("session_id"),
            )

            if return_code is not None:
                return return_code

            now = time.time()
            if now >= deadline:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    return_code = proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return_code = proc.wait(timeout=5.0)
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=_resolve_evaluation_timeout_seconds(job))

            if (now - last_heartbeat_epoch) >= heartbeat_timeout_s:
                reason = (
                    "evaluation heartbeat timeout after "
                    f"{int(heartbeat_timeout_s)}s without status update"
                )
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc.wait(timeout=5.0)
                _mark_evaluation_status_terminal(
                    job["out_dir"],
                    reason=reason,
                    exit_code=1,
                )
                return 1

            time.sleep(poll_interval_s)


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
    session_id = str(job.get("session_id") or Path(out_dir).name)
    launcher_pid = os.getpid()
    launcher_parent_pid = os.getppid()
    excluded_runtime_pids = {launcher_pid, launcher_parent_pid}

    lock_acquired = False
    try:
        preexisting_launcher_status = _load_launcher_status(status_path)
        preexisting_eval_status = _load_evaluation_status(out_dir)
        stale_session_reason = _preexisting_stale_session_reason(
            job=job,
            launcher_status=preexisting_launcher_status,
            evaluation_status=preexisting_eval_status,
        )
        if stale_session_reason == "evaluation already completed for this session":
            _safe_update_results_payload(job=job, completed=True, reason=None)
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_completed",
                reason=None,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=True,
                evaluation_completed=True,
                evaluation_exit_code=0,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 0
        if stale_session_reason:
            _safe_update_results_payload(job=job, completed=False, reason=stale_session_reason)
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_skipped",
                reason=stale_session_reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1

        _update_launcher_status(
            status_path,
            session_id=session_id,
            state="pending_parent_shutdown",
            reason=None,
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
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
                session_id=session_id,
                state="evaluation_failed",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=1,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1

        tracked_runtime_processes = _normalize_tracked_processes(
            job.get("tracked_runtime_processes", {}),
            exclude_pids=excluded_runtime_pids,
        )
        observed_runtime_processes, parent_exit_timed_out = _wait_for_parent_exit(
            parent_pid,
            exclude_pids=excluded_runtime_pids,
        )
        tracked_runtime_processes.update(observed_runtime_processes)
        tracked_runtime_processes = _normalize_tracked_processes(
            tracked_runtime_processes,
            exclude_pids=excluded_runtime_pids,
        )
        if parent_exit_timed_out:
            reason = (
                "parent process did not exit before launcher timeout "
                f"({int(PARENT_EXIT_TIMEOUT_S)}s)"
            )
            _safe_update_results_payload(job=job, completed=False, reason=reason)
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
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
            session_id=session_id,
            state="waiting_for_runtime_quiescence",
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
            evaluation_started=False,
            evaluation_completed=False,
            evaluation_exit_code=None,
            finished_at=None,
            tracked_runtime_pids=sorted(int(pid) for pid in tracked_runtime_processes),
        )

        time.sleep(POST_PARENT_COOLDOWN_S)

        alive_runtime_processes = _wait_for_runtime_quiescence(
            tracked_runtime_processes,
            exclude_pids=excluded_runtime_pids,
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
                session_id=session_id,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                alive_runtime_pids=alive_runtime_pids,
            )
            return 1

        _update_launcher_status(
            status_path,
            session_id=session_id,
            state="waiting_for_artifacts",
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
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
                session_id=session_id,
                state="evaluation_completed",
                reason=None,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=True,
                evaluation_completed=True,
                evaluation_exit_code=0,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 0

        if not _artifacts_ready(job):
            reason = (
                "final evaluation artifacts missing after parent shutdown: "
                + "; ".join(_missing_artifacts(job))
            )
            _safe_update_results_payload(job=job, completed=False, reason=reason)
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
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
                    session_id=session_id,
                    state="evaluation_completed",
                    reason=None,
                    parent_pid=parent_pid,
                    launcher_pid=launcher_pid,
                    evaluation_started=True,
                    evaluation_completed=True,
                    evaluation_exit_code=0,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                return 0

            reason = "evaluation already in progress or completed"
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_skipped",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=False,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1

        _update_launcher_status(
            status_path,
            session_id=session_id,
            state="launching_evaluation",
            reason=None,
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
            evaluation_started=True,
            evaluation_completed=False,
            evaluation_exit_code=None,
            finished_at=None,
        )
        _write_json_atomic(
            out_dir / "evaluation_status.json",
            {
                "session_id": session_id,
                "pid": None,
                "completed": False,
                "reason": "evaluation subprocess starting",
                "exit_code": None,
                "artifacts_written": False,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
                "evaluation_metric_keys": [],
                "progress": 0.0,
                "progress_bar": _render_progress_bar(0.0),
                "completed_scenarios": 0,
                "total_scenarios": 0,
                "total_episodes": 0,
                "current_phase": "starting",
                "current_scenario_idx": 0,
                "current_map": None,
                "current_profile": None,
                "progress_mode": "exact",
                "heartbeat_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "heartbeat_epoch_s": time.time(),
            },
        )
        try:
            return_code = _launch_evaluation(job)
        except subprocess.TimeoutExpired:
            timeout_s = _resolve_evaluation_timeout_seconds(job)
            reason = (
                "evaluation subprocess timeout "
                f"after {int(timeout_s)}s"
            )
            _mark_evaluation_status_terminal(
                out_dir,
                reason=reason,
                exit_code=1,
            )
            _safe_update_results_payload(job=job, completed=False, reason=reason)
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_failed",
                reason=reason,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
                evaluation_started=True,
                evaluation_completed=False,
                evaluation_exit_code=None,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return 1
        eval_status = _load_evaluation_status(out_dir)

        if return_code == 0 and eval_status.get("completed", False):
            _safe_update_results_payload(job=job, completed=True, reason=None)
            _update_launcher_status(
                status_path,
                session_id=session_id,
                state="evaluation_completed",
                reason=None,
                parent_pid=parent_pid,
                launcher_pid=launcher_pid,
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
            session_id=session_id,
            state="evaluation_failed",
            reason=reason,
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
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
            session_id=session_id,
            state="evaluation_failed",
            reason=reason,
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
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
            session_id=session_id,
            state="evaluation_failed",
            reason=reason,
            parent_pid=parent_pid,
            launcher_pid=launcher_pid,
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
