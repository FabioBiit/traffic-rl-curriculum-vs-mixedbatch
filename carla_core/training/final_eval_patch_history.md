# Final Eval Patch History

This file tracks attempted final-evaluation patches to avoid repeating
previously inconclusive changes.

## 2026-04-01

1. Move final eval out of the training process into:
   - train parent
   - final eval launcher
   - isolated eval subprocess
   Outcome: required; kept.

2. Schedule launcher before risky training teardown and write artifacts before
   teardown.
   Outcome: required; kept.

3. Add fast-exit in the parent to avoid CARLA teardown crashes.
   Outcome: parent still logs Windows access violations, but launcher scheduling
   survives; kept as current workaround.

4. Fix malformed checkpoint path in final eval job.
   Outcome: required; kept.

5. Register RLlib custom env/model inside the eval subprocess.
   Outcome: required; kept.

6. Replace Unicode eval progress bar with ASCII and force UTF-8 for eval child
   stdio.
   Outcome: removed Windows cp1252 heartbeat failure; kept.

7. Add launcher timeout for the eval subprocess.
   Outcome: required; kept.

8. Reload CARLA target world in the eval subprocess before building RLlib.
   Outcome: required for clean simulator takeover; kept.

9. Isolate Windows subprocesses more strongly using:
   - CREATE_NEW_PROCESS_GROUP
   - DETACHED_PROCESS
   - CREATE_BREAKAWAY_FROM_JOB
   - CREATE_DEFAULT_ERROR_MODE
   Applied to:
   - parent -> launcher
   - launcher -> eval child
   Outcome: current active patch under validation.

10. Use the windowed Python interpreter (`pythonw.exe`) for:
    - parent -> launcher
    - launcher -> eval child
    and explicitly disconnect subprocess stdin with `DEVNULL` plus `close_fds=True`.
    Outcome: current active patch under validation.

11. Write a launcher-side `evaluation_status.json` stub before spawning the
    eval child, so immediate child startup failures do not erase the failure
    point from disk.
    Outcome: current active patch under validation.

12. Add file-based eval heartbeat/progress reporting through
    `evaluation_status.json`, emitted by the eval child during scenario
    execution and monitored by the launcher.
    Outcome: current active patch under validation.

13. Add same-session stale-eval guard keyed by `session_id` to prevent
    relaunching a second final eval for the same run directory when an
    incomplete record already exists.
    Outcome: current active patch under validation.

14. Exclude the launcher PID from the tracked runtime-process quiescence gate.
    Outcome: required after the launcher started skipping eval because it was
    counting itself as an active training-runtime child.

15. Exclude the launcher's immediate parent PID as well, to handle Windows
    venv `pythonw.exe` bootstrap/redirector processes that can remain visible
    in the tracked descendant set after the real launcher process starts.
    Outcome: current active patch under validation.

16. Stop using `pythonw.exe` for the eval child that initializes Ray.
    Keep the eval child on `python.exe` so Ray dashboard/agent processes have
    valid stdio handles; `pythonw.exe` caused `sys.stdout is None` in
    `dashboard_agent`.
    Outcome: current active patch under validation.

17. Reload the CARLA world before each final-eval scenario and force Python
    garbage collection after each scenario-local `Algorithm.stop()`.
    Outcome: added to harden scenario-to-scenario simulator reset safety after
    `libcarla` aborts during the second scenario.
