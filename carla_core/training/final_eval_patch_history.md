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
