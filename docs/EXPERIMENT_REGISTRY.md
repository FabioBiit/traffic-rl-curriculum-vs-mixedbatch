# Experiment Registry

Last updated: 2026-05-17
Branch: EVO/new-main
Trunk at time of writing: C0 + C1 + D2

This file is the detailed audit trail for all experimental candidates.
`CLAUDE.md` and `AGENTS.md` carry the summary table; this file has
implementation logic, affected files, key parameters, and pseudocode
per candidate.

---

## Summary Table

| Sigla | Status | Run ID | Description |
|-------|--------|--------|-------------|
| C0 | accepted/supporting | — | Diagnostic logging: `continuous_route_progress`, `no_wp_steps`, `stuck_cause`, `dist_to_next_wp`, `speed_kmh`. Measurement only, no policy change. |
| C1 | accepted/trunk | — | Geometric route-aware vehicle obs (preview longitudinal/lateral, heading error, curvature, lateral error). Obs remain 44D. |
| C2v2-A | not promoted | 20260514_001215 | Movement reward. Vehicle SR up vs baseline but stuck+timeout worsened. |
| D1 | rejected | 20260514_073353 | Observation/reward candidate. SR down, stuck+timeout up. |
| D2 | accepted/trunk | 20260514_095921 | Reward shaping: `target_min_speed=8.0`, start/unblock shaping, `no_wp_steps>100` penalty, `safe_to_push=hazard_risk<0.75`, collision `reward-=50.0`. Reduces stuck+timeout, raises speed; safety remains weak. |
| D1+D2 | rejected | 20260514_133823 | Combined D1+D2. Worsened primary targets vs D2 alone. |
| D2-Safety | rejected/reverted | 20260514_155151 | D2 safety variant. SR down, no safety improvement. Reverted. |
| D3 | rejected/reverted | 20260514_190424 | Early vehicle-stuck termination (`no_wp_steps>=300`, `route<0.3`, `hazard<0.75`). SR -2.90 pp vs D2, stuck+timeout +7.07 pp. Reverted. |
| Path curriculum easy-only | candidate evidence only | 20260514_211642 | Lock easy, `15m/15m`. Does not test budget or sampling weights. |
| Full path curriculum | pending/conditional | — | `difficulty=path`, no lock, `15/35/60m`, budget `0.30/0.32/0.38`, weights `1.00/1.07/1.27`. |
| H1+H1.1 | not promoted / not reverted | 20260515_175921 | Critic fix: `vf_clip_param` 10->1e6 + `vf_loss_coeff` 0.5->0.05. Mechanism confirmed (vehicle `vf_explained_var` ~0->0.87) but vehicle gate FAILS 3 of 4. Not promoted; kept as base for H2 (reverting restores a non-functional critic). |
| H2 | not promoted / not reverted | 20260515_211055 | `gamma` 0.99->0.997 on the H1+H1.1 base. Hypothesis (longer horizon propagates the -50 collision penalty) falsified; vehicle gate FAILS 3 of 4. Not promoted; `gamma=0.997` kept as base for H3. |
| H3 | not promoted / not reverted | 20260516_144007 | `entropy_coeff` schedule `[[0,0.03],[250000,0.005]]` on the H1+H1.1+H2 base. Mechanism confirmed (vehicle final `entropy` 4.78->3.25) but vehicle gate FAILS 3 of 4; all deltas within run-to-run noise. Not promoted; schedule kept as base for R3. |
| R3 | not promoted / not reverted / hypothesis falsified | 20260516_200545 | Vehicle collision penalty `carla_multi_agent_env.py:1748` `-50.0`->`-500.0` (vehicle only). Adapted R3 gate FAILS (SR +1.27, collision -0.23, stuck+timeout +2.28, offroad -3.32 pp vs H3). Hypothesis "collision tunable via penalty magnitude" falsified. Not promoted; `-500` kept as base for the next candidate `R1`. |
| R1 | pending (next) | — | Reward shaping: remove the `route_completion < 0.3` gate in `_vehicle_reward` (~1760/1782/1788) so start/unblock and speed shaping stay active for the whole route. Targets the dominant stuck+timeout failure. Checkpoint-comparable; single-knob A/B vs run `20260516_200545`. |
| R2 | pending | — | Reward shaping: gate the `+0.1` smooth-steering bonus in `_vehicle_reward` (~1803) on `speed_kmh > 5.0`. Checkpoint-comparable. |
| Route-len bugfix | pending | — | Env bugfix: enforce the docstring's `2.0x` upper bound on vehicle route length in `route_planner.py` `plan_vehicle_route` (~184); only the `0.5x` lower bound is currently checked. Checkpoint-comparable; separate A/B (changes the route-length distribution). |
| O1+O2 | pending (obs change; applied last) | — | Vehicle obs `44D -> 47D`: O1 adds normalized `no_wp_steps` + `loop_penalty_active` flag, O2 adds normalized time-remaining. Markov state-aliasing fixes, not hazard/perception features. Not checkpoint-comparable; one retrain-from-scratch 47D variant. |

---

## C0 — Diagnostic Episode Logging Fields

**Type:** Diagnostic / Measurement
**Status:** Accepted/Supporting
**Files:** `carla_core/envs/carla_multi_agent_env.py` (~line 547)

**What was added:**
Five fields added to `agent_info` in `step()` for every vehicle agent:

```python
agent_info["continuous_route_progress"] = continuous fractional waypoint progress
agent_info["no_wp_steps"]               = steps since last waypoint advance
agent_info["stuck_cause"]               = category: low_route_completion | loop_penalty | no_waypoint_advance
agent_info["dist_to_next_wp"]           = Euclidean distance to next target waypoint (m)
agent_info["speed_kmh"]                 = vehicle speed converted to km/h
```

**Key constraint:** No policy change. These fields improve measurement quality only and
are logged via `_terminated_agent_infos` for analysis.

---

## C1 — Geometric Route-Aware Observations

**Type:** Observation change (dimensionality-preserving)
**Status:** Accepted/Trunk
**Files:** `carla_core/envs/carla_multi_agent_env.py` (~lines 1152–1450)
**Vehicle obs dimensionality:** 44D (unchanged)

**Key functions:**

| Function | Obs index | Description |
|----------|-----------|-------------|
| `_fill_route_preview()` | [25:31] | 3 lookahead waypoints: (longitudinal, lateral) pairs |
| `_route_heading_error()` | [31] | Ego heading vs route tangent |
| `_route_turn_angle()` | [32] | Upcoming turn curvature |
| `_signed_lateral_error_to_route()` | [33] | Signed lateral offset from route center |

**Pseudocode:**

```python
# Route preview — 3 lookahead waypoints at offsets [1, 3, 5]
for i, offset in enumerate([1, 3, 5]):
    wp = route_waypoints[current_wp_idx + offset]
    dx, dy = wp.x - ego.x, wp.y - ego.y
    longitudinal =  dx * cos(yaw) + dy * sin(yaw)
    lateral      = -dx * sin(yaw) + dy * cos(yaw)
    obs[25 + 2*i]     = clip(longitudinal / 50.0, -1, 1)   # 50m scale for vehicles
    obs[25 + 2*i + 1] = clip(lateral / lane_width, -1, 1)  # default lane_width=4.0m

# Heading error (obs[31])
obs[31] = normalize_angle(ego_yaw - route_tangent_yaw) / pi  # range [-1, 1]

# Turn curvature (obs[32])
obs[32] = upcoming_turn_angle / pi  # range [-1, 1]

# Lateral error (obs[33])
obs[33] = signed_distance_to_route_center / lane_width  # range [-1, 1]
```

---

## C2v2-A — Movement Reward Shaping

**Type:** Reward shaping
**Status:** Not promoted
**Run:** `20260514_001215`

**What was tried:** Added a reward component for forward movement and route progress
at the start of training (early exploration bonus).

**Gate result:** Vehicle SR improved vs baseline but stuck+timeout worsened.
The agent learned to move but not to complete routes. Not promoted.

---

## D1 — Observation/Reward Candidate

**Type:** Observation + reward
**Status:** Rejected
**Run:** `20260514_073353`

**Gate result:** SR down, stuck+timeout up vs baseline. Rejected.
Implementation details unverified from code (run reverted, no surviving branch).

---

## D2 — Reward Shaping: Speed Target + Start/Unblock + No-Advance Pressure

**Type:** Reward shaping
**Status:** Accepted/Trunk
**Run:** `20260514_095921`
**Files:** `carla_core/envs/carla_multi_agent_env.py` (~lines 1718–1813)
**Function:** `_vehicle_reward()` — Reward v5, Section 5 (Start/Unblock)

**Key parameters:**

```
target_min_speed        = 8.0    # km/h — minimum speed target in early route
safe_to_push threshold  = 0.75   # hazard_risk < 0.75 → safe to push forward
early_route threshold   = 0.3    # route_completion < 0.3 = early stage
no_wp_penalty_start     = 100    # no_wp_steps > 100 triggers no-advance penalty
collision_penalty       = -50.0  # reward -= 50.0 on collision
```

**Pseudocode:**

```python
# Section 5: Start/unblock shaping (inside _vehicle_reward)
if not at_traffic_light:
    safe_to_push = hazard_risk < 0.75
    early_route  = route_completion < 0.3

    # Stuck-start bonus: encourage throttle when nearly stopped
    if speed < 2.5 and safe_to_push and early_route:
        urgency = min(no_wp_steps / 200.0, 1.0)
        reward += start_gain * throttle * alignment * urgency

    # Speed-below-target penalty
    if safe_to_push and early_route and speed < target_min_speed:
        reward -= 0.04 * (target_min_speed - speed) * alignment

    # No-advance pressure: penalize prolonged no-waypoint-advance
    if no_wp_steps > 100:
        reward -= min(1.0, 0.004 * (no_wp_steps - 100))

# Safety collision penalty (trunk, unchanged by D2)
reward -= 50.0  # on collision event
```

**Observed effect:** Reduces stuck+timeout rate, raises average speed.
Safety (collision/offroad) remains weak — this is a known limitation of D2.

---

## D1+D2 — Combined D1 and D2

**Type:** Combined reward/obs
**Status:** Rejected
**Run:** `20260514_133823`

**Gate result:** Worsened primary targets (SR, stuck+timeout) vs D2 alone. Rejected.

---

## D2-Safety — D2 Safety Variant

**Type:** Reward shaping variant
**Status:** Rejected/Reverted
**Run:** `20260514_155151`

**What was tried:** Added safety-specific components on top of D2:
hazard margin reward, speed/throttle penalty for unsafe forward push,
brake bonus when hazard detected. Modified `safe_to_push` threshold.

**Gate result:** Did not improve collision/offroad rates; SR dropped vs D2.

**Revert:** `reward -= 50.0` restored, `safe_to_push = hazard_risk < 0.75` restored,
hazard_margin/safety shaping components removed. D2 trunk restored intact.

---

## D3 — Early Vehicle-Stuck Termination

**Type:** Termination logic
**Status:** Rejected/Reverted
**Run:** `20260514_190424`
**Files:** `carla_core/envs/carla_multi_agent_env.py` (logic removed)
**Commit added:** `739b304` | **Commit reverted:** `5a06589`

**Config keys (removed from codebase):**

```yaml
terminate_vehicle_on_early_stuck: True
vehicle_early_stuck_no_wp_steps: 300
vehicle_early_stuck_route_completion_threshold: 0.3
vehicle_early_stuck_hazard_threshold: 0.75
```

**Pseudocode (no longer in code):**

```python
# Helper _is_vehicle_early_stuck(), called in _check_done() for vehicle agents
def _is_vehicle_early_stuck(agent_data) -> bool:
    if not terminate_vehicle_on_early_stuck:
        return False
    if agent_data.no_wp_steps < 300:
        return False
    if not (route_completion < 0.3 or loop_penalty_active):
        return False
    if at_traffic_light:        # do not terminate at red/yellow
        return False
    if hazard_risk >= 0.75:     # obstacle ahead — not a free stuck
        return False
    return True  # → termination_reason = "stuck"
```

**Gate result:** Vehicle SR -2.90 pp vs D2, stuck+timeout +7.07 pp. D2 trunk kept.

---

## Path Curriculum Easy-Only — Exploratory Evidence Run

**Type:** Curriculum (locked, evidence only)
**Status:** Candidate evidence only
**Run:** `carla_mappo_20260514_211642`

**Config:**

```
difficulty = path
--lock-curriculum-level easy
easy route distance = 15m / 15m
```

**Constraint:** `--lock-curriculum-level easy` disables budget allocation and
sampling weight logic. This run tests only `path` easy routing, not the
`easy→medium→hard` progression. Not usable as evidence for curriculum promotion decisions.

---

## Full Path Curriculum — Pending Candidate

**Type:** Curriculum
**Status:** Pending/Conditional

**Config:**

```
difficulty = path
# No --lock-curriculum-level flag
easy_distance   = 15m,  medium_distance = 35m,  hard_distance = 60m
budget_easy     = 0.30, budget_medium   = 0.32, budget_hard   = 0.38
weight_easy     = 1.00, weight_medium   = 1.07, weight_hard   = 1.27  (base)
probation_medium = 1.00, probation_hard = 1.19
```

**Run command:**

```bash
python -m carla_core.training.train_carla_mappo \
    --mode curriculum --difficulty path --timesteps 3000000 --seed 999
```

**Gate to promote:** vehicle SR +2.0 pp, stuck+timeout -2.0 pp vs D2 trunk baseline.
Collision and offroad must not worsen by more than +1.0 pp.

---

## H1+H1.1 — Critic Fix: vf_clip_param + vf_loss_coeff

**Type:** Optimizer / bug fix (checkpoint-comparable, vehicle obs unchanged 44D)
**Status:** Not promoted / not reverted (mechanism confirmed)
**Run:** `20260515_175921`
**Files:** `carla_core/configs/train_mappo.yaml`, `carla_core/training/mappo_runtime.py`

**Change:**

```
vf_clip_param:  10   -> 1000000.0   # H1
vf_loss_coeff:  0.5  -> 0.05        # H1.1
```

**Rationale:** With `vf_clip_param=10` the value target was clipped far below
the true return scale, leaving the vehicle critic non-functional
(`vf_explained_var` ~0; baseline -0.002 — the critic explained ~0% of return
variance).

**Mechanism:** Confirmed from RLlib logs — vehicle `vf_explained_var` ~0 -> 0.87.

**Gate result:** Vehicle gate vs `20260514_211642` (cumulative, from
`episodes.jsonl`) FAILS 3 of 4: SR +1.51 pp, stuck+timeout -1.98 pp,
collision +3.28 pp, offroad -2.81 pp. Confounded (two knobs). Not promoted;
`vf_clip_param` / `vf_loss_coeff` kept (reverting restores a non-functional
critic). Base for H2.

---

## H2 — Discount Factor gamma

**Type:** Optimizer / horizon
**Status:** Not promoted / not reverted (hypothesis falsified)
**Run:** `20260515_211055`
**Files:** `carla_core/configs/train_mappo.yaml`, `carla_core/training/mappo_runtime.py`

**Change:** `gamma: 0.99 -> 0.997` on the H1+H1.1 base.

**Hypothesis:** A longer effective horizon would propagate the -50 collision
penalty further back and reduce the H1 collision regression.

**Gate result:** Vehicle gate vs `20260515_175921` (cumulative, from
`episodes.jsonl`) FAILS 3 of 4: SR +0.14 pp, stuck+timeout -7.48 pp,
collision +5.18 pp, offroad +2.16 pp. The longer horizon amplified the
dominant route-completion incentive and converted passive failure (timeout)
into active failure (collision/offroad) ~1:1; SR flat. Hypothesis falsified.
`gamma=0.997` kept as base for H3 (user decision).

---

## H3 — entropy_coeff Schedule

**Type:** Optimizer / exploration
**Status:** Not promoted / not reverted (mechanism confirmed)
**Run:** `20260516_144007`
**Files:** `carla_core/configs/train_mappo.yaml`, `carla_core/training/mappo_runtime.py`

**Change:** add `entropy_coeff_schedule: [[0, 0.03], [250000, 0.005]]` (was a
constant `entropy_coeff=0.03`) on the H1+H1.1+H2 base. Single-knob A/B vs
`20260515_211055`.

**Hypothesis:** Decaying the entropy coefficient late in training suppresses
the late-training `entropy` blow-up (`175921`->5.43, `211055`->4.78) and
recovers the chunk-4->chunk-6 vehicle SR decay.

**Mechanism:** Confirmed — vehicle final `entropy` 4.78 -> 3.25 (`entropy_coeff`
reached 0.005), `vf_explained_var` 0.92 (critic healthy).

**Gate result:** Vehicle gate vs `20260515_211055` (cumulative, from
`episodes.jsonl`) FAILS 3 of 4: SR -0.32 pp (21.75->21.43), stuck+timeout
-0.37 pp (44.71->44.35), collision -0.99 pp (26.79->25.79), offroad +1.69 pp
(6.75->8.43). Vehicle route-completions identical in absolute count (216 vs
216) — the SR delta is denominator-only. All deltas within the run-to-run
noise visible in the pre-250k chunks (H2 and H3 share an identical config
there). Late-training SR decay only slightly softened (chunk-6 vehicle SR
22.62 vs 19.64); chunk-4 peak unchanged. Integrity OK (2016 records, 0 dups,
0 NaN/inf). Not promoted; schedule retained as base for R3.

---

## R3 — Vehicle Collision Penalty Magnitude

**Type:** Reward shaping
**Status:** Not promoted / not reverted (hypothesis falsified)
**Run:** `20260516_200545` (A/B run completed — 326 episodes)
**Files:** `carla_core/envs/carla_multi_agent_env.py` (line 1748, `_vehicle_reward()`)
**Base:** H3 config `20260516_144007` (`gamma=0.997` + entropy schedule +
`vf_clip_param=1e6` + `vf_loss_coeff=0.05` retained)

**Change:**

```python
# _vehicle_reward(), Section 3 — Collision penalty
# before: reward -= 50.0
reward -= 500.0   # vehicle only; pedestrian collision (:1841) stays -50.0
```

**Rationale (verified from `episodes.jsonl`, run `20260516_144007`):**
- The waypoint bonus is `+100/wp` and is the dominant route signal.
- The 260 H3 vehicle collision episodes have mean `route_completion` 0.29 and
  run ~800 steps — a typical crash episode already earned ~+200-290 in
  `+100/wp` bonuses before crashing. The old `-50` penalty (half a waypoint)
  cancels ~1/5 of that, so a crash is strongly net-positive.
- `-500` (= 5 waypoints) exceeds the route bonus of essentially all crash
  episodes; the expected value of an aggressive push flips negative.
- A fully-stuck episode (1000 steps) already costs ~ -600/-900 via idle
  (`-0.15/step`), no-advance (up to `-1.0/step`) and loop (`-1.0/step`)
  penalties. `-500` brings a crash to ~parity with a freeze, removing the
  cheap escape; it does not create a new freeze incentive because freezing is
  already that expensive.
- `-200` was rejected: it leaves an aggressive push at positive EV and a crash
  cheaper than a freeze.

**Adapted gate (user-approved; deviates from the generic vehicle gate):**
- PRIMARY: vehicle SR >= +2.0 pp AND vehicle collision drops >= -3.0 pp.
- CANARY (must not worsen by more than +1.0 pp): stuck+timeout, offroad.
- Integrity: 6 records/episode; no NaN/inf.
- Deviation rationale: R3 targets the collision axis, so the generic gate's
  "stuck+timeout improves by >= -2.0 pp" would reject a clean
  collision->completion conversion. For R3, stuck+timeout is an
  overcorrection canary, not a required improvement.

**Gate result (vs H3 `20260516_144007`, cumulative from `episodes.jsonl`):**
The adapted gate FAILS. PRIMARY: vehicle SR +1.27 pp (21.43->22.70, need
>= +2.0); collision -0.23 pp (25.79->25.56, need <= -3.0 — clean miss).
CANARY: stuck+timeout +2.28 pp (44.35->46.63 — breached); offroad -3.32 pp
(8.43->5.11 — pass). Integrity OK (326 ep x 6 = 1956 records, 0 dups,
0 NaN/inf); vehicle `vf_explained_var` 0.92->0.88, `entropy` 3.25->2.58.

The hypothesis "the vehicle collision rate is tunable via the
collision-penalty magnitude" is falsified: a 10x penalty left collision flat.
The policy did respond to the reward change (offroad -3.32 pp; stuck -7.17 pp
converted into timeout +9.45 pp; `route%` 0.44->0.50; entropy down) but not on
the collision axis — collision avoidance is not learnable from the current 44D
vehicle observation (a perception limit, not a reward-weight problem). Not
promoted; `-500` retained (not reverted) by user decision as the experimental
base for the next candidate `R1`. Collision-axis reward shaping (penalty
magnitude) is exhausted, but `R1` and `R2` remain untested reward-shaping
candidates targeting the dominant stuck+timeout failure.

**Verification:** `python -m compileall carla_core/envs/carla_multi_agent_env.py`
passed; `git diff --check` clean; the diff is the single line at `:1748`.

---

## R1 — Remove the route_completion < 0.3 Reward Gate

**Type:** Reward shaping
**Status:** Pending (immediate next candidate)
**Files:** `carla_core/envs/carla_multi_agent_env.py` — `_vehicle_reward()` (~1760, ~1782, ~1788)
**Comparability:** Checkpoint-comparable (no obs/architecture change)

**Change:** Drop the `route_completion < 0.3` guard so the start/unblock and
`target_min_speed` shaping stay active for the whole route, not only the first
30%. The `safe_to_push` (hazard < 0.75) and `alignment > 0.25` guards remain.

**Rationale:** The timeout cohort sits at mean `route_completion` ~0.5-0.6 —
past the `0.3` guard — so it currently has no speed incentive. Targets the
dominant stuck+timeout failure, not the collision axis.

**A/B:** Single-knob vs run `20260516_200545` (R3); project gate.
See `PROPOSED_PLAN.md` Punto 3 for the exact surgical edits.

---

## R2 — Gate the Smooth-Steering Bonus on Speed

**Type:** Reward shaping
**Status:** Pending
**Files:** `carla_core/envs/carla_multi_agent_env.py` — `_vehicle_reward()` (~1803)
**Comparability:** Checkpoint-comparable

**Change:** Pay the `+0.1` smooth-steering bonus only when `speed_kmh > 5.0`.
The jerk penalty (`steer_delta > 0.5 -> -0.3`) is unchanged.

**Rationale:** The bonus is currently unconditional, so a stationary vehicle
holding the wheel still collects `+0.1/step`, partly cancelling the `-0.15`
idle penalty. Purely defensive — removes a reward-hacking incentive for
immobility. See `PROPOSED_PLAN.md` Punto 4.

---

## Route-len Bugfix — Enforce the Route-Length Upper Bound

**Type:** Environment bugfix
**Status:** Pending
**Files:** `carla_core/envs/route_planner.py` — `plan_vehicle_route()` (~184)
**Comparability:** Checkpoint-comparable; separate A/B condition (it changes
the realized route-length distribution)

**Change:** Reject routes with `route_len > target_distance_m * 2.0`. The
function docstring already promises validation in `[0.5x, 2.0x]`, but the code
checks only the `0.5x` lower bound, so an "easy 15m" route can be arbitrarily
long.

**Rationale:** Unbounded route length inflates timeouts and decalibrates
curriculum difficulty. See `PROPOSED_PLAN.md` Punto 5.

---

## O1+O2 — Markov State-Aliasing Observations (44D -> 47D)

**Type:** Observation change (dimensionality-changing)
**Status:** Pending (applied last; one retrain-from-scratch 47D variant)
**Files:** `carla_core/envs/carla_multi_agent_env.py` (obs constant + `_get_vehicle_obs`),
`carla_core/agents/centralized_critic.py` (obs constant)
**Comparability:** NOT checkpoint-comparable — breaks checkpoints, requires a
retrain from scratch; not directly comparable to the 44D trunk

**Change:**
- O1 (44D -> 46D): `obs[44] = min(no_wp_steps / 300, 1)`, `obs[45] = float(loop_penalty_active)`.
- O2 (46D -> 47D): `obs[46] = 1 - min(step_count / max_steps, 1)` (time-remaining).

**Rationale:** The reward already penalizes `no_wp_steps > 100` and
`loop_penalty_active`, and the episode is truncated at a fixed horizon, but
none of these is in the observation — a Markov violation that causes
state-aliasing in exactly the stuck/timeout cohorts. These are state-aliasing
fixes, NOT hazard/perception features. See `PROPOSED_PLAN.md` Punti 6-7.
