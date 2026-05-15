# Experiment Registry

Last updated: 2026-05-15
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
