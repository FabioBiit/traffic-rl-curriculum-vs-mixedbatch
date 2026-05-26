<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes - gives risk-scored analysis |
| `get_review_context` | Need source snippets for review - token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

### When running shell commands that may produce verbose output, prefer RTK-prefixed commands

- rtk git status
- rtk git diff
- rtk pytest
- rtk grep
- rtk find
Avoid reading full logs unless necessary.

### Recap Workflow

Use CodeReviewGraph before code changes to understand repo structure and impacted files.
Use RTK for verbose terminal commands such as git diff, git status, pytest, grep, find and logs.
Prefer targeted commands and avoid full raw logs unless necessary.

---

# Project Operating Brief

**Last updated:** 2026-05-26

---

## Role

You are a senior AI/ML Engineer and domain expert in Python engineering,
Reinforcement Learning, MAPPO, and CARLA simulation for multi-agent
autonomous-driving experiments.

**Non-negotiable rules:**
- Be empirical, concise, and gate-driven.
- Never invent metrics, baselines, citations, or results.
- If a value is not verified from files, logs, or tool output, state it is unverified.
- **Never launch CARLA training or evaluation runs.** The user launches them.

---

## Research Question

> Does curriculum learning (`easy → medium → hard`) produce measurably different
> multi-agent driving behavior than mixed/batch training in MARL for urban
> autonomous driving?

Evaluate with agent-level metrics, policy-level breakdowns, and Town03→Town05
generalization. Do not reduce this to a single success-rate leaderboard.

---

## Fixed Experimental Stack

| Item | Value |
|------|-------|
| Simulator | CARLA 0.9.16 |
| Algorithm | MAPPO with CTDE |
| Framework | Ray/RLlib 2.10.0 |
| Runtime | Python 3.11.9, PyTorch 2.7+cu126 |
| Agents | 3 RL vehicles + 3 RL pedestrians |
| Train map | Town03 |
| Test map | Town05 |
| Architecture | Centralized critic, separate vehicle/pedestrian policies; PopArt off, attention off, GNN off |

Do not use MetaDrive assumptions, files, or conclusions for CARLA decisions.

---

## Current Goal

Improve the vehicle policy without changing the MAPPO architecture unless explicitly
requested. Vehicle failure modes: low success rate, high stuck rate, high timeout rate,
low average speed, weak early route progress. Pedestrians are comparatively stronger —
**always report them separately from vehicles**.

---

## Active Configuration

- **Trunk**: `C0 + C1 + D2 + R1`. Reverted: `D3`.
- **Vehicle obs**: `47D` (O1+O2 applied); `global_obs_dim=225`. **Not checkpoint-compatible** with pre-O1+O2 (44D) runs.
- **Pedestrian obs**: `26D`.

**Non-reverted retained knobs** (failed gate; kept by user decision):

| Knob | Value | Reason retained |
|------|-------|----------------|
| `vf_clip_param` | `1e6` (H1) | Reverting restores a non-functional critic (`vf_explained_var` was ~0) |
| `vf_loss_coeff` | `0.05` (H1.1) | Same |
| `gamma` | `0.997` (H2) | User decision; H2 hypothesis falsified |
| `entropy_coeff` | schedule `[[0,0.03],[249000,0.005]]` (fraction-based; validated by EvoEntropy `20260520_133747`) | PASS — see EvoEntropy entry below |
| Collision penalty | `-500.0` at `carla_multi_agent_env.py:1748` (R3) | User decision; hypothesis falsified |

**Key analytical findings:**
- H1+H2+H3 together: three optimizer-side single-knob changes, all mechanism-confirmed, all failing
  the gate. Vehicle SR pinned ~21.5% on the pre-bugfix distribution; binding constraint was the
  vehicle collision rate (~25.8%).
- R3 finding: collision avoidance is not learnable from 44D obs (perception limit, not a
  reward-weight problem). The 10× penalty left collision flat; policy responded on offroad/timeout
  axes only.
- R2 provisional (not promoted): vs R1 `20260517_134652`, SR −1.85 pp, collision +1.43 pp,
  offroad +3.15 pp. Speed improved (+2.21 km/h) but extra mobility converted into offroad/collision.
  Do not stack new candidates on R2 as if it passed the gate.

---

## Current Baselines

- **Post-bugfix vehicle baseline**: run `20260517_212109` (route-len bugfix, commit `24e072e`).
  Vehicle SR 75.1% training-only — task-distribution change, NOT policy improvement.
  Pre-bugfix H/R-series absolute SR (~21–27%) is not comparable across the bugfix.
- **47D vehicle baseline**: run `20260518_195947` (O1+O2 + route-seed fix; easy-locked; HEALTHY;
  vehicle SR plateau ~79%; `vf_explained_var` 0.983). **Open item**: 16.4%
  `route_source=legacy_fallback` on vehicle routes — diagnose before the long run.
- **R1 re-check caveat**: R1 removed the `route_completion < 0.3` guard, which interacts with route
  length; its single-knob verdict is most exposed to re-check on the corrected route distribution.

---

## Curriculum Configuration (`curriculum_batch.yaml`, updated 2026-05-18)

- `difficulty=path`; distances: **`easy=30m / medium=60m / hard=100m`** (source of truth:
  `carla_core/configs/levels.yaml::levels_path`; verified against `run_config.json` of in-flight
  run `20260520_133747`). Previously documented as `15/35/60` — corrected on 2026-05-20 (doc-config
  drift, no code change). A move to `15/30/60` punto fisso, or a truncated-range "forbice"
  `[L,U]` sampler, is under consideration for the next iteration but **not** in the current run.
- Budget shares: `easy=0.30 / medium=0.35 / hard=0.35`.
- Base sampling weights: `1.00 / 1.17 / 1.17`; medium `min_budget_share=0.20`.
- Probation weights: `medium=1.00 / hard=0.85`.
- Unlock metric: balanced **windowed** policy SR (`window_full` required; not cumulative).
  Rationale: cumulative SR permanently dragged by cold-start failures
  (pilot `20260518_195947`: 86% windowed vs 67% cumulative). Reporting metrics remain cumulative.
  First full unlock-path run: **`20260525_205912`** (3M, `--difficulty path`, seed 999;
  curriculum_lock disabled; launched on the unified `EVO/new-main` ≡ `EVO/curriculum-stack`).
  Evaluation pending consolidation.
  Note (2026-05-25): runs `20260520_133747` and `20260525_091300` were both easy-locked
  (`curriculum_lock.enabled=true, level=easy`) and therefore do not exercise the unlock metric.

---

## Next Planned Step

**Branch state (verified 2026-05-26)**: `EVO/new-main` and `EVO/curriculum-stack` point
to the same commit (`d84fa3a`). All P1+P2+P3 commits
(`9c20f97 EvoP1CurriculumRehearsalFloors`, `749c8dc EvoP2UnlockGateMinPolicySR`,
`7790c7d EvoP3RaiseHardForceUnlockCap`) and V1 (`2bbcdbe EvoVehicleSafePushTolerance`)
are present on `EVO/new-main`. No outstanding merge.

**Step 5 launched and completed**: run `20260525_205912` (3M, `--difficulty path
--timesteps 3000000 --seed 999`, curriculum_lock disabled). This is the first run that:
- Exercises the curriculum unlock metric (no `curriculum_lock`, `difficulty=path`,
  distances `30/60/100m`, budget `0.30/0.35/0.35`, balanced windowed-SR unlock).
- Tests V1 + P1+P2+P3 stack at full horizon and on hard routes.

**Immediate next action**: consolidate evaluation of `20260525_205912` (training
`episodes.jsonl` + the run's deterministic `final_eval_job.json`). Report cumulative
agent-level metrics (vehicles / pedestrians separately), per-level breakdown
(easy/medium/hard), Q1→Q4 trajectory, and termination-reason mix incl. the
`route_short` demotion already applied by `episode_classification.py`.

**Caveat for step 5**: with fraction-based entropy schedule, transition fires at 83% of 3M =
step 2.49M; Q4 amounts to ~510K step in low-exploration regime (~3× the 300K Q4 duration).
V1's Q4 was strongly ascending at 300K (+5.1 pp Q3→Q4), but a longer collapsed-entropy phase
may still surface late-training plateau or instability. Watch:
- Late-window vehicle SR slope (Q3→Q4 within 3M frame, i.e. 2.25M→3M).
- Collision/offroad creep under raised `safe_to_push=0.85` once exposed to 100m hard routes
  (V1 was tested only on 30m easy).
- Pedestrian speed convergence on 100m routes (at 2.0 m/s × 50 s = 100 m, ped is at the
  cusp; below ~1.8 m/s may not complete hard routes in `max_steps`).

If step 5 shows any of these regressions, contingency knobs ready:
- **V2** (`no_wp_steps` cap 1.0→2.0 at `carla_multi_agent_env.py:1946`) — anti-stuck strength.
- **V4** (entropy schedule transition 83%→90%, i.e. 2.49M→2.70M for 3M) — extends Q4 exploration.
- **V5** (asymmetric ped band: add `-0.2` penalty for `speed > 2.5` at L2007) — structural ped
  speed anchor; currently peds overshoot bonus-only band.

**Open items** carried into step 5:
- Vehicle `route_source=legacy_fallback` ~16.4–16.7% (stable across 4 runs); diagnose if it
  bites under harder routes (100m).
- Pedestrian `route_source=sidewalk_fallback` ~20% post Ped-route (relabel-only per commit
  `21a6724`); deeper backtracking fix deferred.
- Ped comfort band asymmetric (bonus inside only) — 3-run pattern: `[0.8,1.8]→1.99 m/s`,
  `[1.2,2.6]→2.28 m/s`, `[1.5,2.2]→2.64 m/s`. V1 with band `[1.2, 2.6]` converged at 2.14 m/s
  (best), but the asymmetry remains a latent risk.

**Candidate queue (updated 2026-05-26)**:
`EvoEntropy (PASS, 20260520_133747) → Ped-route+Ped-speed bundle (PARTIAL, 20260525_091300)
→ V3 retune (REJECTED gate2 FAIL, 20260525_125127, reverted) → V1 (PROMOTED, 20260525_162428)
→ Step 5 long run 3M COMPLETED (20260525_205912; eval consolidation pending)
→ [V2/V4/V5 contingent on step 5 evidence] → R-norm v2 (gated on Block 4 evidence)`.
File-overlap ordering enforced: V1/V2 share `carla_multi_agent_env.py` (apply sequentially,
never in parallel); P1/P2 share `curriculum_batch_manager.py`.

---

## Candidate Registry

| Sigla | Status | Run ID | Description |
|-------|--------|--------|-------------|
| C0 | accepted/supporting | — | Diagnostic logging: `continuous_route_progress`, `no_wp_steps`, `stuck_cause`, `dist_to_next_wp`, `speed_kmh`. Measurement only, no policy change. |
| C1 | accepted/trunk | — | Geometric route-aware vehicle obs (preview longitudinal/lateral, heading error, curvature, lateral error). Obs remain 44D. |
| C2v2-A | not promoted | 20260514_001215 | Movement reward. SR up vs baseline but stuck+timeout worsened. |
| D1 | rejected | 20260514_073353 | Obs/reward candidate. SR down, stuck+timeout up. |
| D2 | accepted/trunk | 20260514_095921 | Reward shaping: `target_min_speed=8.0`, start/unblock shaping, `no_wp_steps>100` penalty, `safe_to_push=hazard_risk<0.75`, collision `reward-=50.0`. |
| D1+D2 | rejected | 20260514_133823 | Combined D1+D2. Worsened primary targets vs D2 alone. |
| D2-Safety | rejected/reverted | 20260514_155151 | D2 safety variant. SR down, no safety improvement. |
| D3 | rejected/reverted | 20260514_190424 | Early vehicle-stuck termination (`no_wp_steps>=300`, `route<0.3`, `hazard<0.75`). SR -2.90 pp vs D2, stuck+timeout +7.07 pp. |
| Path curriculum easy-only | candidate evidence only | 20260514_211642 | Lock easy, `15m/15m`. Does not test budget or sampling weights. |
| Full path curriculum | pending/conditional | — | `difficulty=path`, no lock, `30/60/100m`, budget `0.30/0.35/0.35`, windowed-SR unlock metric (updated 2026-05-18). |
| H1+H1.1 | not promoted / not reverted | 20260515_175921 | `vf_clip_param` 10→1e6 + `vf_loss_coeff` 0.5→0.05. Mechanism: `vf_explained_var` ~0→0.87. Gate FAILS 3/4: SR +1.51 pp, stuck+timeout -1.98 pp, collision +3.28 pp. Confounded (two knobs). `vf_clip` retained (reverting restores non-functional critic). |
| H2 | not promoted / hypothesis falsified | 20260515_211055 | `gamma` 0.99→0.997. Gate FAILS 3/4: collision +5.18 pp. Longer horizon amplified route incentive, converted timeout→collision 1:1; SR flat (+0.14 pp). Retained by user decision. |
| H3 | not promoted / mechanism confirmed | 20260516_144007 | Entropy schedule `[[0,0.03],[250000,0.005]]`. Mechanism: entropy 4.78→3.25. Gate FAILS 3/4: SR delta denominator-only (216 completions in both runs); all deltas within run-to-run noise. Retained by user decision. |
| R3 | not promoted / hypothesis falsified | 20260516_200545 | Collision penalty -50→-500. Gate FAILS: collision flat (-0.23 pp). Policy responded on offroad/timeout but not collision — collision not learnable from 44D obs (perception limit, not reward-weight problem). `-500` retained by user decision. |
| R1 | promoted (trunk) | 20260517_134652 | Removed `route_completion < 0.3` guard in `_vehicle_reward` (`safe_to_push` and `alignment > 0.25` guards kept). Gate PASSES 4/4: SR +4.69 pp (22.70→27.38), stuck+timeout -3.45 pp, collision -1.46 pp, offroad +0.22 pp. +45 completions absolute (222→267); speed 11.53→13.04 km/h. Pedestrians (separate): SR -4.03 pp (likely run-to-run noise; R1 does not touch pedestrian reward). |
| R2 | retained provisional / not promoted | 20260517_164707 | Gate smooth-steering bonus on `speed_kmh > 5.0`. Gate FAILS: SR -1.85 pp, collision +1.43 pp, offroad +3.15 pp. Speed +2.21 km/h but extra mobility→offroad/collision. Equal-episode truncation keeps verdict. |
| Route-len bugfix (Punto 5) | kept (env correctness) | 20260517_212109 | `route_planner.py` enforces `<= 2.0x target` (commit `24e072e`). Gate PASSES 4/4 (SR +49.57 pp) — task-distribution change, NOT policy. Prior H/R absolute SR not comparable. |
| Route-seed fix | applied 2026-05-18 / verified | — | `hash(ad.agent_id)` → `SeedSequence([traffic_seed, reset_count, zlib.crc32(agent_id)])`. A/B runs now route-paired. First exercised in `20260518_195947`. |
| O1+O2 | evaluated / healthy / 47D baseline | 20260518_195947 | Vehicle obs 44D→47D: norm `no_wp_steps`, `loop_penalty_active` flag, norm time-remaining. Markov state-aliasing fixes. HEALTHY: plateau ~79% SR, `vf_explained_var` 0.983. Caveat: 16.4% `legacy_fallback`. Not checkpoint-compatible with 44D runs. See `PROPOSED_PLAN.md` Punti 6-7. |
| EvoEntropy (step 1) | promoted / validated | 20260520_133747 | Step 1 of EVO roadmap. Fraction-based entropy schedule `[[0,0.03],[249000,0.005]]` consolidated on post-bugfix 47D / R1 trunk (commit `1f3324d` EvoEntropySchemaCleanFractionBasedSchedule). 300K easy-locked, seed 999. Gate PASS: mechanism (entropy decreases monotonically, KL stable, no instability); episode integrity 6/6, no NaN/inf. Cumulative metrics: veh SR 60.48% (802 compl), ped SR 85.67% (1136 compl), veh stuck 22.47%, veh collision 8.45%, veh offroad 5.73%, ped speed 1.99 m/s. Acts as baseline for Ped-route+Ped-speed comparison. |
| Ped-route + Ped-speed (bundle) | evaluated / gate partial / retained pending V3 | 20260525_091300 | Step 2 of EVO roadmap. Bundle of two commits on `EVO/new-main`: `21a6724` EvoPedRouteRejectShortChains (`route_planner.py`: `min_route_ratio=0.5` rejects short pedestrian chains; per commit message, effect is relabel `sidewalk_distance → sidewalk_fallback`, no deep geometric fix) + `185a9d2` EvoPedSpeedWidenComfortBand (`carla_multi_agent_env.py:2004` `_pedestrian_reward` section 5: comfort band `[0.8, 1.8] → [1.2, 2.6]` m/s). 300K easy-locked, seed 999. **Gate1** (ped SR ≥ −2 pp vs EvoEntropy): **PASS** (+0.04 pp, 85.67% → 85.71%). **Gate2** (ped speed ∈ [1.5, 2.2] m/s): **FAIL** (2.281 m/s, +0.081 m/s over upper bound; baseline 1.990 m/s; Δ +0.291 m/s, +14.6%). Ripple on vehicles (file shared): stuck +4.70 pp (22.47 → 27.18), speed −2.27 km/h (16.03 → 13.76), no_wp_steps +48.5 (215.8 → 264.3); collision −1.71 pp & offroad −1.79 pp interpreted as consequence of vehicle inactivity, not policy improvement. Mechanism (verified at L1911–1946): faster peds saturate `ped_ttc/ped_occ` → `hazard_risk ≥ 0.75` → `safe_to_push=False` → urgency/`target_min_speed` penalty gates off, vehicle finds zero-speed local optimum; `no_wp_steps>100` penalty (cap 1.0, rate 0.004) too slow to break loop (stuck eps `no_wp_steps` mean 745 → 808). Ped `sidewalk_fallback` 7.24% → 20.69% (relabel-only per commit). Bundle retained pending V3 retune. |
| V3 (Ped-speed retune) | rejected / reverted | 20260525_125127 | Narrow diff (`carla_multi_agent_env.py:2004` `_pedestrian_reward` section 5): comfort band `[1.2, 2.6] → [1.5, 2.2]` m/s. 300K easy-locked, seed 999. Episode integrity 6/6 (450 ep × 6 = 2700 records, no NaN/inf). Cumulative: veh SR 62.00% (837 compl), ped SR 85.41% (1153 compl), veh stuck 22.22%, veh speed 11.88 km/h, ped speed **2.641 m/s**. **Gate1** (ped SR ≥ −2 pp vs EvoEntropy 85.67%): **PASS** (Δ −0.26 pp). **Gate2** (ped speed ∈ [1.5, 2.2] m/s): **FAIL** (+0.441 over upper bound; worse than Bundle 2.281). **β** side-check (veh stuck ≤ ~24%): PASS (22.22% ≈ EvoEntropy 22.47%). Trajectory Q1→Q4 reveals plateau: veh SR 15.1→70.1→83.4→**79.3** (−4.1 pp decline, worst of 3 runs); veh speed Q4 16.3 km/h vs EvoE 24.0, Bundle 19.1; ped speed Q1→Q4 1.77→2.94→2.90→**2.96** (band exit by Q2). Diagnosis: narrowing band amplifies pedestrian overshoot — comfort bonus is asymmetric (only positive inside band) and dominated by progress rewards. Fast unpredictable peds saturate `hazard_risk`, gating off vehicle urgency/min_speed → defensive equilibrium → Q4 SR decline. **Reverted per protocol**; band restored to `[1.2, 2.6]`. Motivates V1 (raise `safe_to_push` threshold to keep urgency active under higher hazard). |
| V1 (safe_to_push tolerance) | promoted (trunk) | 20260525_162428 | Narrow diff (`carla_multi_agent_env.py:1927` `_vehicle_reward`): `safe_to_push = hazard_risk < 0.75 → < 0.85`. 300K easy-locked, seed 999. Episode integrity 6/6 (444 ep × 6 = 2664 records, no NaN/inf). Cumulative metrics vs EvoEntropy `20260520_133747`: **Gate PASS 5/5**: veh SR 62.76% (+2.28 pp ≥ +2 pp), veh stuck+timeout 25.53% (−2.97 pp ≤ −2 pp), veh collision 7.43% (−1.02 pp, _improved_), veh offroad 4.28% (−1.45 pp, _improved_), ped SR 85.89% (+0.22 pp). Bonus: veh speed 18.30 km/h (+2.27), ped speed 2.136 m/s (back inside `[1.5, 2.2]` window). Trajectory Q1→Q4 vehicle SR `19.5→73.3→76.6→81.7` (+5.1 pp Q3→Q4, **strongest ascending of 4 runs**); Q4 veh speed 26.38 km/h (highest across V3/Bundle/EvoE/V1). Mechanism confirmed: raising threshold from 0.75 to 0.85 keeps urgency reward and `target_min_speed=8.0` penalty active under hazard ∈ [0.75, 0.85), preventing defensive equilibrium. Safety floor preserved at hazard ≥ 0.85. Merged into the unified `EVO/new-main` ≡ `EVO/curriculum-stack` (P1+P2+P3 + V1 verified on both branches at commit `d84fa3a`); step 5 launched as run `20260525_205912` (3M, `--difficulty path`). |

See `docs/EXPERIMENT_REGISTRY.md` for per-candidate implementation details.

---

## Promotion Gate

Default gate for vehicle-focused changes:

| Metric | Threshold |
|--------|-----------|
| Vehicle success rate | ≥ `+2.0 pp` |
| Vehicle stuck + timeout | ≤ `-2.0 pp` |
| Collision rate change | ≤ `+1.0 pp` |
| Offroad rate change | ≤ `+1.0 pp` |
| Episode integrity | 6 agent-level records per episode; no NaN/inf |

If a candidate fails the gate, revert only that candidate. Keep the accepted trunk intact.

---

## Measurement Rules

1. **Success definition**: `termination_reason == "route_complete"`.
2. **Aggregation**: cumulative agent-level metrics. Never use joint success rate unless explicitly asked.
3. **Integrity**: 6 agent-level records per episode. Deduplicate by `episode_id + agent_id`, keep last.
4. **Always report separately**: `vehicles+pedestrians`, `vehicles`, `pedestrians`.
5. **Key metrics**: success rate, stuck rate, timeout rate, stuck+timeout, collision rate, offroad rate,
   route completion, path efficiency, speed, no-waypoint steps.
6. **Recalculate from disk** when `episodes.jsonl` is live or recently updated.

---

## Do Not Infer

- Final evaluation quality from training episodes alone.
- Curriculum budget or sampling-weight evidence from easy-only locked runs.
- Comparisons between 44D and 47D runs (different obs spaces; not checkpoint-compatible).
- Policy improvements from evaluation/reporting tooling changes.

---

## Technical Constraints

- **Python**: narrow diffs, explicit control flow, deterministic config handling, verification with compile checks.
- **Every proposed change** must state: source, assumption transferred, expected measurable effect,
  files affected, gate.
- **Do not** introduce PopArt, attention, GNNs, new policies, or obs-space changes unless explicitly
  requested.
- **Source hierarchy**: official docs for simulator/API → local repo code for implementation →
  `episodes.jsonl` for empirical truth.
- Never label a method "state of the art" without a concrete source.
- Distinguish change types: bug fix / reward shaping / obs change / curriculum / architecture / eval tooling.
- **Local empirical evidence has priority.** A change that fails the gate is not promoted.

---

## Workflow

1. **Before editing**: `git status --short` to freeze state.
2. **Code exploration**: use graph tools first (see MCP section above); fall back to `rg` only when graph coverage is insufficient.
3. **Changes**: keep narrow and behaviorally isolated; do not revert unrelated user changes.
4. **After each patch**: `python -m compileall <touched files>` + `git diff --check`.
5. **Reports**: update `reports/carla_finetuning_maggio_2026.docx` only with metrics recalculated from disk.
6. **Multi-step work**: give concise preamble before notable tool use; track progress with a numbered TODO list.

### Common Commands

```bash
git status --short
git diff --check

# Compile touched modules
python -m compileall carla_core\envs\carla_multi_agent_env.py carla_core\training\curriculum_batch_manager.py carla_core\training\train_carla_mappo.py

# Full path-curriculum run (user launches; never auto-run)
# Current operating budget is 300k timesteps (in-flight reference: 20260520_133747); 3M is reserved for the final long run.
python -m carla_core.training.train_carla_mappo --mode curriculum --difficulty path --timesteps 300000 --seed 999
```

---

## Track Record Maintenance

After each evaluated run, ablation, or experimental decision:
1. Update `## Active Configuration` (and relevant sections) in this file.
2. Update `reports/carla_finetuning_maggio_2026.docx`.
3. Update `CLAUDE.md` to keep both files synchronized.

**Synchronization rules:**
- Do not write `AGENTS.md` and `CLAUDE.md` simultaneously.
- Before editing either, read the latest delta for both; add only missing information;
  preserve the other agent's changes.
- If one file cannot be updated, stop and report the synchronization blocker.
- All claims must be tied to run IDs and `episodes.jsonl` evidence.
- Update `Last updated` whenever this file changes.
- Language: English only.
