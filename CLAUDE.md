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

---

<metadata>
Last updated: 2026-05-18
Purpose: Repository-level operating instructions for Claude Code.
Language rule: Write all future additions and updates to this file in English.
</metadata>

<role>
Act as a senior AI/ML Engineer and domain expert in Python engineering,
Reinforcement Learning, MAPPO, and CARLA simulation for multi-agent
autonomous-driving experiments.

Be empirical, concise, and gate-driven. Do not invent metrics, baselines,
citations, or results. If a value was not verified from files, logs, or tool
output, state that it is unverified.
</role>

<research_question>
Does curriculum learning from `easy -> medium -> hard` produce measurably
different multi-agent driving behavior than mixed/batch training in MARL for
urban autonomous driving?

Evaluate this through agent-level metrics, policy-level breakdowns, and
Town03-to-Town05 generalization. Do not reduce the thesis question to a single
success-rate leaderboard.
</research_question>

<fixed_stack>
- Simulator: `CARLA 0.9.16`.
- Algorithm: `MAPPO` with CTDE.
- Framework: `Ray/RLlib 2.10.0`.
- Runtime: `Python 3.11.9`, `PyTorch 2.7+cu126`.
- Setup: `3` RL vehicles and `3` RL pedestrians.
- Train map: `Town03`.
- Test map: `Town05`.
- Default architecture: centralized critic, separate vehicle and pedestrian
  policies, PopArt off, attention off, GNN off.
- Do not use MetaDrive assumptions, files, or conclusions for CARLA decisions.
</fixed_stack>

<current_goal>
Improve the vehicle policy without changing the MAPPO architecture unless the
user explicitly requests it. The observed vehicle failure modes are low success
rate, high stuck rate, high timeout rate, low average speed, and weak early
route progress. Pedestrians are comparatively stronger and must always be
reported separately from vehicles.
</current_goal>

<current_known_state>
- Vehicle obs: `44D`, pedestrian obs: `26D`, global obs: `216D`.
- `C0` and `C1` are implemented: diagnostics plus geometric observations
  without changing vehicle observation dimensionality.
- `D2` reward shaping is the current useful trunk among tested reward changes.
- `D3` early vehicle-stuck termination failed the gate and was reverted.
- `H1+H1.1` (`vf_clip_param` 10->1e6, `vf_loss_coeff` 0.5->0.05) is
  mechanistically confirmed (vehicle `vf_explained_var` ~0 -> 0.87, baseline
  critic non-functional) but failed the vehicle gate (collision +3.28 pp) on
  run `20260515_175921`; not promoted but not reverted, and `vf_clip` is kept (not reverted) as
  the base for `H2`.
- `H2` (`gamma` 0.99->0.997) was evaluated on run `20260515_211055` and failed
  the vehicle gate 3 of 4 vs `20260515_175921`: SR +0.14 pp, stuck+timeout
  -7.48 pp, collision +5.18 pp, offroad +2.16 pp. The longer horizon amplified
  the route-completion incentive and converted passive failure (timeout) into
  active failure (collision/offroad) roughly 1:1, leaving SR flat; the H2
  hypothesis (longer horizon propagates the -50 collision penalty) is
  falsified. By user decision `gamma=0.997` is kept (not reverted) as the base
  for `H3`.
- `H3` (`entropy_coeff` constant 0.03 -> schedule `[[0,0.03],[250000,0.005]]`)
  was evaluated on run `20260516_144007` (single-knob A/B vs `20260515_211055`;
  seed 999, easy-lock, 300k). Mechanism confirmed: vehicle final `entropy`
  4.78 -> 3.25 (schedule reached `entropy_coeff=0.005`), `vf_explained_var`
  0.92 (critic healthy). Vehicle gate vs `20260515_211055` (cumulative,
  recomputed from `episodes.jsonl`) FAILS 3 of 4: SR -0.32 pp (21.75->21.43),
  stuck+timeout -0.37 pp (44.71->44.35), collision -0.99 pp (26.79->25.79),
  offroad +1.69 pp (6.75->8.43). Vehicle route-completions are identical in
  absolute count (216 in both runs); the SR delta is denominator-only. All
  deltas sit within the run-to-run noise visible in the pre-250k chunks where
  H2 and H3 share an identical config. Not promoted; by user decision the
  entropy schedule is retained (not reverted) as the base for `R3`. H1+H2+H3
  together: three optimizer-side single-knob changes, all mechanism-confirmed,
  all failing the vehicle gate, vehicle SR pinned ~21.5%; the binding
  constraint is the vehicle collision rate (~25.8%).
- `R3` (vehicle collision penalty `-50.0` -> `-500.0` at
  `carla_multi_agent_env.py:1748`) was evaluated on run `20260516_200545`
  (single-knob A/B vs `20260516_144007`; seed 999, easy-lock, 300k). The
  adapted R3 gate FAILS: vehicle SR +1.27 pp (21.43->22.70), collision
  -0.23 pp (25.79->25.56), stuck+timeout +2.28 pp (44.35->46.63), offroad
  -3.32 pp (8.43->5.11); integrity OK (326 ep x 6 = 1956 records, 0 dups,
  0 NaN/inf); vehicle `vf_explained_var` 0.92->0.88, `entropy` 3.25->2.58.
  The hypothesis "the vehicle collision rate is tunable via the
  collision-penalty magnitude" is falsified: a 10x penalty left the collision
  rate flat. The policy did respond to the reward change (offroad -3.32 pp,
  stuck -7.17 pp converted into timeout +9.45 pp, `route%` 0.44->0.50) but not
  on the collision axis -- evidence that collision avoidance is not learnable
  from the current 44D vehicle observation (a perception limit, not a
  reward-weight problem). R3 is not promoted; by user decision the `-500`
  penalty is retained (not reverted) as the experimental base for the next
  candidate `R1`. Collision-axis reward shaping (penalty magnitude) is
  exhausted, but `R1` and `R2` remain untested reward-shaping candidates that
  target the dominant stuck+timeout failure, not the collision axis.
- `R1` (remove the `route_completion < 0.3` reward gate in `_vehicle_reward`,
  so start/unblock and `target_min_speed` shaping stay active for the whole
  route) was evaluated on run `20260517_134652` (single-knob A/B vs
  `20260516_200545`; seed 999, easy-lock, ~300k). Vehicle gate (cumulative,
  recomputed from `episodes.jsonl`) PASSES 4 of 4: SR +4.69 pp (22.70->27.38),
  stuck+timeout -3.45 pp (46.63->43.18), collision -1.46 pp (25.56->24.10),
  offroad +0.22 pp (5.11->5.33); integrity OK (325 ep x 6 = 1950 records,
  0 dups, 0 NaN/inf). On-mechanism: vehicle route-completions +45 in absolute
  count (222->267, not a denominator effect); the timeout cohort is the
  largest contributor (-2.71 pp / -27 episodes); vehicle speed 11.53->13.04
  km/h; the collision canary improved rather than regressed; no late-training
  SR decay (chunk-6 vehicle SR 38.2 vs R3 27.6). Pedestrians (reported
  separately): SR 89.06->85.03 (-4.03 pp), stuck +4.23 pp -- `R1` does not
  touch `_pedestrian_reward`; most likely run-to-run noise (pedestrian SR has
  ranged 84.7-89.1 across the last four runs, R3 the high point) with a
  possible small MARL shared-environment contribution; to be confirmed with
  extra seeds. `R1` is the first candidate to pass the vehicle gate since
  `D2`; promoted into the trunk and retained as the base for `R2`.
- `R2` (gate the `+0.1` smooth-steering bonus on `speed_kmh > 5.0`) was
  evaluated on run `20260517_164707` against the new `R1` baseline
  `20260517_134652` (same `run_config.json`, seed 999, easy-lock, ~300k).
  Integrity OK (342 ep x 6 = 2052 records, 0 dups, 0 NaN/inf). Vehicle gate
  FAILS formally vs `R1`: SR -1.85 pp (27.38->25.54), stuck+timeout -2.73 pp
  (43.18->40.45), collision +1.43 pp (24.10->25.54), offroad +3.15 pp
  (5.33->8.48). Mechanism is plausible but mixed: vehicle speed rises
  13.04->15.25 km/h and `no_wp_steps` drops 232.94->204.84, so immobility is
  reduced, but the extra mobility converts into offroad/collision rather than
  route completions. Equal-episode truncation of `R2` to the first 325 episodes
  keeps the verdict: SR -2.15 pp, stuck+timeout -2.05 pp, collision +0.92 pp,
  offroad +3.28 pp. User decision: do not revert immediately; retain `R2` as a
  provisional experimental base, but do not promote it as validated trunk unless
  follow-up validation brings collision/offroad back within the +1.0 pp canary.
  `results.json.evaluation` is empty/final evaluation pending, so this is
  training-only evidence.
- Historical comparison of `R2` `20260517_164707` vs old baseline
  `20260511_153859` is not an isolated A/B: the old run differs in route
  distance, traffic density, `gamma`, `vf_clip_param`, `vf_loss_coeff`, entropy
  schedule, and curriculum lock behavior. As historical context only, vehicle
  SR is +5.01 pp and stuck+timeout is -16.69 pp, but collision is +9.13 pp and
  offroad is +2.55 pp; do not use this delta as causal evidence for `R2`.
- Candidate pipeline (`PROPOSED_PLAN.md` order): `R1` is done and promoted
  (above). Remaining: `R2` (gate the smooth-steering bonus on
  `speed_kmh > 5`), the `route_planner` upper-bound bugfix (enforce route
  length `<= 2.0x` of target), and `O1+O2` (vehicle obs `44D -> 47D`: add
  `no_wp_steps` norm, loop flag, normalized time-remaining). `R2` and the
  bugfix are checkpoint-comparable; `O1+O2` change the vehicle observation
  space, are not checkpoint-comparable, and are applied last as a single
  retrain-from-scratch 47D variant. `O1+O2` are Markov state-aliasing fixes
  (the reward uses `no_wp_steps` and `loop_penalty_active`, and the episode is
  truncated at a fixed horizon), not hazard/perception features. The immediate
  next candidate is `R2`, a single-knob A/B on the `R1` base.
- Update after `R2`: the previous immediate-next line is now historical.
  `R2` has been run and is retained provisionally, not promoted. The next
  change should not stack new policy assumptions on top of `R2` as if it had
  passed the gate; treat the next candidate as a separately gated condition and
  keep the deterministic route-seed caveat visible (`hash(ad.agent_id) % 10000`
  still prevents true paired route sequences across processes).
- `Punto 5` (route-length upper-bound bugfix: `route_planner.py`
  `plan_vehicle_route` now rejects routes with `route_len > 2.0x target`,
  honoring the docstring `[0.5x, 2.0x]` contract; one-line change, commit
  `24e072e`) was evaluated on run `20260517_212109` (A/B vs `R2`
  `20260517_164707`; `run_config.json` byte-identical except the timestamp,
  the only code diff since the baseline is the `route_planner` one-liner;
  seed 999, easy-lock, 300k). Integrity OK (529 ep x 6 = 3174 records, 0 dups,
  0 NaN/inf). The default vehicle gate formally PASSES 4 of 4 vs `164707`
  (cumulative, recomputed from `episodes.jsonl`): SR +49.57 pp (25.54->75.11),
  stuck+timeout -21.17 pp (40.45->19.28), collision -21.25 pp (25.54->4.28),
  offroad -7.16 pp (8.48->1.32). This is NOT a policy improvement: policy,
  reward, observations and optimizer are identical between the two runs; only
  the env route generator differs. The +49.57 pp is a task-distribution change
  -- the bugfix removes routes longer than 2x target that were unfinishable
  within the 1000-step episode. Evidence: total episodes 342->529 (same
  timestep budget => shorter episodes), mean vehicle `step_count` 900.4->573.8,
  mean `route%` 49.59->87.08, absolute vehicle timeouts 173->34, absolute
  vehicle collisions 262->68. Vehicle SR by chunk: `164707`
  3.5/17.5/27.5/40.4/31.0/33.3 (peaks ~40, decays); `212109`
  10.9/70.0/91.0/93.3/94.0/92.5 (clean climb, plateau ~93) -- consistent with
  the bug also degrading training quality, though this single A/B cannot
  separate "easier task" from "healthier training". Pedestrians (reported
  separately): SR 84.41->89.60 (+5.20 pp), stuck+timeout -5.20 pp; the bugfix
  does not touch `plan_pedestrian_route_by_distance`, most likely mechanical
  (shorter joint episodes => less time to fail). Final evaluation pending
  (`results.json.evaluation` empty) -> training-only evidence. By user
  decision the bugfix is KEPT as an environment-correctness fix, not a
  promoted policy candidate; `20260517_212109` is the post-bugfix vehicle
  baseline.
- Consequence of `Punto 5`: all prior H/R-series runs (`D2`, `H1`-`H3`, `R3`,
  `R1`, `R2`) were trained and measured on the bug-contaminated route
  distribution; their absolute vehicle SR (~21-27%) was structurally
  depressed. Relative single-knob verdicts may still hold directionally (both
  arms shared the bug) but absolute numbers are not comparable across the
  bugfix; `R1` (removing the `route_completion < 0.3` reward gate) interacts
  with route length and is the verdict most exposed to a re-check on the
  corrected distribution.
- Route-seed determinism bug identified at `carla_multi_agent_env.py:960`: the
  vehicle route RNG seed is `traffic_seed + reset_count*1000 +
  hash(ad.agent_id) % 10000`. Python `hash()` of a `str` is per-process salted
  (`PYTHONHASHSEED`, not pinned in the repo), so route seeds are not
  reproducible across processes and A/B runs are not route-paired even with
  `seed=999` fixed; the additive composition also lets the 3 vehicles collide
  on the same seed at nearby `reset_count`. Single site, vehicles only.
  User-approved fix, not yet applied: replace with
  `np.random.SeedSequence([traffic_seed, reset_count, stable_hash(agent_id)])`
  (`stable_hash` via `zlib.crc32`); an environment-correctness/reproducibility
  fix, no gate (it does not change the route distribution, only its
  reproducibility), to be applied before `O1+O2`.
- Current path curriculum configuration uses `difficulty=path` with route
  distances `15m / 35m / 60m` for both vehicles and pedestrians.
- Current curriculum budget proposal is `easy=0.30`, `medium=0.32`,
  `hard=0.38`.
- Current sampling weights are base `easy=1.00`, `medium=1.07`, `hard=1.27`
  and probation `medium=1.00`, `hard=1.19`.
- `carla_mappo_20260514_211642` was an easy-only locked exploratory run. It
  tested `path` easy `15m/15m`, but it did not test budget constraints or
  sampling weights because `--lock-curriculum-level easy` disables those.

### Candidate Registry Summary

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
| H1+H1.1 | not promoted but not reverted / mechanism confirmed | 20260515_175921 | Critic fix: `vf_clip_param` 10->1e6 (H1) + `vf_loss_coeff` 0.5->0.05 (H1.1); obs unchanged (44D), checkpoint-comparable. Mechanism confirmed from RLlib logs: vehicle `vf_explained_var` ~0 (baseline -0.002, critic explained ~0% of return variance) -> 0.87. Vehicle gate vs `20260514_211642` (cumulative, recomputed from `episodes.jsonl`) FAILS 3 of 4: SR +1.51 pp (20.10->21.61), stuck+timeout -1.98 pp (54.17->52.19), collision +3.28 pp (18.33->21.61), offroad -2.81 pp (7.40->4.59). Blocker: collision regression and late-training degradation (`entropy`->5.43). Confounded (two knobs). Not promoted; `vf_clip` kept, not reverted (reverting restores a non-functional critic). |
| H2 | not promoted / not reverted / hypothesis falsified | 20260515_211055 | `gamma` 0.99->0.997 on the H1+H1.1 base; single-knob A/B vs `20260515_175921` (only `gamma` differs; seed 999, easy-lock, 300k). Intended to reduce the H1+H1.1 collision regression by propagating the -50 penalty over a longer horizon; produced the opposite. Vehicle gate vs `175921` (cumulative, recomputed from `episodes.jsonl`) FAILS 3 of 4: SR +0.14 pp (21.61->21.75), stuck+timeout -7.48 pp (52.19->44.71), collision +5.18 pp (21.61->26.79), offroad +2.16 pp (4.59->6.75). Mechanism: longer horizon amplified the dominant route-completion incentive (speed 10.08->13.81 km/h, timeout -6.53 pp) and converted passive failure into active failure roughly 1:1; SR flat. Critic healthy (`vf_explained_var` 0.94). Hypothesis falsified. Not reverted by user decision: `gamma=0.997` retained as the base for `H3`. |
| H3 | not promoted / not reverted / mechanism confirmed | 20260516_144007 | `entropy_coeff` constant 0.03 -> schedule `[[0,0.03],[250000,0.005]]` on the H1+H1.1+H2 base; single-knob A/B vs `20260515_211055` (only the schedule differs; seed 999, easy-lock, 300k). Mechanism confirmed: vehicle final `entropy` 4.78 -> 3.25 (schedule reached `entropy_coeff=0.005`), `vf_explained_var` 0.92 (critic healthy). Vehicle gate vs `211055` (cumulative, recomputed from `episodes.jsonl`) FAILS 3 of 4: SR -0.32 pp (21.75->21.43), stuck+timeout -0.37 pp (44.71->44.35), collision -0.99 pp (26.79->25.79), offroad +1.69 pp (6.75->8.43). Vehicle route-completions identical in absolute count (216 vs 216); the SR delta is denominator-only (H3 ran 5 more episodes). All deltas within the run-to-run noise visible in the pre-250k chunks (identical config there). Late-training SR decay only slightly softened (chunk-6 vehicle SR 22.62 vs 19.64); chunk-4 peak (~32.7%) unchanged; composition shifted timeout -5.81 pp / stuck +5.45 pp. Integrity OK (336 ep x 6 = 2016 records, 0 dups, 0 NaN/inf). Not promoted; by user decision the entropy schedule is retained (not reverted) as the base for `R3`. |
| R3 | not promoted / not reverted / hypothesis falsified | 20260516_200545 | Reward shaping: vehicle collision penalty at `carla_multi_agent_env.py:1748` raised `reward -= 50.0` -> `reward -= 500.0` (vehicle only; pedestrian `-50.0` at `:1841` unchanged), on the `H3` base; single-knob A/B vs `20260516_144007` (seed 999, easy-lock, 300k). Adapted R3 gate (PRIMARY: vehicle SR >= +2.0 pp AND collision <= -3.0 pp; CANARY: stuck+timeout and offroad must not worsen > +1.0 pp) FAILS: SR +1.27 pp (21.43->22.70), collision -0.23 pp (25.79->25.56), stuck+timeout +2.28 pp (44.35->46.63), offroad -3.32 pp (8.43->5.11). Integrity OK (326 ep x 6 = 1956 records, 0 dups, 0 NaN/inf); vehicle `vf_explained_var` 0.92->0.88, `entropy` 3.25->2.58. Hypothesis "collision rate is tunable via the collision-penalty magnitude" falsified: the 10x penalty left collision flat. The policy responded elsewhere (offroad -3.32 pp; stuck -7.17 pp converted into timeout +9.45 pp; `route%` 0.44->0.50) but not on the collision axis -> collision avoidance is not learnable from the current 44D vehicle obs (perception limit). Not promoted; `-500` retained (not reverted) by user decision as the base for the planned observation (hazard-perception) experiment. |
| R1 | promoted (trunk) | 20260517_134652 | Reward shaping: dropped the `route_completion < 0.3` guard in `_vehicle_reward` so start/unblock and `target_min_speed` shaping stay active for the whole route (`safe_to_push` and `alignment > 0.25` guards kept, coefficients unchanged). Single-knob A/B vs R3 `20260516_200545`. Vehicle gate PASSES 4/4: SR +4.69 pp (22.70->27.38), stuck+timeout -3.45 pp (46.63->43.18), collision -1.46 pp (25.56->24.10), offroad +0.22 pp. +45 route-completions in absolute count (222->267); timeout cohort the largest contributor (-2.71 pp); speed 11.53->13.04 km/h; collision canary improved. Integrity OK (325 ep x 6 = 1950, 0 dups, 0 NaN/inf). Pedestrians (separate): SR -4.03 pp (R1 does not touch pedestrian reward; likely run-to-run noise) -- to confirm with extra seeds. First gate pass since `D2`; promoted into the trunk, base for `R2`. |
| R2 | retained provisional / not promoted | 20260517_164707 | Reward shaping: gate the `+0.1` smooth-steering bonus in `_vehicle_reward` (`carla_multi_agent_env.py:1803`) on `speed_kmh > 5.0` so a stationary vehicle no longer collects it. Checkpoint-comparable. Single-knob comparison vs `R1` baseline `20260517_134652` formally FAILS the vehicle gate: SR -1.85 pp, stuck+timeout -2.73 pp, collision +1.43 pp, offroad +3.15 pp. Mechanism plausible (speed +2.21 km/h, `no_wp_steps` -28.10), but extra mobility increased offroad/collision rather than SR. User decision: keep as provisional base, no immediate revert; not a validated promotion. |
| Route-len bugfix (Punto 5) | evaluated / kept (env correctness fix; not a policy promotion) | 20260517_212109 | Env bugfix: `route_planner.py` `plan_vehicle_route` now enforces the docstring `<= 2.0x target` upper bound (commit `24e072e`; previously only the `0.5x` lower bound was checked). Single-knob A/B vs `R2` `20260517_164707` (`run_config.json` byte-identical). Vehicle gate formally PASSES 4/4 (SR +49.57 pp 25.54->75.11, stuck+timeout -21.17 pp, collision -21.25 pp, offroad -7.16 pp; 3174 records, 0 dups/NaN) -- but the delta is a task-distribution change (removes routes >2x target unfinishable in 1000 steps: episodes 342->529, mean step_count 900->574, route% 50->87), NOT a policy improvement. Kept by user decision as an env-correctness fix; `20260517_212109` is the post-bugfix vehicle baseline; prior H/R absolute numbers not comparable across the bugfix. Final eval pending. See `PROPOSED_PLAN.md` Punto 5. |
| Route-seed determinism fix | pending (user-approved; code not yet applied) | — | Env reproducibility bugfix: `carla_multi_agent_env.py:960` seeds the vehicle route RNG with `hash(ad.agent_id)`; Python `hash()` of a `str` is per-process salted (`PYTHONHASHSEED` not pinned) so route seeds are not reproducible across processes and A/B runs are not route-paired. Fix: `np.random.SeedSequence([traffic_seed, reset_count, zlib.crc32(agent_id)])`. No gate (does not change the route distribution, only reproducibility); to be applied before `O1+O2`. |
| O1+O2 | pending (obs change; applied last) | — | Vehicle obs `44D -> 47D`: O1 adds normalized `no_wp_steps` + `loop_penalty_active` flag, O2 adds normalized time-remaining. Markov state-aliasing fixes (the reward uses these quantities, the obs does not), not hazard/perception features. Not checkpoint-comparable; one retrain-from-scratch 47D variant. See `PROPOSED_PLAN.md` Punti 6-7. |

See `docs/EXPERIMENT_REGISTRY.md` for per-candidate implementation logic and pseudocode.
</current_known_state>

<current_accepted_trunk>
- Active trunk: `C0 + C1 + D2 + R1`.
- Reverted: `D3`.
- `H1+H1.1` is not promoted but not reverted; `vf_clip_param=1000000.0` and
  `vf_loss_coeff=0.05` are retained as the experimental base for `H2` because
  the H1 critic fix is mechanistically confirmed and reverting it restores a
  non-functional critic.
- `H2` (`gamma=0.997`) was evaluated (run `20260515_211055`) and is not
  promoted; the H2 hypothesis is falsified. By user decision it is not
  reverted: `gamma=0.997` is retained as the experimental base for `H3`.
- `H3` (`entropy_coeff` schedule `[[0,0.03],[250000,0.005]]`) was evaluated
  (run `20260516_144007`) and is not promoted; the mechanism is confirmed
  (vehicle `entropy` 4.78->3.25) but the vehicle gate fails 3 of 4. By user
  decision the schedule is not reverted: it is retained as the experimental
  base for `R3`.
- `R3` (vehicle collision penalty `-50.0`->`-500.0` at
  `carla_multi_agent_env.py:1748`) was evaluated (run `20260516_200545`) and
  is not promoted; the adapted R3 gate fails and the hypothesis (collision
  rate tunable via the collision-penalty magnitude) is falsified. By user
  decision the `-500` penalty is not reverted: it is retained as the
  experimental base for the next candidate `R1`.
- `R1` (remove the `route_completion < 0.3` reward gate in `_vehicle_reward`)
  was evaluated (run `20260517_134652`) and PASSES the vehicle gate 4 of 4 vs
  `20260516_200545`: SR +4.69 pp, stuck+timeout -3.45 pp, collision -1.46 pp,
  offroad +0.22 pp. It is the first candidate to pass the gate since `D2`;
  promoted into the trunk and retained as the base for `R2`. Pedestrian SR
  regressed -4.03 pp (`R1` does not touch `_pedestrian_reward`; likely
  run-to-run noise, to be confirmed with extra seeds).
- `R2` (gate smooth-steering bonus on `speed_kmh > 5.0`) was evaluated (run
  `20260517_164707`) and is retained provisionally by user decision but is not
  promoted: vs `R1` baseline `20260517_134652`, vehicle SR -1.85 pp,
  stuck+timeout -2.73 pp, collision +1.43 pp, offroad +3.15 pp. It reduced
  immobility signals (speed +2.21 km/h, `no_wp_steps` -28.10) but breached the
  safety canaries, especially offroad.
- `Punto 5` (route-length upper-bound bugfix in `route_planner.py`
  `plan_vehicle_route`) was evaluated (run `20260517_212109`) and KEPT by user
  decision as an environment-correctness fix -- not a promoted policy
  candidate. It formally passes the vehicle gate 4/4, but the +49.57 pp vehicle
  SR is a task-distribution change (removal of routes > 2x target), not a
  policy improvement. `20260517_212109` is the post-bugfix vehicle baseline;
  prior H/R-series absolute numbers are not comparable across the bugfix.
- Route-seed determinism fix (`carla_multi_agent_env.py:960`,
  `hash(ad.agent_id)` -> `np.random.SeedSequence`) is user-approved but not yet
  applied; an environment-correctness/reproducibility fix (no gate), to be
  applied before `O1+O2` so future A/B runs are route-paired.
- Next planned (updated 2026-05-18): route-seed determinism fix -> `O1+O2`
  (vehicle obs `44D -> 47D`, retrain-from-scratch 47D variant) -> final
  evaluation; this supersedes the earlier `Next planned` lines below. The full
  `difficulty=path` curriculum remains pending/conditional.
- Next planned (`PROPOSED_PLAN.md` order): `R2` (gate the smooth-steering
  bonus on `speed_kmh > 5`) -> `route_planner` upper-bound bugfix -> `O1+O2`
  (vehicle obs `44D -> 47D`, Markov state-aliasing fixes, applied last as one
  retrain-from-scratch 47D variant). The immediate next candidate is `R2`, a
  single-knob A/B on the `R1` base.
- Post-`R2` note: the previous "Next planned" sentence is retained for history
  but is superseded. `R2` is now evaluated and provisional; do not count it as
  accepted trunk evidence until a follow-up validation meets the collision and
  offroad canaries.
- Pending/conditional: full `difficulty=path` curriculum without
  `--lock-curriculum-level`, using route distances `15m / 35m / 60m`, budget
  shares `0.30 / 0.32 / 0.38`, and sampling weights `1.00 / 1.07 / 1.27`.
</current_accepted_trunk>

<do_not_infer>
- Do not infer final evaluation quality from training episodes alone.
- Do not treat easy-only locked runs as evidence for curriculum budget or
  sampling weights.
- Do not compare observation-dimension-changing runs directly with
  checkpoint-compatible runs.
- Do not treat evaluation/reporting tooling improvements as policy
  improvements.
</do_not_infer>

<technical_and_scientific_constraints>
- Use established Python engineering practices: narrow diffs, readable code,
  explicit control flow, deterministic configuration handling, structured
  parsers when available, and verification with targeted tests or compile
  checks.
- For RL, MAPPO, CARLA, and multi-agent autonomous-driving decisions, prefer
  evidence-backed guidance from official documentation, mature open-source
  projects, or recognized technical and academic papers.
- Treat external best practices and state-of-the-art methods as hypotheses to
  adapt, not as automatic prescriptions. Every proposed change must be mapped to
  this repository's fixed setup: CARLA `0.9.16`, MAPPO CTDE, `3` RL vehicles,
  `3` RL pedestrians, Town03 training, and Town05 testing.
- Do not introduce architectural changes such as PopArt, attention, GNNs, new
  policies, or observation-space changes unless explicitly requested or approved
  as a separate experimental variant.
- When proposing a technique from literature or another project, state the
  source category, the assumption being transferred, the expected measurable
  effect, the files likely affected, and the gate that would validate or reject
  it.
- Prefer official documentation for simulator/API behavior, local repository
  code for implementation truth, and run logs/`episodes.jsonl` for empirical
  truth.
- Never present a method as "state of the art" unless it is tied to a concrete
  source or clearly labeled as an unverified hypothesis.
- Keep experimental comparability explicit: distinguish bug fixes, reward
  shaping, observation changes, curriculum changes, architecture changes, and
  evaluation/reporting changes as separate experimental conditions.
- Separate training improvements from evaluation/reporting improvements.
  Evaluation tooling can improve measurement quality, but it must not be counted
  as a policy improvement.
- If external guidance conflicts with local CARLA results, report the conflict
  explicitly and use the local results for promotion decisions.
- A change that is theoretically sound but fails the project gate is not
  promoted. Local empirical evidence has priority over external intuition.
</technical_and_scientific_constraints>

<measurement_rules>
- Primary success definition: `termination_reason == "route_complete"`.
- Primary aggregation: cumulative agent-level metrics.
- Do not use joint success rate unless the user explicitly asks for it.
- Every episode should produce `6` agent-level records.
- If duplicates exist, deduplicate by `episode_id + agent_id`, keeping the last
  record.
- Always report at least these groups when evaluating a run:
  `vehicles+pedestrians`, `vehicles`, and `pedestrians`.
- Key metrics: success rate, stuck rate, timeout rate, stuck+timeout,
  collision rate, offroad rate, route completion, path efficiency, speed, and
  no-waypoint steps when available.
- If `episodes.jsonl` is live or recently updated, recalculate from disk.
</measurement_rules>

<gate_policy>
Promote a candidate only if the evidence supports it. The default gate for
vehicle-focused changes is:

- Vehicle success rate improves by at least `+2.0 pp`.
- Vehicle `stuck + timeout` decreases by at least `-2.0 pp`.
- Collision and offroad rates do not worsen by more than `+1.0 pp`.
- No NaN or inf values appear in observations, rewards, global observations, or
  recorded metrics.
- Episode integrity remains valid: expected `6` agent-level records per
  episode.

If a candidate fails the gate, revert only that candidate and keep the accepted
trunk intact.
</gate_policy>

<code_review_graph_workflow>
This project has a code knowledge graph. Use code-review-graph MCP tools before
Grep/Glob/Read when exploring the codebase.

Use these tools first when possible:
- `query_graph` for `callers_of`, `callees_of`, `imports_of`, and `tests_for`.
- `get_impact_radius` to understand blast radius.
- `detect_changes` and `get_review_context` for code review.
- `get_affected_flows` for impacted execution paths.
- `semantic_search_nodes` for functions/classes by name or keyword.

Fall back to `rg`, file reads, or direct inspection only when the graph does not
cover the target or returns no useful result.
</code_review_graph_workflow>

<tool_and_workflow_rules>
- Before modifying code or configs, freeze state with `git status --short`.
- Use `rg` or `rg --files` before slower search methods.
- Use `apply_patch` for manual edits.
- Keep changes narrow and behaviorally isolated.
- Do not revert unrelated user changes.
- Before proposing or applying experimental changes, list impacted files, risk,
  verification plan, and expected gate.
- After a patch, run static checks that match the touched files. At minimum, use
  `python -m compileall` for touched Python modules and `git diff --check`.
- For reports, update `reports/carla_finetuning_maggio_2026.docx` only with
  metrics recalculated from disk or clearly labeled qualitative decisions.
- Give concise preambles before notable tool use and track multi-step work with
  a TODO list.
</tool_and_workflow_rules>

<common_commands>
- Check repo state:
  `git status --short`
- Check whitespace in diffs:
  `git diff --check`
- Compile touched CARLA/MAPPO modules:
  `python -m compileall carla_core\envs\carla_multi_agent_env.py carla_core\training\curriculum_batch_manager.py carla_core\training\train_carla_mappo.py`
- Candidate full path-curriculum run:
  `python -m carla_core.training.train_carla_mappo --mode curriculum --difficulty path --timesteps 3000000 --seed 999`
</common_commands>

<track_record_maintenance>
- Write all future additions and updates to this `CLAUDE.md` file in English.
- After each evaluated run, ablation, or experimental decision, update:
  - the `current_known_state` section in this file;
  - `reports/carla_finetuning_maggio_2026.docx`;
  - `AGENTS.md` when the shared project instructions also need the same state.
- Keep `CLAUDE.md` and `AGENTS.md` synchronized: any durable project-state,
  instruction, gate, run, ablation, or experimental-decision update must be
  reflected in both files.
- Codex and Claude Code must not write `CLAUDE.md` or `AGENTS.md` at the same
  time. If another agent is actively editing either file, wait until that agent
  finishes before editing either file.
- Before updating either file after another agent has edited them, read the
  latest delta for both `CLAUDE.md` and `AGENTS.md`, preserve the other agent's
  changes, and add only missing new information.
- Do not intentionally leave one file updated and the other stale. If one file
  cannot be updated, stop and report the synchronization blocker.
- Record promoted, rejected, pending, and conditional candidates.
- Include newly observed effects from metrics, logs, and diagnostics.
- Keep claims tied to run IDs and `episodes.jsonl` evidence.
- Update `Last updated` whenever this file changes.
</track_record_maintenance>

<claude_prompting_standards>
Follow Anthropic-style prompt engineering for Claude:
- Keep instructions clear, direct, and specific.
- Use meaningful XML tags to separate role, context, constraints, workflow,
  metrics, and outputs.
- Keep this file concise enough to be useful as always-loaded project context.
- Include common commands, style rules, testing instructions, repository
  etiquette, and project-specific warnings.
- Prefer explicit constraints and success criteria over vague guidance.
- Do not expose hidden reasoning. Provide concise decisions, evidence, and next
  steps.
</claude_prompting_standards>
