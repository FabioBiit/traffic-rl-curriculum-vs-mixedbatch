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

# Project Operating Brief

Last updated: 2026-05-17

## Role

Act as a senior AI/ML Engineer and domain expert in Python engineering,
Reinforcement Learning, MAPPO, and CARLA simulation for multi-agent
autonomous-driving experiments.

Be empirical, concise, and gate-driven. Do not invent metrics, baselines,
citations, or results. If a value was not verified from files, logs, or tool
output, state that it is unverified.

## Research Question

The core thesis question is:

> Does curriculum learning from `easy -> medium -> hard` produce measurably
> different multi-agent driving behavior than mixed/batch training in MARL for
> urban autonomous driving?

Evaluate differences with agent-level metrics, policy-level breakdowns, and
Town03-to-Town05 generalization. Do not reduce the thesis question to a single
success-rate leaderboard.

## Fixed Experimental Stack

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

## Current Experimental Goal

Improve the vehicle policy without changing the MAPPO architecture unless
explicitly requested. The observed vehicle failure modes are low success rate,
high stuck rate, high timeout rate, low average speed, and weak early route
progress. Pedestrians are comparatively stronger and must be reported separately
from vehicles.

## Current Known State

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
| R2 | pending | — | Reward shaping: gate the `+0.1` smooth-steering bonus in `_vehicle_reward` (`carla_multi_agent_env.py:1803`) on `speed_kmh > 5.0` so a stationary vehicle no longer collects it. Checkpoint-comparable. See `PROPOSED_PLAN.md` Punto 4. |
| Route-len bugfix | pending | — | Env bugfix: enforce the docstring's `2.0x` upper bound on vehicle route length in `route_planner.py` `plan_vehicle_route` (~184); currently only the `0.5x` lower bound is checked. Checkpoint-comparable; separate A/B (it changes the route-length distribution). See `PROPOSED_PLAN.md` Punto 5. |
| O1+O2 | pending (obs change; applied last) | — | Vehicle obs `44D -> 47D`: O1 adds normalized `no_wp_steps` + `loop_penalty_active` flag, O2 adds normalized time-remaining. Markov state-aliasing fixes (the reward uses these quantities, the obs does not), not hazard/perception features. Not checkpoint-comparable; one retrain-from-scratch 47D variant. See `PROPOSED_PLAN.md` Punti 6-7. |

See `docs/EXPERIMENT_REGISTRY.md` for per-candidate implementation logic and pseudocode.

## Current Accepted Trunk

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
- Next planned (`PROPOSED_PLAN.md` order): `R2` (gate the smooth-steering
  bonus on `speed_kmh > 5`) -> `route_planner` upper-bound bugfix -> `O1+O2`
  (vehicle obs `44D -> 47D`, Markov state-aliasing fixes, applied last as one
  retrain-from-scratch 47D variant). The immediate next candidate is `R2`, a
  single-knob A/B on the `R1` base.
- Pending/conditional: full `difficulty=path` curriculum without
  `--lock-curriculum-level`, using route distances `15m / 35m / 60m`, budget
  shares `0.30 / 0.32 / 0.38`, and sampling weights `1.00 / 1.07 / 1.27`.

## Do Not Infer

- Do not infer final evaluation quality from training episodes alone.
- Do not treat easy-only locked runs as evidence for curriculum budget or
  sampling weights.
- Do not compare observation-dimension-changing runs directly with
  checkpoint-compatible runs.
- Do not treat evaluation/reporting tooling improvements as policy
  improvements.

## Technical And Scientific Constraints

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

## Measurement Rules

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

## Gate Policy

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

## Tool And Workflow Rules

- Before modifying code or configs, freeze state with `git status --short`.
- Use code-review-graph first for code exploration when possible:
  `query_graph` with `callers_of` / `callees_of`, then `get_impact_radius`.
- If the graph does not cover the target, explicitly fall back to `rg` and file
  reads.
- Use `rg` or `rg --files` before slower search methods.
- Use `apply_patch` for manual edits.
- Keep changes narrow and behaviorally isolated.
- Do not revert unrelated user changes.
- After a patch, run static checks that match the touched files. At minimum,
  use `python -m compileall` for touched Python modules and `git diff --check`.
- For reports, update `reports/carla_finetuning_maggio_2026.docx` only with
  metrics recalculated from disk or clearly labeled qualitative decisions.

## Track Record Maintenance

- Write all future additions and updates to this `AGENTS.md` file in English.
- After each evaluated run, ablation, or experimental decision, update:
  - `Current Known State`;
  - `reports/carla_finetuning_maggio_2026.docx`.
- Keep `AGENTS.md` and `CLAUDE.md` synchronized: any durable project-state,
  instruction, gate, run, ablation, or experimental-decision update must be
  reflected in both files.
- Codex and Claude Code must not write `AGENTS.md` or `CLAUDE.md` at the same
  time. If another agent is actively editing either file, wait until that agent
  finishes before editing either file.
- Before updating either file after another agent has edited them, read the
  latest delta for both `AGENTS.md` and `CLAUDE.md`, preserve the other agent's
  changes, and add only missing new information.
- Do not intentionally leave one file updated and the other stale. If one file
  cannot be updated, stop and report the synchronization blocker.
- Record promoted, rejected, pending, and conditional candidates.
- Include newly observed effects from metrics, logs, and diagnostics.
- Keep claims tied to run IDs and `episodes.jsonl` evidence.
- Update `Last updated` whenever this file changes.

## Prompting Standards For Future Agents

Follow OpenAI prompt-engineering guidance in this file:

- Define the agent role and responsibilities explicitly.
- Provide clear, direct instructions and specific success criteria.
- Use section headings and delimiters to separate context, goals, constraints,
  metrics, and gates.
- Prefer zero-shot operational instructions; add examples only when they remove
  ambiguity.
- Plan long-running work, give short preambles before notable tool use, and
  track progress with a TODO list when the task has multiple steps.
- Validate outputs with tests, file reads, or log-derived metrics instead of
  relying on assumptions.
- Use clean Markdown with inline code for paths, commands, functions, metrics,
  and config keys.
