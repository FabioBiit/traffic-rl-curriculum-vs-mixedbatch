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
Last updated: 2026-05-20
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

Evaluate through agent-level metrics, policy-level breakdowns, and
Town03-to-Town05 generalization. Do not reduce the thesis question to a single
success-rate leaderboard.
</research_question>

<fixed_stack>
- Simulator: `CARLA 0.9.16`.
- Algorithm: `MAPPO` with CTDE.
- Framework: `Ray/RLlib 2.10.0`.
- Runtime: `Python 3.11.9`, `PyTorch 2.7+cu126`.
- Setup: `3` RL vehicles and `3` RL pedestrians.
- Train map: `Town03`. Test map: `Town05`.
- Default architecture: centralized critic, separate vehicle and pedestrian
  policies, PopArt off, attention off, GNN off.
- Do not use MetaDrive assumptions, files, or conclusions for CARLA decisions.
</fixed_stack>

<current_goal>
Improve the vehicle policy without changing the MAPPO architecture unless
explicitly requested. Vehicle failure modes: low success rate, high stuck rate,
high timeout rate, low average speed, weak early route progress. Pedestrians
are comparatively stronger and must always be reported separately.
</current_goal>

<experimental_state>

## Active Configuration

- **Trunk**: `C0 + C1 + D2 + R1`. Reverted: `D3`.
- **Vehicle obs**: `47D` (O1+O2 applied); `global_obs_dim=225`. Not checkpoint-compatible with pre-O1+O2 (44D) runs.
- **Pedestrian obs**: `26D`.

**Non-reverted retained knobs** (failed gate; kept by user decision):

| Knob | Current value | Reason retained |
|------|--------------|----------------|
| `vf_clip_param` | `1e6` (H1) | Reverting restores a non-functional critic (`vf_explained_var` was ~0) |
| `vf_loss_coeff` | `0.05` (H1.1) | Same |
| `gamma` | `0.997` (H2) | User decision; H2 hypothesis falsified |
| `entropy_coeff` | schedule `[[0,0.03],[250000,0.005]]` (H3) | User decision; mechanism confirmed |
| Collision penalty | `-500.0` at `carla_multi_agent_env.py:1748` (R3) | User decision; hypothesis falsified |

**Key analytical findings:**
- H1+H2+H3 together: three optimizer-side single-knob changes, all mechanism-confirmed, all failing
  the gate. Vehicle SR pinned ~21.5% on the pre-bugfix distribution; binding constraint was the
  vehicle collision rate (~25.8%).
- R3 finding: collision avoidance is not learnable from 44D obs (perception limit, not a
  reward-weight problem). The 10× penalty left collision flat; policy responded on offroad/timeout
  axes only.
- R2 provisional (not promoted): vs R1 baseline `20260517_134652`, SR −1.85 pp, collision +1.43 pp,
  offroad +3.15 pp. Speed improved (+2.21 km/h) but extra mobility converted into offroad/collision.
  Do not stack new candidates on R2 as if it passed the gate.

## Current Baselines

- **Post-bugfix vehicle baseline**: run `20260517_212109` (route-len bugfix, commit `24e072e`).
  Vehicle SR 75.1% training-only — task-distribution change, NOT policy. All pre-bugfix H/R-series
  absolute SR (~21–27%) is structurally depressed; do not compare across the bugfix.
- **47D vehicle baseline**: run `20260518_195947` (O1+O2 + route-seed fix; easy-locked; HEALTHY;
  vehicle SR plateau ~79%; `vf_explained_var` 0.983). **Open item**: 16.4%
  `route_source=legacy_fallback` on vehicle routes — diagnose before the long run.
- **R1 re-check caveat**: R1 removed the `route_completion < 0.3` guard, which interacts with route
  length; its single-knob verdict is most exposed to re-check on the corrected route distribution.

## Curriculum Configuration (`curriculum_batch.yaml`, updated 2026-05-18)

- `difficulty=path`; distances: **`easy=30m / medium=60m / hard=100m`** (source of truth:
  `carla_core/configs/levels.yaml::levels_path`; verified against `run_config.json` of in-flight
  run `20260520_133747`). Previously documented as `15/35/60` — corrected on 2026-05-20 (doc-config
  drift, no code change). A move to `15/30/60` punto fisso, or a truncated-range "forbice"
  `[L,U]` sampler, is on the table for the next iteration but **not** in the current run.
- Budget shares: `easy=0.30 / medium=0.35 / hard=0.35`.
- Base sampling weights: `1.00 / 1.17 / 1.17`; medium `min_budget_share=0.20`.
- Probation weights: `medium=1.00 / hard=0.85`.
- Unlock metric: balanced **windowed** policy SR (`window_full` required; not cumulative).
  Rationale: cumulative SR is a lagging integrator dragged by cold-start failures
  (pilot `20260518_195947`: 86% windowed vs 67% cumulative). Reporting remains cumulative.
  First full unlock-path run in flight: `20260520_133747` (300k-step budget).

## Next Step

Launch full `difficulty=path` curriculum long run (no `--lock-curriculum-level`) on 47D obs with
the updated `curriculum_batch.yaml`. **Prerequisite**: diagnose the 16.4% `legacy_fallback` vehicle
routes seen in `20260518_195947`.

**Candidate queue (ordered, aligned with `docs/plans/PROPOSED_PLAN.md`)**:
`Entropy (passed) → Ped-route → Ped-speed → P1 → P2 → P3 → R-norm v2`.
Rationale: Entropy verdict closed; pedestrian-side fixes next (Ped-route then Ped-speed);
curriculum-side stack (P1→P2→P3) follows; R-norm v2 is last and **gated on Block 4 evidence**
(skip if P1+P2+P3+Ped-route flatten the easy→hard collision gradient).
File-overlap ordering enforced: P1/P2 share `curriculum_batch_manager.py`; R-norm / Ped-route /
Ped-speed share `carla_multi_agent_env.py` — run sequentially, never in parallel.

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

See `docs/EXPERIMENT_REGISTRY.md` for per-candidate implementation logic and pseudocode.
</experimental_state>

<gate_policy>
Default gate for vehicle-focused changes:

- Vehicle success rate: minimum `+2.0 pp`.
- Vehicle `stuck + timeout`: minimum `-2.0 pp`.
- Collision and offroad rates: must not worsen by more than `+1.0 pp` each.
- No NaN or inf in observations, rewards, global observations, or recorded metrics.
- Episode integrity: expected `6` agent-level records per episode.

If a candidate fails the gate, revert only that candidate and keep the accepted trunk intact.
</gate_policy>

<measurement_rules>
- Primary success: `termination_reason == "route_complete"`.
- Primary aggregation: cumulative agent-level metrics. Do not use joint success rate unless explicitly asked.
- Every episode: `6` agent-level records. Deduplicate by `episode_id + agent_id`, keep last.
- Always report: `vehicles+pedestrians`, `vehicles`, and `pedestrians` separately.
- Key metrics: success rate, stuck rate, timeout rate, stuck+timeout, collision rate, offroad rate,
  route completion, path efficiency, speed, and no-waypoint steps when available.
- If `episodes.jsonl` is live or recently updated, recalculate from disk.
</measurement_rules>

<constraints>
**Do not infer:**
- Final evaluation quality from training episodes alone.
- Curriculum budget or sampling-weight evidence from easy-only locked runs.
- Comparisons between 44D and 47D runs (not checkpoint-compatible).
- Policy improvements from evaluation/reporting tooling changes.

**Technical and scientific:**
- Use established Python engineering: narrow diffs, explicit control flow, deterministic config
  handling, structured parsers, verification with compile checks.
- Map every proposed technique to this fixed setup: CARLA 0.9.16, MAPPO CTDE, 3 vehicles,
  3 pedestrians, Town03/Town05. State: source category, assumption transferred, expected measurable
  effect, files affected, gate.
- Do not introduce PopArt, attention, GNNs, new policies, or obs-space changes unless explicitly
  requested.
- Prefer: official docs for simulator/API → local repo code for implementation →
  `episodes.jsonl` for empirical truth.
- Never present a method as "state of the art" without a concrete source.
- Distinguish change types: bug fix / reward shaping / obs change / curriculum / architecture /
  eval tooling.
- Local empirical evidence has priority. A change that fails the gate is not promoted.
</constraints>

<workflow>
1. Freeze state before edits: `git status --short`.
2. Use graph tools first (MCP block above). Fall back to `rg` only when graph coverage is insufficient.
3. Keep changes narrow and behaviorally isolated. Do not revert unrelated user changes.
4. After a patch: `python -m compileall <touched files>` + `git diff --check`.
5. Update `reports/carla_finetuning_maggio_2026.docx` only with metrics recalculated from disk
   or clearly labeled qualitative decisions.
6. Multi-step work: give concise preamble before notable tool use; track progress with a TODO list.
</workflow>

<common_commands>
```bash
# Repo state
git status --short
git diff --check

# Compile touched CARLA/MAPPO modules
python -m compileall carla_core\envs\carla_multi_agent_env.py carla_core\training\curriculum_batch_manager.py carla_core\training\train_carla_mappo.py

# Candidate full path-curriculum run (user launches; never run autonomously)
# Current operating budget is 300k timesteps (in-flight reference: 20260520_133747); 3M is reserved for the final long run.
python -m carla_core.training.train_carla_mappo --mode curriculum --difficulty path --timesteps 300000 --seed 999
```
</common_commands>

<track_record_maintenance>
- Language: English only for all additions and updates.
- After each evaluated run or experimental decision, update:
  - `<experimental_state>` in this file.
  - `reports/carla_finetuning_maggio_2026.docx`.
  - `AGENTS.md` (keep both files synchronized).
- Synchronization rule: Codex and Claude Code must not write `CLAUDE.md` and `AGENTS.md`
  simultaneously. Before editing either file, read the latest delta for both; add only missing new
  information; preserve the other agent's changes. If one file cannot be updated, stop and report
  the synchronization blocker.
- Claims must be tied to run IDs and `episodes.jsonl` evidence.
- Update `Last updated` in `<metadata>` whenever this file changes.
</track_record_maintenance>
