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

Last updated: 2026-05-15

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

- `C0` and `C1` are implemented: diagnostics plus geometric observations
  without changing vehicle observation dimensionality.
- `D2` reward shaping is the current useful trunk among tested reward changes.
- `D3` early vehicle-stuck termination failed the gate and was reverted.
- Current path curriculum configuration uses `difficulty=path` with route
  distances `15m / 35m / 60m` for both vehicles and pedestrians.
- Current curriculum budget proposal is `easy=0.30`, `medium=0.32`,
  `hard=0.38`.
- Current sampling weights are base `easy=1.00`, `medium=1.07`, `hard=1.27`
  and probation `medium=1.00`, `hard=1.19`.
- `carla_mappo_20260514_211642` was an easy-only locked exploratory run. It
  tested `path` easy `15m/15m`, but it did not test budget constraints or
  sampling weights because `--lock-curriculum-level easy` disables those.

## Current Accepted Trunk

- Active trunk: `C0 + C1 + D2`.
- Reverted: `D3`.
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
