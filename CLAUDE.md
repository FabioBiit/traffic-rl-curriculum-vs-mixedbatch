# CLAUDE.md

<metadata>
Last updated: 2026-05-15
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
</current_known_state>

<current_accepted_trunk>
- Active trunk: `C0 + C1 + D2`.
- Reverted: `D3`.
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
- Update both `CLAUDE.md` and `AGENTS.md` if and only if there is no active
  Codex session.
- If there is an active Codex session, Claude Code must update only
  `CLAUDE.md` and must not edit `AGENTS.md`.
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
