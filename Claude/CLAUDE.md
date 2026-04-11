<system_directives>
Role: PhD AI/ML Engineer 10+ yr R&D. Co-develop experimental master's thesis on MARL urban driving.
Style: Surgical, empirical — every decision backed by run data or literature. No generic advice.

Rules (non-negotiable):
- Empirical data overrides literature defaults.
- No invented citations.
- Audit repo state before any change; surgical edits only (line numbers, not full rewrites).
- Gate-based flow — freeze before arch changes; no fragmented diffs across messages.
- MetaDrive results ≠ CARLA results (different sim/algo/obs space).
- Never apply changes to repo without explicit confirmation. Always ask before generating files.
- Token-efficient output: bullets/tables preferred, no redundant summaries.
</system_directives>

<research_question>
Does curriculum learning produce measurably different agent behavior than batch/mixed training in MARL urban driving? (CARLA 0.9.16, MAPPO, 3V+3P)
</research_question>

---

<stack>

| Item | Value |
|------|-------|
| Sim | CARLA 0.9.16 |
| Algo | MAPPO CTDE, Ray/RLlib 2.10.0 |
| Policies | `vehicle_policy` 25D · `pedestrian_policy` 19D |
| Critic | Fixed-slot 138D (3×25 + 3×19 + 6 alive_mask), PopArt=off |
| Agents | 3V + 3P, fixed map Town03 (Design B rev v2) |
| Framework | PyTorch 2.7+cu126, PettingZoo, Python 3.11.9 |
| HW local | RTX 3080 Laptop 8 GB VRAM / 16 GB RAM, Windows 11 |
| HW cloud | A100 agosto, Vast.ai/RunPod, budget 150–300 € |
| Repo | `traffic-rl-curriculum-vs-mixedbatch` (GitHub) |
| Target | Laurea novembre 2026 |

</stack>

---

<repo_structure>

```
carla_core/
  agents/centralized_critic.py           # CentralizedCriticModel + Callbacks + PopArt + NaN clamp (Bug3)
  envs/carla_multi_agent_env.py          # PettingZoo ParallelEnv 3V+3P + set_level + F1 + Bug4 + Bug6
  envs/route_planner.py                  # Block 5.1: CARLARoutePlanner (A* GRP)
  training/train_carla_mappo.py          # Training loop + level wiring (5.4) + Bug5 + level_criteria
  training/evaluate_carla_mappo.py       # Final eval via subprocess isolation per episode
  training/curriculum_batch_manager.py   # Block 5.3: EpisodeTracker + CurriculumManager + BatchLevelSampler
  training/mappo_runtime.py              # Shared builder (train + eval)
  configs/train_mappo.yaml               # G2 baseline hyperparams (v8) — DO NOT MODIFY
  configs/multi_agent.yaml
  configs/levels.yaml                    # Design B rev v2: route-reduced (30/80/150m), Town03 fixed, test=Town05
  configs/curriculum_batch.yaml          # promotion/replay/batch config, level_criteria per-level
  configs/eval.yaml                      # reload_world: false, subprocess_timeout: 600s (fix applied)
  scripts/{test_mappo_pipeline.py,compare_results_carla.py}
metadrive_prototype/                     # Archive — results NOT comparable with CARLA runs
```

</repo_structure>

---

<hyperparams>

**G2 baseline (v8)** — `train_mappo.yaml` — IMMUTABLE (frozen at tag `g2-freeze-mlp`):

```yaml
lr: 0.0005        entropy_coeff: 0.03   num_sgd_iter: 15    train_batch_size: 8000
sgd_minibatch_size: 256   clip_param: 0.2   grad_clip: 0.5   vf_clip_param: 10.0
use_kl_loss: true   kl_target: 0.02   kl_coeff: 0.3   gamma: 0.99   gae_lambda: 0.95
```

**Curriculum config** (v4, `curriculum_batch.yaml`):

```yaml
promotion_threshold: 0.45   collision_threshold: 0.30   min_episodes: 50
replay_ratio: 0.05           window_size: 50
level_criteria:
  easy:   min_timesteps: 500000
  medium: min_timesteps: 1500000
```

**Batch config**: K=3 stratified shuffle, seed=42.

</hyperparams>

---

<design_freeze>

**Design B revised v2** — FROZEN:

| Level | Map | NPC | route_m (V) | route_m (P) |
|-------|-----|-----|-------------|-------------|
| Easy | Town03 | 5V+10P | 30 | 15 |
| Medium | Town03 | 15V+30P | 80 | 35 |
| Hard | Town03 | 30V+60P | 150 | 60 |
| Test (eval only) | Town05 | 15V+30P | 120 | 50 |

Note: test map = **Town05** (not Town02/Town10 — confirmed in `levels.yaml` and eval JSONs).

**Attention critic**: secondary contribution (MLP G2 = anchor). **GNN**: optional Phase 3 agosto.

</design_freeze>

---

<gates>

| Gate | Status | Artifact |
|------|--------|----------|
| G1 | PASS | CARLA install, smoke test, RLlib setup |
| G2 | PASS | Tag: `g2-freeze-mlp`, checkpoint `baseline_mlp_g2/` |

</gates>

---

<completed>

| Block | What |
|-------|------|
| 0–4.3 | Docstrings, infos side-channel, callbacks, fixed-slot 138D, PopArt stub, V=25D, P=19D |
| Bug2 | `_terminated_agent_infos` fix — route completion underreporting |
| G2 | Run11 1M steps PASS, checkpoint frozen |
| 5.1–5.4 | Route planner, levels, curriculum/batch manager, training wiring |
| Fix T1/F1 | Test agent_order, reset cleanup traffic |
| Bug3 | NaN clamp on logits+value in `forward()` |
| Bug4 | `reset()` path_eff fields init order |
| Bug5 | `level_timesteps=0` root cause — zero promotions in R1 1.5M |
| Bug6 | `path_efficiency` stuck=1.0 → `if stuck: path_eff = 0.0` |
| Design B rev | Town03 fixed, route-reduced (30/80/150m) |
| Finetuning v3/v4 | `level_criteria` per-level, promotion SR≥0.45 / CR≤0.30 |
| eval.yaml fix | `reload_world: false`, `subprocess_timeout: 600s` |
| **R3** | Curriculum 3M — COMPLETED + EVALUATED (reached Hard) |
| **R3-batch** | Batch 3M — COMPLETED + EVALUATED |

</completed>

---

<run_history>

| Run | Mode | Budget | Status | Notes |
|-----|------|--------|--------|-------|
| R1 | Curriculum | 1.5M | Done | Bug5 blocked promotion; old routes — NOT comparable |
| R2 | Batch | 1M | Done | Old routes — NOT comparable with R3+ |
| **R3** | **Curriculum** | **3M** | **COMPLETED + EVALUATED** | Bug5 fixed, routes reduced; reached Hard |
| **R3-batch** | **Batch** | **3M** | **COMPLETED + EVALUATED** | Same config as R3 |

</run_history>

---

<eval_results>

Eval protocol: 25 ep/level, subprocess isolation, `reload_world: false`, `subprocess_timeout: 600s`.
Eval output dir: `carla_core/results/eval/`.

**R3-batch — Batch (eval 20260410_095446)**

| Level | SR | CR | Stuck | Offroad | RC | PathEff |
|-------|----|----|-------|---------|-----|---------|
| Easy (Town03) | **0.500** | 0.073 | 0.353 | 0.073 | 0.564 | 0.492 |
| Medium (Town03) | **0.387** | 0.053 | 0.393 | 0.167 | 0.544 | 0.495 |
| Hard (Town03) | **0.313** | 0.087 | 0.420 | 0.173 | 0.583 | 0.475 |
| Test (Town05) | **0.333** | 0.107 | 0.533 | 0.013 | 0.542 | 0.424 |

**R3 — Curriculum (eval 20260408_130010)**

| Level | SR | CR | Stuck | Offroad | RC | PathEff |
|-------|----|----|-------|---------|-----|---------|
| Easy (Town03) | **0.413** | 0.080 | 0.420 | 0.000 | 0.600 | 0.489 |
| Medium (Town03) | **0.340** | 0.000 | 0.627 | 0.013 | 0.515 | 0.327 |
| Hard (Town03) | **0.360** | 0.020 | 0.533 | 0.067 | 0.517 | 0.418 |
| Test (Town05) | **0.007** | 0.080 | 0.820 | 0.053 | 0.424 | 0.125 |

**Key observation**: Curriculum generalizes to unseen map (Town05 SR=0.333 vs Batch SR=0.007). Batch has lower CR on Medium/Hard (0.0 vs 0.053/0.087) but catastrophic test generalization failure. Aggregate reward not comparable (cross-level distribution differs).

**Eval log**:

| # | Type | Budget | Mode | Result |
|---|------|--------|------|--------|
| 1–2 | Smoke | 36K | Both | PASS |
| 3–4 | Test | 102K | Both | PASS |
| 5 | R3 curriculum | 102K | Curriculum | PASS (stall ep9 fixed: non-det `reload_world` race condition) |
| 6 | R3-batch | 102K | Batch | PASS |

</eval_results>

---

<next_steps>

**Priority 1 — Block 4.4: Attention Critic** (CURRENT NEXT)
- Inputs: G2 MLP checkpoint `baseline_mlp_g2/`, fixed-slot 138D critic, centralized_critic.py
- Deliverable: attention-based critic replacing MLP critic; same 138D input contract
- Gate: train short pilot run, verify no NaN, compare V-loss vs MLP G2 baseline
- Dependency: none (G2 ✓)

**Priority 2 — R4/R5: Attention Runs**
- Curriculum + Batch with attention critic; same 3M budget as R3/R3-batch
- Dep: Block 4.4 ✓

**Priority 3 — Block 5.0: GNN encoder**
- Dep: R4/R5 ✓

**Priority 4 — R6/R7: GNN encoder Runs**
- Curriculum + Batch with GNN encoder; same 3M budget as R3/R3-batch
- Dep: Block 5.0 ✓

**Pending (non-blocking)**:
- Dockerfile (`dock`)
- `compare_results_carla.py` — full R3 vs R3-batch comparison plots

</next_steps>

---

<carla_constraints>

Critical — do not remove or relax:
- `terminate_on_collision: true` — false → NaN in value estimates
- `world.tick(10.0)` — timeout arg prevents sync deadlock
- `sensor.stop()` before `sensor.destroy()`
- **NO `load_world` / `reload_world` same-map** — non-deterministic stall/SIGABRT (CARLA 0.9.16 Windows). `env.reset()` + F1 handles cleanup.
- `_switch_map()` for cross-map eval only, under subprocess isolation
- Bug3: NaN clamp on logits+value MUST remain in `forward()` — distribution shift from level transitions
- Bug5: use `num_env_steps_sampled_this_iter` (not fallback) for `level_timesteps`
- Bug6: `if termination_reason == "stuck": path_eff = 0.0` — post-termination, pre-agent_info

</carla_constraints>

---

<learnings>

- **Bug5**: dominant R1 blocker was plumbing (zero promotions), not convergence failure
- **Bug6**: degenerate path_eff=1.0 for low-displacement stuck agents → forced 0.0
- **`reload_world` stall**: non-deterministic (~0.2–0.5% per call). 4 test runs (408 calls) passed; stalled at call ~9 of 5th run. Root cause: libcarla C++ actor teardown race. Fix: eliminate the call.
- **Batch vs curriculum reward**: not comparable as aggregate — different level sampling distributions
- **Route reduction**: 50m→30m critical for Easy vehicle convergence
- **Pedestrian converges faster**: simpler dynamics; Easy pedestrian completions inflate batch aggregate reward
- **Test generalization gap**: Curriculum SR_test=0.333 vs Batch SR_test=0.007 — largest behavioral difference observed between training modes

</learnings>

---

<audit_log>

| Date | Scope | Result |
|------|-------|--------|
| 08 Apr | Full repo: Bug2–Bug6, F1, T1, config↔code | ALL PASS |
| 09 Apr | Eval stall ep9/102 | Root cause: non-det `reload_world` race condition |
| 10 Apr | R3 + R3-batch eval completed | Both PASS; eval JSONs in `carla_core/results/eval/` |
| 11 Apr | CLAUDE.md update | Fixed Test map (Town05), replay_ratio (0.05), added eval metrics |

</audit_log>
