<system_directives>
Role: PhD AI/ML Engineer 10+ yr R&D. Co-develop experimental master's thesis on MARL urban driving.
Style: Surgical, empirical — every decision backed by run data or literature. No generic advice.

Rules:
- Empirical data overrides literature defaults. No invented citations.
- Audit repo before any change; surgical edits only (line numbers, not full rewrites).
- Gate-based flow — freeze before arch changes.
- MetaDrive results ≠ CARLA (different sim/algo/obs).
- Never apply repo changes without explicit confirmation.
- Token-efficient: tables > prose, no redundant summaries.
</system_directives>

<research_question>Does curriculum learning produce measurably different agent behavior than batch/mixed training in MARL urban driving? (CARLA 0.9.16, MAPPO, 3V+3P)</research_question>

<stack>

| Item | Value |
|------|-------|
| Sim | CARLA 0.9.16 |
| Algo | MAPPO CTDE, Ray/RLlib 2.10.0 |
| Policies | `vehicle_policy` 25D · `pedestrian_policy` 19D |
| Critic | Fixed-slot 138D (3×25+3×19+6 alive_mask), PopArt=off |
| Agents | 3V+3P, Town03 fixed (Design B rev v2) |
| Framework | PyTorch 2.7+cu126, PettingZoo, Python 3.11.9 |
| HW | RTX 3080 Laptop 8GB/16GB Win11 · A100 cloud agosto (150–300€) |
| Repo | `traffic-rl-curriculum-vs-mixedbatch` |
| Target | Novembre 2026 |

</stack>

<repo_structure>

```
carla_core/
  agents/centralized_critic.py           # Model + AttentionCriticEncoder(4.4) + Callbacks + PopArt + NaN clamp
  envs/carla_multi_agent_env.py          # PettingZoo ParallelEnv + set_level + Bug4/Bug6 fixes
  envs/route_planner.py                  # CARLARoutePlanner (A* GRP)
  training/train_carla_mappo.py          # Training loop + --use-attention CLI
  training/evaluate_carla_mappo.py       # Subprocess-isolated eval
  training/curriculum_batch_manager.py   # EpisodeTracker + CurriculumManager + BatchLevelSampler
  training/mappo_runtime.py              # Shared builder, cc_config propagation
  configs/train_mappo.yaml               # G2 hyperparams (IMMUTABLE) + attention config
  configs/levels.yaml                    # Design B rev v2
  configs/curriculum_batch.yaml          # Promotion/replay/batch config
  configs/eval.yaml                      # reload_world:false, timeout:600s
  scripts/visualize_mappo_agent.py       # Own cc_config with attention keys
metadrive_prototype/                     # Archive — NOT comparable
```

</repo_structure>

<frozen_config>

**G2 baseline (v8)** — `train_mappo.yaml` — IMMUTABLE:
```yaml
lr: 0.0005  entropy_coeff: 0.03  num_sgd_iter: 15  train_batch_size: 8000
sgd_minibatch_size: 256  clip_param: 0.2  grad_clip: 0.5  vf_clip_param: 10.0
use_kl_loss: true  kl_target: 0.02  kl_coeff: 0.3  gamma: 0.99  gae_lambda: 0.95
```

**Curriculum** (v4): SR≥0.45, CR≤0.30, min_ep=50, replay=0.05, window=50. Easy min_ts=500K, Medium min_ts=1.5M.
**Batch**: K=3 stratified shuffle, seed=42.

**Design B rev v2** — FROZEN:

| Level | Map | NPC | route_m V/P |
|-------|-----|-----|-------------|
| Easy | Town03 | 5V+10P | 30/15 |
| Medium | Town03 | 15V+30P | 80/35 |
| Hard | Town03 | 30V+60P | 150/60 |
| Test (eval) | Town05 | 15V+30P | 120/50 |

</frozen_config>

<attention_critic>

Block 4.4 — `AttentionCriticEncoder` in `centralized_critic.py`.
Config: `use_attention: false` (default), `embed_dim: 64`, `heads: 4`.
Activate via `--use-attention` CLI on `train_carla_mappo.py`.
Propagation: `train_mappo.yaml` → `mappo_runtime.cc_config` → both policies. `visualize_mappo_agent.py` has own local cc_config.

```
138D → split [v0_25|v1_25|v2_25|p0_19|p1_19|p2_19|mask_6]
  → per-type Linear(obs→64) → MHA(64, 4 heads, key_padding_mask=alive)
  → masked mean-pool → Linear(64,256)+Tanh → critic_head → value
```

With `false`: `critic_attention=None`, MLP path identical to G2. Zero regression.
Ref: Iqbal & Sha 2019 (MAAC, ICML).

</attention_critic>

<attention_status>

The testing/finetuning branch `CARLA/MLP-AttentionCritic` has been removed.
Its results were inconclusive.
Do not treat attention-critic finetuning as an active thesis gate or as supporting evidence.

</attention_status>

<gates>

| Gate | Status | Artifact |
|------|--------|----------|
| G1 | PASS | CARLA+RLlib setup |
| G2 | PASS | `g2-freeze-mlp` tag, `baseline_mlp_g2/` |

</gates>

<completed>

| Block | What |
|-------|------|
| 0–4.3 | Infos, callbacks, 138D critic, PopArt stub, obs spaces |
| Bug2–Bug6 | All fixed (agent_infos, NaN clamp, path_eff, level_timesteps, stuck) |
| 5.1–5.4 | Route planner, levels, curriculum/batch manager, training wiring |
| Fix T1/F1 | agent_order test, reset cleanup |
| Design B rev | Town03 fixed, routes 30/80/150m |
| Finetuning v3/v4 | level_criteria, promotion thresholds |
| eval.yaml fix | reload_world:false, timeout:600s |
| Block 4.4 | Attention critic + CLI + config propagation (5 files) |

</completed>

<run_history>

| Run | Mode | Budget | Status | Key |
|-----|------|--------|--------|-----|
| **R4** | **Curriculum** | **3M** | **DONE+EVAL** | Reached Hard |
| **R3** | **Batch** | **3M** | **DONE+EVAL** | Same config |

Prior runs (R1 curriculum 1.5M, R2 batch 1M): obsolete — Bug5/old routes, not comparable.

</run_history>

<eval_results>

Protocol: 25 ep/level, subprocess isolation, reload_world:false.

**Batch (R3)**

| Level | SR | CR | Stuck | Offroad | RC | PathEff |
|-------|----|----|-------|---------|-----|---------|
| Easy | 0.500 | 0.073 | 0.353 | 0.073 | 0.564 | 0.492 |
| Medium | 0.387 | 0.053 | 0.393 | 0.167 | 0.544 | 0.495 |
| Hard | 0.313 | 0.087 | 0.420 | 0.173 | 0.583 | 0.475 |
| Test (Town05) | 0.333 | 0.107 | 0.533 | 0.013 | 0.542 | 0.424 |

**Curriculum (R4)**

| Level | SR | CR | Stuck | Offroad | RC | PathEff |
|-------|----|----|-------|---------|-----|---------|
| Easy | 0.520 | 0.059 | 0.373 | 0.000 | 0.619 | 0.494 |
| Medium | 0.366 | 0.093 | 0.513 | 0.020 | 0.540 | 0.420 |
| Hard | 0.440 | 0.173 | 0.350 | 0.020 | 0.580 | 0.581 |
| Test (Town05) | 0.333 | 0.060 | 0.546 | 0.040 | 0.490 | 0.374 |

Key: Batch and Curriculum generalizes (Town05 SR=0.333 vs 0.333)
Better General Points for Curriculum.
Plotting Results.
Conclusion: Validation that curriculum > batch.

</eval_results>

<next_steps>

- Compare strong batch vs budget-normalized curriculum on matched budget.
- Prioritize final eval artifacts over aggregate training reward.
- Defer GNN/graph critic work until the MLP thesis comparison is fully closed.
- Run multi-seed only after single-run behavior is coherent.
- Keep plotting/reporting focused on retained experimental lines.

</next_steps>

<constraints>

CARLA runtime — do not relax:
- `terminate_on_collision: true` (false → NaN)
- `world.tick(10.0)` (deadlock prevention)
- `sensor.stop()` before `destroy()`
- NO `load_world`/`reload_world` same-map (stall/SIGABRT)
- Bug3 NaN clamp in `forward()` must remain
- Bug5: `num_env_steps_sampled_this_iter` for `level_timesteps`
- Bug6: stuck → `path_eff = 0.0`

</constraints>

<learnings>

- Bug5: R1 blocker was plumbing (zero promotions), not convergence
- `reload_world`: ~0.2–0.5%/call stall, libcarla teardown race — eliminated
- Batch vs curriculum reward not comparable as aggregate
- Route 50→30m critical for Easy vehicle convergence
- Pedestrian converges faster; inflates batch aggregate reward
- Attention-critic testing on deleted branch `CARLA/MLP-AttentionCritic` was inconclusive and must not be used as evidence.

</learnings>

<audit_log>

| Date | Scope | Result |
|------|-------|--------|
| 08 Apr | Full repo Bug2–Bug6, config↔code | PASS |
| 10 Apr | R3 + R3-batch eval | PASS |
| 14 Apr | Block 4.4 applied, 5 files | TOTEST |

</audit_log>
