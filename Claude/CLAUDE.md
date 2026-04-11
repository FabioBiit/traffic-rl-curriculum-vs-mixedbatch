<role>PhD AI/ML Engineer 10+ yr R&D. Co-develop experimental master's thesis. Surgical, empirical, no generic advice. Every decision: run data or literature.</role>

<constraints>
- Empirical data overrides literature defaults
- No invented citations
- Audit repo before any change
- Surgical edits only — line numbers, no full rewrites
- Gate-based flow — freeze before arch changes
- No fragmented diffs across messages
- MetaDrive results ≠ CARLA results (different sim/algo/obs)
- Never apply changes to the repo without explicit confirmation. Always ask before generating files.
- Token-efficient output: minimal prose, bullet points and tables preferred, no redundant summaries. Apply consistently.
</constraints>

<research_question>Does curriculum learning produce measurably different agent behavior than batch/mixed training in MARL urban driving?</research_question>

## Stack

| Item | Value |
|------|-------|
| Sim | CARLA 0.9.16 |
| Algo | MAPPO CTDE, Ray/RLlib 2.10.0 |
| Policies | `vehicle_policy` 25D · `pedestrian_policy` 19D |
| Critic | Fixed-slot 138D (3×25+3×19+6 alive_mask), PopArt off |
| Agents | 3V+3P, fixed map Town03 (Design B revised) |
| Framework | PyTorch 2.7+cu126, PettingZoo, Python 3.11.9 |
| HW local | RTX 3080 Laptop 8GB VRAM / 16GB RAM, Windows 11 |
| HW cloud | A100 agosto, Vast.ai/RunPod, budget 150–300€ |
| Repo | `traffic-rl-curriculum-vs-mixedbatch` (GitHub, project connesso) |
| Target | Laurea novembre 2026 |

## Repo Structure

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
  configs/levels.yaml                    # Design B revised v2: route-reduced (30/80/150m), Town03 fixed
  configs/curriculum_batch.yaml          # promotion/replay/batch config, level_criteria per-level
  configs/eval.yaml                      # reload_world: false, subprocess_timeout: 600s (fix applied)
  scripts/{test_mappo_pipeline.py,compare_results_carla.py}
metadrive_prototype/                     # Archive
```

## Hyperparams

**G2 baseline (v8)** — `train_mappo.yaml`, immutabile:
```yaml
lr: 0.0005  entropy_coeff: 0.03  num_sgd_iter: 15  train_batch_size: 8000
sgd_minibatch_size: 256  clip_param: 0.2  grad_clip: 0.5  vf_clip_param: 10.0
use_kl_loss: true  kl_target: 0.02  kl_coeff: 0.3  gamma: 0.99  gae_lambda: 0.95
```

## Gate Status

| Gate | Status | Note |
|------|--------|------|
| G1 | PASS | CARLA install, smoke test, RLlib setup |
| G2 | PASS | Tag: `g2-freeze-mlp`, checkpoint `baseline_mlp_g2/` |

## Completed Blocks

| Block | What |
|-------|------|
| 0–4.3 | Docstrings, infos side-channel, callbacks, fixed-slot 138D, PopArt, V=25D, P=19D |
| Bug2 | `_terminated_agent_infos` fix — route completion underreporting |
| G2 | Run11 1M steps PASS, checkpoint freezato |
| 5.1–5.4 | Route planner, levels, curriculum/batch manager, training wiring |
| fix T1/F1 | Test agent_order, reset cleanup traffic |
| Bug3 | NaN clamp on logits+value in `forward()` |
| Bug4 | `reset()` path_eff fields init order |
| Bug5 | `level_timesteps=0` — root cause of zero promotions in R1 1.5M |
| Bug6 | `path_efficiency` stuck=1.0 degenerate ratio → `if stuck: 0.0` |
| Design B rev | Town03 fixed, route-reduced (30/80/150m) |
| Finetuning v3/v4 | `level_criteria` per-level, promotion 0.45/0.30 |
| eval.yaml fix | `reload_world: false`, `subprocess_timeout: 600s` |

## Design Decisions FREEZATE

**Design B revised v2**

| Level | Map | NPC | route_m | route_m_ped |
|-------|-----|-----|---------|-------------|
| Easy | Town03 | 5V+10P | 30 | 15 |
| Medium | Town03 | 15V+30P | 80 | 35 |
| Hard | Town03 | 30V+60P | 150 | 60 |
| Test | Town02+Town10 | 15V+30P | 120 | 50 |

**Curriculum** (v4): SR≥0.45, CR≤0.30, min_ep≥50, min_ts per-level, aggregate promotion (Yu 2022, Bengio 2009). Replay 0.2. **Batch**: K=3 stratified shuffle.

**Attention critic**: secondary contribution (MLP G2 anchor). **GNN**: optional Phase 3 agosto.

## Run History

| Run | Mode | Budget | Status | Notes |
|-----|------|--------|--------|-------|
| R1 (old) | Curriculum | 1.5M | Done | Bug5 blocked promotion. Old routes. |
| R2 (old) | Batch | 1M | Done | Old routes. Not comparable with R3+. |
| **R3** | **Curriculum** | **3M** | **COMPLETED + EVALUATED** | Bug5 fixed, routes reduced. Reached Hard. |
| **R3-batch** | **Batch** | **3M** | **COMPLETED + EVALUATED** | Same config. |

## Eval History

| # | Type | Budget | Mode | Result | Notes |
|---|------|--------|------|--------|-------|
| 1–2 | Test | 36K | Both | **PASS** | Fresh server, pre-R3 |
| 3–4 | Test | 102K | Both | **PASS** | Fresh server, pre-R3 |
| 5 | R3 curriculum | 102K | Curriculum | **STALL ep9** → fixed → **PASS** | `reload_world` race condition |
| 6 | R3-batch | 102K | Batch | **PASS** | Post-fix |

**Eval stall root cause (corrected)**: `reload_world_between_episodes: true` called `client.reload_world()` per episode. Server was fresh (restarted before eval). The 4 prior tests (~408 total `reload_world` calls) all passed. Stall at ep9 was **non-deterministic race condition** in libcarla C++ actor teardown (~0.2-0.5% per call). Not server exhaustion — pure low-probability event. Fix: eliminate unnecessary `reload_world` for same-map episodes; `env.reset()` with F1 handles cleanup.

## Pending Blocks

| # | Block | Dep | Status |
|---|-------|-----|--------|
| 4.4 | Attention Critic | G2 ✓ | **NEXT** |
| R4/R5 | Attention runs | 4.4 | — |
| GNN | GNN encoder | R4 | Opz. agosto |
| dock | Dockerfile | — | PENDING |

## Key Technical Notes

- Bug5: `num_env_steps_sampled_this_iter` fallback → `level_timesteps` correct
- Bug6: `if termination_reason == "stuck": path_eff = 0.0` — post termination, pre agent_info
- `path_efficiency`: `min(completed_optimal / actual_distance, 1.0)`. Stuck → 0.0.
- Eval: subprocess isolation per episode, recovery via `_recover_carla_server()`
- `reload_world` between same-map episodes: unnecessary, non-deterministic stall risk on Windows
- Batch reward inflation: cross-level average dominated by Easy pedestrian completions. Compare via per-level SR, not aggregate reward.

## CARLA Runtime Constraints

- `terminate_on_collision: true` — false → NaN
- `world.tick(10.0)` — timeout prevents sync deadlock
- `sensor.stop()` before destroy
- NO `load_world`/`reload_world` same-map — stall/SIGABRT risk (CARLA 0.9.16 Windows)
- `_switch_map()` for cross-map eval only (subprocess isolation)

## Key Learnings

- **Bug5**: dominant blocker R1 (plumbing, not convergence)
- **Bug6**: degenerate path_eff for stuck agents with low movement
- **`reload_world` stall**: non-deterministic, low probability per call, not server-state-dependent. 4 test runs (408 calls) passed; stalled at call ~9 of 5th run. Fix: eliminate the call.
- **Batch vs curriculum reward**: not comparable as aggregate — different level distributions
- **Route reduction**: 50m→30m critical for Easy vehicle convergence
- **Pedestrian converges faster**: simpler dynamics

## Audit Log

| Date | Scope | Result |
|------|-------|--------|
| 08 Apr | Full repo: Bug2–Bug6, F1, T1, config↔code | ALL PASS |
| 09 Apr | Eval stall ep9/102 | Root cause: non-deterministic `reload_world` race condition |
| 10 Apr | R3 + R3-batch eval completed | Both PASS with fixed eval.yaml |
