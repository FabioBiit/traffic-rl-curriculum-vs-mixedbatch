# Project: traffic-rl-curriculum-vs-mixedbatch

MARL master's thesis on urban driving (CARLA + MAPPO). Target: Nov 2026.

## Role

PhD AI/ML Engineer (10+ yr R&D). Co-develop an experimental master's thesis on MARL urban driving. Surgical, empirical style. Every decision backed by run data or literature.

## Research Question

Does curriculum learning produce measurably different agent behavior than batch/mixed training in MARL urban driving? (CARLA 0.9.16, MAPPO, 3V+3P)

## Rules

- Empirical data overrides literature defaults. No invented citations.
- Audit repo before any change; surgical diffs with file + line anchors; no full-file rewrites.
- Gate-based flow; freeze before architecture changes.
- MetaDrive results ≠ CARLA (distinct sim / algo / obs).
- No repo changes without explicit user confirmation; ask before generating files.
- Token-efficient: tables > prose, no redundant summaries.
- Output discipline for evo proposals: plan ≤100 words; each evo ≤50 words; file + line anchor per evo; validate up to 3× vs repo before presenting.
- Mark assumptions as `Inference`; missing evidence as `Not found in repo`.

## Stack

| Item | Value |
|------|-------|
| Sim | CARLA 0.9.16 |
| Algo | MAPPO CTDE, Ray/RLlib 2.10.0 |
| Policies | `vehicle_policy` 25D · `pedestrian_policy` 19D |
| Critic | Fixed-slot 138D (3×25 + 3×19 + 6 alive_mask), PopArt=off |
| Agents | 3V + 3P, Town03 training (Design B rev v2) |
| Framework | PyTorch 2.7+cu126, PettingZoo, Python 3.11.9 |
| HW | RTX 3080 Laptop 8GB/16GB Win11 · A100 cloud Aug (150–300€) |
| Target | Nov 2026 |

## Repository Layout

```
carla_core/
  agents/centralized_critic.py     — Model + encoders + callbacks + PopArt
  envs/
    carla_multi_agent_env.py       — PettingZoo ParallelEnv + set_level
    route_planner.py               — A* GlobalRoutePlanner wrapper
  training/
    train_carla_mappo.py           — main loop + CLI
    evaluate_carla_mappo.py        — subprocess-isolated eval
    curriculum_batch_manager.py    — CurriculumManager + BatchLevelSampler
    mappo_runtime.py               — shared config builder
  configs/
    train_mappo.yaml               — G2 hyperparams (IMMUTABLE)
    levels.yaml                    — Design B rev v2
    curriculum_batch.yaml          — promotion / replay / batch
    eval.yaml                      — reload_world:false, timeout:600s
  scripts/visualize_mappo_agent.py — own local cc_config
metadrive_prototype/                — archive (NOT comparable)
```

## Frozen Config

**G2 baseline v8** — `train_mappo.yaml` — IMMUTABLE:

```yaml
lr: 0.0005  entropy_coeff: 0.03  num_sgd_iter: 15  train_batch_size: 8000
sgd_minibatch_size: 256  clip_param: 0.2  grad_clip: 0.5  vf_clip_param: 10.0
use_kl_loss: true  kl_target: 0.02  kl_coeff: 0.3  gamma: 0.99  gae_lambda: 0.95
```

**Curriculum v4**: SR≥0.45, CR≤0.30, min_ep=50, replay=0.05, window=50. Easy min_ts=500K, Medium min_ts=1.5M.

**Batch**: K=3 stratified shuffle, seed=42.

**Design B rev v2** (FROZEN):

| Level | Map | NPC | route_m V/P |
|-------|-----|-----|-------------|
| Easy | Town03 | 5V + 10P | 30/15 |
| Medium | Town03 | 15V + 30P | 80/35 |
| Hard | Town03 | 30V + 60P | 150/60 |
| Test (eval) | Town05 | 15V + 30P | 120/50 |

## Critic Encoders (2×2 orthogonal matrix)

| `use_gnn` | `use_attention` | Encoder | Block | Status |
|-----------|-----------------|---------|-------|--------|
| F | F | MLP flat | baseline | ✅ R3/R4 |
| F | T | MLP + self-attention (agent tokens) | 4.4 | wired, untested on main |
| T | F | GNN (GraphConv mean aggregation) | 4.5a | wired, untested |
| T | T | GAT (graph attention) | 4.5b | wired, untested |

CLI: `--use-attention` + `--use-gnn` orthogonal, combinable. Propagation: `train_mappo.yaml → mappo_runtime.cc_config → both policies`. Pure PyTorch (no `torch_geometric`).

Shared token pipeline: `138D → split slots → per-type Linear → aggregation{MHA | GraphConv | GAT} → masked mean-pool → Linear(embed→256)+Tanh → critic_head → value`. MLP flat path bypasses tokenization.

Refs: Iqbal & Sha 2019 (MAAC) · Hamilton et al. 2017 (GraphSAGE) · Veličković et al. 2018 (GAT).

## Gates

| Gate | Status |
|------|--------|
| G1 | PASS (CARLA + RLlib setup) |
| G2 | PASS (`g2-freeze-mlp` tag, `baseline_mlp_g2/`) |

## Completed

- Blocks 0–4.3 (infos, callbacks, 138D critic, PopArt stub, obs spaces)
- Bug2–Bug6 fixes (agent_infos, NaN clamp, path_eff, level_timesteps, stuck)
- 5.1–5.4 (routing, levels, managers, wiring)
- Fix T1/F1 (agent_order test, reset cleanup)
- Design B rev v2 freeze (Town03, routes 30/80/150m)
- Finetuning v3/v4 (level_criteria, promotion thresholds)
- `eval.yaml` fix (reload_world:false, timeout:600s)
- Block 4.4 (Attention wired, 5 files, untested on main branch)
- Block 4.5 (GNN + GAT encoders, 7 evos + refactor `mappo_runtime` as canonical `_build_mappo_config`, wired, untested)

## Run History

| Run | Mode | Encoder | Budget | Status |
|-----|------|---------|--------|--------|
| R3 | Batch | MLP | 3M | DONE+EVAL |
| R4 | Curriculum | MLP | 3M | DONE+EVAL |

R1/R2 obsolete (Bug5 + old routes).

## Eval Results (25 ep/level, subprocess-isolated, `reload_world:false`)

### R3 Batch (MLP)

| Level | SR | CR | Stuck | Offroad | RC | PathEff |
|-------|----|----|-------|---------|----|---------|
| Easy | 0.500 | 0.073 | 0.353 | 0.073 | 0.564 | 0.492 |
| Medium | 0.387 | 0.053 | 0.393 | 0.167 | 0.544 | 0.495 |
| Hard | 0.313 | 0.087 | 0.420 | 0.173 | 0.583 | 0.475 |
| Test | 0.333 | 0.107 | 0.533 | 0.013 | 0.542 | 0.424 |

### R4 Curriculum (MLP)

| Level | SR | CR | Stuck | Offroad | RC | PathEff |
|-------|----|----|-------|---------|----|---------|
| Easy | 0.520 | 0.059 | 0.373 | 0.000 | 0.619 | 0.494 |
| Medium | 0.366 | 0.093 | 0.513 | 0.020 | 0.540 | 0.420 |
| Hard | 0.440 | 0.173 | 0.350 | 0.020 | 0.580 | 0.581 |
| Test | 0.333 | 0.060 | 0.546 | 0.040 | 0.490 | 0.374 |

## Next Steps

### Runs queue (Block 4.5 applied — 7 evos + refactor)

```bash
# R5 MLP+Attn cur    --mode curriculum --timesteps 3000000 --use-attention
# R6 MLP+Attn batch  --mode batch      --timesteps 3000000 --use-attention
# R7 GNN cur         --mode curriculum --timesteps 3000000 --use-gnn
# R8 GAT cur         --mode curriculum --timesteps 3000000 --use-gnn --use-attention
```

Per-run gate: no NaN in 50K + V-loss finite vs R3/R4.

### Post-runs

Dockerfile → multi-seed (≥5 seeds, cluster) → `compare_results_carla.py` (MLP vs MLP+Attn vs GNN vs GAT) → thesis write-up (Sept).

## Runtime Constraints (CARLA)

- `terminate_on_collision: true` (false → NaN)
- `world.tick(10.0)` (deadlock prevention)
- `sensor.stop()` before `destroy()`
- NO `load_world` / `reload_world` same-map (stall / SIGABRT on Windows)
- Bug3 NaN clamp in `forward()` must remain
- Bug5: `num_env_steps_sampled_this_iter` for `level_timesteps`
- Bug6: stuck → `path_eff = 0.0`
- Encoder flags orthogonal (`use_attention` + `use_gnn` combinable → GAT); no mutual exclusion at model or CLI level.

## Working Style

- Gate-based workflow; freeze repo state before arch changes.
- No conceptual repository changes without explicit user confirmation.
- Prefer surgical diffs with exact file and line references.
- Do not rewrite full files unless explicitly requested.
- Review each fix up to 3 validation passes before considering stable.
- Always check for regressions, dead code, and ambiguous naming.

## Output Contract

- Return exactly the sections requested, in the requested order.
- Default: short, high-density sections.
- Use bullets or tables when they improve scannability.
- Mark assumptions as `Inference`; missing evidence as `Not found in repo`.
- For code tasks include: affected files, what changed, regression risk, verification performed.
- For review tasks: findings first, summaries second.

## Learnings

- Bug5 was R1 blocker: plumbing (zero promotions), not convergence.
- `reload_world`: 0.2–0.5% stall per call + libcarla teardown race — eliminated.
- Batch vs curriculum aggregate reward not directly comparable.
- Route 50→30m critical for Easy vehicle convergence.
- Pedestrian converges faster → inflates batch aggregate reward.
- GAT in pure PyTorch avoids `torch_geometric` CUDA/Windows install issues.

## Audit Log

| Date | Scope | Result |
|------|-------|--------|
| 08 Apr | Full repo Bug2–Bug6, config ↔ code | PASS |
| 10 Apr | R3 + R4 eval | PASS |
| 14 Apr | Block 4.4 applied, 5 files | wired, untested |
| 18 Apr | Block 4.5 design redefined (2×2 matrix) | 7 evos specified, pending application |
| 19 Apr | Block 4.5 + refactor `mappo_runtime` (canonical `_build_mappo_config`) | applied, 3-pass validation PASS |
