# traffic-rl-curriculum-vs-mixedbatch
This repository contains a thesis project on RL for mixed urban traffic (vehicles + pedestrians). It compares mixed-batch training vs curriculum learning (easy→medium→hard) using vector observations, measuring collision rate, success rate, path efficiency, and convergence speed.

# MARL Collision Avoidance & Pathfinding in Simulated Urban Environments

## Thesis Project — 2026

### Overview
Multi-Agent Reinforcement Learning (MAPPO) for collision avoidance and pathfinding
in 3D urban environments with vehicles and pedestrians.

**Research Question:** Does Curriculum Learning (easy→medium→hard) outperform
Batch Training (all maps simultaneously) for multi-agent navigation in mixed
vehicle-pedestrian urban scenarios?

### Architecture
- **Algorithm:** MAPPO (Multi-Agent PPO) with centralized critic
- **Policies:** Separate `vehicle_policy` and `pedestrian_policy`
- **Policy Network:** MLP baseline → Attention mechanism (conditional on gate G4)
- **Simulator:** CARLA (prototyping on MetaDrive)
- **Framework:** RLlib (Ray) — Plan B: PettingZoo + SB3

### Project Structure
```
traffic-rl-curriculum-vs-mixedbatch/
├── envs/                    # Environment wrappers
│   ├── metadrive_env.py     # MetaDrive wrapper (prototyping)
│   └── carla_env.py         # CARLA multi-agent wrapper
├── agents/                  # Neural network architectures
│   ├── mlp_policy.py        # MLP baseline policy
│   └── attention_policy.py  # Attention mechanism (gate G4)
├── training/                # Training scripts
│   ├── train_metadrive.py   # MetaDrive PPO training
│   ├── train_carla_single.py # CARLA single-agent
│   └── train_carla_mappo.py # CARLA multi-agent MAPPO
├── evaluation/              # Evaluation & metrics
│   ├── metrics.py           # Collision rate, success rate, etc.
│   └── plotting.py          # Visualization scripts
├── experiments/             # Experiment results & configs
│   ├── curriculum/          # Curriculum learning runs
│   └── batch/               # Batch training runs
├── configs/                 # YAML configuration files
│   ├── metadrive_config.yaml
│   └── carla_mappo_config.yaml
├── docs/                    # Documentation
│   ├── paper_notes/         # Structured notes per paper
│   └── setup_guide.md       # Installation guide
├── scripts/                 # Utility scripts
├── requirements.txt
└── README.md
```

### Gate Decisions
| Gate | Date       | Question                          | Pass → | Fail →              |
|------|------------|-----------------------------------|--------|----------------------|
| G1   | 31 Mar     | RL loop works? CARLA installed?   | April  | Fix setup            |
| G2   | 30 Apr     | RLlib + CARLA communicate?        | May    | Switch to PettingZoo |
| G3   | 31 May     | Multi-agent trains without crash? | Scale  | Reduce to 1+1 RL    |
| G4   | 30 Jun     | MAPPO+MLP converges on 2 maps?   | Add Attention | Stabilize     |
| G5   | 31 Jul     | Attention works on 3 maps?        | Freeze (+GNN?) | Freeze       |
| G6   | 31 Aug     | All experiments done?             | Write  | Complete by Sep 7    |

### Timeline
- **March:** Research + MetaDrive prototyping
- **April:** CARLA wrapper + single agent
- **May:** Multi-agent MAPPO
- **June-July:** Scaling + experiments (+ exams)
- **August:** Full experiments
- **September:** Thesis writing (+ AI exam)
- **1-9 October:** Final revision → **SUBMIT**
