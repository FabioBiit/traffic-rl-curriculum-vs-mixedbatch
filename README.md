# traffic-rl-curriculum-vs-mixedbatch
Thesis repository on RL for mixed urban traffic (vehicles + pedestrians), with experimental comparison between mixed-batch training and curriculum learning.

## MARL Collision Avoidance And Pathfinding In Simulated Urban Environments
### Thesis Project - 2026

### Overview
The long-term target is MAPPO on CARLA.
MetaDrive is used as prototype phase to validate training loop, metrics, and experiment protocol.

Research question:
Does curriculum learning (easy -> medium -> hard) outperform mixed-batch training for urban navigation tasks?

### Architecture
- Algorithm target: MAPPO (centralized critic) for CARLA phase
- Prototype algorithm: PPO on MetaDrive
- Simulator target: CARLA
- Prototype simulator: MetaDrive
- Framework target: RLlib (Ray), with alternatives if needed

### Current Project Structure
```text
traffic-rl-curriculum-vs-mixedbatch/
|-- metadrive_prototype/ (prototype)
|   |-- envs/                  # MetaDrive env factory and curriculum manager
|   |-- training/              # train_experiment.py (batch vs curriculum)
|   |-- scripts/               # compare_results.py, verify_setup.py
|   |-- evaluation/            # Placeholder package
|   |-- experiments/           # Batch/Curriculum run outputs
|   `-- results/               # Plots and comparison artifacts
|-- carla_core/ (main project)
|   |-- agents/                # CARLA policies
|   |-- configs/               # YAML configs (CARLA only)
|   |-- envs/                  # CARLA Gymnasium wrapper
|   |-- evaluation/            # CARLA metrics and eval
|   |-- experiments/           # CARLA run outputs (batch vs curriculum)
|   |-- results/               # CARLA plots and tables
|   |-- scripts/               # Utility scripts
|   `-- training/              # CARLA train entrypoints

|-- DOCS/                      # Thesis docs and notes (Not Committed)
|-- requirements.txt
`-- README.md
```

### MetaDrive Prototype Commands
```bash
python ./metadrive_prototype/scripts/verify_setup.py
python ./metadrive_prototype/training/train_experiment.py --mode both --timesteps 1500000
python ./metadrive_prototype/scripts/compare_results.py \
  --batch metadrive_prototype/experiments/batch/<run>/results.json \
  --curriculum metadrive_prototype/experiments/curriculum/<run>/results.json \
  --output metadrive_prototype/results/plots/<name>
```

### Gate Decisions
| Gate | Date   | Question                           | Pass ->         | Fail ->              |
|------|--------|------------------------------------|-----------------|----------------------|
| G1   | 31 Mar | RL loop works? CARLA installed?   | Start CARLA dev | Fix setup/runtime    |
| G2   | 30 Apr | RLlib + CARLA communicate?        | Continue RLlib  | Switch to PettingZoo |
| G3   | 31 May | Multi-agent trains without crash? | Scale scenarios | Reduce scope         |
| G4   | 30 Jun | MAPPO + MLP converges on 2 maps?  | Add attention   | Stabilize baseline   |
| G5   | 31 Jul | Attention works on 3 maps?        | Freeze design   | Freeze baseline      |
| G6   | 31 Aug | All experiments completed?        | Start writing   | Complete by Sep 7    |

### Timeline
- March: Research and MetaDrive prototyping
- April: CARLA wrapper and single-agent baseline
- May: Multi-agent MAPPO loop
- June-July: Scaling and experiments
- August: Full experiments
- September: Thesis writing
- October 1-9: Final revision and submission
