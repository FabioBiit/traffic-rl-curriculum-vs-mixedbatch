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
