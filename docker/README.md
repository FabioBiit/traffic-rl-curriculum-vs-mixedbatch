# Docker Setup

## Prerequisiti
- Docker Engine 24+ con Compose v2
- NVIDIA Container Toolkit (per il profilo GPU)
- Windows: Docker Desktop con WSL2 backend; Windows 11 + WSLg per la GUI

## Build

```bash
docker compose build trainer
```

## Training MetaDrive

```bash
docker compose run --rm trainer \
    -m metadrive_prototype.training.train_experiment \
    --mode batch --timesteps 100000
```

Con rendering offscreen (xvfb):

```bash
docker compose run --rm -e USE_XVFB=1 trainer \
    -m metadrive_prototype.training.train_experiment \
    --mode batch --timesteps 100000
```

## Training CARLA

```bash
# Avvia il server CARLA in background
docker compose up -d carla-server

# Attendi che sia pronto (cerca "WorldOnTick" nei log)
docker compose logs -f carla-server

# Avvia il training
docker compose run --rm trainer \
    carla_core/training/train_carla_mappo.py \
    --difficulty path --timesteps 100000
```

## Evaluation e compare

```bash
docker compose run --rm trainer \
    carla_core/training/evaluate_carla_mappo.py --config carla_core/configs/eval.yaml

docker compose run --rm trainer \
    carla_core/training/compare_results_carla.py
```

## Profilo CPU (no GPU)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml \
    run --rm trainer \
    -m metadrive_prototype.training.train_experiment --mode batch
```

## X11 / GUI forwarding

### Linux

```bash
xhost +local:docker
docker compose run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    trainer carla_core/training/visualize_agent.py
```

### Windows (WSLg / Windows 11)

Da un terminale WSL2 (WSLg espone `$DISPLAY` automaticamente):

```bash
docker compose run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    trainer carla_core/training/visualize_agent.py
```

## Output persistente

I volumi montati bidirezionalmente host ↔ container:

| Host | Container |
|------|-----------|
| `./carla_core/experiments` | `/app/carla_core/experiments` |
| `./carla_core/results` | `/app/carla_core/results` |
| `./carla_core/configs` | `/app/carla_core/configs` |
| `./metadrive_prototype/experiments` | `/app/metadrive_prototype/experiments` |
| `./metadrive_prototype/results` | `/app/metadrive_prototype/results` |
