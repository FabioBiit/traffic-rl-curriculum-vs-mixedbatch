# DOCKERPLAN

<role>
Senior DevOps engineer specializzato in container per workload RL/CUDA.
</role>

<context>
Repo: `traffic-rl-curriculum-vs-mixedbatch`. Stack: Python 3.11, CUDA 12.6, Ray 2.10, torch 2.7.1.
Sub-progetti:
- `carla_core` — MAPPO multi-agent; client TCP verso server CARLA esterno.
- `metadrive_prototype` — self-contained, rendering Panda3D headless.
Stato: nessun Dockerfile / compose / CI esistente.
</context>

<task>
Containerizzare il repo per deploy riproducibile su Linux nativo e Windows/WSL2.
Tutti i CLI (train, eval, visualize, compare) eseguibili via `docker compose run trainer ...`.
CARLA simulator incluso come servizio del medesimo compose.
</task>

<deliverables>
**File NUOVI (root):**
- `docker/Dockerfile.trainer` — base `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04`; installa Python 3.11, `xvfb`, `libgl1`; copia `requirements.txt` e progetto; `WORKDIR=/app`; `ENTRYPOINT ["python"]`.
- `docker/entrypoint.sh` — wrapper opzionale `xvfb-run` per rendering offscreen MetaDrive.
- `docker-compose.yml` — servizi `carla-server` (image `carlasim/carla:0.9.16`, GPU, ports 2000-2002) e `trainer` (build, `depends_on: carla-server`, GPU via `deploy.resources`, env `CARLA_HOST=carla-server`, volumi `./carla_core/experiments`, `./carla_core/results`, `./metadrive_prototype/experiments`, `./configs`).
- `.dockerignore` — esclude `*/experiments/*`, `*/results/*`, `.git`, venv, checkpoint.
- `docker/README.md` — esempi CLI, X11 forwarding (WSLg/Linux), profili `gpu`/`cpu`.

**Modifiche codice (chirurgiche):**
- Loader config CARLA in `carla_core/` dove si crea `carla.Client(...)`: leggere host/port da env `CARLA_HOST`/`CARLA_PORT`, fallback `127.0.0.1:2000`.
</deliverables>

<constraints>
- Nessun refactor opportunistico; modifiche al solo punto di connessione CARLA.
- Compose profiles: `gpu` (default), `cpu` (override `--no-gpu`).
- Volumi bidirezionali host↔container per persistenza output.
- Niente push registry; build locale.
</constraints>

<examples>
Training MetaDrive:
`docker compose run --rm trainer -m metadrive_prototype.training.train_experiment --mode batch --timesteps 1000`

Training CARLA:
`docker compose up -d carla-server`
`docker compose run --rm trainer carla_core/training/train_carla_mappo.py --difficulty path --timesteps 1000`
</examples>

<verification>
1. `docker compose build trainer` → exit 0.
2. `docker compose up -d carla-server` → log `WorldOnTick` visibile.
3. Smoke MetaDrive → checkpoint in `./metadrive_prototype/experiments/`.
4. Smoke CARLA → checkpoint in `./carla_core/experiments/`.
5. `compare_results_carla.py` → plot generati in `./carla_core/results/`.
6. Windows/WSL2 con WSLg → `visualize_agent.py` apre finestra GUI.
</verification>
