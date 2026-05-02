# Refactor Brief — CARLA MAPPO

<role>
Senior Python engineer su `carla_core/` (RLlib MAPPO + CARLA). Applica solo edit chirurgici e mirati. Per ogni task esegui un loop di integrazione (max 3 iter): implementa → verifica → se fallisce patch+retest, se passa procedi al merge. Stop dopo 3 iter senza pass.
</role>

<context>
- `BatchLevelSampler` (`curriculum_batch_manager.py:758-813`) usa shuffle stratificato a finestra, non i.i.d.
- `train_carla_mappo.py::main` ha `exp_seed=42` hardcoded (default YAML) e nessuno switch CLI di difficoltà architetturale.
- `levels.yaml` espone un solo blocco `levels:` dove map+traffic+route variano insieme.
- `_load_levels_yaml()` ritorna `data["levels"]`; `get_level_configs(env_cfg)` mergia `env_cfg["levels"]` sopra il default da disco.
</context>

<task id="1" branch="task1/batch-random">
<files>`curriculum_batch_manager.py`, `configs/curriculum_batch.yaml`</files>
<changes>
1. `BatchLevelSampler` (L754-813): mantieni solo `levels` + `_rng`. `__init__(self, levels, seed)`. `sample() → self._rng.choice(self.levels)`. `summary() → {"levels": list(self.levels)}`. Elimina `window_size, _window, _cursor, _total_samples, _counts, _refill_window, counts_balanced`. Aggiorna docstring modulo (L9) e classe (L759-764).
2. `curriculum_batch.yaml`: rimuovi `batch.window_size`.
</changes>
<verify>300 sample i.i.d. con seed fisso → chi-squared p>0.05; grep simboli rimossi vuoto; `results.json["batch_sampling"]` regge; curriculum mode + `EpisodeTracker` intatti.</verify>
</task>

<task id="2" branch="task2/cli-seed-difficulty">
<files>`train_carla_mappo.py`, `configs/levels.yaml`, `configs/train_mappo.yaml`, `configs/curriculum_batch.yaml`</files>
<changes>
1. argparse (~L582): `--seed:int`, `--difficulty {path,traffic,mixed}` (required).
2. (~L618): `exp_seed = args.seed`. Propagazione esistente a `env_cfg["traffic"]["seed"]`, samplers, `_build_mappo_config`, `torch/np/random.seed`.
3. Pre `level_manager`: `lv = load_yaml(base/"configs/levels.yaml"); env_cfg["levels"] = lv[f"levels_{args.difficulty}"]`. Merge via `get_level_configs`.
4. `levels.yaml`: aggiungi `levels_path` (varia solo `route_distance_m`), `levels_traffic` (varia solo `n_*_npc`), `levels_mixed` (= attuale). Tutti i campi pieni. Mantieni `levels:` e `test:`.
5. Cleanup YAML: rimuovi `experiment.seed`, `curriculum.teacher_seed`, `batch.seed`.
</changes>
<verify>`--seed 7 --mode batch --difficulty path --timesteps 2000` → `run_config.json[seed]==7`, `n_vehicles_npc` costante fra easy/medium/hard; eval e visualize non regrediti (usano `levels:` default); grep `"= 42\|seed: 42"` in `carla_core/` vuoto.</verify>
</task>

<constraints>
Branch isolati, nessuna linea condivisa (Task1 → `BatchLevelSampler` + `batch.window_size`; Task2 → chiavi YAML disgiunte). PEP 8, no dead imports, no commented-out code, no scope creep. NON eseguire fino ad approvazione esplicita.
</constraints>
