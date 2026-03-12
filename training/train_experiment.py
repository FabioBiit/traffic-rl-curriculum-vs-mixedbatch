"""
Settimana 2 - Esperimento Curriculum vs Batch su MetaDrive
==========================================================
Lancia i due approcci di training e salva i risultati per il confronto.

Esegui con:
    python ./training/train_experiment.py --mode batch
    python ./training/train_experiment.py --mode curriculum
    python ./training/train_experiment.py --mode both

Quick validation (500K step per verificare che il curriculum funzioni):
    python ./training/train_experiment.py --mode curriculum --timesteps 500000

Confronta su TensorBoard:
    tensorboard --logdir=experiments/

Confronta risultati:
    python ./evaluation/compare_results.py

Changelog v2.0 (12 Marzo 2026):
- Curriculum training con replay mechanism (revisione livelli precedenti)
- Progress tracker globale con tempo trascorso, ETA, e stato promozione
- Evaluation aumentata a 50 episodi (da 20)
- CurriculumManager usa nuovi criteri: soglia 0.6, collision threshold,
  min_timesteps, replay_ratio
- Timeseries logging piu' dettagliato (include livello replay)
- JSON output con schema strutturato per analisi
"""

import os
import sys
import json
import yaml
import time
import random
import argparse
import numpy as np
from datetime import datetime, timedelta

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Aggiungi la root del progetto al path per gli import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multi_level_env import (
    create_env,
    LEVEL_CONFIGS,
    EpisodeTracker,
    CurriculumManager,
)


# ============================================================
# COSTANTI GLOBALI
# ============================================================

SEED = 42
DEVICE = "cpu"

# Budget totale identico per entrambi gli approcci
TOTAL_TIMESTEPS = 1_500_000

# Dimensione di ogni blocco di training
# Tra un blocco e l'altro possiamo cambiare ambiente
BLOCK_SIZE = 50_000

# Episodi per la valutazione finale
# v2.0: aumentato da 20 a 50 per affidabilita' statistica
EVAL_EPISODES = 50

# PPO — identico per batch e curriculum (variabile sperimentale isolata)
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "verbose": 0,
    "device": DEVICE,
}

# Curriculum — soglie di promozione v2.0
CURRICULUM_CONFIG = {
    "promotion_threshold": 0.6,     # v2.0: alzata da 0.3 a 0.6
    "collision_threshold": 0.3,     # v2.0: nuovo — max collision rate per promuovere
    "min_episodes": 50,
    "min_timesteps": 200_000,       # v2.0: nuovo — minimo 200K step per livello
    "window_size": 50,
    "replay_ratio": 0.2,           # v2.0: nuovo — 20% blocchi dedicati a revisione
}


# ============================================================
# SEED
# ============================================================

def set_global_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PROGRESS TRACKER — Tempo trascorso, ETA, stato globale
# ============================================================

class ProgressTracker:
    """
    Traccia il progresso globale dell'esperimento.
    Mostra: blocco corrente, step totali, tempo trascorso, ETA stimata.
    
    Per curriculum, mostra anche il livello corrente e lo stato promozione.
    """

    def __init__(self, mode, total_steps, block_size):
        self.mode = mode
        self.total_steps = total_steps
        self.block_size = block_size
        self.total_blocks = total_steps // block_size
        self.start_time = time.time()
        self.block_times = []  # durata di ogni blocco per stima ETA

    def format_time(self, seconds):
        """Formatta secondi in ore:minuti:secondi."""
        if seconds < 0:
            return "??:??:??"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h{m:02d}m{s:02d}s"

    def block_start(self):
        """Chiama all'inizio di ogni blocco."""
        self._block_start = time.time()

    def block_end(self):
        """Chiama alla fine di ogni blocco. Ritorna la durata."""
        duration = time.time() - self._block_start
        self.block_times.append(duration)
        return duration

    def get_eta(self, steps_done):
        """Stima il tempo rimanente basandosi sulla media dei blocchi completati."""
        if not self.block_times:
            return "??:??:??"
        avg_block_time = sum(self.block_times) / len(self.block_times)
        remaining_blocks = (self.total_steps - steps_done) / self.block_size
        eta_seconds = avg_block_time * remaining_blocks
        return self.format_time(eta_seconds)

    def get_elapsed(self):
        """Tempo trascorso dall'inizio."""
        return self.format_time(time.time() - self.start_time)

    def print_status(self, block_num, steps_done, level, is_replay=False,
                     block_episodes=0, block_sr=None, block_cr=None,
                     curriculum_manager=None, tracker=None):
        """
        Stampa una riga di stato compatta e informativa.
        
        Formato:
        [BATCH 12/30] 550K/1.5M | easy | Ep:45 SR:62% CR:8% | Elapsed: 1h23m | ETA: 2h10m
        """
        mode_upper = self.mode.upper()
        steps_k = f"{steps_done / 1000:.0f}K"
        total_k = f"{self.total_steps / 1000:.0f}K"
        if self.total_steps >= 1_000_000:
            steps_k = f"{steps_done / 1_000_000:.1f}M" if steps_done >= 100_000 else f"{steps_done / 1000:.0f}K"
            total_k = f"{self.total_steps / 1_000_000:.1f}M"

        replay_tag = " [REPLAY]" if is_replay else ""
        level_str = f"{level.upper()}{replay_tag}"

        elapsed = self.get_elapsed()
        eta = self.get_eta(steps_done)

        # Riga principale
        parts = [f"[{mode_upper} {block_num}/{self.total_blocks}]"]
        parts.append(f"{steps_k}/{total_k}")
        parts.append(f"{level_str}")

        if block_episodes > 0:
            parts.append(f"Ep:{block_episodes}")
            if block_sr is not None:
                parts.append(f"SR:{block_sr:.0%}")
            if block_cr is not None:
                parts.append(f"CR:{block_cr:.0%}")

        parts.append(f"Elapsed:{elapsed}")
        parts.append(f"ETA:{eta}")

        print(" | ".join(parts))

        # Per curriculum, mostra stato promozione ogni 5 blocchi
        if curriculum_manager and tracker and block_num % 5 == 0:
            status = curriculum_manager.promotion_status(tracker)
            if not status["is_final_level"]:
                checks = []
                checks.append(f"SR:{status['success_rate_current']:.0%}/{status['success_rate_required']:.0%}"
                              + (" OK" if status["success_rate_ok"] else ""))
                checks.append(f"CR:{status['collision_rate_current']:.0%}/{status['collision_rate_max']:.0%}"
                              + (" OK" if status["collision_rate_ok"] else ""))
                checks.append(f"Steps:{status['timesteps_current']//1000}K/{status['timesteps_required']//1000}K"
                              + (" OK" if status["timesteps_ok"] else ""))
                checks.append(f"Ep:{status['episodes_current']}/{status['episodes_required']}"
                              + (" OK" if status["episodes_ok"] else ""))
                print(f"  Promotion check: {' | '.join(checks)}")


# ============================================================
# CALLBACK — Traccia metriche e episodi per ogni blocco
# ============================================================

class ExperimentCallback(BaseCallback):
    """
    Callback che traccia metriche per l'esperimento.
    Aggiorna l'EpisodeTracker ad ogni fine episodio.
    """

    def __init__(self, tracker, level_name="unknown"):
        super().__init__()
        self.tracker = tracker
        self.level_name = level_name
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                self.tracker.record(info)
                self.block_episodes += 1

                if info.get("arrive_dest", False):
                    self.block_successes += 1
                if info.get("crash", False) or info.get("crash_vehicle", False):
                    self.block_collisions += 1

                # Log su TensorBoard
                if self.tracker.total_episodes % 10 == 0:
                    self.logger.record("thesis/success_rate",
                                       self.tracker.cumulative_success_rate)
                    self.logger.record("thesis/collision_rate",
                                       self.tracker.cumulative_collision_rate)
                    self.logger.record("thesis/window_success_rate",
                                       self.tracker.window_success_rate)
                    self.logger.record("thesis/window_collision_rate",
                                       self.tracker.window_collision_rate)
                    self.logger.record("thesis/current_level",
                                       self.level_name)
        return True

    def reset_block_stats(self):
        """Resetta i contatori del blocco corrente."""
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0


# ============================================================
# TIMESERIES COLLECTOR — Raccoglie dati per JSON output
# ============================================================

class TimeseriesCollector:
    """Raccoglie snapshot delle metriche ad ogni blocco per il JSON finale."""

    def __init__(self):
        self.data = []

    def record(self, timestep, episode, level, tracker, callback,
               reward_mean=None, reward_std=None, ep_length_mean=None,
               is_replay=False):
        """Registra uno snapshot delle metriche correnti."""
        self.data.append({
            "timestep": timestep,
            "episode": episode,
            "level": level,
            "is_replay": is_replay,
            "success_rate": round(tracker.cumulative_success_rate, 4),
            "collision_rate": round(tracker.cumulative_collision_rate, 4),
            "window_success_rate": round(tracker.window_success_rate, 4),
            "window_collision_rate": round(tracker.window_collision_rate, 4),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "episode_length_mean": ep_length_mean,
            "path_efficiency_mean": None,  # placeholder per futuro
        })


# ============================================================
# BATCH TRAINING
# ============================================================

def train_batch(total_steps, block_size, ppo_config, run_dir):
    """
    Training Batch (Domain Randomization).
    Ad ogni blocco sceglie una mappa random tra Easy/Medium/Hard.
    """
    print("\n" + "=" * 60)
    print("BATCH TRAINING (Domain Randomization)")
    print(f"Budget totale: {total_steps:,} step")
    print(f"Blocco: {block_size:,} step")
    print(f"Blocchi totali: {total_steps // block_size}")
    print("=" * 60)

    levels = list(LEVEL_CONFIGS.keys())
    tracker = EpisodeTracker(window_size=50)
    timeseries = TimeseriesCollector()
    progress = ProgressTracker("batch", total_steps, block_size)
    steps_done = 0
    model = None
    block_num = 0
    level_history = []

    _cycle = list(levels)  # inizializzazione prima del while
    while steps_done < total_steps:
        block_num += 1
        remaining = total_steps - steps_done
        current_block = min(block_size, remaining)

        # Shuffle a blocchi (stratified random sampling):
        # ogni ciclo di len(levels) blocchi contiene tutti i livelli in ordine casuale.
        # Garantisce bilanciamento perfetto senza ordine prevedibile.
        if (block_num - 1) % len(levels) == 0:
            _cycle = list(levels)
            random.shuffle(_cycle)
        level = _cycle[(block_num - 1) % len(levels)]
        level_history.append(level)
        progress.block_start()

        # Crea ambiente per questo livello
        env = create_env(level)

        if model is None:
            model = PPO("MlpPolicy", env, tensorboard_log=run_dir, **ppo_config)
        else:
            model.set_env(env)

        callback = ExperimentCallback(tracker, level_name=level)

        try:
            model.learn(
                total_timesteps=current_block,
                callback=[callback],
                reset_num_timesteps=False,
                progress_bar=False,
            )
        except Exception as e:
            print(f"ERRORE nel blocco {block_num}: {e}")
            import traceback
            traceback.print_exc()
            env.close()
            break

        block_duration = progress.block_end()
        steps_done += current_block

        # Calcola metriche blocco
        block_sr = None
        block_cr = None
        if callback.block_episodes > 0:
            block_sr = callback.block_successes / callback.block_episodes
            block_cr = callback.block_collisions / callback.block_episodes

        # Progress tracker
        progress.print_status(
            block_num, steps_done, level,
            block_episodes=callback.block_episodes,
            block_sr=block_sr, block_cr=block_cr,
        )

        # Timeseries
        timeseries.record(
            timestep=steps_done,
            episode=tracker.total_episodes,
            level=level,
            tracker=tracker,
            callback=callback,
        )

        env.close()

    # Salva modello finale
    if model is not None:
        model.save(os.path.join(run_dir, "final_model"))

    # Report finale
    summary = tracker.summary()
    summary["mode"] = "batch"
    summary["total_steps"] = steps_done
    summary["level_history"] = level_history
    summary["level_distribution"] = {
        lv: level_history.count(lv) for lv in levels
    }
    summary["wall_clock_seconds"] = time.time() - progress.start_time
    summary["timeseries"] = timeseries.data

    return model, summary


# ============================================================
# CURRICULUM TRAINING (v2.0 con replay)
# ============================================================

def train_curriculum(total_steps, block_size, ppo_config, curriculum_config, run_dir):
    """
    Training Curriculum (Easy -> Medium -> Hard) con replay mechanism.
    
    v2.0 cambiamenti:
    - Usa CurriculumManager.get_block_level() per decidere se fare
      training sul livello corrente o replay su un livello precedente.
    - Soglia promozione alzata a 0.6 con collision threshold a 0.3.
    - Richiede min_timesteps (200K) oltre a min_episodes (50).
    - Il tracker tiene traccia dei timesteps per livello.
    """
    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING v2.0 (Easy -> Medium -> Hard + Replay)")
    print(f"Budget totale: {total_steps:,} step")
    print(f"Soglia promozione: {curriculum_config['promotion_threshold']:.0%}")
    print(f"Soglia collision: {curriculum_config['collision_threshold']:.0%}")
    print(f"Min timesteps/livello: {curriculum_config['min_timesteps']:,}")
    print(f"Min episodi/livello: {curriculum_config['min_episodes']}")
    print(f"Replay ratio: {curriculum_config['replay_ratio']:.0%}")
    print("=" * 60)

    tracker = EpisodeTracker(window_size=curriculum_config["window_size"])
    manager = CurriculumManager(
        promotion_threshold=curriculum_config["promotion_threshold"],
        collision_threshold=curriculum_config["collision_threshold"],
        min_episodes=curriculum_config["min_episodes"],
        min_timesteps=curriculum_config["min_timesteps"],
        replay_ratio=curriculum_config["replay_ratio"],
        window_size=curriculum_config["window_size"],
    )

    timeseries = TimeseriesCollector()
    progress = ProgressTracker("curriculum", total_steps, block_size)
    steps_done = 0
    model = None
    block_num = 0
    current_env = None
    current_env_level = None  # quale livello ha l'env aperto attualmente

    # Tracking distribuzione livelli
    level_timesteps = {lv: 0 for lv in LEVEL_CONFIGS.keys()}

    while steps_done < total_steps:
        block_num += 1
        remaining = total_steps - steps_done
        current_block = min(block_size, remaining)

        # Determina il livello per questo blocco (potrebbe essere replay)
        block_level, is_replay = manager.get_block_level()

        progress.block_start()

        # Crea/cambia ambiente se necessario
        if current_env_level != block_level:
            if current_env is not None:
                current_env.close()
            current_env = create_env(block_level)
            current_env_level = block_level

            if model is None:
                model = PPO("MlpPolicy", current_env, tensorboard_log=run_dir, **ppo_config)
            else:
                model.set_env(current_env)

        # Callback — per blocchi di replay, non aggiorniamo il tracker del livello corrente
        # ma loggiamo comunque le metriche
        callback = ExperimentCallback(tracker, level_name=block_level)

        try:
            model.learn(
                total_timesteps=current_block,
                callback=[callback],
                reset_num_timesteps=False,
                progress_bar=False,
            )
        except Exception as e:
            print(f"ERRORE nel blocco {block_num}: {e}")
            import traceback
            traceback.print_exc()
            if current_env is not None:
                current_env.close()
            break

        steps_done += current_block
        level_timesteps[block_level] = level_timesteps.get(block_level, 0) + current_block

        # Aggiorna timesteps del tracker SOLO se non e' replay
        # (il tracker misura il progresso sul livello corrente per promozione)
        if not is_replay:
            tracker.add_timesteps(current_block)

        block_duration = progress.block_end()

        # Calcola metriche blocco
        block_sr = None
        block_cr = None
        if callback.block_episodes > 0:
            block_sr = callback.block_successes / callback.block_episodes
            block_cr = callback.block_collisions / callback.block_episodes

        # Progress tracker
        progress.print_status(
            block_num, steps_done, block_level,
            is_replay=is_replay,
            block_episodes=callback.block_episodes,
            block_sr=block_sr, block_cr=block_cr,
            curriculum_manager=manager,
            tracker=tracker,
        )

        # Timeseries
        timeseries.record(
            timestep=steps_done,
            episode=tracker.total_episodes,
            level=block_level,
            tracker=tracker,
            callback=callback,
            is_replay=is_replay,
        )

        # Controlla promozione (solo se il blocco era sul livello corrente, non replay)
        if not is_replay and manager.should_promote(tracker):
            old_level = manager.current_level
            new_level = manager.promote(tracker, global_timestep=steps_done)
            print(f"\n{'>'*20} PROMOZIONE: {old_level.upper()} -> {new_level.upper()} {'<'*20}")
            promo = manager.promotion_history[-1]
            print(f"  SR alla promozione: {promo['success_rate_at_promotion']:.1%}")
            print(f"  CR alla promozione: {promo['collision_rate_at_promotion']:.1%}")
            print(f"  Timesteps sul livello: {promo['timesteps_on_level']:,}")
            print(f"  Episodi sul livello: {promo['episodes_on_level']}")
            print()

            # Chiudi env corrente — verra' ricreato al prossimo blocco
            if current_env is not None:
                current_env.close()
                current_env = None
                current_env_level = None

    # Chiudi ambiente se ancora aperto
    if current_env is not None:
        current_env.close()

    # Salva modello finale
    if model is not None:
        model.save(os.path.join(run_dir, "final_model"))

    # Report finale
    summary = tracker.summary()
    summary["mode"] = "curriculum"
    summary["total_steps"] = steps_done
    summary["curriculum"] = manager.summary()
    summary["level_distribution"] = level_timesteps
    summary["wall_clock_seconds"] = time.time() - progress.start_time
    summary["timeseries"] = timeseries.data

    return model, summary


# ============================================================
# VALUTAZIONE SU TUTTE LE MAPPE
# ============================================================

def evaluate_model(model, levels_to_test, n_episodes=None):
    """
    Valuta un modello addestrato su una lista di livelli.
    
    v2.0: n_episodes aumentato a EVAL_EPISODES (50) di default.
    
    Args:
        model: modello PPO addestrato
        levels_to_test: lista di nomi livelli
        n_episodes: episodi per livello (default EVAL_EPISODES)
    
    Returns:
        dizionario con le metriche per ogni livello
    """
    if n_episodes is None:
        n_episodes = EVAL_EPISODES

    results = {}

    for level in levels_to_test:
        print(f"\nValutazione su: {level.upper()} ({n_episodes} episodi)...")
        env = create_env(level)

        successes = 0
        collisions = 0
        total_reward = 0
        total_steps = 0
        episode_lengths = []

        obs = env.reset()
        episodes_done = 0
        ep_reward = 0
        ep_steps = 0

        while episodes_done < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

            if done[0]:
                episodes_done += 1
                total_reward += ep_reward
                total_steps += ep_steps
                episode_lengths.append(ep_steps)
                ep_reward = 0
                ep_steps = 0

                episode_info = info[0]
                if episode_info.get("arrive_dest", False):
                    successes += 1
                if episode_info.get("crash", False) or episode_info.get("crash_vehicle", False):
                    collisions += 1

                obs = env.reset()

        env.close()

        results[level] = {
            "success_rate": successes / n_episodes,
            "collision_rate": collisions / n_episodes,
            "avg_reward": round(total_reward / n_episodes, 2),
            "avg_episode_length": round(total_steps / n_episodes, 1),
            "episodes": n_episodes,
        }

        print(f"  Success: {results[level]['success_rate']:.1%} | "
              f"Collision: {results[level]['collision_rate']:.1%} | "
              f"Avg Reward: {results[level]['avg_reward']:.1f} | "
              f"Avg Length: {results[level]['avg_episode_length']:.0f}")

    return results


# ============================================================
# SALVATAGGIO RISULTATI (v2.0 — JSON + TXT)
# ============================================================

def save_results(run_dir, mode, train_summary, eval_results, config):
    """
    Salva tutti i risultati in formato strutturato.
    
    v2.0: aggiunto output JSON con schema formale.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ---- JSON strutturato (per analisi programmatica) ----
    json_output = {
        "meta": {
            "experiment_id": os.path.basename(run_dir),
            "mode": mode,
            "simulator": "metadrive",
            "algorithm": "PPO",
            "seed": config.get("seed", SEED),
            "total_timesteps_budget": config.get("total_timesteps", TOTAL_TIMESTEPS),
            "total_timesteps_actual": train_summary["total_steps"],
            "total_episodes": train_summary["total_episodes"],
            "wall_clock_seconds": round(train_summary.get("wall_clock_seconds", 0), 2),
            "timestamp": timestamp_str,
        },
        "config": {
            "ppo": config.get("ppo", PPO_CONFIG),
            "levels": config.get("levels", LEVEL_CONFIGS),
        },
        "training_summary": {
            "cumulative_success_rate": train_summary["cumulative_success_rate"],
            "cumulative_collision_rate": train_summary["cumulative_collision_rate"],
        },
        "level_distribution": train_summary.get("level_distribution", {}),
        "timeseries": train_summary.get("timeseries", []),
        "evaluation": eval_results,
    }

    if mode == "curriculum":
        json_output["config"]["curriculum"] = config.get("curriculum", CURRICULUM_CONFIG)
        curriculum = train_summary.get("curriculum", {})
        json_output["training_summary"]["final_level"] = curriculum.get("final_level", "N/A")
        json_output["training_summary"]["levels_completed"] = curriculum.get("levels_completed", 0)
        json_output["training_summary"]["total_levels"] = curriculum.get("total_levels", 3)
        json_output["curriculum_history"] = curriculum.get("promotion_history", [])

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)

    # ---- Config YAML ----
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ---- Report leggibile TXT ----
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"REPORT: {mode.upper()} TRAINING v2.0\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"Status: COMPLETATO\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total steps: {train_summary['total_steps']:,}\n")
        f.write(f"Total episodes: {train_summary['total_episodes']}\n")
        f.write(f"Wall clock: {train_summary.get('wall_clock_seconds', 0):.0f}s\n")
        f.write(f"Cumulative success rate: {train_summary['cumulative_success_rate']:.1%}\n")
        f.write(f"Cumulative collision rate: {train_summary['cumulative_collision_rate']:.1%}\n\n")

        if mode == "batch":
            f.write("Level distribution (blocchi):\n")
            for lv, count in train_summary.get("level_distribution", {}).items():
                f.write(f"  {lv}: {count}\n")
        elif mode == "curriculum":
            curriculum = train_summary.get("curriculum", {})
            f.write(f"Final level reached: {curriculum.get('final_level', 'N/A')}\n")
            f.write(f"Levels completed: {curriculum.get('levels_completed', 0)}/{curriculum.get('total_levels', 3)}\n")
            f.write(f"Promotion threshold: {curriculum.get('promotion_threshold', 'N/A')}\n")
            f.write(f"Collision threshold: {curriculum.get('collision_threshold', 'N/A')}\n")
            f.write(f"Min timesteps: {curriculum.get('min_timesteps', 'N/A')}\n")
            f.write(f"Replay ratio: {curriculum.get('replay_ratio', 'N/A')}\n\n")
            for promo in curriculum.get("promotion_history", []):
                f.write(f"Promotion: {promo['from']} -> {promo['to']} "
                        f"(after {promo.get('timesteps_on_level', 'N/A')} steps, "
                        f"{promo['episodes_on_level']} episodes, "
                        f"SR: {promo['success_rate_at_promotion']:.1%}, "
                        f"CR: {promo.get('collision_rate_at_promotion', 'N/A')})\n")

        f.write(f"\n{'=' * 60}\n")
        f.write(f"VALUTAZIONE FINALE ({EVAL_EPISODES} episodi per livello)\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"{'Livello':<10} {'Success':<12} {'Collision':<12} {'Avg Reward':<12} {'Avg Length':<12}\n")
        f.write(f"{'-' * 60}\n")
        for level, metrics in eval_results.items():
            f.write(f"{level:<10} {metrics['success_rate']:<12.1%} "
                    f"{metrics['collision_rate']:<12.1%} "
                    f"{metrics['avg_reward']:<12.1f} "
                    f"{metrics['avg_episode_length']:<12.0f}\n")

    print(f"\nRisultati salvati in: {run_dir}")
    print(f"  JSON: {json_path}")


# ============================================================
# MAIN
# ============================================================

def run_experiment(mode, run_dir):
    """Esegue un singolo esperimento (batch o curriculum)."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(run_dir, f"{mode}_run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    set_global_seed(SEED)

    config = {
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "block_size": BLOCK_SIZE,
        "eval_episodes": EVAL_EPISODES,
        "ppo": PPO_CONFIG,
        "levels": LEVEL_CONFIGS,
        "mode": mode,
    }

    if mode == "curriculum":
        config["curriculum"] = CURRICULUM_CONFIG

    # Training
    print("\n" + "#" * 60)
    print(f"  ESPERIMENTO: {mode.upper()} v2.0")
    print(f"  Directory: {experiment_dir}")
    print(f"  Eval episodes: {EVAL_EPISODES}")
    print("#" * 60)

    if mode == "batch":
        model, train_summary = train_batch(
            TOTAL_TIMESTEPS, BLOCK_SIZE, PPO_CONFIG, experiment_dir
        )
    elif mode == "curriculum":
        model, train_summary = train_curriculum(
            TOTAL_TIMESTEPS, BLOCK_SIZE, PPO_CONFIG, CURRICULUM_CONFIG, experiment_dir
        )
    else:
        raise ValueError(f"Mode non valido: {mode}. Usa 'batch' o 'curriculum'.")

    # Valutazione su tutte le mappe + test
    print("\n" + "=" * 60)
    print(f"VALUTAZIONE FINALE ({EVAL_EPISODES} episodi per livello)")
    print("=" * 60)

    eval_levels = ["easy", "medium", "hard", "test"]
    eval_results = evaluate_model(model, eval_levels)

    # Salva tutto
    save_results(experiment_dir, mode, train_summary, eval_results, config)

    return experiment_dir, train_summary, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esperimento Curriculum vs Batch v2.0")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["batch", "curriculum", "both"],
                        help="Modalita: 'batch', 'curriculum', o 'both'")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override budget totale di step")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Override numero episodi valutazione (default 50)")
    args = parser.parse_args()

    if args.timesteps:
        TOTAL_TIMESTEPS = args.timesteps
    if args.eval_episodes:
        EVAL_EPISODES = args.eval_episodes

    run_dir = "experiments"

    if args.mode == "both":
        print("\n" + "#" * 60)
        print("LANCIO ENTRAMBI GLI ESPERIMENTI v2.0")
        print(f"Budget per esperimento: {TOTAL_TIMESTEPS:,} step")
        print(f"Eval episodes: {EVAL_EPISODES}")
        print("#" * 60)

        # Batch
        batch_dir, batch_summary, batch_eval = run_experiment("batch", run_dir)

        # Curriculum
        curriculum_dir, curriculum_summary, curriculum_eval = run_experiment("curriculum", run_dir)

        # Confronto finale a schermo
        print("\n\n" + "=" * 60)
        print("  CONFRONTO FINALE: BATCH vs CURRICULUM")
        print("=" * 60)

        print(f"\n{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} {'Batch CR':<12} {'Curric CR':<12}")
        print("-" * 60)

        for level in ["easy", "medium", "hard", "test"]:
            b_sr = batch_eval[level]["success_rate"]
            c_sr = curriculum_eval[level]["success_rate"]
            b_cr = batch_eval[level]["collision_rate"]
            c_cr = curriculum_eval[level]["collision_rate"]
            print(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} {b_cr:<12.1%} {c_cr:<12.1%}")

        # Training summary
        print(f"\nTraining Summary:")
        print(f"  Batch - Episodes: {batch_summary['total_episodes']}, "
              f"SR: {batch_summary['cumulative_success_rate']:.1%}, "
              f"CR: {batch_summary['cumulative_collision_rate']:.1%}")
        print(f"  Curriculum - Episodes: {curriculum_summary['total_episodes']}, "
              f"SR: {curriculum_summary['cumulative_success_rate']:.1%}, "
              f"CR: {curriculum_summary['cumulative_collision_rate']:.1%}")

        if "curriculum" in curriculum_summary:
            curr = curriculum_summary["curriculum"]
            print(f"  Curriculum reached: {curr['final_level']} "
                  f"({curr['levels_completed']}/{curr['total_levels']} levels)")
            if curr.get("promotion_history"):
                for p in curr["promotion_history"]:
                    print(f"    {p['from']}->{p['to']}: "
                          f"SR={p['success_rate_at_promotion']:.1%}, "
                          f"steps={p.get('timesteps_on_level', 'N/A')}")

        # Salva confronto
        comparison_path = os.path.join(run_dir, "batch_vs_curriculum_comparison.txt")
        with open(comparison_path, "w") as f:
            f.write("CONFRONTO BATCH vs CURRICULUM v2.0\n")
            f.write(f"Budget per esperimento: {TOTAL_TIMESTEPS:,} step\n")
            f.write(f"Eval episodes: {EVAL_EPISODES}\n")
            f.write(f"Seed: {SEED}\n\n")
            f.write(f"{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} {'Batch CR':<12} {'Curric CR':<12}\n")
            f.write(f"{'-'*58}\n")
            for level in ["easy", "medium", "hard", "test"]:
                b_sr = batch_eval[level]["success_rate"]
                c_sr = curriculum_eval[level]["success_rate"]
                b_cr = batch_eval[level]["collision_rate"]
                c_cr = curriculum_eval[level]["collision_rate"]
                f.write(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} {b_cr:<12.1%} {c_cr:<12.1%}\n")

        print(f"\nConfronto salvato in: {comparison_path}")

    else:
        run_experiment(args.mode, run_dir)

    print("\n\nProssimi step:")
    print("1. TensorBoard: tensorboard --logdir=experiments/")
    print("2. Confronto dettagliato: python evaluation/compare_results.py")
