"""
Settimana 2 - Esperimento Curriculum vs Batch su MetaDrive
==========================================================
Lancia i due approcci di training e salva i risultati per il confronto.

Esegui con:
    python ./metadrive_prototype/training/train_experiment.py --mode batch
    python ./metadrive_prototype/training/train_experiment.py --mode curriculum
    python ./metadrive_prototype/training/train_experiment.py --mode both

Quick validation (500K step per verificare che il curriculum funzioni):
    python ./metadrive_prototype/training/train_experiment.py --mode curriculum --timesteps 500000

Confronta su TensorBoard:
    tensorboard --logdir=metadrive_prototype/experiments/

Confronta risultati:
    python ./metadrive_prototype/scripts/compare_results.py --batch metadrive_prototype/experiments/batch/batch_run_XXXX_XXXX/results.json --curriculum metadrive_prototype/experiments/curriculum/curriculum_run_XXXX_XXXX/results.json

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
import math
import random
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Aggiungi la root del progetto al path per gli import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
EXPERIMENTS_DIR_HINT = "metadrive_prototype/experiments"
COMPARE_SCRIPT_HINT = "metadrive_prototype/scripts/compare_results.py"

from envs.multi_level_env import (
    create_env,
    LEVEL_CONFIGS,
    EpisodeTracker,
    CurriculumManager,
)
from training.common import set_global_seed, PPO_CONFIG_BASE, episode_outcome


# ============================================================
# COSTANTI GLOBALI
# ============================================================

SEED = 42
DEVICE = "cpu"

# Budget totale identico per entrambi gli approcci
TOTAL_TIMESTEPS = 1_500_000

# Dimensione di ogni blocco di training
# Tra un blocco e l'altro possiamo cambiare ambiente

# FINETUNING RUN 1
BLOCK_SIZE = 25_000 # 50_000 -> 25_000

# Episodi per la valutazione finale
# v2.0: aumentato da 20 a 50 per affidabilita' statistica
EVAL_EPISODES = 50

# PPO — identico per batch e curriculum (variabile sperimentale isolata)
PPO_CONFIG = {**PPO_CONFIG_BASE, "verbose": 0, "device": DEVICE}

# FINETUNING RUN 1 + RUN 2 + RUN 3
# Curriculum v2.0
CURRICULUM_CONFIG = {
    "promotion_threshold": 0.55,    # 0.6 -> 0.55
    "collision_threshold": 0.35,    # 0.3 -> 0.35
    "min_episodes": 50,
    "min_timesteps": 150_000,       # 200_000 -> 150_000
    "window_size": 50,
    "replay_ratio": 0.10,           # 0.20 -> 0.10 # 0.0 Per spegnere replay, 0.2 per 20% blocchi di revisione # 0.0 Per TEST ON vs OFF
    "max_blocks_without_replay": 4  # 2 -> 4 # v2.1: forzatura replay anti-forgetting
}

# Cap valutazione per evitare stalli (v2.1)
MAX_CAP = 3000  # max step per episodio in valutazione

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
        self.total_blocks = math.ceil(total_steps / block_size)
        self.start_time = time.time()
        self.block_times = []  # durata di ogni blocco per stima ETA

    def format_time(self, seconds):
        """Formatta secondi in ore:minuti:secondi."""
        if seconds <= 0:
            return "0h00m00s"
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
        remaining_steps = max(0, self.total_steps - steps_done)
        remaining_blocks = remaining_steps / self.block_size
        eta_seconds = max(0.0, avg_block_time * remaining_blocks)
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
    Aggiorna i tracker ad ogni fine episodio e raccoglie
    metriche di reward/lunghezza episodio per blocco.
    """

    def __init__(
        self,
        global_tracker,
        level_name="unknown",
        promotion_tracker=None,
        track_promotion=False,
    ):
        super().__init__()
        self.global_tracker = global_tracker
        self.promotion_tracker = promotion_tracker
        self.track_promotion = track_promotion
        self.level_name = level_name
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0
        self.block_rewards = []
        self.block_episode_lengths = []

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue

            info = self.locals["infos"][i]

            # Sempre aggiorna metriche globali del run.
            self.global_tracker.record(info)

            # Aggiorna metriche di promozione solo su blocchi non-replay.
            if self.track_promotion and self.promotion_tracker is not None:
                self.promotion_tracker.record(info)

            self.block_episodes += 1

            success, collision = episode_outcome(info)
            if success:
                self.block_successes += 1
            if collision:
                self.block_collisions += 1

            # Metriche episodio dal Monitor wrapper (SB3).
            episode_info = info.get("episode")
            if episode_info is not None:
                reward = episode_info.get("r")
                ep_len = episode_info.get("l")
                if reward is not None:
                    self.block_rewards.append(float(reward))
                if ep_len is not None:
                    self.block_episode_lengths.append(float(ep_len))

            # Log su TensorBoard
            if self.global_tracker.total_episodes % 10 == 0:
                self.logger.record("thesis/success_rate",
                                   self.global_tracker.cumulative_success_rate)
                self.logger.record("thesis/collision_rate",
                                   self.global_tracker.cumulative_collision_rate)
                self.logger.record("thesis/window_success_rate",
                                   self.global_tracker.window_success_rate)
                self.logger.record("thesis/window_collision_rate",
                                   self.global_tracker.window_collision_rate)
                self.logger.record("thesis/current_level", self.level_name)

                if self.promotion_tracker is not None:
                    self.logger.record("thesis/promotion_window_success_rate",
                                       self.promotion_tracker.window_success_rate)
                    self.logger.record("thesis/promotion_window_collision_rate",
                                       self.promotion_tracker.window_collision_rate)
        return True

    @property
    def block_reward_mean(self):
        if not self.block_rewards:
            return None
        return float(np.mean(self.block_rewards))

    @property
    def block_reward_std(self):
        if not self.block_rewards:
            return None
        return float(np.std(self.block_rewards))

    @property
    def block_ep_length_mean(self):
        if not self.block_episode_lengths:
            return None
        return float(np.mean(self.block_episode_lengths))

    def reset_block_stats(self):
        """Resetta i contatori del blocco corrente."""
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0
        self.block_rewards = []
        self.block_episode_lengths = []


# ============================================================
# TIMESERIES COLLECTOR — Raccoglie dati per JSON output
# ============================================================

class TimeseriesCollector:
    """Raccoglie snapshot delle metriche ad ogni blocco per il JSON finale."""

    def __init__(self):
        self.data = []

    def record(self, timestep, episode, level, tracker,
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
    print(f"Blocchi totali: {math.ceil(total_steps / block_size)}")
    print("=" * 60)

    levels = list(LEVEL_CONFIGS.keys())
    tracker = EpisodeTracker(window_size=50)
    timeseries = TimeseriesCollector()
    progress = ProgressTracker("batch", total_steps, block_size)
    steps_done = 0
    model = None
    block_num = 0
    level_history = []
    level_block_counts = {lv: 0 for lv in levels}
    level_timesteps = {lv: 0 for lv in levels}
    training_status = "COMPLETATO"

    _cycle = list(levels)  # inizializzazione prima del while
    while steps_done < total_steps:
        block_start_steps = steps_done
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
        level_block_counts[level] = level_block_counts.get(level, 0) + 1
        progress.block_start()

        env = None
        callback = None
        try:
            # Crea ambiente per questo livello
            env = create_env(level)

            if model is None:
                model = PPO("MlpPolicy", env, tensorboard_log=run_dir, **ppo_config)
            else:
                model.set_env(env)

            callback = ExperimentCallback(
                global_tracker=tracker,
                level_name=level,
            )

            model.learn(
                total_timesteps=current_block,
                callback=[callback],
                reset_num_timesteps=False,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            training_status = "INTERROTTO"
            if model is not None:
                steps_done = int(model.num_timesteps)
            print("\nTraining interrotto da tastiera.")
            if env is not None:
                env.close()
            break
        except Exception as e:
            training_status = "ERRORE"
            if model is not None:
                steps_done = int(model.num_timesteps)
            print(f"ERRORE nel blocco {block_num}: {e}")
            import traceback
            traceback.print_exc()
            if env is not None:
                env.close()
            break

        progress.block_end()
        steps_done = int(model.num_timesteps)
        current_block = max(0, steps_done - block_start_steps)
        level_timesteps[level] = level_timesteps.get(level, 0) + current_block

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
            reward_mean=callback.block_reward_mean,
            reward_std=callback.block_reward_std,
            ep_length_mean=callback.block_ep_length_mean,
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
    summary["level_distribution"] = level_block_counts  # legacy: blocchi per livello
    summary["level_distribution_blocks"] = level_block_counts
    summary["level_distribution_timesteps"] = level_timesteps
    summary["wall_clock_seconds"] = time.time() - progress.start_time
    summary["timeseries"] = timeseries.data
    summary["status"] = training_status

    return model, summary, training_status


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
    print(
        "Max blocchi senza replay: "
        f"{curriculum_config.get('max_blocks_without_replay', 2)}"
    )
    print("=" * 60)

    global_tracker = EpisodeTracker(window_size=curriculum_config["window_size"])
    promotion_tracker = EpisodeTracker(window_size=curriculum_config["window_size"])
    manager = CurriculumManager(
        promotion_threshold=curriculum_config["promotion_threshold"],
        collision_threshold=curriculum_config["collision_threshold"],
        min_episodes=curriculum_config["min_episodes"],
        min_timesteps=curriculum_config["min_timesteps"],
        replay_ratio=curriculum_config["replay_ratio"],
        max_blocks_without_replay=curriculum_config.get("max_blocks_without_replay", 2),
        window_size=curriculum_config["window_size"],
    )

    timeseries = TimeseriesCollector()
    progress = ProgressTracker("curriculum", total_steps, block_size)
    steps_done = 0
    model = None
    block_num = 0
    current_env = None
    current_env_level = None  # quale livello ha l'env aperto attualmente
    training_status = "COMPLETATO"

    # Tracking distribuzione livelli
    level_timesteps = {lv: 0 for lv in manager.levels}
    level_block_counts = {lv: 0 for lv in manager.levels}

    while steps_done < total_steps:
        block_start_steps = steps_done
        block_num += 1
        remaining = total_steps - steps_done
        current_block = min(block_size, remaining)

        # Determina il livello per questo blocco (potrebbe essere replay)
        block_level, is_replay = manager.get_block_level()
        level_block_counts[block_level] = level_block_counts.get(block_level, 0) + 1

        progress.block_start()

        pending_env = None
        callback = None
        try:
            # Crea/cambia ambiente se necessario
            if current_env_level != block_level:
                if current_env is not None:
                    current_env.close()
                    current_env = None
                    current_env_level = None

                pending_env = create_env(block_level)

                if model is None:
                    model = PPO("MlpPolicy", pending_env, tensorboard_log=run_dir, **ppo_config)
                else:
                    model.set_env(pending_env)

                current_env = pending_env
                current_env_level = block_level
                pending_env = None

            callback = ExperimentCallback(
                global_tracker=global_tracker,
                promotion_tracker=promotion_tracker,
                track_promotion=(not is_replay),
                level_name=block_level,
            )

            model.learn(
                total_timesteps=current_block,
                callback=[callback],
                reset_num_timesteps=False,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            training_status = "INTERROTTO"
            if model is not None:
                steps_done = int(model.num_timesteps)
            print("\nTraining interrotto da tastiera.")
            if pending_env is not None:
                pending_env.close()
            if current_env is not None:
                current_env.close()
                current_env = None
                current_env_level = None
            break
        except Exception as e:
            training_status = "ERRORE"
            if model is not None:
                steps_done = int(model.num_timesteps)
            print(f"ERRORE nel blocco {block_num}: {e}")
            import traceback
            traceback.print_exc()
            if pending_env is not None:
                pending_env.close()
            if current_env is not None:
                current_env.close()
                current_env = None
                current_env_level = None
            break

        steps_done = int(model.num_timesteps)
        current_block = max(0, steps_done - block_start_steps)
        level_timesteps[block_level] = level_timesteps.get(block_level, 0) + current_block

        # Aggiorna timesteps SOLO sul tracker usato per promozione
        # (i blocchi replay non contano per avanzare livello).
        if not is_replay:
            promotion_tracker.add_timesteps(current_block)

        progress.block_end()

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
            tracker=promotion_tracker,
        )

        # Timeseries
        timeseries.record(
            timestep=steps_done,
            episode=global_tracker.total_episodes,
            level=block_level,
            tracker=global_tracker,
            reward_mean=callback.block_reward_mean,
            reward_std=callback.block_reward_std,
            ep_length_mean=callback.block_ep_length_mean,
            is_replay=is_replay,
        )

        # Controlla promozione (solo se il blocco era sul livello corrente, non replay)
        if not is_replay and manager.should_promote(promotion_tracker):
            old_level = manager.current_level
            new_level = manager.promote(promotion_tracker, global_timestep=steps_done)
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
    summary = global_tracker.summary()
    summary["mode"] = "curriculum"
    summary["total_steps"] = steps_done
    summary["curriculum"] = manager.summary()
    summary["level_distribution"] = level_timesteps  # legacy: timesteps per livello
    summary["level_distribution_blocks"] = level_block_counts
    summary["level_distribution_timesteps"] = level_timesteps
    summary["wall_clock_seconds"] = time.time() - progress.start_time
    summary["timeseries"] = timeseries.data
    summary["status"] = training_status

    return model, summary, training_status


# ============================================================
# VALUTAZIONE SU TUTTE LE MAPPE
# ============================================================

def evaluate_model(model, levels_to_test, n_episodes=None, max_steps_per_episode=MAX_CAP, heartbeat_interval=200):
    """
    Valuta un modello addestrato su una lista di livelli.
    
    v2.0: n_episodes aumentato a EVAL_EPISODES (50) di default.
    v2.1: aggiunto cap max_steps_per_episode in eval per evitare stalli.
    
    Args:
        model: modello PPO addestrato
        levels_to_test: lista di nomi livelli
        n_episodes: episodi per livello (default EVAL_EPISODES)
        max_steps_per_episode: cap di step per episodio in valutazione.
        heartbeat_interval: ogni quanti step stampare heartbeat intra-episodio.
    
    Returns:
        dizionario con le metriche per ogni livello (anche parziali se interrotto)
    """
    if n_episodes is None:
        n_episodes = EVAL_EPISODES

    results = {}

    for level in levels_to_test:
        print(f"\nValutazione su: {level.upper()} ({n_episodes} episodi)...")
        env = create_env(level)

        successes = 0
        collisions = 0
        total_reward = 0.0
        total_steps = 0
        timeouts = 0
        episodes_done = 0
        ep_reward = 0.0
        ep_steps = 0

        def build_metrics(episodes_count):
            denom = max(episodes_count, 1)
            return {
                "success_rate": successes / denom,
                "collision_rate": collisions / denom,
                "avg_reward": round(total_reward / denom, 2),
                "avg_episode_length": round(total_steps / denom, 1),
                "episodes": episodes_count,
                "timeouts": timeouts,
            }

        pbar = None
        try:
            pbar = tqdm(
                total=n_episodes,
                desc=f"Eval {level.upper()}",
                unit="ep",
                leave=True,
            )
            obs = env.reset()

            while episodes_done < n_episodes:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward[0])
                ep_steps += 1

                if heartbeat_interval and ep_steps % heartbeat_interval == 0:
                    tqdm.write(
                        f"[EVAL {level.upper()}] Ep {episodes_done + 1}/{n_episodes} "
                        f"step {ep_steps}..."
                    )

                forced_timeout = (
                    max_steps_per_episode is not None
                    and ep_steps >= max_steps_per_episode
                    and not done[0]
                )

                if done[0] or forced_timeout:
                    episodes_done += 1
                    pbar.update(1)
                    total_reward += ep_reward
                    total_steps += ep_steps

                    if done[0]:
                        episode_info = info[0]
                        success, collision = episode_outcome(episode_info)
                        if success:
                            successes += 1
                        if collision:
                            collisions += 1
                    else:
                        timeouts += 1
                        obs = env.reset()

                    ep_reward = 0.0
                    ep_steps = 0

        except KeyboardInterrupt:
            print("\nValutazione interrotta da tastiera. Salvataggio risultati parziali...")
            if episodes_done > 0:
                results[level] = build_metrics(episodes_done)
                results[level]["partial"] = True
            return results
        except Exception as e:
            print(f"ERRORE durante valutazione su {level.upper()}: {e}")
            import traceback
            traceback.print_exc()
            if episodes_done > 0:
                results[level] = build_metrics(episodes_done)
                results[level]["partial"] = True
            return results
        finally:
            if pbar is not None:
                pbar.close()
            env.close()

        results[level] = build_metrics(episodes_done)
        results[level]["partial"] = False

        timeout_note = f" | Timeouts: {results[level]['timeouts']}" if results[level]["timeouts"] > 0 else ""
        print(f"Success: {results[level]['success_rate']:.1%} | "
              f"Collision: {results[level]['collision_rate']:.1%} | "
              f"Avg Reward: {results[level]['avg_reward']:.1f} | "
              f"Avg Length: {results[level]['avg_episode_length']:.0f}{timeout_note}")

    return results


# ============================================================
# SALVATAGGIO RISULTATI (v2.0 — JSON + TXT)
# ============================================================

def save_results(run_dir, mode, train_summary, eval_results, config, training_status):
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
            "status": training_status,
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
        "level_distribution_blocks": train_summary.get("level_distribution_blocks", {}),
        "level_distribution_timesteps": train_summary.get("level_distribution_timesteps", {}),
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

        f.write(f"Status: {training_status}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total steps: {train_summary['total_steps']:,}\n")
        f.write(f"Total episodes: {train_summary['total_episodes']}\n")
        f.write(f"Wall clock: {train_summary.get('wall_clock_seconds', 0):.0f}s\n")
        f.write(f"Cumulative success rate: {train_summary['cumulative_success_rate']:.1%}\n")
        f.write(f"Cumulative collision rate: {train_summary['cumulative_collision_rate']:.1%}\n\n")

        if mode == "batch":
            f.write("Level distribution (blocchi):\n")
            for lv, count in train_summary.get(
                "level_distribution_blocks", train_summary.get("level_distribution", {})
            ).items():
                f.write(f"  {lv}: {count}\n")
            f.write("\nLevel distribution (timesteps):\n")
            for lv, steps in train_summary.get("level_distribution_timesteps", {}).items():
                f.write(f"  {lv}: {steps:,}\n")
        elif mode == "curriculum":
            curriculum = train_summary.get("curriculum", {})
            f.write(f"Final level reached: {curriculum.get('final_level', 'N/A')}\n")
            f.write(f"Levels completed: {curriculum.get('levels_completed', 0)}/{curriculum.get('total_levels', 3)}\n")
            f.write(f"Promotion threshold: {curriculum.get('promotion_threshold', 'N/A')}\n")
            f.write(f"Collision threshold: {curriculum.get('collision_threshold', 'N/A')}\n")
            f.write(f"Min timesteps: {curriculum.get('min_timesteps', 'N/A')}\n")
            f.write(f"Replay ratio: {curriculum.get('replay_ratio', 'N/A')}\n")
            f.write(f"Max blocchi senza replay: {curriculum.get('max_blocks_without_replay', 'N/A')}\n\n")
            f.write("Level distribution (timesteps):\n")
            for lv, steps in train_summary.get(
                "level_distribution_timesteps", train_summary.get("level_distribution", {})
            ).items():
                f.write(f"  {lv}: {steps:,}\n")
            f.write("Level distribution (blocchi):\n")
            for lv, count in train_summary.get("level_distribution_blocks", {}).items():
                f.write(f"  {lv}: {count}\n")
            f.write("\n")
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

def run_experiment(mode, run_dir, allow_eval_on_incomplete=False):
    """Esegue un singolo esperimento (batch o curriculum)."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(run_dir, mode, f"{mode}_run_{timestamp}")
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
        model, train_summary, training_status = train_batch(
            TOTAL_TIMESTEPS, BLOCK_SIZE, PPO_CONFIG, experiment_dir
        )
    elif mode == "curriculum":
        model, train_summary, training_status = train_curriculum(
            TOTAL_TIMESTEPS, BLOCK_SIZE, PPO_CONFIG, CURRICULUM_CONFIG, experiment_dir
        )
    else:
        raise ValueError(f"Mode non valido: {mode}. Usa 'batch' o 'curriculum'.")

    # Valutazione su tutte le mappe + test
    print("\n" + "=" * 60)
    print(f"VALUTAZIONE FINALE ({EVAL_EPISODES} episodi per livello)")
    print("=" * 60)

    eval_levels = ["easy", "medium", "hard", "test"]
    if model is None:
        print("Training non ha prodotto un modello. Valutazione saltata.")
        eval_results = {}
    elif training_status != "COMPLETATO" and not allow_eval_on_incomplete:
        print(
            f"Training status={training_status}. "
            "Valutazione saltata (usa --eval-on-incomplete per forzare)."
        )
        eval_results = {}
    else:
        eval_results = evaluate_model(model, eval_levels)

    # Salva tutto
    save_results(experiment_dir, mode, train_summary, eval_results, config, training_status)

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
    parser.add_argument("--eval-on-incomplete", action="store_true",
                        help="Esegue la valutazione anche se il training e' INTERROTTO/ERRORE")
    args = parser.parse_args()

    if args.timesteps:
        TOTAL_TIMESTEPS = args.timesteps
    if args.eval_episodes:
        EVAL_EPISODES = args.eval_episodes

    run_dir = EXPERIMENTS_DIR

    if args.mode == "both":
        print("\n" + "#" * 60)
        print("LANCIO ENTRAMBI GLI ESPERIMENTI v2.0")
        print(f"Budget per esperimento: {TOTAL_TIMESTEPS:,} step")
        print(f"Eval episodes: {EVAL_EPISODES}")
        print("#" * 60)

        # Batch
        batch_dir, batch_summary, batch_eval = run_experiment(
            "batch", run_dir, allow_eval_on_incomplete=args.eval_on_incomplete
        )

        # Curriculum
        curriculum_dir, curriculum_summary, curriculum_eval = run_experiment(
            "curriculum", run_dir, allow_eval_on_incomplete=args.eval_on_incomplete
        )

        # Confronto finale a schermo
        print("\n\n" + "=" * 60)
        print("  CONFRONTO FINALE: BATCH vs CURRICULUM")
        print("=" * 60)

        levels = ["easy", "medium", "hard", "test"]
        if all(level in batch_eval for level in levels) and all(level in curriculum_eval for level in levels):
            print(f"\n{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} {'Batch CR':<12} {'Curric CR':<12}")
            print("-" * 60)

            for level in levels:
                b_sr = batch_eval[level]["success_rate"]
                c_sr = curriculum_eval[level]["success_rate"]
                b_cr = batch_eval[level]["collision_rate"]
                c_cr = curriculum_eval[level]["collision_rate"]
                print(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} {b_cr:<12.1%} {c_cr:<12.1%}")
        else:
            print("\nConfronto metriche per livello non disponibile (una delle due run non ha valutazione completa).")

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
            if all(level in batch_eval for level in levels) and all(level in curriculum_eval for level in levels):
                f.write(f"{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} {'Batch CR':<12} {'Curric CR':<12}\n")
                f.write(f"{'-'*58}\n")
                for level in levels:
                    b_sr = batch_eval[level]["success_rate"]
                    c_sr = curriculum_eval[level]["success_rate"]
                    b_cr = batch_eval[level]["collision_rate"]
                    c_cr = curriculum_eval[level]["collision_rate"]
                    f.write(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} {b_cr:<12.1%} {c_cr:<12.1%}\n")
            else:
                f.write("Confronto per livello non disponibile: valutazione incompleta in almeno una run.\n")

        print(f"\nConfronto salvato in: {comparison_path}")

    else:
        run_experiment(args.mode, run_dir, allow_eval_on_incomplete=args.eval_on_incomplete)

    print("\n\nProssimi step:")
    print(f"1. TensorBoard: tensorboard --logdir={EXPERIMENTS_DIR_HINT}")
    print(f"2. Confronto dettagliato: python ./{COMPARE_SCRIPT_HINT}")
