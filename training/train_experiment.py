"""
        Esperimento Curriculum vs Batch su MetaDrive
==========================================================
Lancia i due approcci di training e salva i risultati per il confronto.

Esegui con:
    python ./training/train_experiment.py --mode batch
    python ./training/train_experiment.py --mode curriculum
    python ./training/train_experiment.py --mode both

Confronta su TensorBoard:
    tensorboard --logdir=experiments/

Confronta risultati:
    python ./scripts/compare_results.py --batch experiments/<batch_dir>/results.json --curriculum experiments/<curriculum_dir>/results.json

Output per ogni run:
    - results.json         → dati strutturati con timeseries (contratto stabile per compare_results.py)
    - final_model.zip      → modello SB3 salvato
    - TensorBoard logs     → serie temporali per visualizzazione interattiva
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime

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
# Ogni blocco = 1 datapoint nella timeseries JSON
BLOCK_SIZE = 50_000

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

# Curriculum — soglie di promozione
CURRICULUM_CONFIG = {
    "promotion_threshold": 0.3,
    "min_episodes": 50,
    "window_size": 50,
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
# CALLBACK — Traccia metriche e episodi per ogni blocco
# ============================================================

class ExperimentCallback(BaseCallback):
    """
    Callback che traccia metriche per l'esperimento.
    Aggiorna l'EpisodeTracker ad ogni fine episodio.
    Raccoglie anche reward e episode length per la timeseries JSON.
    """

    def __init__(self, tracker, level_name="unknown"):
        super().__init__()
        self.tracker = tracker
        self.level_name = level_name

        # Contatori per il blocco corrente
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0

        # Reward e episode length per il blocco corrente
        # Servono per calcolare media e std nella timeseries
        self.block_rewards = []
        self.block_episode_lengths = []

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

                # Reward e lunghezza episodio dal Monitor wrapper
                # Monitor salva queste info in "episode" quando l'episodio finisce
                episode_info = info.get("episode")
                if episode_info is not None:
                    self.block_rewards.append(episode_info["r"])
                    self.block_episode_lengths.append(episode_info["l"])

                # Log su TensorBoard ogni 10 episodi
                if self.tracker.total_episodes % 10 == 0:
                    self.logger.record("thesis/success_rate",
                                       self.tracker.cumulative_success_rate)
                    self.logger.record("thesis/collision_rate",
                                       self.tracker.cumulative_collision_rate)
                    self.logger.record("thesis/window_success_rate",
                                       self.tracker.window_success_rate)
                    self.logger.record("thesis/current_level",
                                       self.level_name)
        return True

    def get_block_snapshot(self):
        """
        Ritorna uno snapshot delle metriche del blocco corrente.
        Chiamato alla fine di ogni blocco per popolare la timeseries.
        """
        snapshot = {
            "block_episodes": self.block_episodes,
            "block_successes": self.block_successes,
            "block_collisions": self.block_collisions,
        }

        if len(self.block_rewards) > 0:
            snapshot["reward_mean"] = float(np.mean(self.block_rewards))
            snapshot["reward_std"] = float(np.std(self.block_rewards))
        else:
            snapshot["reward_mean"] = None
            snapshot["reward_std"] = None

        if len(self.block_episode_lengths) > 0:
            snapshot["episode_length_mean"] = float(np.mean(self.block_episode_lengths))
        else:
            snapshot["episode_length_mean"] = None

        return snapshot

    def reset_block_stats(self):
        """Resetta i contatori del blocco corrente."""
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0
        self.block_rewards = []
        self.block_episode_lengths = []


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
    steps_done = 0
    model = None
    block_num = 0
    level_history = []
    timeseries = []

    # Contatori timestep per livello (per level_distribution)
    level_timesteps = {lv: 0 for lv in levels}
    level_blocks = {lv: 0 for lv in levels}

    while steps_done < total_steps:
        block_num += 1
        remaining = total_steps - steps_done
        current_block = min(block_size, remaining)

        # Scegli livello random
        level = random.choice(levels)
        level_history.append(level)

        print(f"\nBlocco {block_num}: {level.upper()} ({current_block:,} step, "
              f"totale: {steps_done:,}/{total_steps:,})")

        # Crea ambiente per questo livello
        env = create_env(level)

        if model is None:
            # Primo blocco: crea il modello
            model = PPO("MlpPolicy", env, tensorboard_log=run_dir, **ppo_config)
        else:
            # Blocchi successivi: cambia ambiente, mantieni i pesi
            model.set_env(env)

        # Callback
        callback = ExperimentCallback(tracker, level_name=level)

        # Training del blocco
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

        # Aggiorna contatori per livello
        level_timesteps[level] += current_block
        level_blocks[level] += 1

        # Report blocco a schermo
        if callback.block_episodes > 0:
            block_sr = callback.block_successes / callback.block_episodes
            block_cr = callback.block_collisions / callback.block_episodes
            print(f"Episodi: {callback.block_episodes} | "
                  f"Success: {block_sr:.1%} | Collision: {block_cr:.1%}")

        # Snapshot per timeseries JSON
        steps_done += current_block
        block_snapshot = callback.get_block_snapshot()
        timeseries.append({
            "timestep": steps_done,
            "episode": tracker.total_episodes,
            "level": level,
            "success_rate": round(tracker.cumulative_success_rate, 4),
            "collision_rate": round(tracker.cumulative_collision_rate, 4),
            "window_success_rate": round(tracker.window_success_rate, 4),
            "window_collision_rate": round(tracker.window_collision_rate, 4),
            "reward_mean": block_snapshot["reward_mean"],
            "reward_std": block_snapshot["reward_std"],
            "episode_length_mean": block_snapshot["episode_length_mean"],
            "path_efficiency_mean": None,  # Non disponibile in MetaDrive
        })

        env.close()

    # Salva modello finale
    if model is not None:
        model.save(os.path.join(run_dir, "final_model"))

    # Costruisci summary
    summary = {
        "cumulative_success_rate": round(tracker.cumulative_success_rate, 4),
        "cumulative_collision_rate": round(tracker.cumulative_collision_rate, 4),
        "total_episodes": tracker.total_episodes,
        "total_steps": steps_done,
    }

    level_distribution = {
        lv: {"blocks": level_blocks[lv], "timesteps": level_timesteps[lv]}
        for lv in levels
    }

    return model, summary, timeseries, level_distribution, level_history


# ============================================================
# CURRICULUM TRAINING
# ============================================================

def train_curriculum(total_steps, block_size, ppo_config, curriculum_config, run_dir):
    """
    Training Curriculum (Easy -> Medium -> Hard).
    Promuove al livello successivo quando il success rate
    sulla finestra mobile supera la soglia.
    """
    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING (Easy -> Medium -> Hard)")
    print(f"Budget totale: {total_steps:,} step")
    print(f"Soglia promozione: {curriculum_config['promotion_threshold']:.0%}")
    print(f"Episodi minimi: {curriculum_config['min_episodes']}")
    print("=" * 60)

    tracker = EpisodeTracker(window_size=curriculum_config["window_size"])
    manager = CurriculumManager(
        promotion_threshold=curriculum_config["promotion_threshold"],
        min_episodes=curriculum_config["min_episodes"],
    )

    steps_done = 0
    model = None
    block_num = 0
    current_env = None
    timeseries = []
    curriculum_history = []

    # Contatori timestep per livello
    level_timesteps = {lv: 0 for lv in manager.levels}

    while steps_done < total_steps:
        block_num += 1
        remaining = total_steps - steps_done
        current_block = min(block_size, remaining)
        level = manager.current_level

        print(f"\nBlocco {block_num}: {level.upper()} ({current_block:,} step, "
              f"totale: {steps_done:,}/{total_steps:,})")

        # Crea ambiente se necessario (primo blocco o dopo promozione)
        if current_env is None:
            current_env = create_env(level)

            if model is None:
                model = PPO("MlpPolicy", current_env, tensorboard_log=run_dir, **ppo_config)
            else:
                model.set_env(current_env)

        # Callback
        callback = ExperimentCallback(tracker, level_name=level)

        # Training del blocco
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

        # Aggiorna contatori
        level_timesteps[level] += current_block
        steps_done += current_block

        # Report blocco a schermo
        if callback.block_episodes > 0:
            block_sr = callback.block_successes / callback.block_episodes
            block_cr = callback.block_collisions / callback.block_episodes
            print(f"Episodi: {callback.block_episodes} | "
                  f"Success: {block_sr:.1%} | Collision: {block_cr:.1%} | "
                  f"Window SR: {tracker.window_success_rate:.1%}")

        # Snapshot per timeseries JSON
        block_snapshot = callback.get_block_snapshot()
        timeseries.append({
            "timestep": steps_done,
            "episode": tracker.total_episodes,
            "level": level,
            "success_rate": round(tracker.cumulative_success_rate, 4),
            "collision_rate": round(tracker.cumulative_collision_rate, 4),
            "window_success_rate": round(tracker.window_success_rate, 4),
            "window_collision_rate": round(tracker.window_collision_rate, 4),
            "reward_mean": block_snapshot["reward_mean"],
            "reward_std": block_snapshot["reward_std"],
            "episode_length_mean": block_snapshot["episode_length_mean"],
            "path_efficiency_mean": None,  # Non disponibile in MetaDrive
        })

        # Controlla promozione
        if manager.should_promote(tracker):
            old_level = manager.current_level
            new_level = manager.promote(tracker)
            print(f"\n>>> PROMOZIONE: {old_level.upper()} -> {new_level.upper()} <<<")
            print(f"Success rate alla promozione: "
                  f"{manager.promotion_history[-1]['success_rate_at_promotion']:.1%}")

            # Registra promozione con timestep (mancava nel codice originale)
            curriculum_history.append({
                "from": old_level,
                "to": new_level,
                "timestep_at_promotion": steps_done,
                "episode_at_promotion": tracker.total_episodes,
                "success_rate_at_promotion": round(
                    manager.promotion_history[-1]["success_rate_at_promotion"], 4
                ),
            })

            # Chiudi vecchio ambiente, ne creeremo uno nuovo al prossimo blocco
            current_env.close()
            current_env = None

    # Chiudi ambiente se ancora aperto
    if current_env is not None:
        current_env.close()

    # Salva modello finale
    if model is not None:
        model.save(os.path.join(run_dir, "final_model"))

    # Costruisci summary
    summary = {
        "cumulative_success_rate": round(tracker.cumulative_success_rate, 4),
        "cumulative_collision_rate": round(tracker.cumulative_collision_rate, 4),
        "total_episodes": tracker.total_episodes,
        "total_steps": steps_done,
        "final_level": manager.current_level,
        "levels_completed": manager.current_index,
        "total_levels": len(manager.levels),
    }

    level_distribution = {
        lv: {"timesteps": level_timesteps[lv]}
        for lv in manager.levels
    }

    return model, summary, timeseries, curriculum_history, level_distribution


# ============================================================
# VALUTAZIONE SU TUTTE LE MAPPE
# ============================================================

def evaluate_model(model, levels_to_test, n_episodes=20):
    """
    Valuta un modello addestrato su una lista di livelli.
    Ritorna un dizionario con le metriche per ogni livello.
    """
    results = {}

    for level in levels_to_test:
        print(f"\nValutazione su: {level.upper()} ({n_episodes} episodi)...")
        env = create_env(level)

        successes = 0
        collisions = 0
        total_reward = 0
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
            "success_rate": round(successes / n_episodes, 4),
            "collision_rate": round(collisions / n_episodes, 4),
            "avg_reward": round(total_reward / n_episodes, 2),
            "avg_episode_length": round(float(np.mean(episode_lengths)), 1),
            "episodes": n_episodes,
        }

        print(f"Success: {results[level]['success_rate']:.1%} | "
              f"Collision: {results[level]['collision_rate']:.1%} | "
              f"Avg Reward: {results[level]['avg_reward']:.1f} | "
              f"Avg Length: {results[level]['avg_episode_length']:.0f}")

    return results


# ============================================================
# SALVATAGGIO RISULTATI — JSON (contratto stabile)
# ============================================================

def save_results_json(run_dir, mode, summary, timeseries, evaluation,
                      config, wall_clock_seconds, timestamp_start, timestamp_end,
                      curriculum_history=None, level_distribution=None):
    """
    Salva i risultati in formato JSON strutturato.
    Questo e' il contratto stabile tra training e compare_results.py.
    Lo stesso schema verra' usato anche per CARLA.
    """
    results = {
        "meta": {
            "experiment_id": os.path.basename(run_dir),
            "mode": mode,
            "simulator": "metadrive",
            "algorithm": "PPO",
            "seed": config["seed"],
            "total_timesteps_budget": config["total_timesteps"],
            "total_timesteps_actual": summary["total_steps"],
            "total_episodes": summary["total_episodes"],
            "wall_clock_seconds": round(wall_clock_seconds, 2),
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
        },
        "config": {
            "ppo": config["ppo"],
            "levels": config["levels"],
        },
        "training_summary": {
            "cumulative_success_rate": summary["cumulative_success_rate"],
            "cumulative_collision_rate": summary["cumulative_collision_rate"],
        },
        "curriculum_history": curriculum_history or [],
        "level_distribution": level_distribution or {},
        "timeseries": timeseries,
        "evaluation": evaluation,
    }

    # Aggiungi config curriculum se presente
    if "curriculum" in config:
        results["config"]["curriculum"] = config["curriculum"]

    # Aggiungi info extra curriculum nel summary
    if mode == "curriculum":
        results["training_summary"]["final_level"] = summary.get("final_level")
        results["training_summary"]["levels_completed"] = summary.get("levels_completed")
        results["training_summary"]["total_levels"] = summary.get("total_levels")

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"JSON salvato: {json_path}")
    return json_path


# ============================================================
# MAIN
# ============================================================

def run_experiment(mode, run_dir):
    """Esegue un singolo esperimento (batch o curriculum)."""

    timestamp_start = datetime.now().isoformat(timespec="seconds")
    wall_clock_start = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(run_dir, mode, f"{mode}_run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    set_global_seed(SEED)

    config = {
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "block_size": BLOCK_SIZE,
        "ppo": PPO_CONFIG,
        "levels": LEVEL_CONFIGS,
        "mode": mode,
    }

    if mode == "curriculum":
        config["curriculum"] = CURRICULUM_CONFIG

    # Training
    print("\n" + "#" * 60)
    print(f"ESPERIMENTO: {mode.upper()}")
    print(f"Directory: {experiment_dir}")
    print("#" * 60)

    if mode == "batch":
        model, summary, timeseries, level_distribution, level_history = train_batch(
            TOTAL_TIMESTEPS, BLOCK_SIZE, PPO_CONFIG, experiment_dir
        )
        curriculum_history = None
    elif mode == "curriculum":
        model, summary, timeseries, curriculum_history, level_distribution = train_curriculum(
            TOTAL_TIMESTEPS, BLOCK_SIZE, PPO_CONFIG, CURRICULUM_CONFIG, experiment_dir
        )
    else:
        raise ValueError(f"Mode non valido: {mode}. Usa 'batch' o 'curriculum'.")

    # Valutazione su tutte le mappe + test
    print("\n" + "=" * 60)
    print("VALUTAZIONE FINALE")
    print("=" * 60)

    eval_levels = ["easy", "medium", "hard", "test"]
    evaluation = evaluate_model(model, eval_levels, n_episodes=20)

    # Timing
    wall_clock_end = time.time()
    wall_clock_seconds = wall_clock_end - wall_clock_start
    timestamp_end = datetime.now().isoformat(timespec="seconds")

    # Salva JSON strutturato (contratto stabile per compare_results.py)
    save_results_json(
        run_dir=experiment_dir,
        mode=mode,
        summary=summary,
        timeseries=timeseries,
        evaluation=evaluation,
        config=config,
        wall_clock_seconds=wall_clock_seconds,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        curriculum_history=curriculum_history if mode == "curriculum" else None,
        level_distribution=level_distribution,
    )

    return experiment_dir, summary, evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esperimento Curriculum vs Batch")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["batch", "curriculum", "both"],
                        help="Modalita: 'batch', 'curriculum', o 'both'")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override budget totale di step")
    args = parser.parse_args()

    if args.timesteps:
        TOTAL_TIMESTEPS = args.timesteps

    run_dir = "experiments"

    if args.mode == "both":
        # Lancia entrambi in sequenza
        print("\n" + "#" * 60)
        print("LANCIO ENTRAMBI GLI ESPERIMENTI")
        print(f"Budget per esperimento: {TOTAL_TIMESTEPS:,} step")
        print("#" * 60)

        # Batch
        batch_dir, batch_summary, batch_eval = run_experiment("batch", run_dir)

        # Curriculum
        curriculum_dir, curriculum_summary, curriculum_eval = run_experiment("curriculum", run_dir)

        # Confronto finale a schermo
        print("\n\n" + "=" * 60)
        print("CONFRONTO FINALE: BATCH vs CURRICULUM")
        print("=" * 60)

        print(f"\n{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} "
              f"{'Batch CR':<12} {'Curric CR':<12}")
        print("-" * 60)

        for level in ["easy", "medium", "hard", "test"]:
            b_sr = batch_eval[level]["success_rate"]
            c_sr = curriculum_eval[level]["success_rate"]
            b_cr = batch_eval[level]["collision_rate"]
            c_cr = curriculum_eval[level]["collision_rate"]

            print(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} {b_cr:<12.1%} {c_cr:<12.1%}")

        # Training summary
        print(f"\nTraining Summary:")
        print(f"Batch - Episodes: {batch_summary['total_episodes']}, "
              f"SR: {batch_summary['cumulative_success_rate']:.1%}")
        print(f"Curriculum - Episodes: {curriculum_summary['total_episodes']}, "
              f"SR: {curriculum_summary['cumulative_success_rate']:.1%}")

        if curriculum_summary.get("final_level"):
            print(f"Curriculum reached: {curriculum_summary['final_level']} "
                  f"({curriculum_summary['levels_completed']}/"
                  f"{curriculum_summary['total_levels']} levels)")

        # Indica come confrontare
        print(f"\nConfronto dettagliato con grafici:")
        print(f"python scripts/compare_results.py "
              f"--batch {batch_dir}/results.json "
              f"--curriculum {curriculum_dir}/results.json")

    else:
        experiment_dir, summary, evaluation = run_experiment(args.mode, run_dir)
        print(f"\nProssimi step:")
        print(f"1. TensorBoard: tensorboard --logdir={experiment_dir}")
        print(f"2. Lancia l'altro esperimento e poi confronta con:")
        print(f"python scripts/compare_results.py "
              f"--batch experiments/<batch_dir>/results.json "
              f"--curriculum experiments/<curriculum_dir>/results.json")