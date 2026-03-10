"""
Settimana 2 - Esperimento Curriculum vs Batch su MetaDrive
==========================================================
Lancia i due approcci di training e salva i risultati per il confronto.

Esegui con:
    python ./training/train_experiment.py --mode batch
    python ./training/train_experiment.py --mode curriculum
    python ./training/train_experiment.py --mode both

Confronta su TensorBoard:
    tensorboard --logdir=experiments/

Confronta risultati:
    python ./evaluation/compare_results.py
"""

import os
import sys
import yaml
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
                    self.logger.record("thesis/current_level",
                                       self.level_name)
        return True

    def reset_block_stats(self):
        """Resetta i contatori del blocco corrente."""
        self.block_episodes = 0
        self.block_successes = 0
        self.block_collisions = 0


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

        # Report blocco
        if callback.block_episodes > 0:
            block_sr = callback.block_successes / callback.block_episodes
            block_cr = callback.block_collisions / callback.block_episodes
            print(f"Episodi: {callback.block_episodes} | "
                  f"Success: {block_sr:.1%} | Collision: {block_cr:.1%}")

        env.close()
        steps_done += current_block

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

    return model, summary


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

        steps_done += current_block

        # Report blocco
        if callback.block_episodes > 0:
            block_sr = callback.block_successes / callback.block_episodes
            block_cr = callback.block_collisions / callback.block_episodes
            print(f"Episodi: {callback.block_episodes} | "
                  f"Success: {block_sr:.1%} | Collision: {block_cr:.1%} | "
                  f"Window SR: {tracker.window_success_rate:.1%}")

        # Controlla promozione
        if manager.should_promote(tracker):
            old_level = manager.current_level
            new_level = manager.promote(tracker)
            print(f"\n>>> PROMOZIONE: {old_level.upper()} -> {new_level.upper()} <<<")
            print(f"Success rate alla promozione: "
                  f"{manager.promotion_history[-1]['success_rate_at_promotion']:.1%}")

            # Chiudi vecchio ambiente, ne creeremo uno nuovo al prossimo blocco
            current_env.close()
            current_env = None

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

    return model, summary


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

        obs = env.reset()
        episodes_done = 0
        ep_reward = 0

        while episodes_done < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]

            if done[0]:
                episodes_done += 1
                total_reward += ep_reward
                ep_reward = 0

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
            "avg_reward": total_reward / n_episodes,
            "episodes": n_episodes,
        }

        print(f"Success: {results[level]['success_rate']:.1%} | "
              f"Collision: {results[level]['collision_rate']:.1%} | "
              f"Avg Reward: {results[level]['avg_reward']:.1f}")

    return results


# ============================================================
# SALVATAGGIO RISULTATI
# ============================================================

def save_results(run_dir, mode, train_summary, eval_results, config):
    """Salva tutti i risultati in un formato strutturato."""

    # Config completa
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Report leggibile
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"REPORT: {mode.upper()} TRAINING\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"Status: COMPLETATO\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total steps: {train_summary['total_steps']:,}\n")
        f.write(f"Total episodes: {train_summary['total_episodes']}\n")
        f.write(f"Cumulative success rate: {train_summary['cumulative_success_rate']:.1%}\n")
        f.write(f"Cumulative collision rate: {train_summary['cumulative_collision_rate']:.1%}\n\n")

        # Info specifica per modalita
        if mode == "batch":
            f.write("Level distribution:\n")
            for lv, count in train_summary.get("level_distribution", {}).items():
                f.write(f"  {lv}: {count} blocchi\n")
        elif mode == "curriculum":
            curriculum = train_summary.get("curriculum", {})
            f.write(f"Final level reached: {curriculum.get('final_level', 'N/A')}\n")
            f.write(f"Levels completed: {curriculum.get('levels_completed', 0)}/{curriculum.get('total_levels', 3)}\n\n")
            for promo in curriculum.get("promotion_history", []):
                f.write(f"Promotion: {promo['from']} -> {promo['to']} "
                        f"(after {promo['episodes_on_level']} episodes, "
                        f"SR: {promo['success_rate_at_promotion']:.1%})\n")

        f.write(f"\n{'=' * 60}\n")
        f.write("VALUTAZIONE FINALE\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"{'Livello':<10} {'Success':<12} {'Collision':<12} {'Avg Reward':<12}\n")
        f.write(f"{'-' * 60}\n")
        for level, metrics in eval_results.items():
            f.write(f"{level:<10} {metrics['success_rate']:<12.1%} "
                    f"{metrics['collision_rate']:<12.1%} "
                    f"{metrics['avg_reward']:<12.1f}\n")

    # Risultati strutturati (per compare_results.py)
    structured = {
        "mode": mode,
        "training": train_summary,
        "evaluation": eval_results,
        "config": config,
    }
    with open(os.path.join(run_dir, "results.yaml"), "w") as f:
        yaml.dump(structured, f, default_flow_style=False)

    print(f"\nRisultati salvati in: {run_dir}")


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
        "ppo": PPO_CONFIG,
        "levels": LEVEL_CONFIGS,
        "mode": mode,
    }

    if mode == "curriculum":
        config["curriculum"] = CURRICULUM_CONFIG

    # Training
    print("\n" + "#" * 60)
    print(f"  ESPERIMENTO: {mode.upper()}")
    print(f"Directory: {experiment_dir}")
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
    print("VALUTAZIONE FINALE")
    print("=" * 60)

    eval_levels = ["easy", "medium", "hard", "test"]
    eval_results = evaluate_model(model, eval_levels, n_episodes=20)

    # Salva tutto
    save_results(experiment_dir, mode, train_summary, eval_results, config)

    return experiment_dir, train_summary, eval_results


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
        print("  CONFRONTO FINALE: BATCH vs CURRICULUM")
        print("=" * 60)

        print(f"\n{'Livello':<10} {'Batch SR':<12} {'Curric SR':<12} {'Batch CR':<12} {'Curric CR':<12}")
        print("-" * 60)

        for level in ["easy", "medium", "hard", "test"]:
            b_sr = batch_eval[level]["success_rate"]
            c_sr = curriculum_eval[level]["success_rate"]
            b_cr = batch_eval[level]["collision_rate"]
            c_cr = curriculum_eval[level]["collision_rate"]

            # Indica il vincitore
            sr_winner = "<" if b_sr < c_sr else ">" if b_sr > c_sr else "="
            print(f"{level:<10} {b_sr:<12.1%} {c_sr:<12.1%} {b_cr:<12.1%} {c_cr:<12.1%}")

        # Training summary
        print(f"\nTraining Summary:")
        print(f"Batch - Episodes: {batch_summary['total_episodes']}, "
              f"SR: {batch_summary['cumulative_success_rate']:.1%}")
        print(f"Curriculum - Episodes: {curriculum_summary['total_episodes']}, "
              f"SR: {curriculum_summary['cumulative_success_rate']:.1%}")

        if "curriculum" in curriculum_summary:
            curr = curriculum_summary["curriculum"]
            print(f"Curriculum reached: {curr['final_level']} "
                  f"({curr['levels_completed']}/{curr['total_levels']} levels)")

        # Salva confronto
        comparison_path = os.path.join(run_dir, "batch_vs_curriculum_comparison.txt")
        with open(comparison_path, "w") as f:
            f.write("CONFRONTO BATCH vs CURRICULUM\n")
            f.write(f"Budget per esperimento: {TOTAL_TIMESTEPS:,} step\n")
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