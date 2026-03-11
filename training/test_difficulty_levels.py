"""
        Test Livelli di Difficolta su MetaDrive
=====================================================
Lancia 3 training con difficolta crescente per verificare
che la domanda sperimentale (curriculum vs batch) abbia senso.

Esegui con:
    python ./training/test_difficulty_levels.py

Confronta i risultati su TensorBoard:
    tensorboard --logdir=experiments/difficulty_test/
"""

import os
import yaml
import numpy as np
from copy import deepcopy
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from metadrive.envs import MetaDriveEnv

SEED = 42 # Seed globale per riproducibilita
TIMESTEPS = 500_000 # LastTrain: 500000 # Timesteps per ogni livello (da aumentare per performance migliori)
DEVICE = "cpu" # "cpu" per MLP semplice, "cuda" per reti più grandi (da modificare quando passeremo a RLlib + CARLA)
LOG_DIR = "experiments/difficulty_test"

# ============================================================
# SEED
# ============================================================

def set_global_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3 LIVELLI DI DIFFICOLTA
# ============================================================

LEVELS = {
    "easy": {
        "map": "SSS",
        "traffic_density": 0.05,
        "num_scenarios": 5,
        "accident_prob": 0.0,
        "start_seed": 0
    },
    "medium": {
        "map": "SCSC",
        "traffic_density": 0.15,
        "num_scenarios": 10,
        "accident_prob": 0.1,
        "start_seed": 0
    },
    "hard": {
        "map": "SCRCSRC",
        "traffic_density": 0.3,
        "num_scenarios": 15,
        "accident_prob": 0.2,
        "start_seed": 0
    },
}

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
    "device": DEVICE
}

# ============================================================
# CALLBACK
# ============================================================

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.collisions = 0
        self.successes = 0
        self.total_episodes = 0

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.total_episodes += 1
                info = self.locals["infos"][i]
                if info.get("arrive_dest", False):
                    self.successes += 1
                if info.get("crash", False) or info.get("crash_vehicle", False):
                    self.collisions += 1
                if self.total_episodes % 10 == 0 and self.total_episodes > 0:
                    self.logger.record("thesis/success_rate",
                                       self.successes / self.total_episodes)
                    self.logger.record("thesis/collision_rate",
                                       self.collisions / self.total_episodes)
        return True


# ============================================================
# TRAINING PER LIVELLO
# ============================================================

def train_level(level_name, env_config):
    print("=" *50)
    print(f"LIVELLO: {level_name.upper()}")
    print(f"Mappa: {env_config['map']} | Traffico: {env_config['traffic_density']}")
    print(f"Scenari: {env_config['num_scenarios']} | Incidenti: {env_config['accident_prob']}")
    print("=" *50)

    set_global_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_DIR,f"{level_name}",f"{level_name}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Salva config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump({"env": env_config, "ppo": PPO_CONFIG, "timesteps": TIMESTEPS}, f)

    # Crea environment
    def make_env():
        env = MetaDriveEnv({**env_config, "use_render": False})
        return Monitor(env)

    env = DummyVecEnv([make_env])
    metrics_cb = MetricsCallback()

    # Training
    model = PPO("MlpPolicy", env, tensorboard_log=run_dir, **PPO_CONFIG)

    print(f"Training per {TIMESTEPS:,} timesteps...")
    try:
        model.learn(total_timesteps=TIMESTEPS, callback=[metrics_cb], progress_bar=True)
    except KeyboardInterrupt:
        print("Interrotto!")
    finally:
        # Report
        sr = metrics_cb.successes / max(metrics_cb.total_episodes, 1)
        cr = metrics_cb.collisions / max(metrics_cb.total_episodes, 1)

        print(f"\nRISULTATI {level_name.upper()}:")
        print(f"Episodi: {metrics_cb.total_episodes}")
        print(f"Success rate: {sr:.1%}")
        print(f"Collision rate: {cr:.1%}")

        # Salva report
        with open(os.path.join(run_dir, "report.txt"), "w") as f:
            f.write(f"Level: {level_name}\n")
            f.write(f"Map: {env_config['map']}\n")
            f.write(f"Traffic: {env_config['traffic_density']}\n")
            f.write(f"Episodi: {metrics_cb.total_episodes}\n")
            f.write(f"Success rate: {sr:.1%}\n")
            f.write(f"Collision rate: {cr:.1%}\n")
            f.write(f"Timesteps: {TIMESTEPS}\n")

        model.save(os.path.join(run_dir, "final_model"))
        env.close()

    return {"level": level_name, "episodes": metrics_cb.total_episodes,
            "success_rate": sr, "collision_rate": cr}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("TEST LIVELLI DI DIFFICOLTA - Validazione Domanda Sperimentale")
    print(f"Timesteps per livello: {TIMESTEPS:,}")
    print(f"Livelli: {list(LEVELS.keys())}")
    print("=" * 50)

    results = []
    for name, config in LEVELS.items():
        result = train_level(name, config)
        results.append(result)

    # Confronto finale
    print("\n\n" + "=" * 50)
    print("CONFRONTO FINALE")
    print("=" * 50)
    print(f"{'Livello':<10} {'Episodi':<10} {'Success':<12} {'Collision':<12}")
    print("=" * 50)
    for r in results:
        print(f"{r['level']:<10} {r['episodes']:<10} {r['success_rate']:<12.1%} {r['collision_rate']:<12.1%}")

    print(f"\nSe i numeri sono diversi tra i livelli, la domanda sperimentale (curriculum vs batch) ha senso!")
    print("=" * 50)

    # Salva confronto
    with open(os.path.join(LOG_DIR, "comparison.txt"), "w") as f:
        f.write("CONFRONTO LIVELLI DI DIFFICOLTA\n")
        f.write(f"Timesteps per livello: {TIMESTEPS}\n\n")
        for r in results:
            f.write(f"{r['level']}: success={r['success_rate']:.1%}, collision={r['collision_rate']:.1%}, episodes={r['episodes']}\n")