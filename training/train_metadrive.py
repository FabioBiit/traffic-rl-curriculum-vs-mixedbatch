"""
Settimana 1 - Primo Training PPO su MetaDrive
==============================================
Esegui con:
    python training/train_metadrive.py
    python training/train_metadrive.py --timesteps 500000
    python training/train_metadrive.py --eval <path_modello.zip>
    python training/train_metadrive.py --render --timesteps 50000

Visualizza risultati:
    tensorboard --logdir=experiments/metadrive_baseline/
"""

import os
import yaml
import argparse
import numpy as np
from copy import deepcopy
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from metadrive.envs import MetaDriveEnv


# ============================================================
# CONFIGURAZIONE
# ============================================================

CONFIG = {
    # MetaDrive environment
    "env": {
        "use_render": False,       # False per training veloce, True per vedere
        "traffic_density": 0.1,    # 10% traffico (basso per iniziare)
        "map": "SSS",              # 3 segmenti dritti (mappa semplice)
        "start_seed": 0,           # Seed di MetaDrive per generazione mappa
        "num_scenarios": 5,        # Varianti della mappa (rotazione ad ogni reset)
        "accident_prob": 0.0,      # Nessun incidente random
    },

    # PPO hyperparameters (valori standard dal paper Schulman 2017)
    "ppo": {
        "learning_rate": 3e-4,     # Standard PPO - Dice al modello quanto aggiornare i pesi ad ogni step
        "n_steps": 2048,           # Step raccolti prima di ogni update
        "batch_size": 64,          # Mini-batch per gradient descent
        "n_epochs": 10,            # Passate sui dati per ogni update
        "gamma": 0.99,             # Discount factor (alto = pianifica a lungo)
        "gae_lambda": 0.95,        # GAE lambda (bias-variance tradeoff)
        "clip_range": 0.2,         # PPO clipping (limita update della policy)
        "ent_coef": 0.01,          # Entropy bonus (incentiva esplorazione)
        "verbose": 1,              # Stampa metriche durante training
        "device": "cpu",           # cpu per MLP piccola, cuda per reti grandi (Da modificare quando passeremo a RLlib + CARLA)
    },

    # Training settings
    "training": {
        "total_timesteps": 100_000,
        "eval_freq": 5_000,
        "n_eval_episodes": 5,
        "log_dir": "experiments/metadrive_baseline",
        "seed": 42,                # Seed globale per riproducibilita
        "checkpoint_freq": 50_000, # Salva checkpoint ogni N step
    },
}


# ============================================================
# SEED GLOBALE - Riproducibilita
# ============================================================

def set_global_seed(seed):
    """Fissa tutti i seed per garantire risultati riproducibili."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CALLBACK - Metriche specifiche per la tesi
# ============================================================

class ThesisMetricsCallback(BaseCallback):
    """Logga collision rate e success rate durante il training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.collisions = 0
        self.successes = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.total_episodes += 1
                info = self.locals["infos"][i]

                if info.get("arrive_dest", False):
                    self.successes += 1
                if info.get("crash", False) or info.get("crash_vehicle", False):
                    self.collisions += 1

                if self.total_episodes % 10 == 0 and self.total_episodes > 0:
                    sr = self.successes / self.total_episodes
                    cr = self.collisions / self.total_episodes
                    self.logger.record("thesis/success_rate", sr)
                    self.logger.record("thesis/collision_rate", cr)
                    self.logger.record("thesis/total_episodes", self.total_episodes)

        return True


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================

def make_env(config, render=False):
    """Crea un ambiente MetaDrive wrappato con Monitor."""
    def _init():
        env_config = deepcopy(config)  # deepcopy evita side-effect
        env_config["use_render"] = render
        env = MetaDriveEnv(env_config)
        env = Monitor(env)
        return env
    return _init


# ============================================================
# TRAINING
# ============================================================

def train(config):
    """Lancia il training PPO su MetaDrive."""

    # Setup directory
    log_dir = config["training"]["log_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Seed globale per riproducibilita
    seed = config["training"]["seed"]
    set_global_seed(seed)

    # Salva config per riproducibilita
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Crea environment
    print("\n[1/4] Creazione environment MetaDrive...")
    env = DummyVecEnv([make_env(config["env"])])
    print("      OK!")

    # Crea modello PPO
    print("\n[2/4] Creazione modello PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        verbose=config["ppo"]["verbose"],
        tensorboard_log=run_dir,
        device=config["ppo"]["device"],
    )
    print(f"      Device: {model.device}")
    print(f"      Policy: {model.policy.__class__.__name__}")
    print(f"      Seed: {seed}")
    print(f"      Observation space: {env.observation_space.shape}")
    print(f"      Action space: {env.action_space.shape}")

    # Callbacks
    metrics_cb = ThesisMetricsCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=config["training"]["checkpoint_freq"],
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="model",
    )

    # Training con crash recovery
    total_steps = config["training"]["total_timesteps"]
    print(f"\n[3/4] Training per {total_steps:,} timesteps...")
    print(f"      TensorBoard: tensorboard --logdir={run_dir}")
    print("      " + "=" * 50)

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=[metrics_cb, checkpoint_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrotto! Salvataggio modello parziale...")
    except Exception as e:
        print(f"\n\nErrore durante il training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Salva sempre il modello
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)

        # Report a schermo
        print("\n[4/4] REPORT FINALE")
        print("=" * 50)
        print(f"  Episodi totali:  {metrics_cb.total_episodes}")
        if metrics_cb.total_episodes > 0:
            print(f"  Success rate:    {metrics_cb.successes / metrics_cb.total_episodes:.1%}")
            print(f"  Collision rate:  {metrics_cb.collisions / metrics_cb.total_episodes:.1%}")
        print(f"  Modello:         {final_path}")
        print(f"  Run dir:         {run_dir}")
        print("=" * 50)

        # Salva report su file
        with open(os.path.join(run_dir, "report.txt"), "w") as f:
            f.write(f"Episodi totali: {metrics_cb.total_episodes}\n")
            if metrics_cb.total_episodes > 0:
                f.write(f"Success rate: {metrics_cb.successes / metrics_cb.total_episodes:.1%}\n")
                f.write(f"Collision rate: {metrics_cb.collisions / metrics_cb.total_episodes:.1%}\n")
            f.write(f"Timesteps: {total_steps}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Modello: {final_path}\n")

        env.close()

    return model, run_dir


# ============================================================
# VALUTAZIONE VISIVA
# ============================================================

def evaluate(model_path, config, n_episodes=3):
    """Carica un modello e mostralo in azione con rendering."""
    print(f"\nCaricamento modello: {model_path}")
    model = PPO.load(model_path)

    env = MetaDriveEnv({**config["env"], "use_render": True})

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            if info.get("arrive_dest"):
                status = "GOAL!"
            elif info.get("crash") or info.get("crash_vehicle"):
                status = "CRASH"
            elif info.get("out_of_road"):
                status = "FUORI STRADA"
            else:
                status = "TIMEOUT"

            print(f"  Ep {ep + 1}: {status} | Reward: {total_reward:.1f} | Steps: {steps}")
    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training su MetaDrive")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path al modello da valutare visivamente")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps")
    parser.add_argument("--render", action="store_true",
                        help="Abilita rendering durante training (lento)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, CONFIG)
    else:
        if args.timesteps:
            CONFIG["training"]["total_timesteps"] = args.timesteps
        if args.render:
            CONFIG["env"]["use_render"] = True

        model, run_dir = train(CONFIG)

        print("\nProssimi step:")
        print(f"  1. TensorBoard: tensorboard --logdir={run_dir}")
        print(f"  2. Guarda il modello: python training/train_metadrive.py --eval {run_dir}/final_model.zip")
        print(f"  3. Piu timesteps: python training/train_metadrive.py --timesteps 2000000")