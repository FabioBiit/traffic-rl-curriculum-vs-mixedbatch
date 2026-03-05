"""
Settimana 1 - Primo Training PPO su MetaDrive
==============================================
Il tuo "hello world" del Reinforcement Learning.

Esegui con:
    python .\training/train_metadrive.py

Visualizza risultati:
    tensorboard --logdir=experiments/metadrive_baseline/
"""

import os
import yaml
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from metadrive.envs import MetaDriveEnv


# ============================================================
# CONFIGURAZIONE
# ============================================================

CONFIG = {
    "env": {
        "use_render": False,
        "traffic_density": 0.1,
        "map": "SSS",
        "start_seed": 0,
        "num_scenarios": 5,
        "accident_prob": 0.0,
    },
    "ppo": {
        "learning_rate": 3e-4, # Di quanto il modello aggiorna i pesi ad ogni step
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
    },
    "training": {
        "total_timesteps": 100_000,
        "eval_freq": 5_000,
        "n_eval_episodes": 5,
        "log_dir": "experiments/metadrive_baseline",
    },
}


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
    def _init():
        env_config = config.copy()
        env_config["use_render"] = render
        env = MetaDriveEnv(env_config)
        env = Monitor(env)
        return env
    return _init


# ============================================================
# TRAINING
# ============================================================

def train(config):
    log_dir = config["training"]["log_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Salva config per riproducibilita
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Crea environment (uno solo - MetaDrive non supporta istanze multiple)
    print("\n[1/4] Creazione environment MetaDrive...")
    env = DummyVecEnv([make_env(config["env"])])
    print("OK!")

    # Crea modello PPO
    print("\n[2/4] Creazione modello PPO...")
    model = PPO(
        "MlpPolicy",
        env,
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
        device="cpu",
    )
    print(f"Device: {model.device}")
    print(f"Policy: {model.policy.__class__.__name__}")

    obs_space = env.observation_space
    act_space = env.action_space
    print(f"Observation space: {obs_space.shape}")
    print(f"Action space: {act_space.shape}")

    # Callback (solo metriche tesi, no eval separato)
    metrics_cb = ThesisMetricsCallback()

    # Training
    total_steps = config["training"]["total_timesteps"]
    print(f"\n[3/4] Training per {total_steps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir={run_dir}")
    print("      " + "=" * 50)

    model.learn(
        total_timesteps=total_steps,
        callback=[metrics_cb],
        progress_bar=True,
    )

    # Salva modello finale
    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)

    # Report
    print("\n[4/4] REPORT FINALE")
    print("=" * 50)
    print(f"Episodi totali: {metrics_cb.total_episodes}")
    if metrics_cb.total_episodes > 0:
        print(f"Success rate: {metrics_cb.successes / metrics_cb.total_episodes:.1%}")
        print(f"Collision rate: {metrics_cb.collisions / metrics_cb.total_episodes:.1%}")
    print(f"Modello: {final_path}")
    print(f"Run dir: {run_dir}")
    print("=" * 50)

    env.close()
    return model, run_dir


# ============================================================
# VALUTAZIONE VISIVA
# ============================================================

def evaluate(model_path, config, n_episodes=3):
    print(f"\nCaricamento modello: {model_path}")
    model = PPO.load(model_path)

    env = MetaDriveEnv({**config["env"], "use_render": True})

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

        status = "GOAL!" if info.get("arrive_dest") else "CRASH" if info.get("crash") else "TIMEOUT"
        print(f"  Ep {ep + 1}: {status} | Reward: {total_reward:.1f} | Steps: {steps}")

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
        print(f"  1. Guarda i risultati: tensorboard --logdir={run_dir}")
        print(f"  2. Vedi il modello: python training/train_metadrive.py --eval {run_dir}/best_model/best_model.zip")
        print(f"  3. Piu timesteps: python training/train_metadrive.py --timesteps 500000")