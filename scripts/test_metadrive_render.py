"""
Test rapido: MetaDrive renderizza correttamente?
"""
import time

from envs.multi_level_env import create_eval_env, LEVEL_CONFIGS


def main():
    # Riusa la factory condivisa e mantiene il setup "easy" del test rapido.
    quick_level_configs = {
        "easy": {
            **LEVEL_CONFIGS["easy"],
            "traffic_density": 0.1,
            "num_scenarios": 1,
        }
    }
    env = create_eval_env("easy", level_configs=quick_level_configs)

    try:
        obs, info = env.reset()
        print("Finestra aperta? Guarda la barra delle applicazioni.")
        print("L'agente guidera a caso per 30 secondi...")

        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        env.close()
        print("Test completato.")


if __name__ == "__main__":
    main()
