"""
Test rapido: MetaDrive renderizza correttamente?
"""
from metadrive.envs import MetaDriveEnv
import time


def main():
    env = MetaDriveEnv({
        "use_render": True,
        "traffic_density": 0.1,
        "map": "SSS",
        "num_scenarios": 1,
    })

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