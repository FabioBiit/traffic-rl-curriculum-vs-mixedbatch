"""
Test rapido: MetaDrive renderizza correttamente?
"""
from metadrive.envs import MetaDriveEnv
import time

env = MetaDriveEnv({
    "use_render": True,
    "traffic_density": 0.1,
    "map": "SSS",
    "num_scenarios": 1,
})

obs, info = env.reset()
print("Finestra aperta? Guarda la barra delle applicazioni.")
print("L'agente guidera' a caso per 30 secondi...")

for step in range(1000):
    action = env.action_space.sample()  # azione random
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.03)  # 30fps circa

env.close()
print("Test completato.")