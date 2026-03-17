"""
Test CarlaMultiAgentEnv — 1V + 1P, random actions, 3 episodes.

Prerequisiti:
    - Server CARLA in esecuzione
    - pip install carla==0.9.16 pettingzoo gymnasium pyyaml

Uso:
    python carla_core/scripts/test_multi_agent_env.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from carla_core.envs.carla_multi_agent_env import CarlaMultiAgentEnv


def main():
    env = CarlaMultiAgentEnv()

    print(f"Agents: {env.possible_agents}")
    for a in env.possible_agents:
        print(f"  {a}: obs={env.observation_space(a).shape}, act={env.action_space(a).shape}")

    try:
        for ep in range(3):
            obs, infos = env.reset()
            done_agents = set()
            ep_rewards = {a: 0.0 for a in env.possible_agents}
            step = 0

            while env.agents:
                actions = {}
                for a in env.agents:
                    actions[a] = env.action_space(a).sample()

                obs, rewards, terms, truncs, infos = env.step(actions)
                step += 1

                for a in rewards:
                    ep_rewards[a] += rewards[a]

            print(f"\nEp {ep+1}/3 | Steps: {step}")
            for a in env.possible_agents:
                print(f"  {a}: reward={ep_rewards[a]:.1f}")

    finally:
        env.close()

    print("\n[OK] CarlaMultiAgentEnv funziona.")


if __name__ == "__main__":
    main()
