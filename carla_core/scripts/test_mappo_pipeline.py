"""
Test MAPPO Pipeline — Dry Run su MPE (no CARLA)
================================================
Verifica che CentralizedCriticModel + CentralizedCriticCallbacks
funzionino correttamente su un ambiente PettingZoo toy.

Usa simple_spread_v3 (3 agenti) con 2 policy separate:
  - agent_0 → policy_A (simula vehicle_policy)
  - agent_1, agent_2 → policy_B (simula pedestrian_policy)

Test:
  1. Model registrato e istanziato correttamente
  2. Callbacks iniettano global_obs nel SampleBatch
  3. GAE ricalcolato senza errori
  4. Training 3 iterazioni senza crash

Uso:
    python carla_core/scripts/test_mappo_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import ray
from pettingzoo.mpe import simple_spread_v3
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from carla_core.agents.centralized_critic import (
    CentralizedCriticModel,
    CentralizedCriticCallbacks,
)


def env_creator(cfg):
    return ParallelPettingZooEnv(simple_spread_v3.parallel_env(N=3, max_cycles=25))


def policy_mapping_fn(agent_id, *args, **kwargs):
    # agent_0 → policy_A, agent_1/2 → policy_B
    if agent_id == "agent_0":
        return "policy_A"
    return "policy_B"


def main():
    print("=" * 55)
    print("TEST MAPPO PIPELINE (CentralizedCritic + Callbacks)")
    print("=" * 55)

    ray.init(num_cpus=2, num_gpus=0, log_to_driver=False)

    try:
        register_env("mpe_cc_test", env_creator)
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

        # Get per-agent spaces
        raw = simple_spread_v3.parallel_env(N=3, max_cycles=25)
        agents = raw.possible_agents
        obs_space = raw.observation_space(agents[0])
        act_space = raw.action_space(agents[0])
        obs_dim = obs_space.shape[0]
        raw.close()

        # global_obs = 3 agents * obs_dim + alive_mask (3 agents)
        global_obs_dim = 3 * obs_dim + 3
        cc_cfg = {
            "hidden_size": 64,
            "n_hidden_layers": 1,
            "global_obs_dim": global_obs_dim,
        }

        print(f"  Agents: {agents}")
        print(f"  Obs dim: {obs_dim} | Global obs dim: {global_obs_dim}")
        print(f"  Policies: policy_A (agent_0), policy_B (agent_1,2)")

        config = (
            PPOConfig()
            .environment(env="mpe_cc_test")
            .multi_agent(
                policies={
                    "policy_A": (
                        None, obs_space, act_space,
                        {"model": {"custom_model": "cc_model",
                                   "custom_model_config": cc_cfg}},
                    ),
                    "policy_B": (
                        None, obs_space, act_space,
                        {"model": {"custom_model": "cc_model",
                                   "custom_model_config": cc_cfg}},
                    ),
                },
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["policy_A", "policy_B"],
            )
            .callbacks(CentralizedCriticCallbacks)
            .resources(num_gpus=0)
            .rollouts(num_rollout_workers=0)
            .training(
                train_batch_size=512,
                sgd_minibatch_size=128,
                num_sgd_iter=3,
                lr=3e-4,
            )
            .framework("torch")
        )

        print("\nBuild algorithm...")
        algo = config.build()

        n_iters = 3
        for i in range(1, n_iters + 1):
            print(f"\n  Iteration {i}/{n_iters}...")
            result = algo.train()

            pol_rew = result.get("policy_reward_mean", {})
            ts = result.get("timesteps_total", 0)
            eps = result.get("episodes_total", 0)

            print(f"    Steps: {ts} | Episodes: {eps}")
            for pid, rew in pol_rew.items():
                print(f"    {pid}: reward={rew:.2f}")

        algo.stop()

        print(f"\n{'=' * 55}")
        print("[OK] MAPPO pipeline funziona:")
        print("  - CentralizedCriticModel istanziato")
        print("  - CentralizedCriticCallbacks eseguiti")
        print("  - GAE ricalcolato con global_obs")
        print("  - 2 policy separate trainano")
        print(f"{'=' * 55}")

    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
