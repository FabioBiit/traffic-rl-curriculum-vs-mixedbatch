"""
Test RLlib Multi-Agent — Gate G1
================================
Verifica che Ray/RLlib funzioni con un ambiente multi-agent.
Usa PettingZoo MPE (simple_spread) come ambiente toy.

Prerequisiti:
    pip install "ray[rllib]==2.10.0" pettingzoo supersuit

Esegui con:
    python carla_core/scripts/test_rllib_mpe.py
"""

import sys

# ---- Import check ----
missing = []
for mod in ["ray", "ray.rllib", "pettingzoo", "supersuit"]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)

if missing:
    print(f"[ERRORE] Moduli mancanti: {', '.join(missing)}")
    print("Installa con: pip install 'ray[rllib]==2.10.0' pettingzoo supersuit")
    sys.exit(1)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_spread_v3


def env_creator(config):
    """Factory per ambiente PettingZoo compatibile con RLlib."""
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25)
    return ParallelPettingZooEnv(env)


def main():
    print("=" * 50)
    print("TEST RLLIB MULTI-AGENT (MPE simple_spread)")
    print("=" * 50)

    # Init Ray
    ray.init(num_cpus=2, num_gpus=0, log_to_driver=False)
    print(f"Ray {ray.__version__} inizializzato")

    try:
        # Registra ambiente
        from ray.tune.registry import register_env
        register_env("mpe_spread", env_creator)

        # Crea ambiente temporaneo per ottenere spaces

        # Spazi per-policy: devono essere SINGLE-AGENT (non Dict multi-agent)
        raw_env = simple_spread_v3.parallel_env(N=3, max_cycles=25)
        first_agent = raw_env.possible_agents[0]
        single_obs_space = raw_env.observation_space(first_agent)
        single_act_space = raw_env.action_space(first_agent)
        raw_env.close()

        # Solo per debug/stampa (spazi aggregati del wrapper RLlib)
        tmp_env = env_creator({})
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        agents = tmp_env.get_agent_ids()
        tmp_env.close()

        print(f"Agenti: {agents}")
        print(f"Obs space: {obs_space}")
        print(f"Act space: {act_space}")
        print(f"Single-agent Obs space: {single_obs_space}")
        print(f"Single-agent Act space: {single_act_space}")


        # Config PPO multi-agent
        config = (
            PPOConfig()
            .environment(env="mpe_spread")
            .multi_agent(
                policies={
                    "shared_policy": (None, single_obs_space, single_act_space, {}),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .resources(num_gpus=0)
            .rollouts(num_rollout_workers=1)
            .training(
                train_batch_size=512,
                sgd_minibatch_size=128,
                num_sgd_iter=5,
                lr=3e-4,
            )
        )

        # Build e train 1 iterazione
        print("\nBuild algoritmo...")
        algo = config.build()

        print("Training 1 iterazione...")
        result = algo.train()

        # Report
        reward = result.get("episode_reward_mean", "N/A")
        eps = result.get("episodes_this_iter", "N/A")
        timesteps = result.get("timesteps_total", "N/A")

        print(f"\n{'='*50}")
        print("TEST COMPLETATO")
        print(f"{'='*50}")
        print(f"  Reward medio: {reward}")
        print(f"  Episodi: {eps}")
        print(f"  Timesteps: {timesteps}")
        print(f"\n[OK] RLlib multi-agent funziona. Gate G1 (parte RLlib): PASS")

        algo.stop()

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
