"""
carla_core.agents — Policy models for CARLA MAPPO.
"""

from carla_core.agents.centralized_critic import (
    CentralizedCriticModel,
    CentralizedCriticCallbacks,
    GLOBAL_OBS,
    compute_global_obs_dim_with_mask
)

__all__ = [
    "CentralizedCriticModel",
    "CentralizedCriticCallbacks",
    "GLOBAL_OBS",
    "compute_global_obs_dim_with_mask"
]
