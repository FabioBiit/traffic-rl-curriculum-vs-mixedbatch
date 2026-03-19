"""
carla_core.agents — Policy models for CARLA MAPPO.
"""

from carla_core.agents.centralized_critic import (
    CentralizedCriticModel,
    CentralizedCriticCallbacks,
    GLOBAL_OBS,
)

__all__ = [
    "CentralizedCriticModel",
    "CentralizedCriticCallbacks",
    "GLOBAL_OBS",
]
