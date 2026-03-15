"""
Utility condivise per script di training.
Centralizza config PPO base e funzione seed per ridurre duplicazioni.
"""

import random
import numpy as np

# FINETUNING RUN 1 + RUN 2
PPO_CONFIG_BASE = {
    "learning_rate": 2e-4, # 3e-4 -> 2e-4 run1
    "n_steps": 4096, # 2048 -> 4096 run2
    "batch_size": 128, # 64 -> 128 run2
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005 # 0.01 -> 0.005 run2
}


def set_global_seed(seed):
    """Fissa seed Python/NumPy/PyTorch per riproducibilita."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_outcome(info):
    """
    Estrae flag standardizzati di outcome episodio da un dict info.
    Ritorna (success, collision).
    """
    success = bool(info.get("arrive_dest", False))
    collision = bool(info.get("crash", False) or info.get("crash_vehicle", False))
    return success, collision
