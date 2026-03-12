"""
Utility condivise per script di training.
Centralizza config PPO base e funzione seed per ridurre duplicazioni.
"""

import random
import numpy as np


PPO_CONFIG_BASE = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
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
