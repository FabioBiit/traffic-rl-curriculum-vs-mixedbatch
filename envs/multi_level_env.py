"""
Multi-Level Environment Manager per MetaDrive
==============================================
Gestisce la creazione di ambienti MetaDrive per livelli diversi.
Non e' un wrapper Gym — e' un factory + tracker, perche MetaDrive
(Panda3D) non supporta istanze multiple simultanee.

Usato da training/train_experiment.py per batch e curriculum.
"""

import numpy as np
from copy import deepcopy
from collections import deque

from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================
# CONFIGURAZIONE LIVELLI
# ============================================================

LEVEL_CONFIGS = {
    "easy": {
        "map": "SSS",
        "traffic_density": 0.05,
        "num_scenarios": 5,
        "accident_prob": 0.0,
        "start_seed": 0,
    },
    "medium": {
        "map": "SCSC",
        "traffic_density": 0.15,
        "num_scenarios": 10,
        "accident_prob": 0.1,
        "start_seed": 0,
    },
    "hard": {
        "map": "SCRCSRC",
        "traffic_density": 0.3,
        "num_scenarios": 15,
        "accident_prob": 0.2,
        "start_seed": 0,
    },
}

# Mappa di test — mai vista durante il training
TEST_CONFIG = {
    "map": "SCCSS",
    "traffic_density": 0.2,
    "num_scenarios": 10,
    "accident_prob": 0.15,
    "start_seed": 100,
}


# ============================================================
# FACTORY — Crea e distrugge ambienti per livello
# ============================================================

def create_env(level_name, level_configs=None):
    """
    Crea un DummyVecEnv per il livello specificato.
    
    Args:
        level_name: "easy", "medium", "hard", o "test"
        level_configs: dizionario custom dei livelli (opzionale)
    
    Returns:
        DummyVecEnv wrappato con Monitor
    """
    if level_configs is None:
        level_configs = LEVEL_CONFIGS

    if level_name == "test":
        config = deepcopy(TEST_CONFIG)
    else:
        if level_name not in level_configs:
            raise ValueError(f"Livello '{level_name}' non trovato. Disponibili: {list(level_configs.keys())}")
        config = deepcopy(level_configs[level_name])

    config["use_render"] = False

    def _make():
        env = MetaDriveEnv(config)
        return Monitor(env)

    return DummyVecEnv([_make])


def create_eval_env(level_name, level_configs=None):
    """
    Crea un MetaDriveEnv diretto (senza DummyVecEnv) per valutazione visiva.
    
    Args:
        level_name: "easy", "medium", "hard", o "test"
        level_configs: dizionario custom dei livelli (opzionale)
    
    Returns:
        MetaDriveEnv con rendering attivo
    """
    if level_configs is None:
        level_configs = LEVEL_CONFIGS

    if level_name == "test":
        config = deepcopy(TEST_CONFIG)
    else:
        config = deepcopy(level_configs[level_name])

    config["use_render"] = True
    return MetaDriveEnv(config)


# ============================================================
# EPISODE TRACKER — Traccia metriche per finestra di episodi
# ============================================================

class EpisodeTracker:
    """
    Traccia success rate e collision rate su una finestra mobile
    di episodi. Usato per determinare la promozione nel curriculum.
    
    Args:
        window_size: numero di episodi nella finestra mobile
    """

    def __init__(self, window_size=50):
        self.window_size = window_size
        self.successes = deque(maxlen=window_size)
        self.collisions = deque(maxlen=window_size)
        self.total_episodes = 0
        self.total_successes = 0
        self.total_collisions = 0

    def record(self, info):
        """Registra il risultato di un episodio."""
        success = info.get("arrive_dest", False)
        collision = info.get("crash", False) or info.get("crash_vehicle", False)

        self.successes.append(1 if success else 0)
        self.collisions.append(1 if collision else 0)
        self.total_episodes += 1
        if success:
            self.total_successes += 1
        if collision:
            self.total_collisions += 1

    @property
    def window_success_rate(self):
        """Success rate sulla finestra mobile corrente."""
        if len(self.successes) == 0:
            return 0.0
        return sum(self.successes) / len(self.successes)

    @property
    def window_collision_rate(self):
        """Collision rate sulla finestra mobile corrente."""
        if len(self.collisions) == 0:
            return 0.0
        return sum(self.collisions) / len(self.collisions)

    @property
    def cumulative_success_rate(self):
        """Success rate cumulativo dall'inizio."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_successes / self.total_episodes

    @property
    def cumulative_collision_rate(self):
        """Collision rate cumulativo dall'inizio."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_collisions / self.total_episodes

    @property
    def window_full(self):
        """True se la finestra ha abbastanza episodi per essere affidabile."""
        return len(self.successes) >= self.window_size

    def reset(self):
        """Resetta il tracker per un nuovo livello."""
        self.successes.clear()
        self.collisions.clear()
        # Non resettare i totali — servono per il report finale

    def summary(self):
        """Ritorna un dizionario con tutte le metriche."""
        return {
            "total_episodes": self.total_episodes,
            "total_successes": self.total_successes,
            "total_collisions": self.total_collisions,
            "cumulative_success_rate": self.cumulative_success_rate,
            "cumulative_collision_rate": self.cumulative_collision_rate,
            "window_success_rate": self.window_success_rate,
            "window_collision_rate": self.window_collision_rate,
            "window_size": len(self.successes),
        }


# ============================================================
# CURRICULUM MANAGER — Gestisce la promozione tra livelli
# ============================================================

class CurriculumManager:
    """
    Gestisce la progressione Easy -> Medium -> Hard.
    Promuove al livello successivo quando il success rate
    sulla finestra mobile supera la soglia.
    
    Args:
        levels: lista ordinata dei nomi dei livelli
        promotion_threshold: success rate minimo per promuovere
        min_episodes: episodi minimi prima di valutare la promozione
    """

    def __init__(self, levels=None, promotion_threshold=0.3, min_episodes=50):
        self.levels = levels or ["easy", "medium", "hard"]
        self.promotion_threshold = promotion_threshold
        self.min_episodes = min_episodes
        self.current_index = 0
        self.promotion_history = []

    @property
    def current_level(self):
        return self.levels[self.current_index]

    @property
    def is_final_level(self):
        return self.current_index >= len(self.levels) - 1

    def should_promote(self, tracker):
        """
        Controlla se l'agente deve essere promosso al livello successivo.
        
        Args:
            tracker: EpisodeTracker con le metriche correnti
        
        Returns:
            True se deve essere promosso
        """
        if self.is_final_level:
            return False
        if tracker.total_episodes < self.min_episodes:
            return False
        if not tracker.window_full:
            return False
        return tracker.window_success_rate >= self.promotion_threshold

    def promote(self, tracker):
        """Promuovi al livello successivo. Ritorna il nuovo livello."""
        self.promotion_history.append({
            "from": self.current_level,
            "to": self.levels[self.current_index + 1],
            "episodes_on_level": tracker.total_episodes,
            "success_rate_at_promotion": tracker.window_success_rate,
        })
        self.current_index += 1
        tracker.reset()  # Resetta la finestra per il nuovo livello
        return self.current_level

    def summary(self):
        """Ritorna un dizionario con la storia del curriculum."""
        return {
            "final_level": self.current_level,
            "levels_completed": self.current_index,
            "total_levels": len(self.levels),
            "promotion_history": self.promotion_history,
        }