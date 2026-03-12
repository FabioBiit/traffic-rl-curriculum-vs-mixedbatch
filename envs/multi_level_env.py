"""
Multi-Level Environment Manager per MetaDrive
==============================================
Gestisce la creazione di ambienti MetaDrive per livelli diversi.
Non e' un wrapper Gym — e' un factory + tracker, perche MetaDrive
(Panda3D) non supporta istanze multiple simultanee.

Usato da training/train_experiment.py per batch e curriculum.

Changelog v2.0 (12 Marzo 2026):
- CurriculumManager: aggiunto replay mechanism (revisione livelli precedenti)
- CurriculumManager: aggiunto min_timesteps oltre a min_episodes
- CurriculumManager: soglia di promozione alzata a 0.6 (default)
- CurriculumManager: aggiunto collision_threshold (max collision rate per promuovere)
- EpisodeTracker: aggiunto tracking timesteps
- EpisodeTracker: aggiunta window_collision_rate nel check promozione
"""

from copy import deepcopy
from collections import deque

from metadrive.envs import MetaDriveEnv
from training.common import episode_outcome


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
    # Import lazy: evita inizializzazione SB3/Torch quando serve solo
    # create_eval_env (es. test rendering).
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

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
    
    Changelog v2.0:
    - Aggiunto tracking dei timesteps (totali e per livello)
    - I totali NON vengono resettati al cambio livello (servono per report)
    - La finestra mobile viene resettata al cambio livello
    
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
        # v2.0: tracking timesteps
        self.level_timesteps = 0  # timesteps sul livello corrente (resettato a promozione)
        self.level_episodes = 0   # episodi sul livello corrente (resettato a promozione)

    def record(self, info):
        """Registra il risultato di un episodio."""
        success, collision = episode_outcome(info)

        self.successes.append(1 if success else 0)
        self.collisions.append(1 if collision else 0)
        self.total_episodes += 1
        self.level_episodes += 1
        if success:
            self.total_successes += 1
        if collision:
            self.total_collisions += 1

    def add_timesteps(self, n):
        """Aggiorna il contatore di timesteps per il livello corrente."""
        self.level_timesteps += n

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
        """
        Resetta il tracker per un nuovo livello.
        Resetta: finestra mobile, contatori livello.
        NON resetta: totali cumulativi (servono per report finale).
        """
        self.successes.clear()
        self.collisions.clear()
        self.level_timesteps = 0
        self.level_episodes = 0

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
            "level_timesteps": self.level_timesteps,
            "level_episodes": self.level_episodes,
        }


# ============================================================
# CURRICULUM MANAGER — Gestisce la promozione tra livelli
# ============================================================

class CurriculumManager:
    """
    Gestisce la progressione Easy -> Medium -> Hard con meccanismo
    di replay per contrastare il catastrophic forgetting.
    
    Promuove al livello successivo quando TUTTE le condizioni sono soddisfatte:
    1. Window success rate >= promotion_threshold
    2. Window collision rate <= collision_threshold (se specificato)
    3. Episodi sul livello corrente >= min_episodes
    4. Timesteps sul livello corrente >= min_timesteps
    5. La finestra mobile e' piena
    
    Dopo la promozione, una percentuale dei blocchi viene dedicata
    alla revisione dei livelli precedenti (replay_ratio).
    
    Changelog v2.0 (12 Marzo 2026):
    - Aggiunto min_timesteps: l'agente deve accumulare abbastanza
      esperienza (in step, non solo episodi) prima di poter essere promosso.
      Motivazione: nel run 1.5M, Easy->Medium avveniva dopo soli 100K step
      (527 episodi), insufficienti per consolidare la policy.
    - Aggiunto collision_threshold: il success rate da solo non basta.
      Un agente con 60% SR ma 40% CR non e' pronto per il livello successivo.
    - Aggiunto replay_ratio: dopo ogni promozione, una percentuale dei
      blocchi futuri viene dedicata a rivedere i livelli precedenti.
      Motivazione: nel run 1.5M, l'agente dimenticava completamente
      Easy dopo la promozione a Medium (catastrophic forgetting).
    - Alzata soglia default da 0.3 a 0.6: 30% window SR era troppo basso
      per garantire competenza solida sul livello.
    
    Args:
        levels: lista ordinata dei nomi dei livelli
        promotion_threshold: success rate minimo per promuovere (default 0.6)
        collision_threshold: collision rate massimo per promuovere (default 0.3)
        min_episodes: episodi minimi sul livello prima di valutare promozione
        min_timesteps: timesteps minimi sul livello prima di valutare promozione
        replay_ratio: frazione di blocchi da dedicare a livelli precedenti (0.0-1.0)
        window_size: dimensione finestra mobile per EpisodeTracker
    """

    def __init__(
        self,
        levels=None,
        promotion_threshold=0.6,
        collision_threshold=0.3,
        min_episodes=50,
        min_timesteps=200_000,
        replay_ratio=0.2,
        window_size=50,
    ):
        self.levels = levels or ["easy", "medium", "hard"]
        self.promotion_threshold = promotion_threshold
        self.collision_threshold = collision_threshold
        self.min_episodes = min_episodes
        self.min_timesteps = min_timesteps
        self.replay_ratio = replay_ratio
        self.window_size = window_size
        self.current_index = 0
        self.promotion_history = []
        # v2.0: contatore per gestire il replay
        self._block_counter = 0

    @property
    def current_level(self):
        return self.levels[self.current_index]

    @property
    def is_final_level(self):
        return self.current_index >= len(self.levels) - 1

    @property
    def completed_levels(self):
        """Lista dei livelli gia' completati (promossi)."""
        return self.levels[:self.current_index]

    def get_block_level(self):
        """
        Determina su quale livello trainare il prossimo blocco.
        
        Implementa il meccanismo di replay: dopo la promozione,
        una percentuale dei blocchi viene dedicata a livelli precedenti
        per contrastare il catastrophic forgetting.
        
        Returns:
            str: nome del livello per il prossimo blocco
            bool: True se questo e' un blocco di replay
        """
        self._block_counter += 1

        # Se siamo al primo livello, niente replay possibile
        if self.current_index == 0 or self.replay_ratio <= 0:
            return self.current_level, False

        # Replay: ogni N blocchi, ne dedichiamo uno a un livello precedente
        # Esempio: replay_ratio=0.2 -> 1 blocco su 5 e' replay
        replay_interval = max(1, int(1.0 / self.replay_ratio))

        if self._block_counter % replay_interval == 0:
            # Scegli un livello precedente (round-robin)
            completed = self.completed_levels
            replay_index = (self._block_counter // replay_interval) % len(completed)
            replay_level = completed[replay_index]
            return replay_level, True

        return self.current_level, False

    def should_promote(self, tracker):
        """
        Controlla se l'agente deve essere promosso al livello successivo.
        
        TUTTE le condizioni devono essere soddisfatte:
        1. Non siamo gia' all'ultimo livello
        2. Episodi sul livello >= min_episodes
        3. Timesteps sul livello >= min_timesteps
        4. La finestra mobile e' piena
        5. Window success rate >= promotion_threshold
        6. Window collision rate <= collision_threshold
        
        Args:
            tracker: EpisodeTracker con le metriche correnti
        
        Returns:
            True se deve essere promosso
        """
        if self.is_final_level:
            return False

        if tracker.level_episodes < self.min_episodes:
            return False

        if tracker.level_timesteps < self.min_timesteps:
            return False

        if not tracker.window_full:
            return False

        if tracker.window_success_rate < self.promotion_threshold:
            return False

        if tracker.window_collision_rate > self.collision_threshold:
            return False

        return True

    def promote(self, tracker, global_timestep=None):
        """
        Promuovi al livello successivo. Ritorna il nuovo livello.
        
        Registra nella history: livello di partenza, arrivo,
        episodi e timesteps sul livello, metriche alla promozione.
        """
        self.promotion_history.append({
            "from": self.current_level,
            "to": self.levels[self.current_index + 1],
            "episodes_on_level": tracker.level_episodes,
            "timesteps_on_level": tracker.level_timesteps,
            "success_rate_at_promotion": tracker.window_success_rate,
            "collision_rate_at_promotion": tracker.window_collision_rate,
            "timestep_at_promotion": global_timestep
        })
        self.current_index += 1
        tracker.reset()  # Resetta la finestra per il nuovo livello
        return self.current_level

    def promotion_status(self, tracker):
        """
        Ritorna un dizionario con lo stato di ciascun criterio di promozione.
        Utile per debug e per capire cosa manca alla promozione.
        """
        return {
            "is_final_level": self.is_final_level,
            "episodes_ok": tracker.level_episodes >= self.min_episodes,
            "episodes_current": tracker.level_episodes,
            "episodes_required": self.min_episodes,
            "timesteps_ok": tracker.level_timesteps >= self.min_timesteps,
            "timesteps_current": tracker.level_timesteps,
            "timesteps_required": self.min_timesteps,
            "window_full": tracker.window_full,
            "success_rate_ok": tracker.window_success_rate >= self.promotion_threshold,
            "success_rate_current": tracker.window_success_rate,
            "success_rate_required": self.promotion_threshold,
            "collision_rate_ok": tracker.window_collision_rate <= self.collision_threshold,
            "collision_rate_current": tracker.window_collision_rate,
            "collision_rate_max": self.collision_threshold,
        }

    def summary(self):
        """Ritorna un dizionario con la storia del curriculum."""
        return {
            "final_level": self.current_level,
            "levels_completed": self.current_index,
            "total_levels": len(self.levels),
            "promotion_threshold": self.promotion_threshold,
            "collision_threshold": self.collision_threshold,
            "min_episodes": self.min_episodes,
            "min_timesteps": self.min_timesteps,
            "replay_ratio": self.replay_ratio,
            "promotion_history": self.promotion_history,
        }
