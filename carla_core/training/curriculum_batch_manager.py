"""
CurriculumManager & BatchLevelSampler — CARLA MAPPO (Block 5.3 + 5.3b)
=======================================================================
Port from MetaDrive prototype, decoupled from simulator and framework.

Components:
  1. EpisodeTracker   — windowed SR/CR metrics for promotion decisions
  2. CurriculumManager — Easy→Medium→Hard progression with replay guard
  3. BatchLevelSampler — stratified shuffle w/o replacement, cap uniforme

Integration:
  Caller (training loop) reads metrics, calls these classes, then
  calls env.set_level(level) + env.reset(). No direct env dependency.

References:
  - MetaDrive prototype: metadrive_prototype/envs/multi_level_env.py
  - Design B: levels = map+traffic combo (levels.yaml)
"""

import logging
import random
from collections import deque
from copy import deepcopy

logger = logging.getLogger(__name__)


# ====================================================================
# EPISODE TRACKER
# ====================================================================

class EpisodeTracker:
    """Windowed success/collision tracker for promotion decisions.

    Tracks both a sliding window (for promotion checks) and cumulative
    totals (for final reporting). Window and level counters reset on
    level change; cumulative totals persist.

    Args:
        window_size: sliding window length in episodes.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.successes = deque(maxlen=window_size)
        self.collisions = deque(maxlen=window_size)
        self.total_episodes = 0
        self.total_successes = 0
        self.total_collisions = 0
        self.level_timesteps = 0
        self.level_episodes = 0

    # --- Recording ---

    def record(self, success: bool, collision: bool):
        """Record one episode outcome."""
        self.successes.append(1 if success else 0)
        self.collisions.append(1 if collision else 0)
        self.total_episodes += 1
        self.level_episodes += 1
        if success:
            self.total_successes += 1
        if collision:
            self.total_collisions += 1

    def record_counts(self, *, successes: int = 0, collisions: int = 0, total: int = 0):
        """Record multiple outcomes using aggregated counts."""
        success_count = max(int(successes), 0)
        collision_count = max(int(collisions), 0)
        total_count = max(int(total), success_count + collision_count)
        other_count = max(total_count - success_count - collision_count, 0)

        if total_count <= 0:
            return

        self.successes.extend([1] * success_count)
        self.successes.extend([0] * (collision_count + other_count))
        self.collisions.extend([0] * success_count)
        self.collisions.extend([1] * collision_count)
        self.collisions.extend([0] * other_count)

        self.total_episodes += total_count
        self.level_episodes += total_count
        self.total_successes += success_count
        self.total_collisions += collision_count

    def record_from_custom_metrics(self, custom_metrics: dict):
        """Record from RLlib custom_metrics dict (success_rate, collision_rate).

        Treats the per-iteration aggregate as a single observation.
        For finer granularity, use record() per episode instead.
        """
        sr = custom_metrics.get("success_rate_mean", custom_metrics.get("success_rate"))
        cr = custom_metrics.get("collision_rate_mean", custom_metrics.get("collision_rate"))
        if sr is not None:
            self.record(success=(sr >= 0.5), collision=(cr is not None and cr >= 0.5))

    def add_timesteps(self, n: int):
        """Accumulate timesteps for the current level."""
        self.level_timesteps += n

    # --- Properties ---

    @property
    def window_success_rate(self) -> float:
        if len(self.successes) == 0:
            return 0.0
        return sum(self.successes) / len(self.successes)

    @property
    def window_collision_rate(self) -> float:
        if len(self.collisions) == 0:
            return 0.0
        return sum(self.collisions) / len(self.collisions)

    @property
    def cumulative_success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_successes / self.total_episodes

    @property
    def cumulative_collision_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_collisions / self.total_episodes

    @property
    def window_full(self) -> bool:
        return len(self.successes) >= self.window_size

    # --- Lifecycle ---

    def reset(self):
        """Reset window + level counters for a new level. Keeps cumulative totals."""
        self.successes.clear()
        self.collisions.clear()
        self.level_timesteps = 0
        self.level_episodes = 0

    def summary(self) -> dict:
        return {
            "total_episodes": self.total_episodes,
            "total_successes": self.total_successes,
            "total_collisions": self.total_collisions,
            "cumulative_success_rate": self.cumulative_success_rate,
            "cumulative_collision_rate": self.cumulative_collision_rate,
            "window_success_rate": self.window_success_rate,
            "window_collision_rate": self.window_collision_rate,
            "window_size": len(self.successes),
            "window_capacity": self.window_size,
            "level_timesteps": self.level_timesteps,
            "level_episodes": self.level_episodes,
        }


# ====================================================================
# CURRICULUM MANAGER
# ====================================================================

class CurriculumManager:
    """Easy → Medium → Hard progression with replay for forgetting guard.

    After promotion, a fraction of blocks replays only the immediately
    previous level to prevent catastrophic forgetting without fully
    regressing to the easiest stage.

    The manager does NOT call env.set_level() — it returns the level name
    and the caller is responsible for applying it.

    Args:
        levels: ordered level names (default: ["easy", "medium", "hard"]).
        promotion_threshold: min window SR to promote (default 0.6).
        collision_threshold: max window CR to promote (default 0.3).
        min_episodes: min episodes on level before promotion check.
        min_timesteps: min timesteps on level before promotion check.
        replay_ratio: target replay fraction after first promotion.
        max_blocks_without_replay: forced replay after this many consecutive non-replay blocks.
        level_criteria: per-level override dict {level: {promotion_threshold, ...}}.
        window_size: EpisodeTracker window size (for reference).
    """

    def __init__(
        self,
        levels=None,
        promotion_threshold=0.6,
        collision_threshold=0.3,
        min_episodes=50,
        min_timesteps=200_000,
        replay_ratio=0.2,
        max_blocks_without_replay=2,
        level_criteria=None,
        window_size=50,
        replay_trigger_delta_sr=0.05,
        replay_trigger_delta_cr=0.05,
        replay_warmup_blocks_after_promotion=1,
    ):
        self.levels = levels or ["easy", "medium", "hard"]
        self.promotion_threshold = promotion_threshold
        self.collision_threshold = collision_threshold
        self.min_episodes = min_episodes
        self.min_timesteps = min_timesteps
        self.replay_ratio = replay_ratio
        self.max_blocks_without_replay = max(0, int(max_blocks_without_replay))
        self.level_criteria = level_criteria or {}
        self.window_size = window_size
        self.replay_trigger_delta_sr = max(0.0, float(replay_trigger_delta_sr))
        self.replay_trigger_delta_cr = max(0.0, float(replay_trigger_delta_cr))
        self.replay_warmup_blocks_after_promotion = max(0, int(replay_warmup_blocks_after_promotion))
        self.current_index = 0
        self.promotion_history = []
        self._blocks_since_replay = 0
        self._replay_credit = 0.0
        self._warmup_replays_remaining = 0

    # --- Properties ---

    @property
    def current_level(self) -> str:
        return self.levels[self.current_index]

    @property
    def is_final_level(self) -> bool:
        return self.current_index >= len(self.levels) - 1

    def get_replay_candidates(self) -> list:
        """Return replayable levels for the current stage."""
        if self.current_index <= 0:
            return []
        return [self.levels[self.current_index - 1]]

    def _criteria_for(self, level_name: str | None = None) -> dict:
        level_name = level_name or self.current_level
        criteria = self.level_criteria.get(level_name, {})
        return {
            "promotion_threshold": criteria.get("promotion_threshold", self.promotion_threshold),
            "collision_threshold": criteria.get("collision_threshold", self.collision_threshold),
            "min_timesteps": criteria.get("min_timesteps", self.min_timesteps),
            "max_timesteps": criteria.get("max_timesteps"),
        }

    def baseline_for_level(self, level_name: str):
        for item in reversed(self.promotion_history):
            if item.get("from") == level_name:
                return item
        return None

    def should_replay(self, tracker: EpisodeTracker | None) -> bool:
        if tracker is None or not tracker.window_full:
            return False

        replay_candidates = self.get_replay_candidates()
        if not replay_candidates:
            return False

        baseline = self.baseline_for_level(replay_candidates[0])
        if baseline is None:
            return False

        sr_drop = baseline["success_rate_at_promotion"] - tracker.window_success_rate
        cr_rise = tracker.window_collision_rate - baseline["collision_rate_at_promotion"]
        return (
            sr_drop >= self.replay_trigger_delta_sr
            or cr_rise >= self.replay_trigger_delta_cr
        )

    # --- Core API ---

    def get_episode_level(self, replay_tracker: EpisodeTracker | None = None):
        """Determine level for the next episode/block.

        Returns:
            (level_name: str, is_replay: bool)
        """
        if self.current_index == 0 or self.replay_ratio <= 0:
            return self.current_level, False

        replay_candidates = self.get_replay_candidates()
        if not replay_candidates:
            return self.current_level, False

        if self._warmup_replays_remaining > 0:
            self._warmup_replays_remaining -= 1
            self._blocks_since_replay = 0
            return replay_candidates[0], True

        forgetting_active = self.should_replay(replay_tracker)
        if not forgetting_active:
            self._blocks_since_replay = 0
            self._replay_credit = 0.0
            return self.current_level, False

        self._replay_credit += self.replay_ratio
        should_force = (
            self.max_blocks_without_replay > 0
            and self._blocks_since_replay >= self.max_blocks_without_replay
        )
        should_replay_from_credit = self._replay_credit >= 1.0

        if should_force or should_replay_from_credit:
            self._blocks_since_replay = 0
            if should_replay_from_credit:
                self._replay_credit = max(0.0, self._replay_credit - 1.0)
            return replay_candidates[0], True

        self._blocks_since_replay += 1
        return self.current_level, False

    def should_promote(self, tracker: EpisodeTracker, stage_timesteps: int | None = None) -> bool:
        """Check all promotion criteria against tracker metrics.

        All conditions must be met:
          1. Not already at final level
          2. level_episodes >= min_episodes
          3. level_timesteps >= min_timesteps
          4. Window full
          5. window_success_rate >= threshold
          6. window_collision_rate <= threshold
        """
        if self.is_final_level:
            return False

        criteria = self._criteria_for()
        req_sr = criteria["promotion_threshold"]
        req_cr = criteria["collision_threshold"]
        req_ts = criteria["min_timesteps"]
        req_max_ts = criteria["max_timesteps"]
        current_stage_timesteps = tracker.level_timesteps if stage_timesteps is None else int(stage_timesteps)

        if tracker.level_episodes < self.min_episodes:
            return False
        if req_max_ts is not None and current_stage_timesteps >= req_max_ts:
            return True
        if tracker.level_timesteps < req_ts:
            return False
        if not tracker.window_full:
            return False
        if tracker.window_success_rate < req_sr:
            return False
        if tracker.window_collision_rate > req_cr:
            return False

        return True

    def promote(
        self,
        tracker: EpisodeTracker,
        global_timestep=None,
        reason="thresholds_met",
        stage_timesteps: int | None = None,
    ) -> str:
        """Promote to next level. Returns new level name."""
        self.promotion_history.append({
            "from": self.current_level,
            "to": self.levels[self.current_index + 1],
            "episodes_on_level": tracker.level_episodes,
            "timesteps_on_level": tracker.level_timesteps,
            "stage_timesteps_at_promotion": (
                tracker.level_timesteps if stage_timesteps is None else int(stage_timesteps)
            ),
            "success_rate_at_promotion": tracker.window_success_rate,
            "collision_rate_at_promotion": tracker.window_collision_rate,
            "timestep_at_promotion": global_timestep,
            "reason": reason,
        })
        self.current_index += 1
        self._blocks_since_replay = 0
        self._replay_credit = 0.0
        self._warmup_replays_remaining = self.replay_warmup_blocks_after_promotion
        tracker.reset()
        logger.info(
            "Promoted to %s at timestep %s (SR=%.2f, CR=%.2f, reason=%s)",
            self.current_level, global_timestep,
            self.promotion_history[-1]["success_rate_at_promotion"],
            self.promotion_history[-1]["collision_rate_at_promotion"],
            self.promotion_history[-1]["reason"],
        )
        return self.current_level

    def promotion_status(self, tracker: EpisodeTracker, stage_timesteps: int | None = None) -> dict:
        """Diagnostic dict showing which promotion criteria are met/unmet."""
        criteria = self._criteria_for()
        req_sr = criteria["promotion_threshold"]
        req_cr = criteria["collision_threshold"]
        req_ts = criteria["min_timesteps"]
        req_max_ts = criteria["max_timesteps"]
        current_stage_timesteps = tracker.level_timesteps if stage_timesteps is None else int(stage_timesteps)

        return {
            "is_final_level": self.is_final_level,
            "episodes_ok": tracker.level_episodes >= self.min_episodes,
            "episodes_current": tracker.level_episodes,
            "episodes_required": self.min_episodes,
            "timesteps_ok": tracker.level_timesteps >= req_ts,
            "timesteps_current": tracker.level_timesteps,
            "timesteps_required": req_ts,
            "stage_timesteps_current": current_stage_timesteps,
            "max_timesteps_cap": req_max_ts,
            "max_timesteps_reached": (
                req_max_ts is not None and current_stage_timesteps >= req_max_ts
            ),
            "window_full": tracker.window_full,
            "success_rate_ok": tracker.window_success_rate >= req_sr,
            "success_rate_current": tracker.window_success_rate,
            "success_rate_required": req_sr,
            "collision_rate_ok": tracker.window_collision_rate <= req_cr,
            "collision_rate_current": tracker.window_collision_rate,
            "collision_rate_max": req_cr,
        }

    def summary(self) -> dict:
        return {
            "final_level": self.current_level,
            "levels_completed": self.current_index,
            "total_levels": len(self.levels),
            "promotion_threshold": self.promotion_threshold,
            "collision_threshold": self.collision_threshold,
            "min_episodes": self.min_episodes,
            "min_timesteps": self.min_timesteps,
            "replay_ratio": self.replay_ratio,
            "max_blocks_without_replay": self.max_blocks_without_replay,
            "replay_trigger_delta_sr": self.replay_trigger_delta_sr,
            "replay_trigger_delta_cr": self.replay_trigger_delta_cr,
            "replay_warmup_blocks_after_promotion": self.replay_warmup_blocks_after_promotion,
            "replay_candidates": self.get_replay_candidates(),
            "promotion_history": deepcopy(self.promotion_history),
        }


# ====================================================================
# BATCH LEVEL SAMPLER
# ====================================================================

class BatchLevelSampler:
    """Stratified shuffle without replacement — Design B batch training.

    Each window of K levels is a shuffled permutation of all training levels.
    Within a window, each level appears exactly once (no replacement).
    Across windows, a per-level counter guarantees uniform distribution
    with max imbalance of 1 episode.

    Args:
        levels: training level names (default: ["easy", "medium", "hard"]).
        window_size: K — levels per window (default: len(levels)).
        seed: random seed for reproducibility.
    """

    def __init__(self, levels=None, window_size=None, seed=42):
        self.levels = levels or ["easy", "medium", "hard"]
        self.window_size = window_size or len(self.levels)
        self._rng = random.Random(seed)
        self._counts = {lv: 0 for lv in self.levels}
        self._window = []
        self._cursor = 0
        self._total_samples = 0
        self._refill_window()

    def _refill_window(self):
        """Create a new shuffled window of levels."""
        self._window = list(self.levels)
        self._rng.shuffle(self._window)
        self._cursor = 0

    def sample(self) -> str:
        """Return next level from current window. Refill when exhausted."""
        if self._cursor >= len(self._window):
            self._refill_window()

        level = self._window[self._cursor]
        self._cursor += 1
        self._counts[level] += 1
        self._total_samples += 1
        return level

    def counts_balanced(self, max_diff: int = 1) -> bool:
        """Check if per-level counts are within max_diff of each other."""
        if not self._counts:
            return True
        vals = list(self._counts.values())
        return (max(vals) - min(vals)) <= max_diff

    def summary(self) -> dict:
        return {
            "levels": list(self.levels),
            "window_size": self.window_size,
            "total_samples": self._total_samples,
            "counts": dict(self._counts),
            "balanced": self.counts_balanced(),
        }
