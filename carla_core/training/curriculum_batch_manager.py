"""
CurriculumManager & BatchLevelSampler — CARLA MAPPO (Block 5.3 + 5.3b)
=======================================================================
Port from MetaDrive prototype, decoupled from simulator and framework.

Components:
  1. EpisodeTracker   — windowed SR/CR metrics for curriculum competence checks
  2. CurriculumManager — budget-normalized distributional teacher
  3. BatchLevelSampler — i.i.d. uniform random sampling

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
    """Windowed success/collision tracker for curriculum competence checks.

    Tracks both a sliding window (for unlock checks) and cumulative
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
    """Budget-normalized distributional teacher with competence-based unlocks.

    The manager unlocks harder levels when the immediately preceding level
    reaches competence criteria, then keeps sampling distributional across the
    unlocked levels while enforcing cumulative budget constraints relative to
    the total training budget.
    """

    def __init__(
        self,
        levels=None,
        total_budget_timesteps=1_000_000,
        default_success_rate_threshold=0.45,
        default_collision_rate_threshold=0.30,
        default_min_episodes=50,
        unlock_criteria=None,
        budget_constraints=None,
        base_sampling_weights=None,
        probation_sampling_weights=None,
        probation_blocks_after_unlock=2,
        probation_blocks_after_cap_pressure=1,
        teacher_seed=42,
        window_size=50,
    ):
        self.levels = levels or ["easy", "medium", "hard"]
        self.total_budget_timesteps = max(int(total_budget_timesteps), 1)
        self.default_success_rate_threshold = float(default_success_rate_threshold)
        self.default_collision_rate_threshold = float(default_collision_rate_threshold)
        self.default_min_episodes = max(int(default_min_episodes), 1)
        self.unlock_criteria = unlock_criteria or {}
        self.window_size = max(int(window_size), 1)
        self.probation_blocks_after_unlock = max(0, int(probation_blocks_after_unlock))
        self.probation_blocks_after_cap_pressure = max(0, int(probation_blocks_after_cap_pressure))
        self._rng = random.Random(teacher_seed)
        self._validate_share(
            self.default_success_rate_threshold,
            "curriculum.success_rate_threshold",
            allow_none=False,
        )
        self._validate_share(
            self.default_collision_rate_threshold,
            "curriculum.collision_rate_threshold",
            allow_none=False,
        )

        default_weights = {
            level_name: 1.0 + (0.25 * idx)
            for idx, level_name in enumerate(self.levels)
        }
        self.base_sampling_weights = self._normalize_weights(
            base_sampling_weights or default_weights,
            allowed_levels=self.levels,
        )
        self.probation_sampling_weights = self._normalize_weights(
            probation_sampling_weights or {"medium": 1.0, "hard": 1.35},
            allowed_levels=self.levels,
        )

        budget_constraints = budget_constraints or {}
        self._validate_share(
            budget_constraints.get("easy_max_share"),
            "curriculum.budget_constraints.easy_max_share",
        )
        self._validate_share(
            budget_constraints.get("medium_max_share"),
            "curriculum.budget_constraints.medium_max_share",
        )
        self._validate_share(
            budget_constraints.get("hard_min_share"),
            "curriculum.budget_constraints.hard_min_share",
        )
        self.easy_max_share = self._clip_share(budget_constraints.get("easy_max_share"))
        self.medium_max_share = self._clip_share(budget_constraints.get("medium_max_share"))
        self.hard_min_share = self._clip_share(
            budget_constraints.get("hard_min_share"),
            default=0.0,
        )
        self._validate_configuration()

        self.current_level = self.levels[0]
        self.unlocked_levels = [self.levels[0]]
        self._excluded_from_sampling = set()
        self._timesteps_by_level = {level_name: 0 for level_name in self.levels}
        self._blocks_by_level = {level_name: 0 for level_name in self.levels}
        self._sample_counts = {level_name: 0 for level_name in self.levels}
        self._sample_counts[self.current_level] = 1
        self.unlock_history = []
        self.probation_history = []
        self._probation_remaining = 0
        self._cap_pressure_active = False
        self._last_probabilities = {self.levels[0]: 1.0}
        self._last_constraints = {}

    @property
    def total_assigned_timesteps(self) -> int:
        return int(sum(self._timesteps_by_level.values()))

    @property
    def hard_unlocked(self) -> bool:
        return "hard" in self.unlocked_levels

    def _clip_share(self, value, default=None) -> float | None:
        if value is None:
            return default
        return max(0.0, min(float(value), 1.0))

    def _validate_share(self, value, field_name: str, *, allow_none: bool = True) -> float | None:
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"{field_name} cannot be None")
        try:
            share = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a float-compatible value") from exc
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"{field_name} must be within [0.0, 1.0], got {share}")
        return share

    def _validate_configuration(self):
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("curriculum.levels must contain unique level names")

        if (
            self.medium_max_share is not None
            and self.hard_min_share is not None
            and self.medium_max_share + self.hard_min_share > 1.0 + 1e-9
        ):
            raise ValueError(
                "curriculum.budget_constraints.medium_max_share + "
                "curriculum.budget_constraints.hard_min_share must be <= 1.0"
            )

        for target_level in self.levels[1:]:
            criteria = self.unlock_criteria.get(target_level, {})
            min_budget_share = criteria.get(
                "min_budget_share",
                criteria.get("min_timesteps_share", 0.0),
            )
            self._validate_share(
                min_budget_share,
                f"curriculum.unlock_criteria.{target_level}.min_budget_share",
                allow_none=False,
            )
            self._validate_share(
                criteria.get("force_unlock_global_share_cap"),
                f"curriculum.unlock_criteria.{target_level}.force_unlock_global_share_cap",
            )
            collision_rate_threshold = criteria.get(
                "collision_rate_threshold",
                criteria.get("collision_threshold"),
            )
            self._validate_share(
                collision_rate_threshold,
                f"curriculum.unlock_criteria.{target_level}.collision_rate_threshold",
            )
            self._validate_share(
                criteria.get("success_rate_threshold"),
                f"curriculum.unlock_criteria.{target_level}.success_rate_threshold",
            )

    def _normalize_weights(self, weights: dict, allowed_levels: list[str]) -> dict:
        normalized = {}
        for level_name in allowed_levels:
            normalized[level_name] = max(0.0, float(weights.get(level_name, 0.0)))
        if sum(normalized.values()) <= 0.0:
            equal_weight = 1.0 / max(len(allowed_levels), 1)
            return {level_name: equal_weight for level_name in allowed_levels}
        return normalized

    def _criteria_for(self, target_level: str) -> dict:
        criteria = self.unlock_criteria.get(target_level, {})
        min_budget_share = criteria.get(
            "min_budget_share",
            criteria.get("min_timesteps_share", 0.0),
        )
        collision_rate_threshold = criteria.get(
            "collision_rate_threshold",
            criteria.get("collision_threshold", self.default_collision_rate_threshold),
        )
        return {
            "min_episodes": max(int(criteria.get("min_episodes", self.default_min_episodes)), 1),
            "min_budget_share": max(0.0, float(min_budget_share)),
            "force_unlock_global_share_cap": self._clip_share(
                criteria.get("force_unlock_global_share_cap")
            ),
            "success_rate_threshold": float(
                criteria.get("success_rate_threshold", self.default_success_rate_threshold)
            ),
            "collision_rate_threshold": float(collision_rate_threshold),
        }

    def _unlock_reason(self, tracker: EpisodeTracker | None, target_level: str) -> tuple[str | None, dict]:
        if tracker is None:
            return None, {}

        criteria = self._criteria_for(target_level)
        required_timesteps = int(criteria["min_budget_share"] * self.total_budget_timesteps)
        force_unlock_share_cap = criteria["force_unlock_global_share_cap"]
        force_unlock_timesteps = None
        if force_unlock_share_cap is not None:
            force_unlock_timesteps = int(force_unlock_share_cap * self.total_budget_timesteps)

        if tracker.level_episodes < criteria["min_episodes"]:
            return None, criteria
        if not tracker.window_full:
            return None, criteria
        if tracker.window_collision_rate > criteria["collision_rate_threshold"]:
            return None, criteria
        if (
            tracker.level_timesteps >= required_timesteps
            and tracker.window_success_rate >= criteria["success_rate_threshold"]
        ):
            return "competence_unlocked", criteria
        if (
            force_unlock_timesteps is not None
            and self.total_assigned_timesteps >= force_unlock_timesteps
        ):
            return "forced_global_budget_cap", criteria
        return None, criteria

    def _sampling_levels(self) -> list[str]:
        return [
            level_name
            for level_name in self.unlocked_levels
            if level_name not in self._excluded_from_sampling
        ]

    def _activate_probation(self, *, reason: str, global_timestep=None, blocks: int):
        blocks = max(0, int(blocks))
        if blocks <= 0:
            return
        self._probation_remaining = max(self._probation_remaining, blocks)
        self.probation_history.append({
            "reason": reason,
            "timestep": global_timestep,
            "probation_blocks": blocks,
        })

    def _maybe_unlock_levels(self, trackers: dict | None = None, global_timestep=None) -> list[str]:
        trackers = trackers or {}
        events = []

        for idx in range(1, len(self.levels)):
            target_level = self.levels[idx]
            if target_level in self.unlocked_levels:
                continue

            source_level = self.levels[idx - 1]
            if source_level not in self.unlocked_levels:
                break

            tracker = trackers.get(source_level)
            unlock_reason, criteria = self._unlock_reason(tracker, target_level)
            if unlock_reason is None:
                break

            unlock_event = {
                "from": source_level,
                "to": target_level,
                "episodes_on_level": tracker.level_episodes,
                "timesteps_on_level": tracker.level_timesteps,
                "success_rate_at_unlock": tracker.window_success_rate,
                "collision_rate_at_unlock": tracker.window_collision_rate,
                "timestep_at_unlock": global_timestep,
                "assigned_budget_share_at_unlock": (
                    float(self.total_assigned_timesteps) / float(self.total_budget_timesteps)
                ),
                "reason": unlock_reason,
                "criteria_snapshot": deepcopy(criteria),
            }
            self.unlocked_levels.append(target_level)
            self.unlock_history.append(unlock_event)
            events.append(f"unlock:{target_level}")

            logger.info(
                "Unlocked %s from %s at timestep %s (SR=%.2f, CR=%.2f)",
                target_level,
                source_level,
                global_timestep,
                tracker.window_success_rate,
                tracker.window_collision_rate,
            )

            if target_level == "hard":
                self._excluded_from_sampling.add("easy")
                self._activate_probation(
                    reason="unlock_hard",
                    global_timestep=global_timestep,
                    blocks=self.probation_blocks_after_unlock,
                )

        return events

    def record_execution(self, level_name: str, timesteps: int):
        timesteps = max(int(timesteps), 0)
        if level_name not in self._timesteps_by_level or timesteps <= 0:
            return
        self._timesteps_by_level[level_name] += timesteps
        self._blocks_by_level[level_name] += 1

    def _build_base_probabilities(self, levels: list[str], weights: dict) -> dict:
        if not levels:
            return {}

        total_weight = sum(max(weights.get(level_name, 0.0), 0.0) for level_name in levels)
        if total_weight <= 0.0:
            equal_prob = 1.0 / len(levels)
            return {level_name: equal_prob for level_name in levels}

        return {
            level_name: max(weights.get(level_name, 0.0), 0.0) / total_weight
            for level_name in levels
        }

    def _remaining_budget_timesteps(self) -> int:
        return max(self.total_budget_timesteps - self.total_assigned_timesteps, 0)

    def _dynamic_ceiling(self, level_name: str, max_share: float | None) -> float | None:
        if max_share is None:
            return None

        remaining_budget = self._remaining_budget_timesteps()
        if remaining_budget <= 0:
            return 0.0

        remaining_allowance = max(
            int(round(max_share * self.total_budget_timesteps)) - self._timesteps_by_level[level_name],
            0,
        )
        return max(0.0, min(float(remaining_allowance) / float(remaining_budget), 1.0))

    def _dynamic_hard_floor(self) -> float:
        if not self.hard_unlocked or self.hard_min_share <= 0.0:
            return 0.0

        remaining_budget = self._remaining_budget_timesteps()
        if remaining_budget <= 0:
            target_hard_budget = int(round(self.hard_min_share * self.total_budget_timesteps))
            return 1.0 if self._timesteps_by_level.get("hard", 0) < target_hard_budget else 0.0

        required_hard_budget = max(
            int(round(self.hard_min_share * self.total_budget_timesteps)) - self._timesteps_by_level.get("hard", 0),
            0,
        )
        return max(0.0, min(float(required_hard_budget) / float(remaining_budget), 1.0))

    def _constraint_state(self, raw_probabilities: dict) -> dict:
        easy_ceiling = None
        medium_ceiling = None

        if "medium" in self.unlocked_levels and "easy" not in self._excluded_from_sampling:
            easy_ceiling = self._dynamic_ceiling("easy", self.easy_max_share)
        if self.hard_unlocked:
            medium_ceiling = self._dynamic_ceiling("medium", self.medium_max_share)

        hard_floor = self._dynamic_hard_floor()
        return {
            "remaining_budget_timesteps": self._remaining_budget_timesteps(),
            "easy_dynamic_ceiling": easy_ceiling,
            "medium_dynamic_ceiling": medium_ceiling,
            "hard_dynamic_floor": hard_floor,
            "easy_ceiling_active": (
                easy_ceiling is not None
                and "easy" in raw_probabilities
                and raw_probabilities["easy"] > easy_ceiling + 1e-9
            ),
            "medium_ceiling_active": (
                medium_ceiling is not None
                and "medium" in raw_probabilities
                and raw_probabilities["medium"] > medium_ceiling + 1e-9
            ),
            "hard_floor_active": (
                "hard" in raw_probabilities
                and hard_floor > raw_probabilities["hard"] + 1e-9
            ),
        }

    def _project_probabilities(
        self,
        levels: list[str],
        base_probabilities: dict,
        min_probabilities: dict,
        max_probabilities: dict,
    ) -> dict:
        if not levels:
            return {}

        probabilities = {level_name: 0.0 for level_name in levels}
        for level_name in levels:
            floor = max(0.0, float(min_probabilities.get(level_name, 0.0)))
            ceiling = max(0.0, float(max_probabilities.get(level_name, 1.0)))
            probabilities[level_name] = min(floor, ceiling)

        assigned_mass = sum(probabilities.values())
        if assigned_mass >= 1.0:
            if assigned_mass <= 0.0:
                return {levels[0]: 1.0}
            return {
                level_name: probabilities[level_name] / assigned_mass
                for level_name in levels
            }

        free_levels = {
            level_name
            for level_name in levels
            if probabilities[level_name] + 1e-9 < max(0.0, float(max_probabilities.get(level_name, 1.0)))
        }

        while free_levels:
            remaining_mass = max(1.0 - sum(probabilities.values()), 0.0)
            if remaining_mass <= 1e-9:
                break

            total_weight = sum(base_probabilities.get(level_name, 0.0) for level_name in free_levels)
            if total_weight <= 0.0:
                equal_weight = 1.0 / len(free_levels)
                weight_lookup = {level_name: equal_weight for level_name in free_levels}
            else:
                weight_lookup = {
                    level_name: base_probabilities.get(level_name, 0.0) / total_weight
                    for level_name in free_levels
                }

            saturated = []
            for level_name in list(free_levels):
                ceiling = max(0.0, float(max_probabilities.get(level_name, 1.0)))
                candidate = probabilities[level_name] + (remaining_mass * weight_lookup[level_name])
                if candidate >= ceiling - 1e-9:
                    probabilities[level_name] = ceiling
                    saturated.append(level_name)

            if not saturated:
                for level_name in free_levels:
                    probabilities[level_name] += remaining_mass * weight_lookup[level_name]
                break

            for level_name in saturated:
                free_levels.discard(level_name)

        total_probability = sum(probabilities.values())
        if total_probability <= 0.0:
            equal_prob = 1.0 / len(levels)
            return {level_name: equal_prob for level_name in levels}

        if abs(total_probability - 1.0) > 1e-9:
            slack_level = max(levels, key=lambda level_name: base_probabilities.get(level_name, 0.0))
            probabilities[slack_level] += max(1.0 - total_probability, 0.0)
            total_probability = sum(probabilities.values())

        return {
            level_name: probabilities[level_name] / total_probability
            for level_name in levels
        }

    def _sample_level(self, probabilities: dict) -> str:
        threshold = self._rng.random()
        cumulative = 0.0
        ordered_levels = list(probabilities.keys())
        fallback_level = ordered_levels[-1]
        for level_name in ordered_levels:
            cumulative += probabilities[level_name]
            if threshold <= cumulative:
                return level_name
        return fallback_level

    def get_episode_level(self, trackers: dict | None = None, global_timestep=None):
        events = self._maybe_unlock_levels(trackers=trackers, global_timestep=global_timestep)

        base_levels = self._sampling_levels()
        probation_active = self._probation_remaining > 0 and self.hard_unlocked

        general_probabilities = self._build_base_probabilities(
            base_levels,
            self.base_sampling_weights,
        )
        general_constraints = self._constraint_state(general_probabilities)

        cap_pressure_now = self.hard_unlocked and (
            general_constraints["medium_ceiling_active"]
            or general_constraints["hard_floor_active"]
        )
        if cap_pressure_now and not self._cap_pressure_active:
            self._activate_probation(
                reason="cap_pressure",
                global_timestep=global_timestep,
                blocks=self.probation_blocks_after_cap_pressure,
            )
            events.append("probation:cap_pressure")
            probation_active = self._probation_remaining > 0
        self._cap_pressure_active = cap_pressure_now

        if probation_active:
            sampling_levels = [
                level_name
                for level_name in ("medium", "hard")
                if level_name in base_levels
            ]
            base_probabilities = self._build_base_probabilities(
                sampling_levels,
                self.probation_sampling_weights,
            )
        else:
            sampling_levels = base_levels
            base_probabilities = general_probabilities

        constraint_state = self._constraint_state(base_probabilities)
        min_probabilities = {}
        max_probabilities = {}

        if "easy" in sampling_levels and constraint_state["easy_dynamic_ceiling"] is not None:
            max_probabilities["easy"] = constraint_state["easy_dynamic_ceiling"]
        if "medium" in sampling_levels and constraint_state["medium_dynamic_ceiling"] is not None:
            max_probabilities["medium"] = constraint_state["medium_dynamic_ceiling"]
        if "hard" in sampling_levels:
            min_probabilities["hard"] = constraint_state["hard_dynamic_floor"]

        probabilities = self._project_probabilities(
            sampling_levels,
            base_probabilities,
            min_probabilities=min_probabilities,
            max_probabilities=max_probabilities,
        )

        next_level = self._sample_level(probabilities)
        self.current_level = next_level
        self._sample_counts[next_level] += 1
        self._last_probabilities = deepcopy(probabilities)
        self._last_constraints = deepcopy(constraint_state)

        if self._probation_remaining > 0:
            self._probation_remaining -= 1

        diagnostics = {
            "events": events,
            "probabilities": deepcopy(probabilities),
            "constraints": deepcopy(constraint_state),
            "probation_active": probation_active,
            "sampling_levels": list(sampling_levels),
        }
        return next_level, diagnostics

    def summary(self) -> dict:
        budget_shares = {
            level_name: (
                float(self._timesteps_by_level[level_name]) / float(self.total_budget_timesteps)
            )
            for level_name in self.levels
        }
        sample_total = max(sum(self._sample_counts.values()), 1)
        return {
            "teacher_type": "budget_normalized_distributional",
            "current_level": self.current_level,
            "levels": list(self.levels),
            "window_size": self.window_size,
            "unlocked_levels": list(self.unlocked_levels),
            "sampling_levels": self._sampling_levels(),
            "excluded_levels": sorted(self._excluded_from_sampling),
            "total_budget_timesteps": self.total_budget_timesteps,
            "allocated_timesteps": self.total_assigned_timesteps,
            "timesteps_by_level": dict(self._timesteps_by_level),
            "blocks_by_level": dict(self._blocks_by_level),
            "budget_share_by_level": budget_shares,
            "sample_counts": dict(self._sample_counts),
            "sample_share_by_level": {
                level_name: float(self._sample_counts[level_name]) / float(sample_total)
                for level_name in self.levels
            },
            "base_sampling_weights": deepcopy(self.base_sampling_weights),
            "probation_sampling_weights": deepcopy(self.probation_sampling_weights),
            "unlock_criteria": {
                level_name: self._criteria_for(level_name)
                for level_name in self.levels[1:]
            },
            "budget_constraints": {
                "easy_max_share": self.easy_max_share,
                "medium_max_share": self.medium_max_share,
                "hard_min_share": self.hard_min_share,
            },
            "unlock_defaults": {
                "success_rate_threshold": self.default_success_rate_threshold,
                "collision_rate_threshold": self.default_collision_rate_threshold,
                "min_episodes": self.default_min_episodes,
            },
            "unlock_history": deepcopy(self.unlock_history),
            "promotion_history": deepcopy(self.unlock_history),
            "probation_remaining": self._probation_remaining,
            "probation_history": deepcopy(self.probation_history),
            "last_probabilities": deepcopy(self._last_probabilities),
            "last_constraints": deepcopy(self._last_constraints),
        }

# ====================================================================
# BATCH LEVEL SAMPLER
# ====================================================================

class BatchLevelSampler:
    """I.i.d. uniform random sampling — Design B batch training.

    Each call to sample() draws a level uniformly at random with replacement.

    Args:
        levels: training level names (default: ["easy", "medium", "hard"]).
        seed: random seed for reproducibility.
    """

    def __init__(self, levels=None, seed=42):
        self.levels = levels or ["easy", "medium", "hard"]
        self._rng = random.Random(seed)

    def sample(self) -> str:
        return self._rng.choice(self.levels)

    def summary(self) -> dict:
        return {"levels": list(self.levels)}
