"""
Unit tests for carla_core.envs.episode_classification.classify_termination_reason.

Pure logic — no CARLA, no fixtures, no network. Run with:
    python -m unittest carla_core.scripts.verify-check-test.test_episode_classification

Or directly:
    python carla_core/scripts/verify-check-test/test_episode_classification.py
"""

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from carla_core.envs.episode_classification import (
    classify_termination_reason,
    TERMINATION_REASONS,
)


def _kwargs(**overrides):
    """Default valid kwargs; tests override what they exercise."""
    base = dict(
        agent_type="vehicle",
        term=False,
        trunc=False,
        collision_flag=False,
        current_wp_idx=0,
        num_waypoints=10,
        is_offroad=False,
        route_completion=0.0,
        loop_penalty_active=False,
        route_target_m=30.0,
        route_optimal_length_m=30.0,
    )
    base.update(overrides)
    return base


class TestClassifyTerminationReason(unittest.TestCase):

    def test_alive_when_neither_term_nor_trunc(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(term=False, trunc=False)),
            "alive",
        )

    def test_collision_takes_priority_over_route_complete(self):
        # Even if current_wp_idx >= num_waypoints, collision wins.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                term=True,
                collision_flag=True,
                current_wp_idx=10,
                num_waypoints=10,
            )),
            "collision",
        )

    def test_collision_takes_priority_over_offroad(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                term=True,
                collision_flag=True,
                is_offroad=True,
            )),
            "collision",
        )

    def test_vehicle_route_complete_at_target_length(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                route_target_m=30.0,
                route_optimal_length_m=30.0,
            )),
            "route_complete",
        )

    def test_pedestrian_route_complete_at_target_length(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="pedestrian",
                term=True,
                current_wp_idx=8,
                num_waypoints=8,
                route_target_m=30.0,
                route_optimal_length_m=29.0,
            )),
            "route_complete",
        )

    # ---- NEW guard: route_short demotion ----

    def test_pedestrian_route_short_when_optimal_is_zero(self):
        # The buggy episode: route_optimal_length_m=0, target=100, reached end.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="pedestrian",
                term=True,
                current_wp_idx=1,
                num_waypoints=1,
                route_target_m=100.0,
                route_optimal_length_m=0.0,
            )),
            "route_short",
        )

    def test_pedestrian_route_short_below_min_ratio(self):
        # 26m / 100m = 0.26 < 0.5 → route_short, not route_complete.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="pedestrian",
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                route_target_m=100.0,
                route_optimal_length_m=26.0,
            )),
            "route_short",
        )

    def test_vehicle_route_short_below_min_ratio(self):
        # Symmetric guard for vehicles on legacy_fallback short routes.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                term=True,
                current_wp_idx=5,
                num_waypoints=5,
                route_target_m=100.0,
                route_optimal_length_m=20.0,
            )),
            "route_short",
        )

    def test_route_complete_when_ratio_above_threshold(self):
        # 60m / 100m = 0.6 >= 0.5 → route_complete (degenerate edge avoided).
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                route_target_m=100.0,
                route_optimal_length_m=60.0,
            )),
            "route_complete",
        )

    def test_route_complete_at_exact_threshold(self):
        # 50m / 100m = 0.5 — NOT < 0.5, so route_complete.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                route_target_m=100.0,
                route_optimal_length_m=50.0,
            )),
            "route_complete",
        )

    def test_route_complete_when_target_zero(self):
        # Pre-Block-5.1 runs with no configured target → no demotion.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                route_target_m=0.0,
                route_optimal_length_m=0.0,
            )),
            "route_complete",
        )

    def test_custom_min_route_ratio(self):
        # Caller can tighten the guard.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                route_target_m=100.0,
                route_optimal_length_m=70.0,
                min_route_ratio=0.8,
            )),
            "route_short",
        )

    # ---- Offroad / stuck / timeout coverage ----

    def test_vehicle_offroad(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                term=True,
                current_wp_idx=3,
                num_waypoints=10,
                is_offroad=True,
            )),
            "offroad",
        )

    def test_vehicle_offroad_only_when_not_reached_end(self):
        # If route_complete and offroad both indicated, route_complete wins
        # (offroad is checked only when current_wp_idx < num_waypoints).
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                term=True,
                current_wp_idx=10,
                num_waypoints=10,
                is_offroad=True,
            )),
            "route_complete",
        )

    def test_pedestrian_ignores_is_offroad(self):
        # Pedestrians are never classified as offroad.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="pedestrian",
                term=True,
                trunc=True,
                current_wp_idx=3,
                num_waypoints=10,
                is_offroad=True,
                route_completion=0.5,
            )),
            "timeout",
        )

    def test_vehicle_stuck_low_route_completion(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                trunc=True,
                current_wp_idx=2,
                num_waypoints=10,
                route_completion=0.2,
            )),
            "stuck",
        )

    def test_vehicle_stuck_via_loop_penalty(self):
        # High route_completion but loop_penalty active → stuck.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                trunc=True,
                current_wp_idx=8,
                num_waypoints=10,
                route_completion=0.8,
                loop_penalty_active=True,
            )),
            "stuck",
        )

    def test_vehicle_timeout(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                trunc=True,
                current_wp_idx=8,
                num_waypoints=10,
                route_completion=0.8,
            )),
            "timeout",
        )

    def test_pedestrian_stuck(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="pedestrian",
                trunc=True,
                current_wp_idx=1,
                num_waypoints=10,
                route_completion=0.1,
            )),
            "stuck",
        )

    def test_pedestrian_timeout(self):
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="pedestrian",
                trunc=True,
                current_wp_idx=7,
                num_waypoints=10,
                route_completion=0.7,
            )),
            "timeout",
        )

    def test_vehicle_terminated_without_specific_cause_falls_back_to_timeout(self):
        # term=True but not collision/route_complete/offroad/trunc — defaults to timeout.
        self.assertEqual(
            classify_termination_reason(**_kwargs(
                agent_type="vehicle",
                term=True,
                current_wp_idx=3,
                num_waypoints=10,
            )),
            "timeout",
        )

    def test_returned_reason_is_in_canonical_set(self):
        # Exhaustively, no path returns an unknown label.
        for term_val, trunc_val, collision, idx, offroad, rc, loop, target, optimal in [
            (False, False, False, 0, False, 0.0, False, 30.0, 30.0),
            (True,  False, True,  0, False, 0.0, False, 30.0, 30.0),
            (True,  False, False, 10, False, 0.0, False, 30.0, 30.0),
            (True,  False, False, 10, False, 0.0, False, 100.0, 10.0),
            (True,  False, False, 3,  True,  0.0, False, 30.0, 30.0),
            (False, True,  False, 1,  False, 0.1, False, 30.0, 30.0),
            (False, True,  False, 7,  False, 0.7, False, 30.0, 30.0),
            (True,  False, False, 3,  False, 0.0, False, 30.0, 30.0),
        ]:
            res = classify_termination_reason(**_kwargs(
                term=term_val,
                trunc=trunc_val,
                collision_flag=collision,
                current_wp_idx=idx,
                num_waypoints=10,
                is_offroad=offroad,
                route_completion=rc,
                loop_penalty_active=loop,
                route_target_m=target,
                route_optimal_length_m=optimal,
            ))
            self.assertIn(res, TERMINATION_REASONS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
