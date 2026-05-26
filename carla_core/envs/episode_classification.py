"""
Episode termination classification — pure logic, no CARLA dependency.
====================================================================
Extracted from CarlaMultiAgentEnv.step() so the classifier can be unit-tested
without a running simulator.

The classifier returns the per-agent termination_reason emitted in episodes.jsonl
and consumed by SR/collision/timeout aggregators downstream
(train_carla_mappo.py, centralized_critic.py, mappo_runtime.py).

Adds `route_short` to guard against degenerate-fallback false successes:
when a pedestrian/vehicle reaches the end of a route whose optimal length is
below `target_distance * min_route_ratio`, the agent is NOT counted as
route_complete. This mirrors the lower-bound guard already enforced by
route_planner.py for vehicle/pedestrian planners.
"""

from __future__ import annotations

TERMINATION_REASONS = (
    "alive",
    "collision",
    "route_complete",
    "route_short",
    "offroad",
    "stuck",
    "timeout",
)


def classify_termination_reason(
    *,
    agent_type: str,
    term: bool,
    trunc: bool,
    collision_flag: bool,
    current_wp_idx: int,
    num_waypoints: int,
    is_offroad: bool,
    route_completion: float,
    loop_penalty_active: bool,
    route_target_m: float,
    route_optimal_length_m: float,
    min_route_ratio: float = 0.5,
) -> str:
    """Return the termination reason for one agent at episode end.

    Args:
        agent_type: "vehicle" or "pedestrian".
        term: terminated flag (from PettingZoo / env _check_done).
        trunc: truncated flag (max steps reached).
        collision_flag: agent collided at least once during the episode.
        current_wp_idx: index of the current target waypoint in the route.
        num_waypoints: length of the agent's planned route.
        is_offroad: precomputed offroad flag (vehicles only; pass False for
            pedestrians or when terminate_on_offroad is disabled).
        route_completion: fraction of route consumed (0.0..1.0).
        loop_penalty_active: True if the loop-penalty term is currently active.
        route_target_m: configured route distance for this episode (m).
        route_optimal_length_m: actual planned route length (m); 0.0 if route
            has fewer than 2 waypoints.
        min_route_ratio: lower bound for `route_optimal_length / target` below
            which a "completed" route is demoted to `route_short`. Default
            mirrors route_planner.CARLARoutePlanner.plan_*_route_by_distance.

    Returns:
        One of TERMINATION_REASONS.
    """
    if not (term or trunc):
        return "alive"
    if collision_flag:
        return "collision"
    if current_wp_idx >= num_waypoints:
        if route_target_m > 0.0:
            ratio = route_optimal_length_m / route_target_m
            if ratio < min_route_ratio:
                return "route_short"
        return "route_complete"
    if agent_type == "vehicle" and is_offroad:
        return "offroad"
    if trunc:
        if route_completion < 0.3 or loop_penalty_active:
            return "stuck"
        return "timeout"
    return "timeout"
