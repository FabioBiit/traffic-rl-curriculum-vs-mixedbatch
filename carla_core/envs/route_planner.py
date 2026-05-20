"""
CARLARoutePlanner — A* route generation by target distance (Block 5.1)
======================================================================
Thin wrapper around CARLA's GlobalRoutePlanner for road-graph A* routing.

Vehicle routes:  A* on road topology → list[carla.Waypoint]
Pedestrian routes: sidewalk waypoint chain by cumulative distance (no A*).

Usage:
    planner = CARLARoutePlanner(carla_map, sampling_resolution=2.0)
    wps = planner.plan_vehicle_route(origin, 200.0, spawn_points, rng)
    ped_wps = planner.plan_pedestrian_route_by_distance(origin, 80.0, carla_map)
"""

import logging
import math
import os
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

try:
    import carla
except ImportError:
    raise ImportError("pip install carla==0.9.16")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import for CARLA agents (requires PythonAPI/carla on PYTHONPATH)
# ---------------------------------------------------------------------------

_GlobalRoutePlanner = None


def _candidate_carla_pythonapi_dirs():
    candidates = []

    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        candidates.append(Path(carla_root) / "PythonAPI" / "carla")

    candidates.append(Path("C:/CARLA_0.9.16/PythonAPI/carla"))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = str(candidate).lower() if os.name == "nt" else str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(candidate)

    return unique_candidates


def _path_on_syspath(path: Path) -> bool:
    target = str(path.resolve()) if path.exists() else str(path)
    if os.name == "nt":
        target = target.lower()

    for entry in sys.path:
        entry_str = str(entry)
        if os.name == "nt":
            entry_str = entry_str.lower()
        if entry_str == target:
            return True
    return False


def _maybe_add_carla_pythonapi_to_syspath():
    for candidate in _candidate_carla_pythonapi_dirs():
        if not candidate.is_dir():
            continue
        if not _path_on_syspath(candidate):
            sys.path.insert(0, str(candidate))
        return str(candidate)
    return None


def _ensure_grp_import():
    global _GlobalRoutePlanner
    if _GlobalRoutePlanner is not None:
        return

    last_exc = None
    try:
        module = import_module("agents.navigation.global_route_planner")
    except ImportError as exc:
        last_exc = exc
        _maybe_add_carla_pythonapi_to_syspath()
        try:
            module = import_module("agents.navigation.global_route_planner")
        except ImportError as retry_exc:
            last_exc = retry_exc
            searched = ", ".join(str(path) for path in _candidate_carla_pythonapi_dirs())
            raise ImportError(
                "Cannot import agents.navigation.global_route_planner. "
                "Tried the current Python path and these CARLA PythonAPI locations: "
                f"{searched}. Set CARLA_ROOT or add PythonAPI/carla to PYTHONPATH. "
                f"Last import error: {last_exc}"
            ) from retry_exc

    _GlobalRoutePlanner = getattr(module, "GlobalRoutePlanner")


# ---------------------------------------------------------------------------
# Route planner
# ---------------------------------------------------------------------------

class CARLARoutePlanner:
    """A* road-graph route planner for CARLA.

    Builds the topology graph once on init. Reuse across resets on the same map.
    Invalidate (create new instance) when switching maps.

    Args:
        carla_map: carla.Map instance from world.get_map().
        sampling_resolution: waypoint spacing in meters for the A* graph.
    """

    def __init__(self, carla_map, sampling_resolution: float = 2.0):
        _ensure_grp_import()
        self._map = carla_map
        self._sampling_resolution = sampling_resolution
        self._grp = _GlobalRoutePlanner(carla_map, sampling_resolution)
        logger.info(
            "CARLARoutePlanner built (map=%s, resolution=%.1fm)",
            carla_map.name, sampling_resolution,
        )

    # ------------------------------------------------------------------
    # Vehicle routing (A* on road graph)
    # ------------------------------------------------------------------

    def plan_vehicle_route(
        self,
        origin_loc,
        target_distance_m: float,
        spawn_points: list,
        rng: np.random.Generator | None = None,
        min_route_ratio: float = 0.5,
        max_route_ratio: float = 2.0,
        max_candidate_attempts: int = 32,
        return_diagnostics: bool = False,
    ):
        """Plan a vehicle route of approximately target_distance_m meters.

        Strategy:
          1. Filter spawn_points within [0.6x, 1.4x] euclidean distance.
          2. Shuffle candidates deterministically with rng.
          3. A* trace_route → extract waypoints.
          4. Keep routes within [min_route_ratio, max_route_ratio] target.
          5. Return the valid route closest to target distance.

        Args:
            origin_loc: carla.Location of the vehicle.
            target_distance_m: desired route length in meters.
            spawn_points: list of carla.Transform (map spawn points).
            rng: numpy Generator for reproducibility.
            min_route_ratio: lower route-length bound relative to target.
            max_route_ratio: upper route-length bound relative to target.
            max_candidate_attempts: maximum destinations to try.
            return_diagnostics: if True, return (waypoints, diagnostics).

        Returns:
            list[carla.Waypoint] or None if no valid route found. If
            return_diagnostics=True, returns (waypoints_or_none, diagnostics).
        """
        if rng is None:
            rng = np.random.default_rng()

        start_t = time.perf_counter()
        attempts_configured = max(1, int(max_candidate_attempts))
        diagnostics = {
            "route_candidate_attempts_configured": attempts_configured,
            "route_candidate_attempts_used": 0,
            "route_candidate_valid_count": 0,
            "route_candidate_rejected_short_count": 0,
            "route_candidate_rejected_long_count": 0,
            "route_candidate_no_route_count": 0,
            "route_planning_latency_ms": 0.0,
        }

        def finish(wps):
            diagnostics["route_planning_latency_ms"] = (
                time.perf_counter() - start_t
            ) * 1000.0
            if return_diagnostics:
                return wps, diagnostics
            return wps

        candidates = self._candidate_destinations(
            origin_loc, target_distance_m, spawn_points, rng
        )
        if not candidates:
            logger.warning("No suitable destination at ~%.0fm from origin", target_distance_m)
            return finish(None)

        best_wps = None
        best_error = float("inf")
        max_attempts = min(attempts_configured, len(candidates))
        lower = target_distance_m * float(min_route_ratio)
        upper = target_distance_m * float(max_route_ratio)

        for dest in candidates[:max_attempts]:
            diagnostics["route_candidate_attempts_used"] += 1
            try:
                raw_route = self._grp.trace_route(origin_loc, dest.location)
            except Exception as e:
                diagnostics["route_candidate_no_route_count"] += 1
                logger.debug("trace_route failed for candidate: %s", e)
                continue

            if not raw_route:
                diagnostics["route_candidate_no_route_count"] += 1
                continue

            # Extract waypoints from (wp, RoadOption) tuples
            wps = [wp for wp, _ in raw_route]

            route_len = _waypoints_length(wps)
            if route_len < lower:
                diagnostics["route_candidate_rejected_short_count"] += 1
                continue
            if route_len > upper:
                diagnostics["route_candidate_rejected_long_count"] += 1
                continue

            diagnostics["route_candidate_valid_count"] += 1
            error = abs(route_len - target_distance_m)
            if error < best_error:
                best_error = error
                best_wps = wps

        if best_wps is None:
            logger.debug(
                "No valid route after %d candidates for target %.0fm",
                diagnostics["route_candidate_attempts_used"],
                target_distance_m,
            )
            return finish(None)

        return finish(best_wps)

    # ------------------------------------------------------------------
    # Pedestrian routing (sidewalk chain by distance — no A*)
    # ------------------------------------------------------------------

    def plan_pedestrian_route_by_distance(
        self,
        origin_loc,
        target_distance_m: float,
        carla_map=None,
        spacing: float = 2.5,
        min_route_ratio: float = 0.5,
    ):
        """Chain sidewalk waypoints until cumulative distance >= target.

        Args:
            origin_loc: carla.Location of the pedestrian.
            target_distance_m: desired route length in meters.
            carla_map: carla.Map (uses self._map if None).
            spacing: step size in meters for wp.next().
            min_route_ratio: lower route-length bound relative to target;
                routes shorter than target * min_route_ratio are rejected.

        Returns:
            list[carla.Waypoint] or None if no sidewalk found or route too short.
        """
        cmap = carla_map or self._map
        start_wp = _get_sidewalk_waypoint(cmap, origin_loc)
        if start_wp is None:
            return None

        wps = []
        current_wp = start_wp
        cumulative = 0.0

        max_iters = int(target_distance_m / spacing * 3)  # safety cap
        for _ in range(max_iters):
            nexts = [
                wp for wp in current_wp.next(spacing)
                if wp.lane_type == carla.LaneType.Sidewalk
            ]
            if not nexts:
                break
            nxt = nexts[0]
            seg = current_wp.transform.location.distance(nxt.transform.location)
            cumulative += seg
            wps.append(nxt)
            current_wp = nxt
            if cumulative >= target_distance_m:
                break

        if not wps:
            return None

        # Ped-route: reject chains shorter than target * min_route_ratio (mirrors
        # the vehicle planner's lower bound — prevents short sidewalk chains).
        if target_distance_m > 0 and cumulative < target_distance_m * float(min_route_ratio):
            return None

        return wps

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _candidate_destinations(self, origin_loc, target_m, spawn_points, rng):
        """Return shuffled spawn points near the target euclidean distance."""
        lo, hi = target_m * 0.6, target_m * 1.4
        candidates = []
        for sp in spawn_points:
            d = origin_loc.distance(sp.location)
            if lo <= d <= hi:
                candidates.append(sp)

        if not candidates:
            # Fallback: widen to [0.3x, 2.0x]
            lo2, hi2 = target_m * 0.3, target_m * 2.0
            candidates = [sp for sp in spawn_points if lo2 <= origin_loc.distance(sp.location) <= hi2]

        if not candidates:
            return []

        order = rng.permutation(len(candidates))
        return [candidates[int(idx)] for idx in order]


# ---------------------------------------------------------------------------
# Helpers (module-level, reusable)
# ---------------------------------------------------------------------------

def _waypoints_length(wps) -> float:
    """Sum of consecutive waypoint-to-waypoint euclidean distances."""
    if len(wps) < 2:
        return 0.0
    total = 0.0
    for i in range(len(wps) - 1):
        total += wps[i].transform.location.distance(wps[i + 1].transform.location)
    return total


def _get_sidewalk_waypoint(carla_map, loc):
    """Resolve closest sidewalk waypoint (same logic as env)."""
    wp = carla_map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Sidewalk)
    if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
        return wp
    wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
    if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
        return wp
    return None
