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
        cross_radius: float = 6.0,
        max_cross: int = 3,
        return_meta: bool = False,
    ):
        """Chain sidewalk waypoints until cumulative distance >= target, crossing
        crosswalks at dead-ends to bridge sidewalk segments across junctions (P1).

        At a sidewalk dead-end the planner walks to the nearest crosswalk within
        ``cross_radius`` (``carla.Map.get_crosswalks``, parsed/cached per map),
        inserts intermediate waypoints across to the far curb, resolves the far
        sidewalk and resumes. Validated offline (audit_sidewalk_chains.py): median
        contiguous reach 40m->109m on Town03; fallback at ratio 0.5 21/40/56% ->
        5.8/11.5/23%.

        Args:
            origin_loc: carla.Location of the pedestrian.
            target_distance_m: desired route length in meters.
            carla_map: carla.Map (uses self._map if None).
            spacing: step size in meters for wp.next().
            min_route_ratio: lower route-length bound relative to target;
                routes shorter than target * min_route_ratio are rejected.
            cross_radius: max distance (m) from a dead-end to a crosswalk to cross.
            max_cross: max crosswalk crossings per route. Default 3 bounds
                pedestrian on-roadway exposure on long (hard, 100m) routes
                where this cap binds hardest; each crossing places the ped on
                the roadway, the main vehicle collision/offroad source (P1xV1).
                Trims the high-crossing tail, spares the typical 1-2-crossing
                routes that carry pedestrian SR.
            return_meta: if True return (wps, n_crossings); else wps (back-compat).

        Returns:
            list[carla.Waypoint] (or (wps, n_crossings) if return_meta); None /
            (None, n) if no sidewalk start or route < target * min_route_ratio.
        """
        cmap = carla_map or self._map
        start_wp = _get_sidewalk_waypoint(cmap, origin_loc)
        if start_wp is None:
            return (None, 0) if return_meta else None

        crosswalks = _get_crosswalks(cmap)
        wps = []
        current_wp = start_wp
        cumulative = 0.0
        n_crossings = 0
        used_cw = set()
        dedup = spacing * 0.6  # 1.5 m at spacing 2.5 — matches the validated audit

        def _cell(loc):
            return (round(loc.x / dedup), round(loc.y / dedup))

        seen = {_cell(start_wp.transform.location)}
        max_iters = int(target_distance_m / spacing * 3)  # safety cap
        steps = 0
        while steps < max_iters and cumulative < target_distance_m:
            nexts = [
                wp for wp in current_wp.next(spacing)
                if wp.lane_type == carla.LaneType.Sidewalk
                and _cell(wp.transform.location) not in seen
            ]
            if nexts:
                nxt = nexts[0]
                cumulative += current_wp.transform.location.distance(nxt.transform.location)
                wps.append(nxt)
                seen.add(_cell(nxt.transform.location))
                current_wp = nxt
                steps += 1
                continue

            # Dead-end: cross the nearest crosswalk to reach the far sidewalk.
            if n_crossings >= max_cross:
                break
            end_loc = current_wp.transform.location
            hit = _nearest_crosswalk(end_loc, crosswalks, used_cw, cross_radius)
            if hit is None:
                break
            _, cw = hit
            exit_v = max(cw["verts"],
                         key=lambda v: math.hypot(v.x - end_loc.x, v.y - end_loc.y))
            cross_d = math.hypot(exit_v.x - end_loc.x, exit_v.y - end_loc.y)
            # Intermediate crossing waypoints (on the roadway) every `spacing`.
            n_seg = max(1, int(cross_d / spacing))
            for k in range(1, n_seg + 1):
                t = k / n_seg
                cw_wp = cmap.get_waypoint(
                    carla.Location(
                        x=end_loc.x + (exit_v.x - end_loc.x) * t,
                        y=end_loc.y + (exit_v.y - end_loc.y) * t,
                        z=end_loc.z + (exit_v.z - end_loc.z) * t,
                    ),
                    project_to_road=True,
                )
                if cw_wp is not None:
                    wps.append(cw_wp)
            cumulative += cross_d
            n_crossings += 1
            used_cw.add(cw["idx"])
            land = _get_sidewalk_waypoint(
                cmap, carla.Location(x=exit_v.x, y=exit_v.y, z=exit_v.z))
            if land is None or _cell(land.transform.location) in seen:
                break
            wps.append(land)
            seen.add(_cell(land.transform.location))
            current_wp = land
            steps += 1

        if not wps:
            return (None, n_crossings) if return_meta else None

        # Reject chains shorter than target * min_route_ratio (prevents short chains).
        if target_distance_m > 0 and cumulative < target_distance_m * float(min_route_ratio):
            return (None, n_crossings) if return_meta else None

        return (wps, n_crossings) if return_meta else wps

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


# ---------------------------------------------------------------------------
# Crosswalk stitching (P1: bridge sidewalk segments across junctions)
# ---------------------------------------------------------------------------

_CROSSWALK_CACHE = {}  # map name -> list of {"idx", "verts", "edges"}


def _parse_crosswalks(raw, eps=0.5):
    """Split ``Map.get_crosswalks()`` flat point list into polygons.

    CARLA concatenates polygons; each polygon's first point is repeated to close
    it. Returns list of {"idx", "verts", "edges"}.
    """
    polys = []
    cur = []
    start = None
    for loc in raw:
        if start is None:
            start = loc
            cur = [loc]
            continue
        if math.hypot(loc.x - start.x, loc.y - start.y) < eps and len(cur) >= 3:
            polys.append(cur)
            start = None
            cur = []
        else:
            cur.append(loc)
    out = []
    for i, verts in enumerate(polys):
        n = len(verts)
        edges = [(verts[k], verts[(k + 1) % n]) for k in range(n)]
        out.append({"idx": i, "verts": verts, "edges": edges})
    return out


def _get_crosswalks(carla_map):
    """Parse and cache crosswalk polygons per map (static geometry)."""
    name = carla_map.name
    if name not in _CROSSWALK_CACHE:
        try:
            raw = list(carla_map.get_crosswalks())
        except (RuntimeError, AttributeError):  # binding/map dependent
            raw = []
        _CROSSWALK_CACHE[name] = _parse_crosswalks(raw)
    return _CROSSWALK_CACHE[name]


def _pt_seg_dist2d(px, py, ax, ay, bx, by):
    """2D distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _nearest_crosswalk(end_loc, crosswalks, used, radius):
    """Nearest crosswalk (min edge distance) to end_loc within radius, not used."""
    best = None
    for cw in crosswalks:
        if cw["idx"] in used:
            continue
        for (a, b) in cw["edges"]:
            d = _pt_seg_dist2d(end_loc.x, end_loc.y, a.x, a.y, b.x, b.y)
            if d <= radius and (best is None or d < best[0]):
                best = (d, cw)
    return best
