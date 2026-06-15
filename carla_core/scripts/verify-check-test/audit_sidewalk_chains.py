#!/usr/bin/env python3
"""Offline Town03 sidewalk-chain audit -- P0-audit (step 0 of Fix-SR-Pedoni).

MEASUREMENT ONLY. No env / planner / policy / reward change. The script connects
to a running CARLA 0.9.16 server, (re)loads a town (default Town03), samples
pedestrian spawn seeds exactly the way the env does
(``world.get_random_location_from_navigation``, carla_multi_agent_env.py:902),
resolves each to a sidewalk waypoint with the same two-step logic as
``_get_sidewalk_waypoint`` (env:933-951 == route_planner.py:348-356), then
measures the reachable contiguous sidewalk length under two strategies:

  GREEDY  -- the CURRENT planner: always take ``next(spacing)[0]`` filtered to
             LaneType.Sidewalk (route_planner.py:285-297). Chains shorter than
             ``target * min_route_ratio`` (0.5) are rejected -> sidewalk_fallback
             (route_planner.py:304).
  BRANCH  -- the PROPOSED P1: explore ALL sidewalk ``next`` branches (bounded
             simple-path DFS, backtracking) and keep the longest reachable chain.
             Upper bound on what a branch-aware planner can achieve per seed.
  STITCH  -- Option A model: greedy sidewalk walk that, at a dead-end, crosses the
             nearest crosswalk (``carla.Map.get_crosswalks``) to the far curb and
             resumes. Tests whether bridging junctions recovers real 60/100 m routes.

Per curriculum target (path 30/60/100 m) it reports:
  * predicted CURRENT fallback rate (greedy chain < 0.5*target OR unresolved) --
    sanity-compare with empirical r0610 sidewalk_fallback share easy/med/hard =
    21.08 / 38.62 / 52.48 %. A close match validates the offline model.
  * branch-aware success and the RECOVERABLE fraction (branch ok where greedy
    fails) = routes P1 would convert from fallback to valid (no distance change).
  * respawn-rate estimate (even branch-aware < lo*target) and an acceptance-band
    sweep over ``lo`` to calibrate the P1 lower bound and test 100 m feasibility.

NOT covered here: the ghost-walker lifecycle (terminated walkers removed from
``self.agents`` but not stopped/destroyed, env:~622) -- that is a static code
fact, audited by reading, not by this geometry script.

The user runs this (it needs a live CARLA server). ``client.load_world`` resets
the world, so do not run it against a sim you need intact; use --no-reload to
audit the currently loaded map instead. A JSON dump is written for re-analysis
without re-querying CARLA.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime

try:
    import carla
except ImportError as exc:  # pragma: no cover - environment dependent
    sys.stderr.write(
        "ERROR: cannot import 'carla'. Run this with the same Python env used "
        "for training (the CARLA 0.9.16 PythonAPI must be importable).\n"
    )
    raise SystemExit(2) from exc

# Empirical r0610 (carla_mappo_20260610_192146) sidewalk_fallback share per level,
# recomputed from episodes.jsonl -- reference for validating the offline model.
EMPIRICAL_FALLBACK_PCT = {30: 21.08, 60: 38.62, 100: 52.48}
TARGET_TO_LEVEL = {30: "easy", 60: "medium", 100: "hard"}


# --------------------------------------------------------------------------- #
# Geometry helpers (faithful mirrors of the production code)
# --------------------------------------------------------------------------- #
def resolve_sidewalk(cmap, loc):
    """Mirror _get_sidewalk_waypoint (env:933-951): exact match then projected."""
    wp = cmap.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Sidewalk)
    if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
        return wp, "exact"
    wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
    if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
        return wp, "projected"
    return None, "none"


def sidewalk_nexts(wp, spacing):
    """next(spacing) filtered to Sidewalk -- route_planner.py:285-288."""
    return [w for w in wp.next(spacing) if w.lane_type == carla.LaneType.Sidewalk]


def greedy_reach(start_wp, spacing, max_steps):
    """CURRENT planner walk: always next[0]. Returns (length_m, deadend, steps)."""
    cur = start_wp
    total = 0.0
    steps = 0
    for _ in range(max_steps):
        nxts = sidewalk_nexts(cur, spacing)
        if not nxts:
            return total, True, steps  # dead-end (route_planner.py:289-290)
        nxt = nxts[0]
        total += cur.transform.location.distance(nxt.transform.location)
        cur = nxt
        steps += 1
        if total >= max_steps * spacing:  # unreachable guard; real stop is the cap
            break
    return total, False, steps  # hit step cap, not a dead-end


def branch_reach(start_wp, spacing, horizon_m, max_nodes, dedup):
    """PROPOSED branch-aware upper bound: longest simple-path length (bounded DFS).

    Backtracking DFS over all sidewalk next() branches; cells deduped on a grid to
    keep paths simple and terminate on loops. Early-exits once ``horizon_m`` is
    reached (longer chains are truncatable, so not needed). ``max_nodes`` caps the
    expansion; ``capped`` flags an under-estimate. Returns (best_m, capped, max_deg).
    """
    seen = set()
    state = {"best": 0.0, "visits": 0, "capped": False, "max_deg": 0}

    def cell(wp):
        loc = wp.transform.location
        return (round(loc.x / dedup), round(loc.y / dedup), round(loc.z / dedup))

    def dfs(wp, cum):
        if cum > state["best"]:
            state["best"] = cum
        if state["best"] >= horizon_m or state["capped"]:
            return
        state["visits"] += 1
        if state["visits"] >= max_nodes:
            state["capped"] = True
            return
        nxts = sidewalk_nexts(wp, spacing)
        if len(nxts) > state["max_deg"]:
            state["max_deg"] = len(nxts)
        for nxt in nxts:
            c = cell(nxt)
            if c in seen:
                continue
            seen.add(c)
            dfs(nxt, cum + wp.transform.location.distance(nxt.transform.location))
            seen.discard(c)
            if state["best"] >= horizon_m or state["capped"]:
                return

    seen.add(cell(start_wp))
    dfs(start_wp, 0.0)
    return state["best"], state["capped"], state["max_deg"]


# --------------------------------------------------------------------------- #
# Crosswalk stitching (Option A model)
# --------------------------------------------------------------------------- #
def parse_crosswalks(raw, eps=0.5):
    """Split ``Map.get_crosswalks()`` flat list into polygons.

    CARLA returns concatenated polygons; each polygon's first point is repeated
    at the end to close it. Returns list of {"idx", "verts", "edges"}.
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


def _pt_seg_dist2d(px, py, ax, ay, bx, by):
    """2D distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def nearest_crosswalk(end_loc, crosswalks, used, radius):
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


def stitched_reach(start_wp, cmap, spacing, max_steps, horizon_m,
                   crosswalks, cross_radius, max_cross, dedup):
    """Greedy sidewalk walk that crosses crosswalks at dead-ends (Option A).

    When the sidewalk chain dead-ends, walk to the nearest crosswalk within
    ``cross_radius``, cross to the far curb (farthest vertex), resolve to the
    sidewalk there and resume. Returns (length_m, crossings, deadends_with_cross).
    """
    def cell(loc):
        return (round(loc.x / dedup), round(loc.y / dedup))

    cur = start_wp
    total = 0.0
    crossings = 0
    cross_hits = 0
    used = set()
    seen = {cell(cur.transform.location)}
    steps = 0
    while steps < max_steps and total < horizon_m:
        nxts = [w for w in sidewalk_nexts(cur, spacing)
                if cell(w.transform.location) not in seen]
        if nxts:
            nxt = nxts[0]
            total += cur.transform.location.distance(nxt.transform.location)
            seen.add(cell(nxt.transform.location))
            cur = nxt
            steps += 1
            continue
        if crossings >= max_cross:
            break
        end_loc = cur.transform.location
        hit = nearest_crosswalk(end_loc, crosswalks, used, cross_radius)
        if hit is None:
            break
        cross_hits += 1
        _, cw = hit
        exit_v = max(cw["verts"], key=lambda v: math.hypot(v.x - end_loc.x, v.y - end_loc.y))
        total += math.hypot(exit_v.x - end_loc.x, exit_v.y - end_loc.y)
        crossings += 1
        used.add(cw["idx"])
        land = carla.Location(x=exit_v.x, y=exit_v.y, z=exit_v.z)
        nwp, _ = resolve_sidewalk(cmap, land)
        if nwp is None or cell(nwp.transform.location) in seen:
            break
        seen.add(cell(nwp.transform.location))
        cur = nwp
        steps += 1
    return total, crossings, cross_hits


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Offline Town03 sidewalk-chain audit (P0-audit).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=20.0, help="client connect/op timeout (s)")
    p.add_argument("--town", default="Town03")
    p.add_argument("--no-reload", action="store_true",
                   help="audit the currently loaded world instead of load_world(town)")
    p.add_argument("--n-seeds", type=int, default=600)
    p.add_argument("--spacing", type=float, default=2.5, help="wp.next() step (matches planner)")
    p.add_argument("--targets", default="30,60,100", help="curriculum path targets (m)")
    p.add_argument("--lo-grid", default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2",
                   help="lower-bound factors of target for the acceptance-band sweep")
    p.add_argument("--pedestrians-seed", type=int, default=999,
                   help="world.set_pedestrians_seed(seed) -> reproducible navmesh sampling (P0)")
    p.add_argument("--max-nodes", type=int, default=6000, help="DFS expansion budget per seed")
    p.add_argument("--dedup-grid", type=float, default=1.5, help="cell size (m) for simple-path dedup")
    p.add_argument("--no-crosswalks", action="store_false", dest="crosswalks", default=True,
                   help="disable crosswalk stitching (Option A model); default ON for the spike")
    p.add_argument("--cross-radius", type=float, default=6.0,
                   help="max distance (m) from a sidewalk dead-end to a crosswalk to cross it")
    p.add_argument("--max-cross", type=int, default=8, help="max crosswalk crossings per route")
    p.add_argument("--out", default=None, help="JSON dump path (default: ./audit_sidewalk_chains_<town>_<ts>.json)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    lo_grid = [float(x) for x in args.lo_grid.split(",") if x.strip()]
    horizon = max(targets) * max(lo_grid)
    max_steps = max(int(max(targets) / args.spacing * 3), int(horizon / args.spacing) + 2)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    if args.no_reload:
        world = client.get_world()
    else:
        world = client.load_world(args.town)
    cmap = world.get_map()
    map_name = cmap.name

    crosswalks = []
    if args.crosswalks:
        raw_cw = list(cmap.get_crosswalks())
        crosswalks = parse_crosswalks(raw_cw)
        print(f"# crosswalks: {len(crosswalks)} polygons parsed from {len(raw_cw)} points")
        if not crosswalks:
            print("# WARN: no crosswalks in this map -> Option A infeasible; stitch == greedy")

    # P0 mechanism: seed the pedestrian/navmesh RNG so get_random_location_from_navigation
    # is reproducible. world.yml (CARLA 0.9.16): "Should be set before pedestrians
    # are spawned" -- here it governs the navmesh location draws below.
    ped_seed_ok = True
    try:
        world.set_pedestrians_seed(int(args.pedestrians_seed))
    except AttributeError:
        ped_seed_ok = False
        sys.stderr.write("WARN: world.set_pedestrians_seed unavailable in this binding; "
                         "sampling is not reproducible.\n")

    print(f"# audit_sidewalk_chains | map={map_name} | n_seeds={args.n_seeds} | "
          f"spacing={args.spacing} | pedestrians_seed={'set' if ped_seed_ok else 'N/A'}")

    records = []
    nav_none = 0
    t0 = time.time()
    for i in range(args.n_seeds):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            nav_none += 1
            continue
        start_wp, how = resolve_sidewalk(cmap, loc)
        rec = {
            "i": i,
            "spawn": [loc.x, loc.y, loc.z],
            "resolve": how,
        }
        if start_wp is None:
            rec.update({"resolved": False, "greedy_m": 0.0, "greedy_deadend": True,
                        "branch_m": 0.0, "branch_capped": False,
                        "stitch_m": 0.0, "stitch_cross": 0, "stitch_deadend_hits": 0})
        else:
            g_len, g_dead, g_steps = greedy_reach(start_wp, args.spacing, max_steps)
            b_len, b_cap, b_deg = branch_reach(start_wp, args.spacing, horizon, args.max_nodes, args.dedup_grid)
            if args.crosswalks and crosswalks:
                s_len, s_cross, s_hits = stitched_reach(
                    start_wp, cmap, args.spacing, max_steps, horizon,
                    crosswalks, args.cross_radius, args.max_cross, args.dedup_grid)
            else:
                s_len, s_cross, s_hits = g_len, 0, 0
            sw = start_wp.transform.location
            rec.update({
                "resolved": True,
                "resolve_dist": loc.distance(sw),
                "greedy_m": g_len, "greedy_deadend": g_dead, "greedy_steps": g_steps,
                "branch_m": b_len, "branch_capped": b_cap, "branch_max_deg": b_deg,
                "stitch_m": s_len, "stitch_cross": s_cross, "stitch_deadend_hits": s_hits,
            })
        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.n_seeds} seeds ({time.time() - t0:.1f}s)", flush=True)

    resolved = [r for r in records if r["resolved"]]
    n_all = len(records) + 0  # navmesh None excluded from the chain stats but counted as failure below
    n_total_samples = len(records) + nav_none

    # --- summary ---
    print(f"\n## sampling: requested={args.n_seeds}  navmesh_none={nav_none}  "
          f"resolved={len(resolved)}/{len(records)} "
          f"({100 * len(resolved) / max(len(records), 1):.1f}% of spawned)")
    if resolved:
        rdist = [r["resolve_dist"] for r in resolved]
        print(f"   resolve branch: exact={sum(1 for r in resolved if r['resolve']=='exact')}  "
              f"projected={sum(1 for r in resolved if r['resolve']=='projected')}  "
              f"| spawn->sidewalk dist p50={pct(rdist,0.5):.2f}m p90={pct(rdist,0.9):.2f}m max={max(rdist):.2f}m")
        gl = [r["greedy_m"] for r in resolved]
        bl = [r["branch_m"] for r in resolved]
        ded = 100 * sum(1 for r in resolved if r["greedy_deadend"]) / len(resolved)
        cap = 100 * sum(1 for r in resolved if r.get("branch_capped")) / len(resolved)
        deg = [r.get("branch_max_deg", 0) for r in resolved]
        print(f"   greedy reach  (m): p10={pct(gl,0.1):.1f} p50={pct(gl,0.5):.1f} "
              f"p90={pct(gl,0.9):.1f} max={max(gl):.1f}  deadend={ded:.1f}%")
        print(f"   branch reach  (m): p10={pct(bl,0.1):.1f} p50={pct(bl,0.5):.1f} "
              f"p90={pct(bl,0.9):.1f} max={max(bl):.1f}  DFS_capped={cap:.1f}%")
        print(f"   branching: mean_max_outdeg={mean(deg):.2f}  "
              f"(>1 means greedy next[0] discards alternatives)")
        if args.crosswalks and crosswalks:
            sl = [r["stitch_m"] for r in resolved]
            ncr = [r["stitch_cross"] for r in resolved]
            used_x = 100 * sum(1 for r in resolved if r["stitch_cross"] > 0) / len(resolved)
            print(f"   stitch reach  (m): p10={pct(sl,0.1):.1f} p50={pct(sl,0.5):.1f} "
                  f"p90={pct(sl,0.9):.1f} max={max(sl):.1f}  used_crossing={used_x:.1f}% of seeds  "
                  f"mean_cross={mean(ncr):.2f}  (vs greedy p50={pct(gl,0.5):.1f})")

    # --- per-target band sweep (rates over ALL samples: navmesh_none + unresolved = failure) ---
    print(f"\n## per-target rates  (denominator = all {n_total_samples} samples; "
          f"unresolved/navmesh_none count as failure)")
    summary = {}
    for T in targets:
        lvl = TARGET_TO_LEVEL.get(int(T), "?")
        # CURRENT planner success = resolved AND greedy >= 0.5*T (route_planner.py:304)
        cur_ok = sum(1 for r in resolved if r["greedy_m"] >= 0.5 * T)
        cur_fallback = 100 * (1 - cur_ok / n_total_samples)
        emp = EMPIRICAL_FALLBACK_PCT.get(int(T))
        emp_s = f"{emp:.2f}%" if emp is not None else "n/a"
        print(f"\n  target={T:.0f}m ({lvl})  predicted CURRENT fallback={cur_fallback:.2f}%  "
              f"| empirical r0610={emp_s}")
        print(f"    {'lo':>4}  {'band':>8}  {'greedy':>8}  {'stitch':>8}  "
              f"{'recover':>8}  {'respawn':>8}   (recover/respawn vs STITCH)")
        rows = []
        for lo in lo_grid:
            thr = lo * T
            g_ok = sum(1 for r in resolved if r["greedy_m"] >= thr)
            s_ok = sum(1 for r in resolved if r["stitch_m"] >= thr)
            b_ok = sum(1 for r in resolved if r["branch_m"] >= thr)
            recover = sum(1 for r in resolved if r["stitch_m"] >= thr and r["greedy_m"] < thr)
            g_rate = 100 * g_ok / n_total_samples
            s_rate = 100 * s_ok / n_total_samples
            rec_rate = 100 * recover / n_total_samples
            respawn = 100 * (1 - s_ok / n_total_samples)
            print(f"    {lo:>4.2f}  {thr:>7.1f}m  {g_rate:>7.1f}%  {s_rate:>7.1f}%  "
                  f"{rec_rate:>7.1f}%  {respawn:>7.1f}%")
            rows.append({"lo": lo, "thr_m": thr, "greedy_ok_pct": g_rate,
                         "stitch_ok_pct": s_rate, "branch_ok_pct": 100 * b_ok / n_total_samples,
                         "recoverable_stitch_pct": rec_rate, "respawn_stitch_pct": respawn})
        summary[str(int(T))] = {"level": lvl, "predicted_current_fallback_pct": cur_fallback,
                                "empirical_fallback_pct": emp, "band_sweep": rows}

    # --- dump ---
    out = args.out or f"audit_sidewalk_chains_{map_name.split('/')[-1]}_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "meta": {
            "map": map_name, "n_seeds_requested": args.n_seeds,
            "navmesh_none": nav_none, "resolved": len(resolved), "spawned": len(records),
            "spacing": args.spacing, "targets": targets, "lo_grid": lo_grid,
            "horizon_m": horizon, "max_steps": max_steps, "max_nodes": args.max_nodes,
            "n_crosswalks": len(crosswalks), "cross_radius": args.cross_radius,
            "max_cross": args.max_cross, "crosswalks_enabled": args.crosswalks,
            "dedup_grid": args.dedup_grid, "pedestrians_seed": args.pedestrians_seed,
            "pedestrians_seed_applied": ped_seed_ok, "generated": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": summary,
        "records": records,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n# wrote {out}  ({len(records)} records, {time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
