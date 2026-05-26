"""
Static (post-hoc) evaluator for a CARLA MAPPO training run.
==========================================================
Reads `episodes.jsonl` from disk and emits the standard cumulative metrics
per <measurement_rules> in CLAUDE.md:
  - 6 agent records per episode, deduped by (episode_id, agent_id), keep last.
  - Vehicles / pedestrians / combined reported separately.
  - Primary success = termination_reason == "route_complete".
  - Adds Block-5.1 correction: route_complete demoted to route_short when
    route_optimal_length / route_target < min_route_ratio (default 0.5).
  - Curriculum extras: per-level breakdown, route_source distribution,
    level unlock indices, quarter-by-quarter trajectory.

This does NOT replace the deterministic evaluator (`final_eval_job` /
`eval_carla_mappo`) — it is a static read of training-time episodes for
fast inspection of gates and trajectories.

Usage
-----
    python evaluate_run_static.py 20260525_205912
    python evaluate_run_static.py carla_mappo_20260525_205912
    python evaluate_run_static.py 20260525_205912 --mode batch
    python evaluate_run_static.py --run-dir "C:\\path\\to\\run_folder"
    python evaluate_run_static.py 20260525_205912 --min-route-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Path discovery ---------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# carla_core/scripts/verify-check-test/<file>  -> repo root is 3 levels up
_REPO_ROOT = _THIS_FILE.parents[3]
_EXPERIMENTS_ROOT = _REPO_ROOT / "carla_core" / "experiments"
_DEFAULT_MODES = ("curriculum", "batch")


def resolve_run_dir(run_id: str | None, run_dir: str | None, mode: str | None) -> Path:
    """Find the run directory from either an explicit path or a timestamp.

    Acceptable run_id forms:
        20260525_205912
        carla_mappo_20260525_205912
    """
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--run-dir not a directory: {p}")
        return p

    if not run_id:
        raise ValueError("Either --run-dir or a positional <run_id> is required.")

    folder_name = run_id if run_id.startswith("carla_mappo_") else f"carla_mappo_{run_id}"
    candidates: list[Path] = []
    modes = (mode,) if mode else _DEFAULT_MODES
    for m in modes:
        cand = _EXPERIMENTS_ROOT / m / folder_name
        if cand.is_dir():
            candidates.append(cand)

    if not candidates:
        searched = ", ".join(str(_EXPERIMENTS_ROOT / m / folder_name) for m in modes)
        raise FileNotFoundError(f"Run not found. Searched: {searched}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous run: matches multiple modes — disambiguate with --mode. "
            f"Found: {[str(c) for c in candidates]}"
        )
    return candidates[0]


# --- Classifier mirror ------------------------------------------------------


def corrected_reason(rec: dict, min_route_ratio: float) -> str:
    """Apply the Block-5.1 post-bugfix correction to the recorded reason.

    Mirrors carla_core.envs.episode_classification.classify_termination_reason:
    a "route_complete" with route_optimal_length / route_target_distance below
    min_route_ratio is demoted to "route_short".
    """
    reason = rec.get("termination_reason", "unknown")
    if reason != "route_complete":
        return reason
    target = float(rec.get("route_target_distance_m") or 0.0)
    optimal = float(rec.get("route_optimal_length_m") or 0.0)
    if target > 0.0 and (optimal / target) < min_route_ratio:
        return "route_short"
    return reason


# --- Loaders ----------------------------------------------------------------


def load_records(path: Path) -> list[dict]:
    """Stream-parse episodes.jsonl and dedupe by (episode_id, agent_id), keep last."""
    seen: dict[tuple[int, str], dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("episode_id"), rec.get("agent_id"))
            seen[key] = rec
    return list(seen.values())


def episode_order(path: Path) -> tuple[list[int], dict[int, str]]:
    """Return episode IDs in insertion order plus their level (first-seen)."""
    order: list[int] = []
    seen: set[int] = set()
    ep_level: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = rec.get("episode_id")
            if eid is None or eid in seen:
                continue
            seen.add(eid)
            order.append(eid)
            ep_level[eid] = rec.get("level", "unknown")
    return order, ep_level


# --- Aggregators ------------------------------------------------------------


def agg_metrics(records: list[dict], label: str, min_route_ratio: float) -> dict:
    n = len(records)
    if n == 0:
        return {"label": label, "n": 0}

    raw_reasons = Counter(r.get("termination_reason", "unknown") for r in records)
    corr_reasons = Counter(corrected_reason(r, min_route_ratio) for r in records)

    def pct(reason: str, source: Counter) -> float:
        return 100.0 * source.get(reason, 0) / n

    def mean(field: str) -> float:
        vals = [r.get(field) for r in records if r.get(field) is not None]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "label": label,
        "n": n,
        "SR_raw": pct("route_complete", raw_reasons),
        "SR_corr": pct("route_complete", corr_reasons),
        "route_short": pct("route_short", corr_reasons),
        "collision": pct("collision", raw_reasons),
        "offroad": pct("offroad", raw_reasons),
        "stuck": pct("stuck", raw_reasons),
        "timeout": pct("timeout", raw_reasons),
        "stuck_timeout": pct("stuck", raw_reasons) + pct("timeout", raw_reasons),
        "route_completion_mean": mean("route_completion"),
        "path_efficiency_mean": mean("path_efficiency"),
        "speed_kmh_mean": mean("speed_kmh"),
        "no_wp_steps_mean": mean("no_wp_steps"),
        "actual_distance_mean": mean("actual_distance_traveled_m"),
        "optimal_length_mean": mean("route_optimal_length_m"),
        "route_under_target_pct": 100.0
        * sum(1 for r in records if r.get("route_under_target_flag") == 1.0)
        / n,
        "route_too_short_pct": 100.0
        * sum(1 for r in records if r.get("route_too_short_flag") == 1.0)
        / n,
    }


def route_source_breakdown(records: list[dict]) -> Counter:
    return Counter(r.get("route_source", "unknown") for r in records)


def quarter_trajectory(
    records: list[dict],
    agent_type: str,
    eps_order: list[int],
    min_route_ratio: float,
) -> list[dict]:
    """Group records into Q1-Q4 by episode-end order."""
    by_ep: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("agent_id", "").startswith(agent_type):
            by_ep[r.get("episode_id")].append(r)

    total = len(eps_order)
    quartiles: list[list[dict]] = [[], [], [], []]
    for i, eid in enumerate(eps_order):
        q = min(3, (i * 4) // total) if total else 0
        quartiles[q].extend(by_ep.get(eid, []))

    return [
        agg_metrics(qrecs, f"Q{i+1}", min_route_ratio)
        for i, qrecs in enumerate(quartiles)
    ]


# --- Reporting --------------------------------------------------------------


def fmt_row(name: str, m: dict) -> str:
    if m["n"] == 0:
        return f"  {name:18s} (no records)"
    return (
        f"  {name:18s} n={m['n']:5d}  "
        f"SRraw={m['SR_raw']:5.2f}  SRcorr={m['SR_corr']:5.2f}  "
        f"short={m['route_short']:5.2f}  "
        f"coll={m['collision']:5.2f}  off={m['offroad']:5.2f}  "
        f"stuck={m['stuck']:5.2f}  to={m['timeout']:5.2f}  "
        f"s+t={m['stuck_timeout']:5.2f}  "
        f"rc={m['route_completion_mean']:.3f}  "
        f"pe={m['path_efficiency_mean']:.3f}"
    )


def report(run_dir: Path, min_route_ratio: float) -> int:
    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"ERROR: {episodes_path} not found", file=sys.stderr)
        return 1

    records = load_records(episodes_path)
    vehicles = [r for r in records if r.get("agent_id", "").startswith("vehicle")]
    peds = [r for r in records if r.get("agent_id", "").startswith("pedestrian")]
    eps_order, ep_level = episode_order(episodes_path)
    eps_total = len(eps_order)

    print("=" * 100)
    print(f"STATIC EVAL  run_dir = {run_dir.name}")
    print(f"             path    = {run_dir}")
    print(f"             min_route_ratio = {min_route_ratio}")
    print("=" * 100)
    print(f"Total agent records (deduped): {len(records)}")
    print(f"Total episodes:                {eps_total}")
    print(f"Expected records (6 per ep):   {eps_total * 6}")
    integrity_ok = len(records) == eps_total * 6
    print(f"Integrity:                     {'OK' if integrity_ok else 'MISMATCH'}")
    print()

    # ---- Cumulative -------------------------------------------------------
    print("CUMULATIVE METRICS")
    print("-" * 100)
    header = (
        f"  {'slice':18s} n={'':6s}{'SRraw':>6s} {'SRcorr':>7s} {'short':>6s} "
        f"{'coll':>5s} {'off':>5s} {'stuck':>6s} {'to':>5s} {'s+t':>5s} "
        f"{'rc':>5s} {'pe':>5s}"
    )
    print(header)
    print(fmt_row("combined", agg_metrics(records, "combined", min_route_ratio)))
    print(fmt_row("vehicles", agg_metrics(vehicles, "vehicles", min_route_ratio)))
    print(fmt_row("pedestrians", agg_metrics(peds, "pedestrians", min_route_ratio)))
    print()

    # ---- Speed / geometry -------------------------------------------------
    veh = agg_metrics(vehicles, "vehicles", min_route_ratio)
    ped = agg_metrics(peds, "pedestrians", min_route_ratio)
    print("SPEED / ROUTE GEOMETRY")
    print("-" * 100)
    if veh["n"]:
        print(
            f"  vehicles    speed_kmh={veh['speed_kmh_mean']:.3f}  "
            f"no_wp_steps={veh['no_wp_steps_mean']:.1f}  "
            f"actual_dist={veh['actual_distance_mean']:.2f}m  "
            f"optimal_len={veh['optimal_length_mean']:.2f}m  "
            f"under_target={veh['route_under_target_pct']:.2f}%  "
            f"too_short={veh['route_too_short_pct']:.2f}%"
        )
    if ped["n"]:
        print(
            f"  pedestrians speed_kmh={ped['speed_kmh_mean']:.3f}  "
            f"no_wp_steps={ped['no_wp_steps_mean']:.1f}  "
            f"actual_dist={ped['actual_distance_mean']:.2f}m  "
            f"optimal_len={ped['optimal_length_mean']:.2f}m  "
            f"under_target={ped['route_under_target_pct']:.2f}%  "
            f"too_short={ped['route_too_short_pct']:.2f}%"
        )
        print(f"  (ped speed in m/s ~= {ped['speed_kmh_mean']/3.6:.3f})")
    print()

    # ---- Per-level --------------------------------------------------------
    levels_seen = [l for l in ("easy", "medium", "hard", "unknown") if any(r.get("level") == l for r in records)]
    if levels_seen:
        print("PER-LEVEL BREAKDOWN (vehicles)")
        print("-" * 100)
        for lvl in levels_seen:
            lvl_recs = [r for r in vehicles if r.get("level") == lvl]
            if lvl_recs:
                print(fmt_row(f"vehicle/{lvl}", agg_metrics(lvl_recs, f"vehicle/{lvl}", min_route_ratio)))
        print()
        print("PER-LEVEL BREAKDOWN (pedestrians)")
        print("-" * 100)
        for lvl in levels_seen:
            lvl_recs = [r for r in peds if r.get("level") == lvl]
            if lvl_recs:
                print(fmt_row(f"ped/{lvl}", agg_metrics(lvl_recs, f"ped/{lvl}", min_route_ratio)))
        print()

    # ---- Route source -----------------------------------------------------
    print("ROUTE SOURCE DISTRIBUTION")
    print("-" * 100)
    if vehicles:
        for src, cnt in sorted(route_source_breakdown(vehicles).items(), key=lambda x: -x[1]):
            print(f"  vehicles    {src:22s} {cnt:5d}  ({100*cnt/len(vehicles):5.2f}%)")
        print()
    if peds:
        for src, cnt in sorted(route_source_breakdown(peds).items(), key=lambda x: -x[1]):
            print(f"  pedestrians {src:22s} {cnt:5d}  ({100*cnt/len(peds):5.2f}%)")
        print()

    # ---- Level distribution + unlock indices ------------------------------
    if levels_seen and len(levels_seen) > 1:
        print("LEVEL DISTRIBUTION (episodes -- exercises unlock metric)")
        print("-" * 100)
        lvl_count = Counter(ep_level.values())
        for lvl in levels_seen:
            if lvl_count.get(lvl, 0):
                print(f"  {lvl:8s} {lvl_count[lvl]:5d}  ({100*lvl_count[lvl]/eps_total:5.2f}%)")
        print()
        first_med = next((i for i, e in enumerate(eps_order) if ep_level.get(e) == "medium"), None)
        first_hard = next((i for i, e in enumerate(eps_order) if ep_level.get(e) == "hard"), None)
        if first_med is not None:
            print(f"  First MEDIUM episode index: {first_med}  ({100*first_med/eps_total:.1f}% of total)")
        else:
            print("  First MEDIUM episode index: never")
        if first_hard is not None:
            print(f"  First HARD   episode index: {first_hard}  ({100*first_hard/eps_total:.1f}% of total)")
        else:
            print("  First HARD   episode index: never")
        print()

    # ---- Quarter trajectory ----------------------------------------------
    if vehicles:
        print("QUARTER TRAJECTORY -- vehicles (cumulative within each Q)")
        print("-" * 100)
        for q in quarter_trajectory(records, "vehicle", eps_order, min_route_ratio):
            print(fmt_row(q["label"], q))
        print()
    if peds:
        print("QUARTER TRAJECTORY -- pedestrians")
        print("-" * 100)
        for q in quarter_trajectory(records, "pedestrian", eps_order, min_route_ratio):
            print(fmt_row(q["label"], q))
        print()

    # ---- Termination breakdown -------------------------------------------
    for tag, recs in (("vehicles", vehicles), ("pedestrians", peds)):
        if not recs:
            continue
        print(f"TERMINATION REASON BREAKDOWN -- {tag}")
        print("-" * 100)
        raw = Counter(r.get("termination_reason", "?") for r in recs)
        corr = Counter(corrected_reason(r, min_route_ratio) for r in recs)
        for reason in sorted(set(raw) | set(corr)):
            rr = raw.get(reason, 0)
            rc = corr.get(reason, 0)
            print(
                f"  {reason:18s} raw={rr:5d} ({100*rr/len(recs):5.2f}%)   "
                f"corrected={rc:5d} ({100*rc/len(recs):5.2f}%)   "
                f"delta={rc-rr:+5d}"
            )
        print()

    # ---- Short-route audit -----------------------------------------------
    print("SHORT-ROUTE AUDIT (route_complete BUT route_optimal/target < threshold)")
    print("-" * 100)
    for tag, recs in (("vehicles", vehicles), ("pedestrians", peds)):
        if not recs:
            continue
        suspicious = [
            r for r in recs
            if r.get("termination_reason") == "route_complete"
            and float(r.get("route_target_distance_m") or 0) > 0
            and float(r.get("route_optimal_length_m") or 0)
                / float(r.get("route_target_distance_m") or 1) < min_route_ratio
        ]
        print(
            f"  {tag:12s} suspicious={len(suspicious):5d}  "
            f"({100*len(suspicious)/len(recs):5.2f}% of {tag})"
        )
        if suspicious:
            lvl_dist = Counter(s.get("level", "?") for s in suspicious)
            src_dist = Counter(s.get("route_source", "?") for s in suspicious)
            print(f"               by level:  {dict(lvl_dist)}")
            print(f"               by source: {dict(src_dist)}")
    print()

    return 0 if integrity_ok else 2


# --- CLI --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static cumulative evaluator for a CARLA MAPPO training run "
        "(reads episodes.jsonl from disk). Does NOT replace the deterministic "
        "eval phase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_run_static.py 20260525_205912\n"
            "  python evaluate_run_static.py carla_mappo_20260525_205912\n"
            "  python evaluate_run_static.py 20260525_205912 --mode batch\n"
            "  python evaluate_run_static.py --run-dir C:\\path\\to\\run_folder\n"
        ),
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        help='Run timestamp (e.g. "20260525_205912") or full folder name '
        '(e.g. "carla_mappo_20260525_205912"). Mutually exclusive with --run-dir.',
    )
    parser.add_argument(
        "--mode",
        choices=_DEFAULT_MODES,
        default=None,
        help='Experiment subdir under carla_core/experiments/ '
        "(auto-search across known modes if omitted).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit absolute path to the run directory. Overrides run_id/mode.",
    )
    parser.add_argument(
        "--min-route-ratio",
        type=float,
        default=0.5,
        help="route_optimal_length / route_target_distance threshold below which "
        "route_complete is demoted to route_short. Default 0.5 (mirrors "
        "carla_core.envs.episode_classification).",
    )
    args = parser.parse_args(argv)
    if not args.run_id and not args.run_dir:
        parser.error("Provide either a positional <run_id> or --run-dir.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = resolve_run_dir(args.run_id, args.run_dir, args.mode)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return report(run_dir, args.min_route_ratio)


if __name__ == "__main__":
    sys.exit(main())
