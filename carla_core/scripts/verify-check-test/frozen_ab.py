#!/usr/bin/env python3
"""Frozen-checkpoint A/B harness for P1-Crosswalk (step #4).

Runs the SAME frozen r0610 checkpoint under two environments — baseline
(branch EVO/new-main: base + ghost-fix + P0) vs candidate (branch
EVO/ped-crosswalk-p1: + P1) — with the SAME seed_base, so the only difference
is the P1 crosswalk routing (env code), isolated. Ghost-fix + P0 are on both
arms. The env *config* (distances 30/60/100, scenarios) is identical (same job);
only the env *code* differs by checked-out branch.

Two phases:

  setup       -- clone r0610's final_eval_job.json into two jobs (baseline /
                 candidate) that differ only in out_dir/results_path/run_name,
                 create the out_dirs + results.json stubs, print the run
                 commands. Run ONCE (does not launch anything).

  consolidate -- read both arms' eval/episodes.jsonl (dedup episode_id+agent_id,
                 keep last; 6 records/episode), compute per-level vehicle &
                 pedestrian metrics, the candidate-minus-baseline deltas, and the
                 binding vehicle gate verdict.

The two eval runs are launched BY THE USER (one per branch); this script never
launches anything.

Binding vehicle gate (per level easy/medium/hard; test=Town05 is generalization,
reported but not binding):
    veh SR delta            >= -2.0 pp
    veh (stuck+timeout)     <= +2.0 pp
    veh collision           <= +1.0 pp
    veh offroad             <= +1.0 pp
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN = REPO / "carla_core" / "experiments" / "curriculum" / "carla_mappo_20260610_192146"
ARMS = ("baseline", "candidate")
ARM_BRANCH = {"baseline": "EVO/new-main", "candidate": "EVO/ped-crosswalk-p1"}
BINDING_LEVELS = ("easy", "medium", "hard")
GATE = {"sr": -2.0, "st": 2.0, "coll": 1.0, "off": 1.0}  # pp thresholds


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def setup(run_dir: Path, episodes: int, seed_base: int):
    template_path = run_dir / "final_eval_job.json"
    if not template_path.exists():
        raise SystemExit(f"template not found: {template_path}")
    template = json.loads(template_path.read_text(encoding="utf-8-sig"))
    ab_root = run_dir / "frozen_ab"

    for arm in ARMS:
        out_dir = ab_root / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        job = copy.deepcopy(template)
        job["out_dir"] = str(out_dir)
        job["results_path"] = str(out_dir / "results.json")
        job["run_name"] = f"frozen_ab_{arm}"
        job["session_id"] = f"frozen_ab_{arm}"
        job["seed_base"] = int(seed_base)
        # checkpoint_path kept = r0610 run dir (same frozen checkpoint both arms)
        job["eval_cfg"]["evaluation"]["episodes_per_map"] = int(episodes)
        (out_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
        # results.json stub (eval requires it to exist; _update_results_json)
        (out_dir / "results.json").write_text(json.dumps({"meta": {}}, indent=2), encoding="utf-8")
        print(f"  [{arm}] job -> {out_dir / 'job.json'}")

    n_scen = len(template["eval_cfg"]["scenarios"]["entries"])
    print(f"\n# scenarios={n_scen}  episodes_per_scenario={episodes}  "
          f"=> {n_scen * episodes} episodes/arm  | seed_base={seed_base}")
    print(f"# checkpoint (both arms) = {template['checkpoint_path']}")
    print("\n=== RUN PROCEDURE (you launch; needs CARLA running) ===")
    for arm in ARMS:
        job = ab_root / arm / "job.json"
        print(f"\n# {arm}: checkout {ARM_BRANCH[arm]} first")
        print(f"git checkout {ARM_BRANCH[arm]}")
        print(f'python -m carla_core.training.evaluate_carla_mappo --job "{job}"')
    print("\n# then consolidate:")
    print(f'python carla_core/scripts/verify-check-test/frozen_ab.py consolidate --run-dir "{run_dir}"')


# --------------------------------------------------------------------------- #
# consolidate
# --------------------------------------------------------------------------- #
# Eval reuses episode_id across scenarios (cross-scenario seed collision) and the
# 'level' field is null in eval logs. Dedup by (episode_id, agent_id, target
# distance) -- NOT (episode_id, agent_id), which would collapse scenarios -- and
# recover the level from route_target_distance_m.
_DIST_LEVEL = {30: "easy", 60: "medium", 100: "hard", 80: "test"}


def _dist(rec):
    return round(float(rec.get("route_target_distance_m") or 0))


def _load(jsonl: Path):
    seen = {}
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seen[(rec.get("episode_id"), rec.get("agent_id"), _dist(rec))] = rec
    return list(seen.values())


def _level_of(rec):
    return _DIST_LEVEL.get(_dist(rec), f"d{_dist(rec)}")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _nan_count(recs):
    bad = 0
    for r in recs:
        for v in r.values():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                bad += 1
    return bad


def _metrics(recs):
    n = len(recs)
    if n == 0:
        return None
    tr = Counter(r.get("termination_reason") for r in recs)
    return {
        "n": n,
        "sr": 100 * tr.get("route_complete", 0) / n,
        "short": 100 * tr.get("route_short", 0) / n,
        "st": 100 * (tr.get("stuck", 0) + tr.get("timeout", 0)) / n,
        "coll": 100 * tr.get("collision", 0) / n,
        "off": 100 * tr.get("offroad", 0) / n,
        "speed": _mean([float(r.get("speed_kmh") or 0) for r in recs]),
    }


def _split(records):
    by = {}
    vehs = [r for r in records if str(r.get("agent_id", "")).startswith("vehicle")]
    peds = [r for r in records if str(r.get("agent_id", "")).startswith("pedestrian")]
    for tag, recs in (("veh", vehs), ("ped", peds)):
        by[tag] = {"all": recs}
        for lvl in set(_level_of(r) for r in recs):
            by[tag][lvl] = [r for r in recs if _level_of(r) == lvl]
    return by


def consolidate(run_dir: Path):
    ab_root = run_dir / "frozen_ab"
    paths = {a: ab_root / a / "eval" / "episodes.jsonl" for a in ARMS}
    for a, p in paths.items():
        if not p.exists():
            raise SystemExit(f"[{a}] eval log not found: {p}\nRun the {a} eval first.")
    data = {a: _load(paths[a]) for a in ARMS}

    print(f"# frozen A/B consolidate | run = {run_dir.name}")
    for a in ARMS:
        recs = data[a]
        ep_keys = Counter((r.get("episode_id"), _dist(r)) for r in recs)
        eps = len(ep_keys)
        bad = sum(1 for k, c in ep_keys.items() if c != 6)
        print(f"  [{a:9s}] records={len(recs)} episodes={eps} !=6:{bad} NaN/inf:{_nan_count(recs)}")

    split = {a: _split(data[a]) for a in ARMS}
    levels = [lvl for lvl in ("easy", "medium", "hard", "test")
              if lvl in split["baseline"]["veh"] or lvl in split["candidate"]["veh"]]

    # ---- vehicle gate ----
    print("\n## VEHICLES (candidate - baseline), binding gate on easy/medium/hard")
    print(f"  {'level':7s} {'baseSR':>7} {'candSR':>7} {'dSR':>6} | "
          f"{'dS+T':>6} {'dColl':>6} {'dOff':>6}  verdict")
    gate_pass = True
    for lvl in levels:
        b = _metrics(split["baseline"]["veh"].get(lvl, []))
        c = _metrics(split["candidate"]["veh"].get(lvl, []))
        if not b or not c:
            print(f"  {lvl:7s} (missing data)")
            continue
        d_sr, d_st = c["sr"] - b["sr"], c["st"] - b["st"]
        d_coll, d_off = c["coll"] - b["coll"], c["off"] - b["off"]
        ok = (d_sr >= GATE["sr"] and d_st <= GATE["st"]
              and d_coll <= GATE["coll"] and d_off <= GATE["off"])
        binding = lvl in BINDING_LEVELS
        if binding and not ok:
            gate_pass = False
        tag = ("PASS" if ok else "FAIL") + ("" if binding else " (info)")
        print(f"  {lvl:7s} {b['sr']:7.2f} {c['sr']:7.2f} {d_sr:+6.2f} | "
              f"{d_st:+6.2f} {d_coll:+6.2f} {d_off:+6.2f}  {tag}  (n {b['n']}/{c['n']})")

    # ---- pedestrian mechanism (report) ----
    print("\n## PEDESTRIANS (report; +5pp med/hard is the finetune gate, not here)")
    print(f"  {'level':7s} {'baseSR':>7} {'candSR':>7} {'dSR':>6} {'bShort':>7} {'cShort':>7}")
    for lvl in levels:
        b = _metrics(split["baseline"]["ped"].get(lvl, []))
        c = _metrics(split["candidate"]["ped"].get(lvl, []))
        if not b or not c:
            continue
        print(f"  {lvl:7s} {b['sr']:7.2f} {c['sr']:7.2f} {c['sr']-b['sr']:+6.2f} "
              f"{b['short']:7.2f} {c['short']:7.2f}  (n {b['n']}/{c['n']})")
    print("\n  ped route_source by arm (all levels):")
    for a in ARMS:
        src = Counter(r.get("route_source", "?") for r in split[a]["ped"]["all"])
        print(f"    {a:9s} { {s: src[s] for s in src} }")

    print("\n==================== GATE VERDICT ====================")
    print(f"  vehicle per-level (easy/medium/hard): {'PASS' if gate_pass else 'FAIL'}"
          f"  (SR>=-2pp, s+t<=+2pp, coll<=+1pp, off<=+1pp)")
    print("======================================================")
    return 0 if gate_pass else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Frozen-checkpoint A/B harness for P1-Crosswalk")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="generate the two A/B jobs + dirs + stubs")
    s.add_argument("--run-dir", default=str(DEFAULT_RUN))
    s.add_argument("--episodes", type=int, default=100, help="episodes per scenario per arm")
    s.add_argument("--seed-base", type=int, default=999)
    c = sub.add_parser("consolidate", help="compare the two arms' eval logs")
    c.add_argument("--run-dir", default=str(DEFAULT_RUN))
    args = p.parse_args(argv)

    if args.cmd == "setup":
        setup(Path(args.run_dir), args.episodes, args.seed_base)
        return 0
    return consolidate(Path(args.run_dir))


if __name__ == "__main__":
    raise SystemExit(main())
