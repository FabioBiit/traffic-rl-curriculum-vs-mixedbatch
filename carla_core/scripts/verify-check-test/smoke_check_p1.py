#!/usr/bin/env python3
"""Smoke check for P0 (set_pedestrians_seed) + P1-Crosswalk after a SHORT run.

Run a short locked training first, e.g.:
    python -m carla_core.training.train_carla_mappo --mode curriculum \
        --difficulty path --lock-curriculum-level hard --timesteps 20000 --seed 999

then point this at the produced run dir (or let it auto-find the latest):
    python carla_core/scripts/verify-check-test/smoke_check_p1.py [run_dir]

It does NOT launch anything. It only reads episodes.jsonl (dedup by
episode_id+agent_id, keep last; 6 records/episode) and verifies P0+P1 ran
correctly. PASS criteria:
  * episode integrity: >=95% of episodes have exactly 6 agent records;
  * no NaN/inf in any numeric field;
  * pedestrian route_source subset of {sidewalk_distance, sidewalk_crosswalk,
    respawn, legacy_chain} AND zero 'sidewalk_fallback' (old degenerate path
    gone) AND 'sidewalk_crosswalk' > 0 (the crossing logic actually fired);
  * vehicle route_source unchanged (informational).
Plus diagnostics: per-source ped route_length_ratio, respawn/crossing shares,
ped/veh SR sanity.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments" / "curriculum"
PED_OK_SOURCES = {"sidewalk_distance", "sidewalk_crosswalk", "respawn", "legacy_chain"}


def find_latest_run() -> Path | None:
    if not EXPERIMENTS.is_dir():
        return None
    runs = [p for p in EXPERIMENTS.glob("carla_mappo_*") if (p / "episodes.jsonl").exists()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "episodes.jsonl").stat().st_mtime)


def load(run_dir: Path):
    seen = {}
    with (run_dir / "episodes.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seen[(rec.get("episode_id"), rec.get("agent_id"))] = rec
    return list(seen.values())


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    return s[int(k)] if f == c else s[f] * (c - k) + s[c] * (k - f)


def sr(recs):
    if not recs:
        return float("nan"), 0
    n = len(recs)
    c = sum(1 for r in recs if r.get("termination_reason") == "route_complete")
    return 100 * c / n, c


def main(argv):
    run_dir = Path(argv[1]) if len(argv) > 1 else find_latest_run()
    if run_dir is None or not (run_dir / "episodes.jsonl").exists():
        print("FAIL: no run dir / episodes.jsonl found. Pass a run dir explicitly.")
        return 1
    print(f"# smoke_check_p1 | run = {run_dir.name}")

    lr = run_dir / "last_result.json"
    if lr.exists():
        try:
            j = json.loads(lr.read_text(encoding="utf-8"))
            print(f"  timesteps_total={j.get('timesteps_total')} "
                  f"episodes_total={j.get('episodes_total')} "
                  f"iter={j.get('training_iteration')}")
        except (json.JSONDecodeError, OSError):
            pass

    recs = load(run_dir)
    peds = [r for r in recs if str(r.get("agent_id", "")).startswith("pedestrian")]
    vehs = [r for r in recs if str(r.get("agent_id", "")).startswith("vehicle")]
    print(f"  records={len(recs)}  pedestrians={len(peds)}  vehicles={len(vehs)}")

    # --- 1. integrity: 6 records / episode ---
    per_ep = Counter(r.get("episode_id") for r in recs)
    n_ep = len(per_ep)
    bad_ep = [e for e, c in per_ep.items() if c != 6]
    integ_ok = n_ep > 0 and (n_ep - len(bad_ep)) / n_ep >= 0.95
    print(f"\n[1] integrity: {n_ep} episodes, {len(bad_ep)} with !=6 records "
          f"-> {'OK' if integ_ok else 'FAIL'}")

    # --- 2. NaN/inf scan ---
    nan_fields = Counter()
    for r in recs:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                nan_fields[k] += 1
    nan_ok = not nan_fields
    print(f"[2] NaN/inf: {'NONE -> OK' if nan_ok else dict(nan_fields)} "
          f"-> {'OK' if nan_ok else 'FAIL'}")

    # --- 3. pedestrian route_source ---
    ped_src = Counter(r.get("route_source", "unknown") for r in peds)
    unexpected = {s for s in ped_src if s not in PED_OK_SOURCES}
    n_fallback = ped_src.get("sidewalk_fallback", 0)
    n_cross = ped_src.get("sidewalk_crosswalk", 0)
    src_ok = (not unexpected) and n_fallback == 0 and n_cross > 0
    print(f"[3] ped route_source: { {s: ped_src[s] for s in ped_src} }")
    print(f"    crosswalk_used={n_cross}  respawn={ped_src.get('respawn', 0)}  "
          f"old_fallback={n_fallback}  unexpected={sorted(unexpected) or 'none'} "
          f"-> {'OK' if src_ok else 'FAIL'}")

    # --- 4. ped route geometry by source ---
    print("[4] ped route_length_ratio (optimal/target) by source:")
    for s in ("sidewalk_distance", "sidewalk_crosswalk", "respawn"):
        sub = [r for r in peds if r.get("route_source") == s]
        if not sub:
            continue
        ratios = [float(r.get("route_length_ratio") or 0) for r in sub]
        under = 100 * sum(1 for r in sub if r.get("route_under_target_flag") == 1.0) / len(sub)
        print(f"    {s:18s} n={len(sub):4d}  ratio med={st.median(ratios):.2f} "
              f"p10={pct(ratios,0.1):.2f} p90={pct(ratios,0.9):.2f}  under_target={under:.0f}%")

    # --- 5. SR sanity (noisy on a short run; informational) ---
    psr, pc = sr(peds)
    vsr, vc = sr(vehs)
    print(f"[5] SR sanity (short run, noisy): ped {psr:.1f}% ({pc}) | veh {vsr:.1f}% ({vc})")
    veh_src = Counter(r.get("route_source", "unknown") for r in vehs)
    print(f"    veh route_source (should be unchanged): { {s: veh_src[s] for s in veh_src} }")

    # --- verdict ---
    ok = integ_ok and nan_ok and src_ok
    print("\n==================== VERDICT ====================")
    print(f"  integrity : {'OK' if integ_ok else 'FAIL'}")
    print(f"  no NaN/inf: {'OK' if nan_ok else 'FAIL'}")
    print(f"  ped routes: {'OK' if src_ok else 'FAIL'} "
          f"(crosswalk fired={n_cross>0}, no old fallback={n_fallback==0}, "
          f"labels clean={not unexpected})")
    print(f"  SMOKE {'PASS' if ok else 'FAIL'}")
    print("=================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
