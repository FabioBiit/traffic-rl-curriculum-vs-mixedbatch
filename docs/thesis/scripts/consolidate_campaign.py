"""Campaign consolidation for thesis Ch. 5 (2x2 design, seed 999).

Recomputes, from disk and per the project measurement rules (dedup by
episode_id+agent_id [+target distance in eval], success = route_complete
only, vehicles and pedestrians separate), everything Ch. 5 needs:

  per run:  integrity (records/episode, bad lines, non-finite scan),
            cumulative per-class metrics, per-level breakdown, Q1-Q4
            quartile trajectories, unlock timing (curriculum runs),
            pedestrian route diagnostics, windowed-SR threshold
            crossings, timesteps_total (from last_result.json);
  per eval: per-scenario per-class metrics (easy/medium/hard/test-Town05).

Add the GNN-B entry to RUNS when its training/eval complete, re-run, and
the JSON + console tables regenerate.

Usage (repo root):  python docs/thesis/scripts/consolidate_campaign.py
Output: docs/thesis/data/campaign_consolidation.json + console summary.
"""
import collections
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "carla_core" / "experiments"

RUNS = {
    "MLP-C": {"dir": EXP / "curriculum/EVAL_DONE_comparison_3M_MLP/carla_mappo_20260622_171626", "arch": "MLP", "regime": "curriculum"},
    "MLP-B": {"dir": EXP / "batch/EVAL_DONE_comparison_3M_MLP/carla_mappo_20260623_171855", "arch": "MLP", "regime": "batch"},
    "GNN-C": {"dir": EXP / "curriculum/GNN/carla_mappo_20260630_181143", "arch": "GNN", "regime": "curriculum"},
    "GNN-B": {"dir": EXP / "batch/carla_mappo_20260714_155209", "arch": "GNN", "regime": "batch"},
}
SCEN = {30.0: "easy", 60.0: "medium", 100.0: "hard", 80.0: "test"}
WINDOW = 100
THRESHOLDS = (0.30, 0.50)


def load(path, eval_mode=False):
    recs, order, bad, nonfinite = {}, [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            for v in r.values():
                if isinstance(v, float) and not math.isfinite(v):
                    nonfinite += 1
            k = (r["episode_id"], r["agent_id"], r.get("route_target_distance_m") if eval_mode else None)
            if k not in recs:
                order.append(k)
            recs[k] = r
    rs = [recs[k] for k in order]
    return rs, bad, nonfinite


def cls_of(r):
    return "veh" if r["agent_id"].startswith("vehicle") else "ped"


def metrics(rs):
    n = len(rs)
    if n == 0:
        return None
    c = collections.Counter(r.get("termination_reason") for r in rs)
    def pct(*keys):
        return round(100.0 * sum(c.get(k, 0) for k in keys) / n, 2)
    def mean(field):
        vals = [float(r.get(field) or 0.0) for r in rs]
        return round(sum(vals) / n, 3)
    return {
        "n": n, "sr": pct("route_complete"), "route_short": pct("route_short"),
        "stuck": pct("stuck"), "timeout": pct("timeout"), "st": pct("stuck", "timeout"),
        "coll": pct("collision"), "off": pct("offroad"),
        "speed": mean("speed_kmh"), "route_completion": mean("route_completion"),
        "path_eff": mean("path_efficiency"), "no_wp": mean("no_wp_steps"),
    }


def episode_order(rs):
    seen, order = set(), []
    for r in rs:
        if r["episode_id"] not in seen:
            seen.add(r["episode_id"])
            order.append(r["episode_id"])
    return order


def analyze_training(path):
    rs, bad, nonfinite = load(path)
    ep_ids = episode_order(rs)
    n_ep = len(ep_ids)
    per_ep = collections.Counter(r["episode_id"] for r in rs)
    out = {
        "episodes": n_ep, "records": len(rs), "bad_lines": bad,
        "nonfinite_values": nonfinite,
        "episodes_not_6": sum(1 for v in per_ep.values() if v != 6),
    }
    ep_index = {e: i for i, e in enumerate(ep_ids)}
    # cumulative + per-level + quartiles, per class
    for cname, pref in (("veh", "vehicle"), ("ped", "pedestrian")):
        sub = [r for r in rs if r["agent_id"].startswith(pref)]
        out[cname] = {"cumulative": metrics(sub)}
        levels = {}
        for lv in ("easy", "medium", "hard"):
            m = metrics([r for r in sub if r.get("level") == lv])
            if m:
                levels[lv] = m
        out[cname]["per_level"] = levels
        qs = {}
        for q in range(4):
            lo, hi = q * n_ep / 4, (q + 1) * n_ep / 4
            m = metrics([r for r in sub if lo <= ep_index[r["episode_id"]] < hi])
            qs[f"Q{q+1}"] = {k: m[k] for k in ("sr", "st", "coll", "off", "speed")} if m else None
        out[cname]["quartiles"] = qs
    # level mix + unlock timing (first-seen episode per level)
    lv_of = {}
    for r in rs:
        if r.get("level") is not None:
            lv_of.setdefault(r["episode_id"], r["level"])
    mix = collections.Counter(lv_of.get(e) for e in ep_ids)
    out["level_mix"] = {k: v for k, v in mix.items()}
    firsts = {}
    for i, e in enumerate(ep_ids):
        lv = lv_of.get(e)
        if lv and lv not in firsts:
            firsts[lv] = {"episode": i + 1, "pct_episodes": round(100.0 * (i + 1) / n_ep, 1)}
    out["first_episode_per_level"] = firsts
    # windowed veh SR threshold crossings (vs cumulative env steps)
    steps_cum, xing = 0, {}
    buf = collections.deque(maxlen=WINDOW)
    veh_by_ep = collections.defaultdict(list)
    step_by_ep = {}
    for r in rs:
        if cls_of(r) == "veh":
            veh_by_ep[r["episode_id"]].append(1.0 if r["termination_reason"] == "route_complete" else 0.0)
        step_by_ep[r["episode_id"]] = max(step_by_ep.get(r["episode_id"], 0), int(r.get("step_count") or 0))
    for e in ep_ids:
        steps_cum += step_by_ep.get(e, 0)
        vals = veh_by_ep.get(e)
        if vals:
            buf.append(sum(vals) / len(vals))
        if len(buf) == WINDOW:
            w = sum(buf) / WINDOW
            for th in THRESHOLDS:
                key = f"veh_win_sr>={int(th*100)}%"
                if key not in xing and w >= th:
                    xing[key] = {"episode": ep_ids.index(e) + 1, "approx_env_steps": steps_cum}
    out["threshold_crossings"] = xing
    out["approx_total_env_steps_from_logs"] = steps_cum
    # pedestrian route diagnostics (for §5.8)
    ped = [r for r in rs if cls_of(r) == "ped"]
    nped = len(ped) or 1
    out["ped_route_diag"] = {
        "route_source": {k: round(100.0 * v / nped, 2) for k, v in
                         collections.Counter(r.get("route_source") for r in ped).items()},
        "under_target_pct": round(100.0 * sum(1 for r in ped if r.get("route_under_target_flag")) / nped, 2),
        "too_short_pct": round(100.0 * sum(1 for r in ped if r.get("route_too_short_flag")) / nped, 2),
    }
    veh = [r for r in rs if cls_of(r) == "veh"]
    nveh = len(veh) or 1
    out["veh_route_source"] = {k: round(100.0 * v / nveh, 2) for k, v in
                               collections.Counter(r.get("route_source") for r in veh).items()}
    return out


def analyze_eval(path):
    rs, bad, nonfinite = load(path, eval_mode=True)
    out = {"records": len(rs), "bad_lines": bad, "nonfinite_values": nonfinite, "scenarios": {}}
    for dist, name in SCEN.items():
        sub = [r for r in rs if r.get("route_target_distance_m") == dist]
        out["scenarios"][name] = {c: metrics([r for r in sub if cls_of(r) == c]) for c in ("veh", "ped")}
    out["overall"] = {c: metrics([r for r in rs if cls_of(r) == c]) for c in ("veh", "ped")}
    return out


def timesteps_total(run_dir):
    lr = run_dir / "last_result.json"
    if lr.exists():
        try:
            d = json.loads(lr.read_text(encoding="utf-8"))
            for k in ("timesteps_total", "num_env_steps_sampled", "counters"):
                v = d.get(k)
                if isinstance(v, dict):
                    v = v.get("num_env_steps_sampled")
                if isinstance(v, (int, float)):
                    return int(v)
        except Exception:
            return None
    return None


def main():
    result = {}
    for name, meta in RUNS.items():
        d = meta["dir"]
        entry = {"run_id": d.name.replace("carla_mappo_", ""), "arch": meta["arch"], "regime": meta["regime"]}
        entry["timesteps_total"] = timesteps_total(d)
        entry["training"] = analyze_training(d / "episodes.jsonl")
        ev = d / "eval" / "episodes.jsonl"
        entry["eval"] = analyze_eval(ev) if ev.exists() and ev.stat().st_size > 0 else None
        result[name] = entry
        t = entry["training"]
        print(f"\n### {name} ({entry['run_id']}, {meta['arch']} x {meta['regime']})  "
              f"timesteps={entry['timesteps_total']}  ep={t['episodes']}  "
              f"integrity: not6={t['episodes_not_6']} bad={t['bad_lines']} nonfin={t['nonfinite_values']}")
        for c in ("veh", "ped"):
            cu = t[c]["cumulative"]
            print(f"  {c} cum: SR {cu['sr']:6.2f}  s+t {cu['st']:6.2f}  coll {cu['coll']:5.2f}  "
                  f"off {cu['off']:5.2f}  speed {cu['speed']:6.2f}  |  per-level SR: "
                  + " / ".join(f"{lv} {t[c]['per_level'].get(lv, {}).get('sr', float('nan')):.2f}"
                               for lv in ("easy", "medium", "hard")))
            print(f"  {c} Q1-Q4 SR: " + " -> ".join(
                f"{t[c]['quartiles'][q]['sr']:.1f}" for q in ("Q1", "Q2", "Q3", "Q4") if t[c]["quartiles"][q]))
        print(f"  level mix: {t['level_mix']}  firsts: {t['first_episode_per_level']}")
        print(f"  thresholds: {t['threshold_crossings']}")
        print(f"  ped route_source: {t['ped_route_diag']['route_source']}  "
              f"under_target {t['ped_route_diag']['under_target_pct']}%  too_short {t['ped_route_diag']['too_short_pct']}%")
        if entry["eval"]:
            for sc in ("easy", "medium", "hard", "test"):
                v, p = entry["eval"]["scenarios"][sc]["veh"], entry["eval"]["scenarios"][sc]["ped"]
                print(f"  EVAL {sc:<7} veh SR {v['sr']:6.2f} s+t {v['st']:6.2f} coll {v['coll']:5.2f} off {v['off']:5.2f}"
                      f"  |  ped SR {p['sr']:6.2f}")
            ov = entry["eval"]["overall"]
            print(f"  EVAL overall veh SR {ov['veh']['sr']:.2f}  ped SR {ov['ped']['sr']:.2f}")

    outp = REPO / "docs/thesis/data/campaign_consolidation.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
