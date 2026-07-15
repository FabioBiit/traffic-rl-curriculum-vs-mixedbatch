"""Fig. (Ch. 5, April pilot): windowed vehicle SR over training episodes for
the two GNN-curriculum pilot runs, showing that run 0504's mid-training
collapse is invisible in cumulative aggregates.

Reads training episodes.jsonl from disk (measurement rules: dedup by
episode_id+agent_id, success = route_complete only, vehicles only).

Usage (repo root):  python docs/thesis/scripts/make_pilot_collapse_fig.py
Output: docs/thesis/latex/figures/pilot_collapse_timeseries.pdf
"""
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RUNS = {
    "GNN-curr 0425 (regular)": REPO / r"carla_core/experiments/curriculum/OTHER_RUNS/carla_mappo_20260425_105247_3M_GNN_NO_ATTENTION/episodes.jsonl",
    "GNN-curr 0504 (collapsed)": REPO / r"carla_core/experiments/curriculum/OTHER_RUNS/carla_mappo_20260504_080613_2.7M_GNN_NO_ATTENTION_EARLY_COLLASSO/episodes.jsonl",
}
WINDOW = 100  # episodes
OUT = REPO / "docs/thesis/latex/figures/pilot_collapse_timeseries.pdf"


def windowed_vehicle_sr(path: Path):
    recs = {}
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r["episode_id"], r["agent_id"])
            if k not in recs:
                order.append(k)
            recs[k] = r
    # per-episode vehicle success share, in first-seen episode order
    ep_order = []
    seen = set()
    for eid, _ in order:
        if eid not in seen:
            seen.add(eid)
            ep_order.append(eid)
    by_ep = collections.defaultdict(list)
    for (eid, aid), r in recs.items():
        if aid.startswith("vehicle"):
            by_ep[eid].append(1.0 if r.get("termination_reason") == "route_complete" else 0.0)
    xs, ys = [], []
    buf = collections.deque(maxlen=WINDOW)
    for i, eid in enumerate(ep_order, 1):
        vals = by_ep.get(eid)
        if not vals:
            continue
        buf.append(sum(vals) / len(vals))
        if len(buf) == WINDOW:
            xs.append(i)
            ys.append(100.0 * sum(buf) / len(buf))
    return xs, ys


def main():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for (label, path), style in zip(RUNS.items(), ("-", "--")):
        xs, ys = windowed_vehicle_sr(path)
        ax.plot(xs, ys, style, linewidth=1.4, label=label)
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"vehicle SR, {WINDOW}-episode window (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
