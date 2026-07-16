"""Fig. (Ch. 5, sample efficiency): windowed SR learning curves per campaign
cell, vehicles and pedestrians, versus cumulative environment steps.

Add GNN-B to RUNS when its training completes and re-run.
Usage (repo root): python docs/thesis/scripts/make_fig5_learning_curves.py
Output: docs/thesis/latex/figures/learning_curves.pdf
"""
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "carla_core" / "experiments"
RUNS = {
    "MLP-curriculum": (EXP / "curriculum/EVAL_DONE_comparison_3M_MLP/carla_mappo_20260622_171626/episodes.jsonl", "tab:blue", "-"),
    "MLP-batch": (EXP / "batch/EVAL_DONE_comparison_3M_MLP/carla_mappo_20260623_171855/episodes.jsonl", "tab:orange", "--"),
    "GNN-curriculum": (EXP / "curriculum/GNN/carla_mappo_20260630_181143/episodes.jsonl", "tab:green", "-"),
    "GNN-batch": (EXP / "batch/carla_mappo_20260714_155209/episodes.jsonl", "tab:red", "--"),
}
WINDOW = 100
OUT = REPO / "docs/thesis/latex/figures/learning_curves.pdf"


def curves(path):
    recs, order = {}, []
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
    ep_order, seen = [], set()
    for eid, _ in order:
        if eid not in seen:
            seen.add(eid)
            ep_order.append(eid)
    succ = {"veh": collections.defaultdict(list), "ped": collections.defaultdict(list)}
    steps = {}
    for (eid, aid), r in recs.items():
        c = "veh" if aid.startswith("vehicle") else "ped"
        succ[c][eid].append(1.0 if r.get("termination_reason") == "route_complete" else 0.0)
        steps[eid] = max(steps.get(eid, 0), int(r.get("step_count") or 0))
    out = {}
    for c in ("veh", "ped"):
        xs, ys, cum = [], [], 0
        buf = collections.deque(maxlen=WINDOW)
        for eid in ep_order:
            cum += steps.get(eid, 0)
            vals = succ[c].get(eid)
            if vals:
                buf.append(sum(vals) / len(vals))
            if len(buf) == WINDOW:
                xs.append(cum / 1e6)
                ys.append(100.0 * sum(buf) / WINDOW)
        out[c] = (xs, ys)
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    for label, (path, color, style) in RUNS.items():
        data = curves(path)
        for ax, c, title in ((axes[0], "veh", "Vehicles"), (axes[1], "ped", "Pedestrians")):
            xs, ys = data[c]
            ax.plot(xs, ys, style, color=color, linewidth=1.4, label=label)
            ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel(f"SR, {WINDOW}-episode window (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
