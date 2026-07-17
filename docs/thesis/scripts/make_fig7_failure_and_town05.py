"""Ch. 5 figures from the consolidated campaign JSON:
  (a) failure_modes.pdf  — stacked vehicle termination composition per cell
                           (training cumulative + final evaluation);
  (b) town05_bars.pdf    — vehicle eval SR, Town03 scenario average vs Town05,
                           per cell (the 'equal at home, apart away' view).
Usage (repo root): python docs/thesis/scripts/make_fig7_failure_and_town05.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
CONS = json.load(open(REPO / "docs/thesis/data/campaign_consolidation.json"))
CELLS = ["MLP-C", "MLP-B", "GNN-C", "GNN-B"]
LABELS = ["MLP\ncurr.", "MLP\nbatch", "GNN\ncurr.", "GNN\nbatch"]
FIGDIR = REPO / "docs/thesis/latex/figures"

CATS = [("sr", "route complete", "#4c9f70"),
        ("stuck", "stuck", "#e0c368"),
        ("timeout", "timeout", "#b8b8b8"),
        ("coll", "collision", "#c05b5b"),
        ("off", "offroad", "#7a5bc0")]


def veh(cell, layer):
    if layer == "train":
        return CONS[cell]["training"]["veh"]["cumulative"]
    return CONS[cell]["eval"]["overall"]["veh"]


def stacked(ax, layer, title):
    x = range(len(CELLS))
    bottom = [0.0] * len(CELLS)
    for key, label, color in CATS:
        vals = [veh(c, layer)[key] for c in CELLS]
        ax.bar(x, vals, bottom=bottom, color=color, width=0.62,
               label=label, edgecolor="white", linewidth=0.4)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylim(0, 102)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
stacked(axes[0], "train", "Training (cumulative)")
stacked(axes[1], "eval", "Final evaluation (400 ep)")
axes[0].set_ylabel("vehicle records (%)")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIGDIR / "failure_modes.pdf")
print("written:", FIGDIR / "failure_modes.pdf")

# --- Town03 avg vs Town05 ---
t03 = [sum(CONS[c]["eval"]["scenarios"][s]["veh"]["sr"] for s in ("easy", "medium", "hard")) / 3
       for c in CELLS]
t05 = [CONS[c]["eval"]["scenarios"]["test"]["veh"]["sr"] for c in CELLS]

fig2, ax = plt.subplots(figsize=(6.4, 3.4))
x = range(len(CELLS))
w = 0.36
ax.bar([i - w / 2 for i in x], t03, width=w, label="Town03 (scenario avg)", color="#5b83c0")
ax.bar([i + w / 2 for i in x], t05, width=w, label="Town05 (test)", color="#c08a5b")
for i in x:
    ax.text(i - w / 2, t03[i] + 1, f"{t03[i]:.1f}", ha="center", fontsize=8)
    ax.text(i + w / 2, t05[i] + 1, f"{t05[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylabel("vehicle SR (%)")
ax.set_ylim(0, 62)
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
fig2.tight_layout()
fig2.savefig(FIGDIR / "town05_bars.pdf")
print("written:", FIGDIR / "town05_bars.pdf")
