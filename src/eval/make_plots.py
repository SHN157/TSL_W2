#!/usr/bin/env python
"""Training and result figures for the report.  Writes PNGs to figures/."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path.home() / "work/signlanguage"
OUT = W / "figures"; OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN = "#2D6BC4", "#C8541A", "#3E8A5B"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"
plt.rcParams.update({"figure.dpi": 200, "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 10, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

def frame(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, dy), textcoords="offset points",
                color=color, fontsize=8, va="center")

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / name)

log = lambda r: pd.read_csv(W / "runs" / r / "log.csv")

# ---------------------------------------------------------------- 1. Phase A: meaning alignment
a = log("A_v1").dropna(subset=["dev_ccos"])
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(a["epoch"], a["dev_ccos"], color=BLUE, lw=2)
ax.axhline(0.357, color=MUTED, lw=1.2, ls="--")
ax.annotate("category-average guess (0.357)", xy=(a["epoch"].max() * 0.02, 0.357),
            xytext=(0, 5), textcoords="offset points", color=MUTED, fontsize=8)
end_label(ax, a["epoch"].values, a["dev_ccos"].values, "dev ccos", BLUE)
frame(ax, "Phase A: sentence-specific alignment (centred cosine)", "epoch", "ccos on dev")
ax.set_xlim(left=0); save(fig, "fig1_phaseA_ccos.png")

# ---------------------------------------------------------------- 2. Phase A: retrieval
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(a["epoch"], 100 * a["dev_r5"], color=GREEN, lw=2)
ax.axhline(16.8, color=MUTED, lw=1.2, ls="--")
ax.annotate("category-average guess (16.8%)", xy=(a["epoch"].max() * 0.02, 16.8),
            xytext=(0, 5), textcoords="offset points", color=MUTED, fontsize=8)
ax.axhline(1.5, color=MUTED, lw=1.0, ls=":")
ax.annotate("random (1.5%)", xy=(a["epoch"].max() * 0.02, 1.5), xytext=(0, 5),
            textcoords="offset points", color=MUTED, fontsize=8)
end_label(ax, a["epoch"].values, (100 * a["dev_r5"]).values, "R@5", GREEN)
frame(ax, "Phase A: retrieving the correct sentence (R@5 of 327)", "epoch", "R@5 on dev (%)")
ax.set_xlim(left=0); save(fig, "fig2_phaseA_r5.png")

# ---------------------------------------------------------------- 3. Phase B: train vs dev CE
b = log("B_v2")
bd = b.dropna(subset=["dev_ce"])
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(b["epoch"], b["train_ce"], color=BLUE, lw=2, label="train")
ax.plot(bd["epoch"], bd["dev_ce"], color=ORANGE, lw=2, label="dev")
end_label(ax, b["epoch"].values, b["train_ce"].values, "train", BLUE)
end_label(ax, bd["epoch"].values, bd["dev_ce"].values, "dev", ORANGE)
frame(ax, "Phase B: training vs dev cross-entropy (memorisation gap)", "epoch", "cross-entropy")
ax.legend(frameon=False, loc="upper right")
ax.set_xlim(left=0); save(fig, "fig3_phaseB_ce.png")

# ---------------------------------------------------------------- 4. Phase B: dev chrF, both runs
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for run, col, lab in [("B_v1", GREEN, "run 1 (40 epochs)"), ("B_v2", BLUE, "run 2 (80 epochs)")]:
    d = log(run).dropna(subset=["dev_chrf"])
    ax.plot(d["epoch"], d["dev_chrf"], color=col, lw=2, label=lab)
    end_label(ax, d["epoch"].values, d["dev_chrf"].values, run, col)
    i = d["dev_chrf"].idxmax()
    ax.plot(d.loc[i, "epoch"], d.loc[i, "dev_chrf"], "o", color=col, ms=6,
            mec="white", mew=2, zorder=5, label="_nolegend_")
frame(ax, "Phase B: dev chrF per epoch (dot = best epoch)", "epoch", "chrF on dev")
ax.legend(frameon=False, loc="lower right")
ax.set_xlim(left=0); save(fig, "fig4_phaseB_chrf.png")

# ---------------------------------------------------------------- 5. test vs ceiling
import sacrebleu
p = pd.read_csv(W / "runs/TEST_B_v2/pred_test.csv")
p["dom"] = p["clip_id"].str[:2].map({"b1": "cooking", "b2": "medical"})
model = {d: sacrebleu.corpus_chrf(g.hyp.tolist(), [g.ref.tolist()]).score for d, g in p.groupby("dom")}
ceiling = {"cooking": 37.9, "medical": 60.1}
doms = ["cooking", "medical"]; x = np.arange(len(doms)); w = 0.36
fig, ax = plt.subplots(figsize=(5.4, 3.4))
b1 = ax.bar(x - w / 2 - 0.01, [model[d] for d in doms], w, color=BLUE, label="our model")
b2 = ax.bar(x + w / 2 + 0.01, [ceiling[d] for d in doms], w, color=MUTED, label="SONAR decoder ceiling")
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.1f", padding=2, color=INK, fontsize=8)
ax.set_xticks(x, ["cooking (n=%d)" % (p.dom == "cooking").sum(),
                  "medical (n=%d)" % (p.dom == "medical").sum()])
frame(ax, "Test set: chrF against the achievable ceiling", "", "chrF")
ax.legend(frameon=False); ax.set_ylim(0, 68); save(fig, "fig5_test_vs_ceiling.png")

# ---------------------------------------------------------------- 6. per-sentence chrF distribution
p["c"] = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(p.hyp, p.ref)]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, (d, col) in zip(axes, [("cooking", ORANGE), ("medical", BLUE)]):
    g = p[p.dom == d]
    ax.hist(g["c"], bins=np.arange(0, 101, 5), color=col, edgecolor="white", lw=0.8)
    ax.axvline(g["c"].median(), color=INK, lw=1.2, ls="--")
    ax.annotate("median %.0f" % g["c"].median(), xy=(g["c"].median(), 0),
                xytext=(4, 30), textcoords="offset points", color=INK, fontsize=8)
    frame(ax, "%s (n=%d)" % (d, len(g)), "sentence chrF", "clips" if d == "cooking" else "")
fig.suptitle("Test set: per-sentence quality", x=0.02, ha="left", color=INK, fontsize=10)
save(fig, "fig6_test_distribution.png")

print("\nall figures in", OUT)