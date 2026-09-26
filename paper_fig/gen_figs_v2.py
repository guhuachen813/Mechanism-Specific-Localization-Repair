# -*- coding: utf-8 -*-
"""Paper v0.2 figures: Fig1 pipeline (ViLMedSAM style), Fig4 synth-vs-real gap, Fig5 fine-tuning cost."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})

# ---------- palette (restrained, Okabe-Ito accents) ----------
GREEN_F, GREEN_E = "#DDEBDD", "#4C8C4A"
PURP_F, PURP_E = "#E6DFF2", "#6A4C93"
RED_F, RED_E = "#F7DDD6", "#C0442C"
GRAY_F, GRAY_E = "#EFEFEF", "#666666"
BLUE = "#0072B2"; LBLUE = "#56B4E9"; ORANGE = "#D55E00"; GRAYB = "#BBBBBB"

def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.02, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r*100}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, mutation_aspect=1))

def arrow(ax, x1, y1, x2, y2, c="#444444", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=c, lw=lw, shrinkA=1, shrinkB=1))

def txt(ax, x, y, s, size=7.5, ha="center", va="center", weight="normal", color="#222222", style="normal"):
    ax.text(x, y, s, size=size, ha=ha, va=va, weight=weight, color=color, style=style)

# ================= Fig 1: pipeline overview =================
fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=300)
ax.set_xlim(0, 104); ax.set_ylim(0, 100); ax.axis("off")

ly = 95
box(ax, 20, ly-2.2, 3.2, 4.0, GREEN_F, GREEN_E); txt(ax, 24.5, ly, "frozen backbone", 7, ha="left")
box(ax, 44, ly-2.2, 3.2, 4.0, PURP_F, PURP_E);  txt(ax, 48.5, ly, "readout repair (PAIR-Loc)", 7, ha="left")
box(ax, 72, ly-2.2, 3.2, 4.0, RED_F, RED_E);    txt(ax, 76.5, ly, "geometry repair (RectNet)", 7, ha="left")

# input
box(ax, 1, 36, 16, 26, "white", "#888888")
txt(ax, 9, 56, "Photographed", 6.8, weight="bold")
txt(ax, 9, 51, "chest radiograph", 6.8, weight="bold")
txt(ax, 9, 44.5, "smartphone re-capture", 5.4, color="#555555")
txt(ax, 9, 40.5, "of a displayed image", 5.4, color="#555555")

# backbone
box(ax, 20, 34, 15.5, 30, GREEN_F, GREEN_E)
txt(ax, 27.7, 59, "\u2744 Frozen", 7, weight="bold", color=GREEN_E)
txt(ax, 27.7, 52.5, "DenseNet-121", 7, weight="bold")
txt(ax, 27.7, 45.5, "features +", 6.6)
txt(ax, 27.7, 41.5, "CAM readout", 6.6)
txt(ax, 27.7, 28.5, "raw CAM: mislocalized", 6.4, color="#8B3A2E", style="italic")

# readout container (PAIR-Loc)
box(ax, 40, 51, 29, 40, PURP_F, PURP_E, ls="--")
txt(ax, 54.5, 87, "READOUT-FAILURE REPAIR", 6.2, weight="bold", color=PURP_E)
txt(ax, 54.5, 81, "PAIR-Loc", 8.5, weight="bold")
txt(ax, 54.5, 73, "zero-init low-rank adapter (r = 16)\non frozen features", 5.9)
txt(ax, 54.5, 64.5, "target: clean-teacher CAM warped\ninto photo frame by pair homography", 5.9)
txt(ax, 54.5, 57.5, "deployment: photograph only", 5.9, style="italic")

# geometry container (RectNet)
box(ax, 40, 5, 29, 43, RED_F, RED_E, ls="--")
txt(ax, 54.5, 44, "GEOMETRY-FAILURE REPAIR", 6.2, weight="bold", color=RED_E)
txt(ax, 54.5, 38, "RectNet", 8.5, weight="bold")
txt(ax, 54.5, 30.5, "corner regression (8 offsets)\nfrom frozen GAP features", 5.9)
txt(ax, 54.5, 22, "homography rectification \u2192 CAM\non rectified image \u2192 warp back", 5.9)
txt(ax, 54.5, 12.5, "no GT registration at inference", 5.9, style="italic")

# routing box
box(ax, 72.5, 24, 22.5, 44, GRAY_F, GRAY_E)
txt(ax, 83.7, 62.5, "Fixed per-finding routing", 6.8, weight="bold")
txt(ax, 83.7, 58, "(experimentally derived)", 5.8, color="#555555")
txt(ax, 83.7, 50.5, "Cardiomegaly, Effusion", 6.0)
txt(ax, 83.7, 47, "\u2192 PAIR-Loc", 6.0)
txt(ax, 83.7, 42, "Atelectasis \u2192 RectNet", 6.0)
txt(ax, 83.7, 37, "other findings \u2192 raw", 6.0)
txt(ax, 83.7, 33.5, "fallback", 6.0)
txt(ax, 83.7, 28.5, "no automatic failure-mode\nclassifier is trained", 5.6, color="#8B3A2E", style="italic")

# output
box(ax, 97.5, 38, 6, 18, "white", "#888888")
txt(ax, 100.5, 49, "Repaired", 6.5, weight="bold")
txt(ax, 100.5, 44.5, "CAM", 6.5, weight="bold")
txt(ax, 100.5, 34.5, "(photo frame)", 5.2, color="#555555")

# arrows
arrow(ax, 17.3, 49, 19.8, 49)
arrow(ax, 35.8, 56, 39.8, 70)
arrow(ax, 35.8, 42, 39.8, 27)
arrow(ax, 69.3, 70, 74, 52)
arrow(ax, 69.3, 28, 74, 42)
arrow(ax, 95.2, 47, 97.3, 47)
fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================= Fig 4: synthetic vs real gap =================
fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=300)
conds = ["Synthetic\ndigital", "Synthetic\nphotographic", "Natural\nre-photograph"]
agree = [0.640, 0.474, 0.106]
drift = ["24.8 px", "35.4 px", "61.5 px"]
colors = [BLUE, LBLUE, ORANGE]
bars = ax.bar(conds, agree, width=0.58, color=colors, edgecolor="white", lw=0.5)
for b, v, d in zip(bars, agree, drift):
    ax.text(b.get_x() + b.get_width()/2, v + 0.018, f"{v:.3f}", ha="center", size=9, weight="bold")
    ax.text(b.get_x() + b.get_width()/2, 0.035, f"centroid\ndrift {d}", ha="center", va="bottom",
            size=7.2, color="white" if v > 0.3 else "#333333")
ax.set_ylabel("CAM agreement with clean CAM\n(agree-IoU, $\\tau$ = 0.45)", size=8.5)
ax.set_ylim(0, 0.75)
ax.tick_params(labelsize=8.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#888888"); ax.spines["bottom"].set_color("#888888")
ax.annotate("4–6× overestimate of\nlocalization stability", xy=(2, 0.13), xytext=(1.32, 0.42),
            size=8, color=ORANGE, weight="bold",
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_gap.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================= Fig 5: fine-tuning robustness cost =================
fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=300)
models = ["DenseNet-121\n(from scratch)", "SAM\n(general VFM)", "MedSAM\n(medical fine-tuned)"]
clean = [0.8288, 0.7282, 0.7285]
real = [0.6592, 0.6504, 0.5526]
deltas = ["Δ −0.170*", "Δ −0.078*", "Δ −0.176*"]
x = np.arange(3); w = 0.34
b1 = ax.bar(x - w/2, clean, w, color=GRAYB, edgecolor="white", label="clean digital")
b2 = ax.bar(x + w/2, real, w, color=BLUE, edgecolor="white", label="real re-photograph")
for i, (c, r, d) in enumerate(zip(clean, real, deltas)):
    ax.text(i - w/2, c + 0.012, f"{c:.3f}", ha="center", size=7.5)
    ax.text(i + w/2, r + 0.012, f"{r:.3f}", ha="center", size=7.5)
    ax.text(i + w/2, r + 0.052, d, ha="center", size=7.5, weight="bold", color="#C0442C")
ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
ax.text(2.42, 0.508, "chance", size=7, color="#888888", ha="right")
ax.set_ylabel("Cardiomegaly AUROC", size=8.5)
ax.set_ylim(0.45, 0.9); ax.set_xticks(x); ax.set_xticklabels(models, size=8)
ax.tick_params(axis="y", labelsize=8.5)
ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.02, 1.04))
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#888888"); ax.spines["bottom"].set_color("#888888")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_finetune.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("figs done")
