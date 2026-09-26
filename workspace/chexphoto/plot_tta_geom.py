"""方向二图件：几何 TTA 的视图预算扫描，与代价框架下的信号对照。

输出 PDF（供 LaTeX 引用）与 PNG（供预览）。

作者：ClawsGO Science Agent
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

OUT = Path("results/figs")
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}
C = {"clean": "#4D4D4D", "natural/oneplus": "#D55E00",
     "synthetic/photographic": "#0072B2", "synthetic/digital": "#009E73"}
LAB = {"random": "随机", "unc": "不确定度", "costrisk": "代价加权置信度",
       "div2": "单次翻转（既有）", "div12": "几何 TTA（K=12）",
       "pairdelta": "配对分歧", "qc17": "QC 17 维",
       "combo": "置信度 + 几何 TTA", "oracle": "知情上界（不可实现）"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


K = pd.read_csv("results/table_tta_geom_K.csv")
S = pd.read_csv("results/table_tta_selective.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.95),
                               gridspec_kw={"width_ratios": [1.0, 1.12]})

# ==================================================== (a) 视图预算扫描
for c in CONDS:
    g = K[K["cond"] == c].sort_values("K")
    # 只给关注的中心条件画 CI 阴影，避免四个宽带互相压盖
    if c == "natural/oneplus":
        ax1.fill_between(g["K"], g["lo"], g["hi"], color=C[c], alpha=0.13,
                         lw=0, zorder=1)
    ax1.plot(g["K"], g["auroc"], marker="o", ms=5, lw=1.9, color=C[c],
             label=SHORT[c], zorder=3)

g = K[K["cond"] == "natural/oneplus"].sort_values("K")
first, best = g.iloc[0], g.loc[g["auroc"].idxmax()]
ax1.annotate(f"{first['auroc']:.3f}", (first["K"], first["auroc"]),
             textcoords="offset points", xytext=(5, -14), fontsize=9.8,
             color=C["natural/oneplus"], fontweight="bold")
ax1.annotate(f"{best['auroc']:.3f}", (best["K"], best["auroc"]),
             textcoords="offset points", xytext=(-2, 11), fontsize=9.8,
             color=C["natural/oneplus"], fontweight="bold", ha="center")
ax1.axvline(best["K"], color="#BBBBBB", ls=":", lw=1.1, zorder=0)
ax1.annotate("甜点", (best["K"], 1.006), textcoords="offset points",
             xytext=(-4, 0), fontsize=9.0, color="#888888", ha="right",
             va="center")

ax1.set_xscale("log", base=2)
ax1.set_xticks([2, 4, 6, 8, 12, 16])
ax1.set_xticklabels(["2", "4", "6", "8", "12", "16"])
ax1.set_xlim(1.85, 17.5)
ax1.set_xlabel("TTA 视图预算 $K$（次前向）", fontsize=11.5)
ax1.set_ylabel("分歧对软错误的 AUROC", fontsize=11.5)
ax1.set_ylim(0.62, 1.045)
ax1.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
ax1.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=9.6, frameon=False, loc="upper left", ncol=2,
           columnspacing=1.0, handlelength=1.3, borderaxespad=0.3)
ax1.set_title("(a) 视图预算扫描：几何变换使真实降质的信号回升",
              fontsize=11.2, pad=8)

# ==================================================== (b) 真实翻拍的 AURC
sub = S[S["cond"] == "natural/oneplus"].set_index("signal")
order = list(sub["aurc"].sort_values().index)
vals = sub.loc[order, "aurc"].to_numpy(float)
los = sub.loc[order, "aurc_lo"].to_numpy(float)
his = sub.loc[order, "aurc_hi"].to_numpy(float)
cols, tcol = [], []
for s in order:
    if s == "oracle":
        cols.append("#0072B2"); tcol.append("white")
    elif s in ("div12", "combo"):
        cols.append("#D55E00"); tcol.append("white")
    else:
        cols.append("#A8B0B8"); tcol.append("#2B2B2B")
y = np.arange(len(order))

ax2.barh(y, vals, color=cols, height=0.64, zorder=3)
ax2.errorbar(vals, y, xerr=[vals - los, his - vals], fmt="none",
             ecolor="#444444", elinewidth=1.0, capsize=2.4, zorder=4)
# 数值贴在条形内部左端，避开误差棒
for yi, v, tc in zip(y, vals, tcol):
    ax2.text(0.014, yi, f"{v:.3f}", va="center", ha="left",
             fontsize=9.2, color=tc, fontweight="bold", zorder=5)
ax2.set_yticks(y)
ax2.set_yticklabels([LAB[s] for s in order], fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("AURC（越低越好）", fontsize=11.5)
ax2.set_xlim(0, 0.99)
ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax2.grid(axis="x", color="#DDDDDD", lw=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.set_title("(b) 真实翻拍：固定复核预算下的选择性风险",
              fontsize=11.2, pad=8)

fig.tight_layout()
save(fig, "fig5_tta_geom")

# 单独再输出一张纯 K 扫描图，便于正文小尺寸排版
fig2, ax = plt.subplots(figsize=(4.6, 3.4))
for c in CONDS:
    g = K[K["cond"] == c].sort_values("K")
    ax.plot(g["K"], g["auroc"], marker="o", ms=4.5, lw=1.8, color=C[c],
            label=SHORT[c], zorder=3)
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 6, 8, 12, 16])
ax.set_xticklabels(["2", "4", "6", "8", "12", "16"])
ax.set_xlabel("视图预算 $K$", fontsize=10)
ax.set_ylabel("对软错误的 AUROC", fontsize=10)
ax.set_ylim(0.63, 0.99)
ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=7.6, frameon=False, loc="lower right", ncol=1,
          columnspacing=0.9, handlelength=1.3, borderaxespad=0.3)
fig2.tight_layout()
save(fig2, "fig5a_K_scan")

print("完成")
