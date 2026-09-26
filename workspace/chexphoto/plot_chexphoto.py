"""CheXphoto 真实降质外部验证——图组。输出 PDF（供 LaTeX 引用）与 PNG（供预览）。

作者：ClawsGO Science Agent
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

T1 = 0.4943421483039856
OUT = Path("results/figs")
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}
# Okabe–Ito，真实降质用朱红突出
C = {"clean": "#4D4D4D", "natural/oneplus": "#D55E00",
     "synthetic/photographic": "#0072B2", "synthetic/digital": "#009E73"}


def softmax_pos(L, T=T1):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T).clone()
    return torch.softmax(X, dim=1).numpy()[:, 1]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


# ============================================================ 载入
df = pd.read_csv("results/chexphoto_preds.csv")
fr = df[df["orient"] == "frontal"].copy()
fr["y"] = fr["label"].astype(int)
fr["p"] = softmax_pos(fr[["l1_0", "l1_1", "l1_2"]].to_numpy(float))

cal = pd.read_csv("results/table_cp_calibration.csv")
wit = pd.read_csv("results/table_cp_within_strict.csv").set_index("cond")

# ============================================================ 图 1 操作点塌缩
fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.2), sharex=True)
bins = np.linspace(0, 1, 41)
for ax, c in zip(axes, CONDS):
    g = fr[fr["cond"] == c]
    p = g["p"].to_numpy()
    ax.hist(p, bins=bins, density=True, color=C[c], alpha=0.82,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color="#CC0000", ls="--", lw=1.3, zorder=5)
    ax.set_ylabel(SHORT[c], rotation=0, ha="right", va="center", fontsize=10.5)
    ax.set_yticks([])
    ax.set_ylim(0, 9.5)
    npos = int((p >= 0.5).sum())
    ax.annotate(f"判为阳性 {npos}/{len(p)}  ·  最大 p={p.max():.3f}",
                xy=(0.985, 0.80), xycoords="axes fraction", ha="right",
                va="top", fontsize=8.6,
                color="#CC0000" if npos == 0 else "#333333",
                fontweight="bold" if npos == 0 else "normal")
axes[-1].set_xlabel("模型阳性概率 $p$（冻结温度 $T=0.4943$）", fontsize=10.5)
axes[0].annotate("决策阈值 0.5", xy=(0.5, 0.93), xycoords=("data", "axes fraction"),
                 xytext=(0.54, 0.93), textcoords=("data", "axes fraction"),
                 fontsize=8.6, color="#CC0000", va="center")
fig.suptitle("真实拍摄降质使操作点整体塌缩到阈值左侧", fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.975))
save(fig, "fig1_collapse")

# ============================================================ 图 2 校准修补
STRAT = ["A 冻结温度", "B ECE最优温度", "D 仅先验偏移", "F 冻结+先验匹配阈值"]
SLAB = {"A 冻结温度": "A 冻结（现状）", "B ECE最优温度": "B ECE 最优温度",
        "D 仅先验偏移": "D 仅先验偏移", "F 冻结+先验匹配阈值": "F 先验匹配阈值（零标注）"}
SCOL = {"A 冻结温度": "#8C8C8C", "B ECE最优温度": "#E69F00",
        "D 仅先验偏移": "#56B4E9", "F 冻结+先验匹配阈值": "#CC79A7"}

fig, ax = plt.subplots(figsize=(7.4, 3.7))
x = np.arange(len(CONDS))
w = 0.20
for i, s in enumerate(STRAT):
    vals, errlo, errhi = [], [], []
    for c in CONDS:
        r = cal[(cal["cond"] == c) & (cal["strategy"] == s)].iloc[0]
        vals.append(r["cost"])
        lo, hi = r["cost_lo"], r["cost_hi"]
        if np.isfinite(lo) and np.isfinite(hi):
            errlo.append(r["cost"] - lo); errhi.append(hi - r["cost"])
        else:
            errlo.append(0.0); errhi.append(0.0)
    # 只有真正算过 CI 的策略才画误差棒；yerr 全零时传 None 以免留下零长端帽
    has_ci = any(a > 0 or b > 0 for a, b in zip(errlo, errhi))
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=SLAB[s], color=SCOL[s],
                  edgecolor="white", linewidth=0.6, zorder=3,
                  yerr=[errlo, errhi] if has_ci else None,
                  error_kw=dict(ecolor="#333333", elinewidth=1.0, capsize=2.4))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.0, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=10)
ax.set_ylabel("决策代价 $(2\\mathrm{FN}+\\mathrm{FP})/N$", fontsize=10.5)
ax.set_ylim(0, 0.76)
ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.4, frameon=False, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.03), columnspacing=1.4, handlelength=1.4)
fig.suptitle("零标注的先验匹配阈值（F）在各条件下都压低代价；温度缩放（B）不降反升",
             fontsize=11.2, y=1.10)
fig.tight_layout()
save(fig, "fig2_calibration")

# ============================================================ 图 4 信号贬值（源码中排在图 3 之后）
fig, ax = plt.subplots(figsize=(7.2, 4.0))
xs = np.arange(len(CONDS))
series = [("TTA 分歧（零标注）", "tta_auc", "tta_lo", "tta_hi", "#D55E00", "o", "hi"),
          ("图像质量特征 QC（17 维）", "qc_auc", "qc_lo", "qc_hi", "#0072B2", "s", "lo")]
for lab, k, klo, khi, col, mk, side in series:
    v = np.array([wit.loc[c, k] for c in CONDS])
    lo = np.array([wit.loc[c, klo] for c in CONDS])
    hi = np.array([wit.loc[c, khi] for c in CONDS])
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], marker=mk, ms=6, lw=1.9,
                capsize=4, color=col, label=lab, zorder=3)
    for xi, vi, hi_i, lo_i in zip(xs, v, hi, lo):
        if side == "hi":
            ax.annotate(f"{vi:.3f}", (xi, hi_i), textcoords="offset points",
                        xytext=(0, 7), ha="center", va="bottom",
                        fontsize=8.2, color=col)
        else:
            ax.annotate(f"{vi:.3f}", (xi, lo_i), textcoords="offset points",
                        xytext=(0, -7), ha="center", va="top",
                        fontsize=8.2, color=col)

# 干净原图的 TTA 水平参照 + 落差标注
ax.axhline(0.899, color="#D55E00", ls="--", lw=0.9, alpha=0.45, zorder=1)
ax.annotate("", xy=(1.05, 0.899), xytext=(1.05, 0.682),
            arrowprops=dict(arrowstyle="<->", color="#D55E00", lw=1.3))
ax.text(1.14, 0.800, "−0.22", fontsize=9.2, color="#D55E00",
        va="center", ha="left", fontweight="bold")

ax.axhline(0.5, color="#999999", ls=":", lw=1.2, zorder=1)
# 放在 x=0.5（干净原图与真实翻拍之间）——两端的误差棒都在 x=0/1，此处仅有连线，不会碰撞
ax.text(0.5, 0.508, "随机水平", fontsize=8.0, color="#999999",
        ha="center", va="bottom", transform=ax.get_yaxis_transform())
ax.set_xticks(xs)
ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=10)
ax.set_xlim(-0.35, 3.35)
ax.set_ylabel("对软错误的 AUROC", fontsize=10.5)
ax.set_ylim(0.40, 1.06)
ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.6, frameon=False, loc="lower center", ncol=2,
          bbox_to_anchor=(0.5, -0.30))
fig.suptitle("路由信号在真实降质上大幅贬值（患者级聚类 95% CI）",
             fontsize=11.5, y=1.0)
fig.tight_layout(rect=(0, 0.02, 1, 0.965))
save(fig, "fig4_signal")

# ============================================================ 图 3 ECE vs 代价悖论
from matplotlib.lines import Line2D


def decollide(ys, gap):
    """把靠得太近的标签在纵向上推开，返回调整后的 y。"""
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    adj = ys.copy()
    for i in range(1, len(order)):
        p, cu = order[i - 1], order[i]
        if adj[cu] - adj[p] < gap:
            adj[cu] = adj[p] + gap
    return adj


fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8))
panels = [("ece", "期望校准误差 ECE", axes[0]),
          ("cost", "决策代价 $(2\\mathrm{FN}+\\mathrm{FP})/N$", axes[1])]
for key, ylab, ax in panels:
    A = {c: cal[(cal["cond"] == c) & (cal["strategy"] == "A 冻结温度")].iloc[0][key]
         for c in CONDS}
    B = {c: cal[(cal["cond"] == c) & (cal["strategy"] == "B ECE最优温度")].iloc[0][key]
         for c in CONDS}
    allv = np.array(list(A.values()) + list(B.values()))
    gap = 0.075 * (allv.max() - allv.min())
    ay = decollide([A[c] for c in CONDS], gap)
    by = decollide([B[c] for c in CONDS], gap)

    for i, c in enumerate(CONDS):
        ax.plot([0, 1], [A[c], B[c]], marker="o", ms=6, lw=1.9, color=C[c], zorder=3)
        ax.annotate(f"{A[c]:.3f}", (0, ay[i]), textcoords="offset points",
                    xytext=(-8, 0), ha="right", va="center",
                    fontsize=7.8, color=C[c])
        ax.annotate(f"{B[c]:.3f}", (1, by[i]), textcoords="offset points",
                    xytext=(8, 0), ha="left", va="center",
                    fontsize=7.8, color=C[c])

    lo = min(allv.min(), ay.min(), by.min())
    hi = max(allv.max(), ay.max(), by.max())
    pad = 0.18 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.42, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A\n冻结温度", "B\nECE 最优温度"], fontsize=9)
    ax.set_ylabel(ylab, fontsize=10)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)

axes[0].set_title("校准指标变好 ↓", fontsize=10.5, color="#2E7D32", pad=8)
axes[1].set_title("决策代价反而变高 ↑", fontsize=10.5, color="#C62828", pad=8)
handles = [Line2D([], [], color=C[c], marker="o", ms=6, lw=1.9, label=SHORT[c])
           for c in CONDS]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.06), columnspacing=1.6,
           handlelength=1.6)
fig.suptitle("温度缩放把 ECE 压下去，却把决策代价抬上来", fontsize=11.5, y=1.01)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
save(fig, "fig3_ece_cost")

print("完成")
