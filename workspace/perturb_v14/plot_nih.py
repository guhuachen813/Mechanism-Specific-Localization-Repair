"""NIH ChestX-ray14 外部验证配图（印刷版式）。

与 plot_nih.py 的关系
---------------------
本脚本产出的图按**最终印刷尺寸**设计：figsize 宽度 6.4 in，去掉留白后
恰好等于 A4 报告正文栏宽（16.2 cm ≈ 6.38 in），因此缩放比约为 1.0，
图中 8 pt 的字在纸面上就是 8 pt。早期版本按 14 in 宽绘制再缩到栏宽，
字高只剩 4–5 pt，属于「代码跑通但纸上读不了」的失败，故重做。

图组
----
figN1 三域坐标：先验、操作点塌缩与代价
figN2 「何时有用」定律的外部检验（核心图）
figN3 真实质量分布与复核收益

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/ClawsGO/project-20260824-a/workspace/analysis_v12/src")
from rc_lib import setup_matplotlib, OKABE_ITO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W = 6.4          # 印刷宽度（英寸），与报告栏宽一致
DPI = 320

SKEY = ["A", "B", "C", "D", "E", "F"]
SNAME = ["A 冻结\n温度", "B ECE\n最优", "C NLL\n最优",
         "D 仅先验\n偏移", "E 先验偏移\n+温度", "F 先验匹配\n阈值"]

FS_T = 8.6       # 子图标题
FS_L = 8.2       # 轴标签
FS_K = 7.4       # 刻度
FS_G = 7.0       # 图例
FS_A = 7.0       # 注记


def style(ax, grid="y"):
    ax.grid(axis=grid, alpha=0.16, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_K, length=3, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def cond_clean(tag):
    d = pd.read_csv(os.path.join(RES, f"table_conditions_A_seed42_{tag}.csv"))
    return d[d["cond"] == "clean"].iloc[0]


# --------------------------------------------------------------------------
def figN1():
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(W, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.46, wspace=0.34,
                          left=0.085, right=0.975, top=0.925, bottom=0.105)

    route = cond_clean("route")
    off = cond_clean("official")
    nihb = pd.read_csv(os.path.join(RES, "table_nih_baseline.csv")).set_index("metric")

    # ---- (a) 三域坐标系 ----
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter([route["prevalence"]], [route["auroc"]], s=72, color=OKABE_ITO["blue"],
               edgecolor="white", linewidth=1.0, zorder=4,
               label=f"CheXpert 域内 route（n={int(route['n'])//1000}k）")
    ax.scatter([off["prevalence"]], [off["auroc"]], s=72, color=OKABE_ITO["sky"],
               edgecolor="white", linewidth=1.0, zorder=4,
               label=f"CheXpert 偏移域 official（n={int(off['n'])}）")
    ax.scatter([0.0418], [nihb.loc["AUROC（排序）", "point"]], s=72,
               color=OKABE_ITO["vermillion"], edgecolor="white", linewidth=1.0, zorder=5,
               label="NIH ChestX-ray14（n=25.6k）")
    ax.plot([0.0418, 0.0418],
            [nihb.loc["AUROC（排序）", "ci_lo"], nihb.loc["AUROC（排序）", "ci_hi"]],
            color=OKABE_ITO["vermillion"], lw=1.4, zorder=3)
    ax.annotate("患者级 95% CI\n宽度是图像级的 2.0 倍",
                xy=(0.050, 0.6810), xytext=(0.108, 0.634), fontsize=FS_A,
                color="#555555", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8,
                                connectionstyle="arc3,rad=0.20"))
    ax.set_xlabel("目标域患病率", fontsize=FS_L)
    ax.set_ylabel("AUROC", fontsize=FS_L)
    ax.set_xlim(-0.035, 0.425); ax.set_ylim(0.615, 1.00)
    ax.set_title("(a) 三个域：先验跨度 4.2%–32.7%，\n跨机构迁移使 AUROC 掉到 0.70",
                 fontsize=FS_T, loc="left")
    ax.legend(frameon=False, fontsize=FS_G, loc="upper right",
              handletextpad=0.4, borderpad=0.1, labelspacing=0.32)
    style(ax, "both")

    # ---- (b) 操作点塌缩（双向） ----
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(3)
    true_p = [route["prevalence"], off["prevalence"], 0.0418]
    pred_p = [route["prevalence"] - route["fn"] + route["fp"],
              off["prevalence"] - off["fn"] + off["fp"],
              nihb.loc["预测阳性率 PPR", "point"]]
    ax.bar(xs - 0.19, true_p, width=0.36, color=OKABE_ITO["grey"], label="真实患病率")
    ax.bar(xs + 0.19, pred_p, width=0.36, color=OKABE_ITO["vermillion"],
           label="预测阳性率（阈值 0.5）")
    for xi, (t, p) in enumerate(zip(true_p, pred_p)):
        ax.annotate(f"×{p/max(t,1e-9):.2f}", (xi + 0.19, p), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=FS_A, color=OKABE_ITO["vermillion"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["域内\nroute", "偏移域\nofficial", "跨机构\nNIH"],
                       fontsize=FS_K)
    ax.set_ylabel("比例", fontsize=FS_L); ax.set_ylim(0, 0.56)
    ax.set_title("(b) 操作点双向塌缩：域内吻合（×0.98），\n"
                 "偏高处少判 4.4 倍，偏低处多判 8.8 倍", fontsize=FS_T, loc="left")
    ax.legend(frameon=False, fontsize=FS_G, loc="upper left",
              handletextpad=0.4, borderpad=0.1, labelspacing=0.32)
    style(ax)

    # ---- (c) 六策略代价风险，三域并排 ----
    ax = fig.add_subplot(gs[1, :])
    nih_c = pd.read_csv(os.path.join(RES, "table_nih_calibfix.csv")).set_index("strategy")
    nih_cost = [nih_c.loc[[s for s in nih_c.index if s.startswith(k)][0], "cost"]
                for k in SKEY]
    cr = pd.read_csv(os.path.join(RES, "table_calibfix_A_seed42_route.csv"))
    co = pd.read_csv(os.path.join(RES, "table_calibfix_A_seed42_official.csv"))
    series = [("CheXpert 域内", [cr[f"cost_{k}"].mean() for k in SKEY], OKABE_ITO["blue"]),
              ("CheXpert 偏移域", [co[f"cost_{k}"].mean() for k in SKEY], OKABE_ITO["sky"]),
              ("NIH ChestX-ray14", nih_cost, OKABE_ITO["vermillion"])]
    x = np.arange(6); w = 0.25
    for k, (nm, v, c) in enumerate(series):
        ax.bar(x + (k - 1) * w, v, width=w, color=c, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(SNAME, fontsize=FS_K)
    ax.set_ylabel("代价风险（FN:FP = 2:1，每张）", fontsize=FS_L)
    ax.set_ylim(0, 0.80)
    ax.set_title("(c) 三个域上 F 都是唯一一致降低代价的策略；C 靠退化成全判阴性取得低代价",
                 fontsize=FS_T, loc="left")
    ax.legend(frameon=False, fontsize=FS_G, loc="upper left",
              handletextpad=0.4, borderpad=0.1, labelspacing=0.32)
    ax.text(5.55, 0.775, "C 档：NIH 灵敏度仅 0.002\n（模型已实际失效）",
            fontsize=FS_A, color="#444444", ha="right", va="top")
    style(ax)

    p = os.path.join(OUT, "figN1_三域坐标.png")
    fig.savefig(p, dpi=DPI); print("saved", p)


# --------------------------------------------------------------------------
def figN2():
    plt = setup_matplotlib()
    cx = pd.read_csv(os.path.join(RES, "table_h1_robust_A_seed42_route.csv"))
    main = pd.read_csv(os.path.join(RES, "table_nih_law_pc1_全部17维.csv"))
    geom = pd.read_csv(os.path.join(RES, "table_nih_law_pc1_几何6维.csv"))
    inten = pd.read_csv(os.path.join(RES, "table_nih_law_pc1_强度11维.csv"))
    whole = pd.read_csv(os.path.join(RES, "table_nih_law_whole.csv"))

    fig, ax = plt.subplots(figsize=(W, 5.0))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.905, bottom=0.30)

    ax.scatter(cx["auc_base_gb"], cx["delta_gb"], s=40, color=OKABE_ITO["blue"],
               edgecolor="white", linewidth=0.7, zorder=4,
               label="CheXpert 域内 · 13 个合成降质档")
    ax.scatter(main["auc_base_gb"], main["delta_gb"], s=62, marker="s",
               color=OKABE_ITO["vermillion"], edgecolor="white", linewidth=0.8, zorder=5,
               label="NIH · 真实质量分层（17 维）")
    ax.scatter(geom["auc_base_gb"], geom["delta_gb"], s=44, marker="^",
               facecolor="none", edgecolor=OKABE_ITO["orange"], linewidth=1.3, zorder=4,
               label="NIH · 几何分层／其余特征预测")
    ax.scatter(inten["auc_base_gb"], inten["delta_gb"], s=44, marker="D",
               facecolor="none", edgecolor=OKABE_ITO["green"], linewidth=1.3, zorder=4,
               label="NIH · 强度分层／几何特征预测")
    ax.scatter(whole["auc_base_gb"], whole["delta_gb"], s=115, marker="*",
               color=OKABE_ITO["purple"], edgecolor="white", linewidth=0.8, zorder=6,
               label="NIH · 整库不分层")

    s_cx = np.polyfit(cx["auc_base_gb"], cx["delta_gb"], 1)
    s_ni = np.polyfit(main["auc_base_gb"], main["delta_gb"], 1)
    xs = np.linspace(0.48, 0.87, 50)
    ax.plot(xs, np.polyval(s_cx, xs), color=OKABE_ITO["blue"], lw=1.3, alpha=0.7, zorder=2)
    ax.plot(xs, np.polyval(s_ni, xs), color=OKABE_ITO["vermillion"], lw=1.3,
            ls="--", alpha=0.7, zorder=2)

    x0 = float(whole["auc_base_gb"].iloc[0])
    y0 = float(whole["delta_gb"].iloc[0])
    y_fit = float(np.polyval(s_cx, x0))
    ax.annotate("", xy=(x0, y0), xytext=(x0, y_fit),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.2))
    ax.text(x0 - 0.006, (y0 + y_fit) / 2, f"同一失效水平\n{y0/y_fit:.1f} 倍",
            fontsize=FS_A + 0.4, color="#333333", ha="right", va="center")

    ax.axhline(0, color="k", lw=0.9)
    r_cx = np.corrcoef(cx["auc_base_gb"], cx["delta_gb"])[0, 1]
    r_ni = np.corrcoef(main["auc_base_gb"], main["delta_gb"])[0, 1]
    def _m(v):          # 真减号，避免与硬编码的 −0.79 混用两种字符
        return f"{v:+.2f}".replace("-", "\u2212")
    ax.text(0.975, 0.965,
            f"CheXpert 域内  r = {_m(r_cx)}（13 档）\n"
            f"NIH 真实质量  r = {_m(r_ni)}（5 层）\n"
            f"两个不相交对照轴  r = −0.79 / −0.84",
            transform=ax.transAxes, fontsize=FS_A, ha="right", va="top", color="#333333")

    ax.set_xlabel("置信度单独检测「模型判错」的 AUROC（该组内）", fontsize=FS_L)
    ax.set_ylabel("加入质量特征的增量 ΔAUC", fontsize=FS_L)
    ax.set_xlim(0.47, 0.88); ax.set_ylim(-0.035, 0.245)
    ax.set_title("「何时有用」定律在真实临床图像上成立：置信度越失效，质量特征越有用",
                 fontsize=FS_T + 0.6, loc="left")
    ax.legend(frameon=False, fontsize=FS_G - 0.4, loc="upper center",
              bbox_to_anchor=(0.5, -0.175), ncol=2, columnspacing=1.1,
              handletextpad=0.4, labelspacing=0.35)
    style(ax, "both")
    p = os.path.join(OUT, "figN2_定律外部检验.png")
    fig.savefig(p, dpi=DPI); print("saved", p)


# --------------------------------------------------------------------------
def figN3():
    plt = setup_matplotlib()
    d = pd.read_csv(os.path.join(HERE, "nih_out", "nih_full_ids.csv"))
    cq = pd.read_csv(os.path.join(HERE, "nih_out", "chexpert_clean_qr.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(W, 4.35))
    fig.subplots_adjust(left=0.115, right=0.90, top=0.905, bottom=0.105, hspace=0.72)

    # ---- (a) 真实质量分布 ----
    ax = axes[0]
    bins = np.arange(0, 0.68, 1 / 9)
    for nm, s, c, ls in (
            ("CheXpert 干净档（224）", cq["qc_quality_risk"], OKABE_ITO["blue"], "-"),
            ("NIH 全部（224）", d["qc_quality_risk"], OKABE_ITO["vermillion"], "-"),
            ("NIH · AP 床旁", d.loc[d["qc_projection_ap"] == 1, "qc_quality_risk"],
             OKABE_ITO["orange"], "--"),
            ("NIH · PA 标准", d.loc[d["qc_projection_ap"] == 0, "qc_quality_risk"],
             OKABE_ITO["green"], "--")):
        ax.hist(s, bins=bins, weights=np.ones(len(s)) / len(s), histtype="step", lw=1.5,
                ls=ls, color=c, label=f"{nm}　均值 {s.mean():.3f}")
    ax.set_xlabel("quality_risk（9 个布尔判据取均值，均在 224 上计算）", fontsize=FS_L)
    ax.set_ylabel("占比", fontsize=FS_L)
    ax.set_ylim(0, 0.86)
    ax.set_title("(a) 真实图像的质量方差真实存在，投射位是其主因（AP 0.265 vs PA 0.146）",
                 fontsize=FS_T, loc="left")
    ax.legend(frameon=False, fontsize=FS_G - 1.1, loc="upper right",
              handletextpad=0.4, borderpad=0.1, labelspacing=0.22)
    style(ax)

    # ---- (b) 复核收益 ----
    ax = axes[1]
    L = d[["l1_0", "l1_1", "l1_2"]].to_numpy(float) / 0.4943421483039856
    e = np.exp(L - L.max(1, keepdims=True)); p = e[:, 1] / e.sum(1)
    y = (d["label"].to_numpy() == 1).astype(int)
    err = ((p >= 0.5).astype(int) != y).astype(int)
    order = np.argsort(-np.maximum(p, 1 - p))
    covs = np.linspace(0.05, 0.95, 19)
    ek, ec = [], []
    for cv in covs:
        k = int(round(len(d) * cv))
        keep, rev = order[:len(d) - k], order[len(d) - k:]
        ek.append(err[keep].mean()); ec.append(err[rev].sum() / max(err.sum(), 1))
    ax.plot(covs, ek, "-o", ms=3.6, color=OKABE_ITO["vermillion"], lw=1.6,
            label="复核后剩余错误率（左轴）")
    ax.axhline(err.mean(), color=OKABE_ITO["grey"], lw=1.1, ls="--",
               label=f"不复核 {err.mean():.3f}")
    ax.set_xlabel("按置信度复核的图像比例", fontsize=FS_L)
    ax.set_ylabel("剩余错误率", fontsize=FS_L)
    ax.set_ylim(0, 0.50); ax.set_xlim(0, 1.0)
    ax2 = ax.twinx()
    ax2.plot(covs, ec, "-s", ms=3.0, color=OKABE_ITO["blue"], lw=1.2,
             label="被复核批次捕获的错误占比（右轴）")
    ax2.set_ylabel("错误捕获率", color=OKABE_ITO["blue"], fontsize=FS_L)
    ax2.tick_params(axis="y", colors=OKABE_ITO["blue"], labelsize=FS_K, length=3, pad=2)
    ax2.set_ylim(0, 1.12); ax2.spines["top"].set_visible(False)
    ax.set_title("(b) 置信度排序仍可迁移：复核最不确定的 25% 即捕获 36% 的错误，\n"
                 "剩余错误率 0.357 → 0.303", fontsize=FS_T, loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.93, fontsize=FS_G - 0.8, loc="center right",
              handletextpad=0.5, borderpad=0.4, labelspacing=0.28)
    style(ax)

    p = os.path.join(OUT, "figN3_真实质量与复核.png")
    fig.savefig(p, dpi=DPI); print("saved", p)


if __name__ == "__main__":
    figN1(); figN2(); figN3()
