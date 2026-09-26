"""扰动实验配图（本地绘制，中文字体已确认可用）。

三张图
------
fig1 降质阶梯：模型退化、信号响应、最优温度漂移
fig2 决定性的 H1 检验：条件内 vs 池化
fig3 操作性：固定复核预算下的代价风险

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

ORDER = ["clean", "jpeg_q50", "jpeg_q30", "blur_1", "blur_3",
         "noise_003", "noise_006", "noise_010", "dark_05", "bright_16",
         "contrast_04", "ds_2", "ds_4", "combo_mild", "combo_sev"]
LABEL = {
    "clean": "干净", "jpeg_q50": "JPEG\n50", "jpeg_q30": "JPEG\n30",
    "blur_1": "模糊\nσ1", "blur_3": "模糊\nσ3",
    "noise_003": "噪声\n.03", "noise_006": "噪声\n.06", "noise_010": "噪声\n.10",
    "dark_05": "偏暗\n×0.5", "bright_16": "偏亮\n×1.6",
    "contrast_04": "低对比\n×0.4", "ds_2": "降采样\n×2", "ds_4": "降采样\n×4",
    "combo_mild": "复合\n轻", "combo_sev": "复合\n重",
}


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def ordered(df):
    d = df.copy()
    d["_o"] = d["cond"].map({c: i for i, c in enumerate(ORDER)})
    return d.sort_values("_o")


def cond_table(tag):
    df = load(f"table_conditions_{tag}.csv")
    return ordered(df) if df is not None else None


def fig1():
    plt = setup_matplotlib()
    reg = cond_table("B_seed42_route")
    off = cond_table("B_seed42_official")
    x = np.arange(len(reg))
    lab = [LABEL[c] for c in reg["cond"]]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4))

    # (a) AUROC
    ax = axes[0, 0]
    ax.bar(x - 0.2, reg["auroc"], width=0.4, color=OKABE_ITO["blue"],
           label="route-validation（域内）")
    ax.bar(x + 0.2, off["auroc"], width=0.4, color=OKABE_ITO["vermillion"],
           label="official-valid（偏移域）")
    ax.axhline(0.5, color="k", lw=0.9, ls="--")
    ax.text(-0.4, 0.515, "随机水平", fontsize=8, ha="left", va="bottom")
    ax.set_ylim(0.4, 1.06); ax.set_ylabel("AUROC")
    ax.set_title("(a) 排序能力：降质下 AUROC 缓慢下滑", fontsize=11, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.4, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.005))
    i4 = list(reg["cond"]).index("ds_4")
    ax.annotate("降采样 ×4：两集合\n退化最一致的档位",
                xy=(i4 + 0.2, off["auroc"].iloc[i4] + 0.005), xytext=(i4 - 3.4, 0.90),
                fontsize=7.6, color="#444444", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))

    # (b) 错误率
    ax = axes[0, 1]
    ax.plot(x, reg["err"], "-o", color=OKABE_ITO["blue"], ms=4.5, lw=1.8,
            label="route-validation")
    ax.plot(x, off["err"], "-s", color=OKABE_ITO["vermillion"], ms=4.5, lw=1.8,
            label="official-valid")
    ax.set_ylabel("全覆盖错误率"); ax.set_ylim(0, 0.92)
    ax.set_title("(b) 判别阈值失效：低对比与复合重度档错误率翻数倍", fontsize=11, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.4)
    for i, c in enumerate(reg["cond"]):
        if c in ("contrast_04", "combo_sev"):
            ax.annotate(f"{reg['err'].iloc[i]*100:.0f}%", (i - 0.0, reg["err"].iloc[i]),
                        textcoords="offset points", xytext=(-4, 8), ha="center",
                        fontsize=8, color=OKABE_ITO["blue"])
            ax.annotate(f"{off['err'].iloc[i]*100:.0f}%", (i + 0.0, off["err"].iloc[i]),
                        textcoords="offset points", xytext=(17, -7), ha="center",
                        fontsize=8, color=OKABE_ITO["vermillion"])

    # (c) 最优温度（直接作用在三分类 logits 上的正确口径）
    ax = axes[1, 0]
    dg_r = load("table_degeneracy_B_seed42_route.csv")
    dg_o = load("table_degeneracy_B_seed42_official.csv")
    ax.plot(x, ordered(dg_r)["T_ece"], "-o", color=OKABE_ITO["purple"], ms=4.5, lw=1.8,
            label="route-validation")
    ax.plot(x, ordered(dg_o)["T_ece"], "-s", color=OKABE_ITO["green"], ms=4.5, lw=1.8,
            label="official-valid")
    ax.axhline(1.0, color="k", lw=1.0, ls=":")
    ax.axhline(0.4943, color=OKABE_ITO["vermillion"], lw=1.4, ls="--")
    ax.axhspan(0.30, 1.0, color="#f4f4f4", zorder=0)
    ax.text(0.12, 12.5, "冻结温度 T = 0.494（源域拟合）", fontsize=8,
            color=OKABE_ITO["vermillion"], ha="left", va="center")
    ax.set_yscale("log"); ax.set_ylim(0.30, 24)
    ax.set_ylabel("ECE 最优有效温度 $T^*$（对数轴）")
    ax.set_title("(c) 机制：偏移域需软 3–15 倍，冻结温度系统性偏锐", fontsize=11, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.4, loc="upper right")
    i10 = list(ordered(dg_o)["cond"]).index("noise_006")
    ax.annotate("噪声档 $T^*$≈7.5：\n即已退化为多数类", xy=(i10, 7.51), xytext=(i10 - 4.6, 3.0),
                fontsize=7.6, color="#444444", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                connectionstyle="arc3,rad=0.22"))

    # (d) 操作点塌缩：预测阳性率 vs 真实患病率（冻结温度）
    ax = axes[1, 1]
    ppr_r = ordered(dg_r)["ppr_frozen"].to_numpy()
    ppr_o = ordered(dg_o)["ppr_frozen"].to_numpy()
    ax.plot(x, ppr_r, "-o", color=OKABE_ITO["blue"], ms=4.5, lw=1.8,
            label="route-validation 预测阳性率")
    ax.plot(x, ppr_o, "-s", color=OKABE_ITO["vermillion"], ms=4.5, lw=1.8,
            label="official-valid 预测阳性率")
    ax.axhline(0.126, color=OKABE_ITO["blue"], lw=1.2, ls=":")
    ax.axhline(0.327, color=OKABE_ITO["vermillion"], lw=1.2, ls=":")
    ax.text(0.05, 0.345, "official 真实患病率 32.7%", fontsize=7.8,
            color=OKABE_ITO["vermillion"])
    ax.text(0.05, 0.030, "route 真实患病率 12.6%", fontsize=7.8,
            color=OKABE_ITO["blue"], va="bottom")
    ax.annotate("", xy=(0.30, 0.327), xytext=(0.30, 0.074),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.1))
    ax.text(0.62, 0.215, "干净档位少判 4.4 倍\n（近「全判阴性」）",
            fontsize=8, color="#333333", ha="left", va="center")
    i4c = list(ordered(dg_r)["cond"]).index("contrast_04")
    ax.annotate("低对比档反向塌缩：\n91% 判阳（患病率 12.6%）",
                xy=(i4c, 0.913), xytext=(i4c - 5.2, 0.62), fontsize=7.6,
                color="#444444", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                connectionstyle="arc3,rad=-0.22"))
    ax.set_ylabel("预测阳性率（阈值 0.5）"); ax.set_ylim(0, 1.06)
    ax.set_title("(d) 操作点双向塌缩：或几乎不判阳，或几乎全判阳", fontsize=11, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", bbox_to_anchor=(0.0, 0.99))

    for axx in axes.ravel():
        axx.grid(axis="y", alpha=0.2, lw=0.6)
        axx.set_axisbelow(True)
        if not hasattr(axx, "_twinned"):
            axx.spines["top"].set_visible(False)
            axx.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_降质阶梯.png")
    fig.savefig(p, dpi=210); print("saved", p)


def fig2():
    plt = setup_matplotlib()
    w_r = load("table_within_condition_B_seed42_route.csv")
    w_7 = load("table_within_condition_B_seed7_official.csv")
    w_r = ordered(w_r);
    w_7 = w_7.copy(); w_7["_o"] = w_7["cond"].map({c: i for i, c in enumerate(ORDER)})
    w_7 = w_7.sort_values("_o")

    fig = plt.figure(figsize=(13.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.25, 1.35], wspace=0.34)

    # (a) 逐条件 Δ
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(w_r))
    cols = [OKABE_ITO["vermillion"] if v > 0 else OKABE_ITO["blue"] for v in w_r["delta_vs_conf"]]
    ax.bar(x, w_r["delta_vs_conf"], width=0.66, color=cols)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in w_r["cond"]], fontsize=7.2)
    ax.set_ylabel("ΔAUC（14 特征 + 置信度 − 置信度）")
    ax.set_title("(a) 条件内增量：只在模型已经崩掉的档位为正", fontsize=11, loc="left")
    for i, v in enumerate(w_r["delta_vs_conf"]):
        if abs(v) > 0.03:
            ax.annotate(f"{v*100:+.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 5 if v > 0 else -11), ha="center", fontsize=7.6)
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    # (b) Δ vs 错误率
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(w_r["err"], w_r["delta_vs_conf"], s=46, color=OKABE_ITO["blue"],
               edgecolor="white", linewidth=0.8, zorder=4, label="seed42 / route")
    ax.scatter(w_7["err"], w_7["delta_vs_conf"], s=46, marker="^",
               color=OKABE_ITO["vermillion"], edgecolor="white", linewidth=0.8,
               zorder=4, label="seed7 / official")
    ax.axhline(0, color="k", lw=0.9)
    r = np.corrcoef(w_r["err"], w_r["delta_vs_conf"])[0, 1]
    ax.set_xlabel("该降质档位的全覆盖错误率"); ax.set_ylabel("条件内 ΔAUC")
    ax.set_title(f"(b) 增量只出现在高错误区（r = {r:+.2f}）", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")

    # (c) 池化增量
    ax = fig.add_subplot(gs[0, 2])
    inc = load("table_incremental_B_seed42_route.csv")
    inc = inc[inc["protocol"] == "按条件分折（跨降质泛化）"].copy()
    inc["_o"] = inc["model"].map({m: i for i, m in enumerate(
        ["+quality", "+tta", "+disagree", "+quality+tta", "+全部四路"])})
    inc = inc.sort_values("_o")
    y = np.arange(len(inc))
    cols = [OKABE_ITO["vermillion"] if lo > 0 else OKABE_ITO["grey"]
            for lo, hi in zip(inc["ci_lo"], inc["ci_hi"])]
    ax.errorbar(inc["delta"], y, xerr=[inc["delta"] - inc["ci_lo"], inc["ci_hi"] - inc["delta"]],
                fmt="o", color=OKABE_ITO["black"], ecolor=OKABE_ITO["grey"],
                capsize=3, ms=5, lw=1.3, zorder=3)
    ax.scatter(inc["delta"], y, c=cols, s=58, zorder=4, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(inc["model"], fontsize=8.6)
    ax.set_xlabel("ΔAUC（相对纯置信度，跨降质分折）")
    ax.set_title("(c) 池化增量与置信区间", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    for axx in fig.axes:
        axx.spines["top"].set_visible(False); axx.spines["right"].set_visible(False)
    p = os.path.join(OUT, "fig2_H1检验.png")
    fig.savefig(p, dpi=210); print("saved", p)


def fig3():
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, tag, name in zip(axes, ("B_seed42_route", "B_seed42_official"),
                             ("route-validation（域内，n=28,860）",
                              "official-valid（偏移域，n=3,030）")):
        op = load(f"table_operational_{tag}.csv")
        x = np.arange(len(op)); w = 0.34
        ax.bar(x - w / 2, op["cost_conf"], width=w, color=OKABE_ITO["grey"],
               label="纯置信度路由")
        ax.bar(x + w / 2, op["cost_conf_quality"], width=w, color=OKABE_ITO["orange"],
               label="置信度 + 质量路由")
        for i, row in op.iterrows():
            sig = "显著" if (row["ci_lo"] > 0 or row["ci_hi"] < 0) else "不显著"
            ax.annotate(f"Δ={row['delta_cost']*1000:+.1f}‰\n{sig}",
                        (i, max(row["cost_conf"], row["cost_conf_quality"])),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"覆盖率 {int(c*100)}%" for c in op["coverage"]], fontsize=9.5)
        ax.set_ylabel("代价风险（FN:FP = 2:1，每接受一张）")
        ax.set_title(name, fontsize=10.5, loc="left")
        ax.grid(axis="y", alpha=0.2, lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.6)
        ax.set_ylim(0, max(op["cost_conf"].max(), op["cost_conf_quality"].max()) * 1.35)
    for axx in axes:
        axx.spines["top"].set_visible(False); axx.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_操作性代价.png")
    fig.savefig(p, dpi=210); print("saved", p)


STRATS = ["A 冻结温度", "B ECE最优温度", "C NLL最优温度",
          "D 仅先验偏移", "E 先验偏移+温度", "F 冻结+先验匹配阈值"]
SKEY = [s[0] for s in STRATS]
SHORT = {"A": "冻结\n(不修正)", "B": "ECE\n最优温度", "C": "NLL\n最优温度",
         "D": "先验\n偏移", "E": "先验偏移\n+温度", "F": "先验匹配\n阈值"}


def fig4():
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))

    # (a) 代价风险对比（official，两 seed 并列）
    ax = axes[0]
    x = np.arange(len(SKEY)); w = 0.38
    series = {}
    for tag, c in (("seed42_official", OKABE_ITO["blue"]),
                   ("seed7_official", OKABE_ITO["vermillion"])):
        d = load(f"table_calibfix_{tag}.csv")
        series[tag] = (c, [d[f"cost_{k}"].mean() for k in SKEY],
                       [int((d[f"tpr_{k}"] <= 1e-9).sum()) for k in SKEY])
        ax.bar(x + (-w / 2 if tag.startswith("seed42") else w / 2),
               series[tag][1], width=w, color=c, label=tag.replace("_", " / "))
    # 标注定位到「该策略两根柱中较高者」之上，避免压住更高的那根
    for xi, k in enumerate(SKEY):
        top = max(series[t][1][xi] for t in series)
        for j, (tag, (c, vals, zero)) in enumerate(series.items()):
            if zero[xi]:
                ax.annotate(f"{zero[xi]}档 TPR=0", (xi, top), textcoords="offset points",
                            xytext=(0, 6 + 15 * j), ha="center", fontsize=6.6, color=c)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in SKEY], fontsize=8)
    ax.set_ylabel("代价风险（FN:FP = 2:1，每张）")
    ax.set_ylim(0, 0.80)
    ax.set_title("(a) 偏移域：温度类修正反而比不修正更糟，\n只有重定位操作点的 D/F 真正降低代价",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.4, loc="upper right")
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    # (b) ECE–灵敏度背离
    ax = axes[1]
    for tag, c, mk in (("seed42_official", OKABE_ITO["blue"], "o"),
                       ("seed7_official", OKABE_ITO["vermillion"], "^")):
        d = load(f"table_calibfix_{tag}.csv")
        e = [d[f"ece_{k}"].mean() for k in SKEY]
        t = [d[f"tpr_{k}"].mean() for k in SKEY]
        ax.scatter(e, t, s=78, color=c, marker=mk, edgecolor="white",
                   linewidth=1.0, zorder=4, label=tag.replace("_", " / "))
        for xi, yi, k in zip(e, t, SKEY):
            ax.annotate(k, (xi, yi), textcoords="offset points", xytext=(7, -3),
                        fontsize=9, fontweight="bold", color=c)
    ax.annotate("", xy=(0.09, 0.70), xytext=(0.19, 0.31),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.4,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(0.128, 0.585, "ECE 越低 →\n灵敏度越差", fontsize=8.4, color="#444444",
            ha="left", va="center")
    ax.set_xlabel("平均 ECE（越低越「校准好」）"); ax.set_ylabel("平均灵敏度 TPR")
    ax.set_xlim(0.06, 0.24); ax.set_ylim(0.05, 0.95)
    ax.set_title("(b) ECE 与灵敏度方向相反：\n校准指标改善不等于模型变好",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.4, loc="lower right")
    ax.grid(alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    # (c) π_t 估计噪声下的稳健性
    ax = axes[2]
    rb = load("table_calibfix_robust_seed42_official.csv")
    x = np.arange(len(rb))
    ax.plot(x, rb["cost_A_fixed0.5"], "-o", color=OKABE_ITO["grey"], ms=4, lw=1.4,
            label="固定阈值 0.5（不修正）")
    ax.plot(x, rb["cost_F_oracle"], "-s", color=OKABE_ITO["vermillion"], ms=4, lw=1.4,
            label="先验匹配阈值（已知 pi_t）")
    ax.plot(x, rb["cost_F_est"], "--^", color=OKABE_ITO["blue"], ms=5, lw=1.6,
            label="先验匹配阈值（pi_t 由半数估计）")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c].replace("\n", "") for c in rb["cond"]],
                       fontsize=6.6, rotation=55, ha="right")
    ax.set_ylabel("平均代价风险（400 次两折）")
    ax.set_title("(c) 稳健性：pi_t 估计有噪声时优势几乎不损\n（0.362 → 0.371，优势 0.159）",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.set_ylim(0.28, 0.72)
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    for axx in fig.axes:
        axx.spines["top"].set_visible(False); axx.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_校准修正对照.png")
    fig.savefig(p, dpi=210); print("saved", p)


def fig5():
    plt = setup_matplotlib()
    dd = load("table_detect_vs_difficulty_seed42_route.csv")
    wc = load("table_within_condition_B_seed42_route.csv")
    dd = ordered(dd); wc = ordered(wc)
    x = np.arange(len(dd))
    lab = [LABEL[c] for c in dd["cond"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.7))

    # (a) 降质识别：布尔均值 vs 多变量
    ax = axes[0]
    ax.bar(x - 0.2, dd["auc_detect_qr"], width=0.4, color=OKABE_ITO["grey"],
           label="quality_risk（17 个布尔判据取均值）")
    ax.bar(x + 0.2, dd["auc_detect_mv"], width=0.4, color=OKABE_ITO["blue"],
           label="同一批 17 维特征的多变量检测器")
    ax.axhline(0.5, color="k", lw=0.9, ls="--")
    ax.text(-0.45, 0.515, "随机水平", fontsize=7.6, ha="left", va="bottom")
    ax.set_ylim(0, 1.34)
    ax.text(0.015, 0.985, "灰柱低于 0.5 的档位 = 布尔聚合判据方向相反（变暗档仅 0.43，"
                          "而多变量检测器为 1.00）",
            transform=ax.transAxes, fontsize=7.4, color="#444444", ha="left", va="top")
    ax.set_ylabel("降质图 vs 干净图的判别 AUROC")
    ax.set_title("(a) regime 识别：特征信息充足，\n是「布尔取均值」的聚合把它丢掉了",
                 fontsize=10.5, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.0)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    # (b) 条件内难度预测：三个信号
    ax = axes[1]
    m = wc.set_index("cond").reindex(dd["cond"]).reset_index()
    ax.bar(x - 0.27, m["auc_quality_risk"], width=0.27, color=OKABE_ITO["grey"],
           label="quality_risk")
    ax.bar(x, m["auc_qc14"], width=0.27, color=OKABE_ITO["orange"],
           label="14 个原始 QC 特征（学习式）")
    ax.bar(x + 0.27, m["auc_conf"], width=0.27, color=OKABE_ITO["vermillion"],
           label="模型置信度")
    ax.axhline(0.5, color="k", lw=0.9, ls="--")
    ax.set_ylim(0.3, 1.06); ax.set_ylabel("条件内预测「模型是否判错」的 AUROC")
    ax.set_title("(b) 逐图难度：同一批特征在条件内接近随机，\n置信度反而稳定在 0.73 以上",
                 fontsize=10.5, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.0)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=3,
              columnspacing=0.9, handlelength=1.2)
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.set_axisbelow(True)

    for axx in fig.axes:
        axx.spines["top"].set_visible(False); axx.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_识别与难度分离.png")
    fig.savefig(p, dpi=210); print("saved", p)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
