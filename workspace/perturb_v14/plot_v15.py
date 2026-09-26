"""v1.5 配图：全部结论改用设计 A（工作分辨率 224 = 模型分辨率）的真实管线。

图组
----
fig1 降质阶梯（设计 A，域内 + 偏移域）
fig2 管线混淆与「何时有用」定律（A vs B 受控对照）
fig3 识别与难度分离（设计 A）
fig4 校准修正对照（设计 A，六策略）
fig5 操作性代价（固定复核预算）

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

# 设计 A 的 13 个档位，按算子种类分组排列（不是按严重度，便于读者按类读图）
ORDER = ["clean", "jpeg_q10", "jpeg_q30", "blur_1", "blur_3",
         "noise_010", "noise_025", "dark_05", "bright_16",
         "contrast_04", "ds_4", "combo_mild", "combo_sev"]
LABEL = {
    "clean": "干净", "jpeg_q50": "JPEG\n50", "jpeg_q30": "JPEG\n30", "jpeg_q10": "JPEG\n10",
    "blur_1": "模糊\nσ1", "blur_3": "模糊\nσ3",
    "noise_003": "噪声\n.03", "noise_006": "噪声\n.06",
    "noise_010": "噪声\n.10", "noise_025": "噪声\n.25",
    "dark_05": "偏暗\n×0.5", "bright_16": "偏亮\n×1.6",
    "contrast_04": "低对比\n×0.4", "ds_2": "降采样\n×2", "ds_4": "降采样\n×4",
    "combo_mild": "复合\n轻", "combo_sev": "复合\n重",
}
CN = {"clean": "干净", "jpeg_q10": "JPEG10", "jpeg_q30": "JPEG30", "blur_1": "模糊σ1",
      "blur_3": "模糊σ3", "noise_010": "噪声.10", "noise_025": "噪声.25",
      "dark_05": "偏暗0.5", "bright_16": "偏亮1.6", "contrast_04": "低对比0.4",
      "ds_4": "降采样×4", "combo_mild": "复合轻", "combo_sev": "复合重"}


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def ordered(df, order=ORDER):
    d = df.copy()
    d["_o"] = d["cond"].map({c: i for i, c in enumerate(order)})
    return d.sort_values("_o").reset_index(drop=True)


def style(ax, grid="y"):
    if grid == "y":
        ax.grid(axis="y", alpha=0.18, lw=0.6)
    elif grid == "x":
        ax.grid(axis="x", alpha=0.18, lw=0.6)
    elif grid == "both":
        ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------------------------
# fig1 —— 降质阶梯
# --------------------------------------------------------------------------
def fig1():
    plt = setup_matplotlib()
    reg = ordered(load("table_conditions_A_seed42_route.csv"))
    off = ordered(load("table_conditions_A_seed42_official.csv"))
    x = np.arange(len(reg))
    lab = [LABEL[c] for c in reg["cond"]]
    B, V = OKABE_ITO["blue"], OKABE_ITO["vermillion"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.6))

    # (a) AUROC
    ax = axes[0, 0]
    ax.bar(x - 0.2, reg["auroc"], width=0.4, color=B, label="route-validation（域内）")
    ax.bar(x + 0.2, off["auroc"], width=0.4, color=V, label="official-valid（偏移域）")
    ax.axhline(0.5, color="k", lw=1.0, ls="--")
    i = list(reg["cond"]).index("noise_010")
    ax.annotate("噪声 σ=0.10 档：\nAUROC 跌到 0.48（低于随机）",
                xy=(i + 0.22, reg["auroc"].iloc[i] + 0.015), xytext=(i + 0.75, 0.945),
                fontsize=7.8, color="#333333", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                connectionstyle="arc3,rad=0.18"))
    ax.set_ylim(0.40, 1.06); ax.set_ylabel("AUROC")
    ax.set_title("(a) 排序能力：除强噪声外，降质只让 AUROC 缓慢下滑", fontsize=10.8, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    style(ax)

    # (b) 错误率
    ax = axes[0, 1]
    ax.plot(x, reg["err"], "-o", color=B, ms=4.5, lw=1.8, label="route-validation")
    ax.plot(x, off["err"], "-s", color=V, ms=4.5, lw=1.8, label="official-valid")
    ax.set_ylabel("全覆盖错误率"); ax.set_ylim(0, 0.95)
    ax.set_title("(b) 判别阈值失效：低对比与复合重度档错误率翻 6 倍", fontsize=10.8, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    for c, col in (("contrast_04", B), ("combo_sev", V)):
        j = list(reg["cond"]).index(c)
        ax.annotate(f"{reg['err'].iloc[j]*100:.0f}%", (j, reg["err"].iloc[j]),
                    textcoords="offset points", xytext=(-6, 6), ha="center", fontsize=8, color=B)
        ax.annotate(f"{off['err'].iloc[j]*100:.0f}%", (j, off["err"].iloc[j]),
                    textcoords="offset points", xytext=(12, -9), ha="center", fontsize=8, color=V)
    style(ax)

    # (c) 置信度的错误检测能力（关键：这里才是真正崩掉的东西）
    ax = axes[1, 0]
    ax.plot(x, reg["auc_err_conf"], "-o", color=B, ms=4.5, lw=1.8, label="route-validation")
    ax.plot(x, off["auc_err_conf"], "-s", color=V, ms=4.5, lw=1.8, label="official-valid")
    ax.axhline(0.5, color="k", lw=1.0, ls="--")
    ax.text(-0.45, 0.515, "随机水平", fontsize=7.4, ha="left", va="bottom")
    ax.set_ylabel("置信度检测「模型判错」的 AUROC"); ax.set_ylim(0.30, 0.92)
    ax.set_title("(c) 真正崩掉的是置信度：低对比档降到 0.50（＝完全失效）",
                 fontsize=10.8, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    j = list(reg["cond"]).index("contrast_04")
    ax.annotate(f"{reg['auc_err_conf'].iloc[j]:.2f}", (j, reg["auc_err_conf"].iloc[j]),
                textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=B)
    style(ax)

    # (d) 操作点塌缩
    ax = axes[1, 1]
    ax.plot(x, _ppr(reg), "-o",
            color=B, ms=4.5, lw=1.8, label="route-validation 预测阳性率")
    ax.plot(x, _ppr(off), "-s", color=V, ms=4.5, lw=1.8, label="official-valid 预测阳性率")
    ax.axhline(0.126, color=B, lw=1.2, ls=":")
    ax.axhline(0.327, color=V, lw=1.2, ls=":")
    ax.text(0.05, 0.345, "official 真实患病率 32.7%", fontsize=7.8, color=V)
    ax.text(0.05, 0.030, "route 真实患病率 12.6%", fontsize=7.8, color=B, va="bottom")
    ax.annotate("", xy=(0.35, 0.325), xytext=(0.35, 0.079),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.2))
    ax.text(0.62, 0.22, "偏移域干净档少判 4.1 倍\n（近乎「全判阴性」）",
            fontsize=8, color="#333333", ha="left", va="center")
    j = list(reg["cond"]).index("contrast_04")
    ax.annotate("低对比档反向塌缩：\n91% 判阳（患病率 12.6%）",
                xy=(j + 0.05, _ppr(reg)[j] - 0.02), xytext=(j - 5.6, 0.63), fontsize=7.8,
                color="#333333", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                connectionstyle="arc3,rad=-0.22"))
    ax.set_ylabel("预测阳性率（阈值 0.5）"); ax.set_ylim(0, 1.06)
    ax.set_title("(d) 操作点双向塌缩：或几乎不判阳，或几乎全判阳", fontsize=10.8, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    style(ax)

    fig.tight_layout()
    p = os.path.join(OUT, "fig1_降质阶梯.png")
    fig.savefig(p, dpi=210); print("saved", p)


def _ppr(d):
    """由混淆比例重建预测阳性率。

    `table_conditions_*` 的 fn / fp 列存的是**占该档样本数的比例**，
    故 PPR = 阳性被正确找回的部分 + 假阳 = (患病率 − fn) + fp。
    """
    return (d["prevalence"].to_numpy() - d["fn"].to_numpy() + d["fp"].to_numpy())


# --------------------------------------------------------------------------
# fig2 —— 管线混淆与「何时有用」定律
# --------------------------------------------------------------------------
def fig2():
    plt = setup_matplotlib()
    pc = load("table_pipeline_contrast.csv")          # A(224/224) vs B(320→224)，同批同档
    a = load("table_h1_robust_A_seed42_route.csv")
    b = load("table_h1_robust_seed42_route.csv")
    ia = load("table_incremental_A_seed42_route.csv")
    ib = load("table_incremental_B_seed42_route.csv")

    fig = plt.figure(figsize=(14.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.25, 1.05], wspace=0.30)

    # (a) 同档 AUROC 对照（哑铃图）
    ax = fig.add_subplot(gs[0, 0])
    pc = pc.copy()
    pc["_o"] = pc["cond"].map({c: i for i, c in enumerate(ORDER)})
    pc = pc.sort_values("_o").reset_index(drop=True)
    y = np.arange(len(pc))
    for yi, r in pc.iterrows():
        hi = max(r["A_auroc"], r["B_auroc"])
        ax.plot([r["A_auroc"], r["B_auroc"]], [yi, yi], color="#bbbbbb", lw=1.6, zorder=1)
    ax.scatter(pc["A_auroc"], y, s=52, color=OKABE_ITO["blue"], zorder=4,
               edgecolor="white", linewidth=0.8, label="设计 A：224 / 224（对齐）")
    ax.scatter(pc["B_auroc"], y, s=52, color=OKABE_ITO["orange"], zorder=4, marker="D",
               edgecolor="white", linewidth=0.8, label="设计 B：320 → 224（错位）")
    ax.set_yticks(y); ax.set_yticklabels([CN[c] for c in pc["cond"]], fontsize=8.4)
    ax.axvline(0.5, color="k", lw=1.0, ls="--")
    ax.set_xlim(0.40, 0.90); ax.set_xlabel("同档模型 AUROC")
    j = list(pc["cond"]).index("noise_010")
    ax.annotate("同一档、同一参数、同一模型：\nAUROC 0.48 vs 0.81",
                xy=(0.60, j), xytext=(0.415, j + 3.4), fontsize=7.8, color="#333333",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("(a) 降质参数不等于模型感受到的降质：\n换工作分辨率，噪声档结论完全反转",
                 fontsize=10.4, loc="left")
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")
    style(ax, "x")

    # (b) 「何时有用」定律：Δ 是残余置信度的单调递减函数
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(a["auc_base_gb"], a["delta_gb"], s=54, color=OKABE_ITO["blue"],
               edgecolor="white", linewidth=0.8, zorder=4, label="设计 A（17816/档）")
    ax.scatter(b["auc_base_gb"], b["delta_gb"], s=54, color=OKABE_ITO["orange"], marker="D",
               edgecolor="white", linewidth=0.8, zorder=4, label="设计 B（1924/档）")
    for d, c in ((a, OKABE_ITO["blue"]), (b, OKABE_ITO["orange"])):
        s = np.polyfit(d["auc_base_gb"], d["delta_gb"], 1)
        xs = np.linspace(0.49, 0.84, 50)
        ax.plot(xs, np.polyval(s, xs), color=c, lw=1.5, ls="-", alpha=0.75,
                zorder=2)
    ax.axhline(0, color="k", lw=1.0)
    ra = np.corrcoef(a["auc_base_gb"], a["delta_gb"])[0, 1]
    rb = np.corrcoef(b["auc_base_gb"], b["delta_gb"])[0, 1]
    ax.text(0.982, 0.965, f"r = {ra:+.2f}（设计 A）\nr = {rb:+.2f}（设计 B）",
            transform=ax.transAxes, fontsize=8.6, ha="right", va="top", color="#333333")
    ax.text(0.982, 0.775, "两管线遵守同一条律，\n只是落点不同",
            transform=ax.transAxes, fontsize=7.8, ha="right", va="top", color="#666666")
    ax.set_xlabel("条件内置信度检测错误的 AUROC（残余置信度）")
    ax.set_ylabel("质量特征的条件内增量 ΔAUC")
    ax.set_ylim(-0.030, 0.115)
    ax.set_title("(b) 「何时有用」定律：置信度越失效，质量特征越有用", fontsize=10.4, loc="left")
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")
    style(ax, "both")

    # (c) 跨条件池化增量：A vs B 成对
    ax = fig.add_subplot(gs[0, 2])
    def pick(d):
        d = d[d["protocol"] == "按条件分折（跨降质泛化）"].set_index("model")
        return [d.loc[m, "delta"] for m in ("+quality", "+tta", "+disagree")]
    da, db = pick(ia), pick(ib)
    names = ["+质量", "+TTA", "+模型分歧"]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, da, width=w, color=OKABE_ITO["blue"], label="设计 A")
    ax.bar(x + w / 2, db, width=w, color=OKABE_ITO["orange"], label="设计 B")
    for xi, (va, vb) in enumerate(zip(da, db)):
        ax.annotate(f"{va*1000:+.1f}", (xi - w / 2, va), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.8, color=OKABE_ITO["blue"])
        ax.annotate(f"{vb*1000:+.1f}", (xi + w / 2, vb), textcoords="offset points",
                    xytext=(0, 4 if vb >= 0 else -12), ha="center", fontsize=7.8,
                    color=OKABE_ITO["orange"])
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.2)
    ax.set_ylabel("ΔAUC（千分点）")
    ax.set_ylim(-0.004, 0.014)
    ax.set_title("(c) 同一批图、同一模型：\n换管线后所有信号的增量一起归零",
                 fontsize=10.4, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    style(ax)

    p = os.path.join(OUT, "fig2_管线混淆与定律.png")
    fig.savefig(p, dpi=210); print("saved", p)


# --------------------------------------------------------------------------
# fig3 —— 识别与难度分离
# --------------------------------------------------------------------------
def fig3():
    plt = setup_matplotlib()
    dd = ordered(load("table_detect_vs_difficulty_A_seed42_route.csv"))
    wc_all = load("table_within_condition_A_seed42_route.csv")
    # 识别检验是「降质图 vs 干净图」的配对设计，没有 clean 档，故按 dd 的档位对齐
    wc = wc_all.set_index("cond").reindex(dd["cond"]).reset_index()
    x = np.arange(len(dd)); lab = [LABEL[c] for c in dd["cond"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7))

    ax = axes[0]
    ax.bar(x - 0.2, dd["auc_detect_qr"], width=0.4, color=OKABE_ITO["grey"],
           label="quality_risk（9 个布尔判据取均值）")
    ax.bar(x + 0.2, dd["auc_detect_mv"], width=0.4, color=OKABE_ITO["blue"],
           label="同一批 17 维特征的多变量检测器")
    ax.axhline(0.5, color="k", lw=1.0, ls="--")
    ax.text(-0.45, 0.515, "随机水平", fontsize=7.4, ha="left", va="bottom")
    ax.set_ylim(0, 1.20)
    ax.text(0.015, 0.975,
            "信息在特征里：多变量检测器 0.79–1.00，布尔取均值只有 0.50–0.97",
            transform=ax.transAxes, fontsize=7.6, color="#444444", ha="left", va="top")
    ax.set_ylabel("降质图 vs 干净图的判别 AUROC")
    ax.set_title("(a) regime 识别：特征信息充足，是布尔取均值的聚合把它丢掉",
                 fontsize=10.5, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.0)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    style(ax)

    ax = axes[1]
    ax.bar(x - 0.27, wc["auc_quality_risk"], width=0.27, color=OKABE_ITO["grey"],
           label="quality_risk")
    ax.bar(x, wc["auc_qc14"], width=0.27, color=OKABE_ITO["orange"],
           label="17 个原始 QC 特征（学习式）")
    ax.bar(x + 0.27, wc["auc_conf"], width=0.27, color=OKABE_ITO["vermillion"],
           label="模型置信度")
    ax.axhline(0.5, color="k", lw=1.0, ls="--")
    ax.set_ylim(0.3, 1.06); ax.set_ylabel("条件内预测「模型是否判错」的 AUROC")
    ax.set_title("(b) 逐图难度：置信度正常时质量特征无用，\n置信度崩掉时质量特征接手",
                 fontsize=10.5, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.0)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=1)
    style(ax)

    fig.tight_layout()
    p = os.path.join(OUT, "fig3_识别与难度分离.png")
    fig.savefig(p, dpi=210); print("saved", p)


# --------------------------------------------------------------------------
# fig4 —— 校准修正对照
# --------------------------------------------------------------------------
SKEY = ["A", "B", "C", "D", "E", "F"]
SNAME = {"A": "冻结温度\n（不修正）", "B": "ECE 最优\n温度", "C": "NLL 最优\n温度",
         "D": "仅先验\n偏移", "E": "先验偏移\n+ 温度", "F": "先验匹配\n阈值"}


def fig4():
    plt = setup_matplotlib()
    da = load("table_calibfix_A_seed42_route.csv")
    do = load("table_calibfix_A_seed42_official.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.7))

    # (a) 代价风险
    ax = axes[0]
    x = np.arange(6); w = 0.38
    series = [("设计 A · 域内 route", da, OKABE_ITO["blue"]),
              ("设计 A · 偏移域 official", do, OKABE_ITO["vermillion"])]
    zeros = {}
    for k, (nm, d, c) in enumerate(series):
        vals = [d[f"cost_{s}"].mean() for s in SKEY]
        ax.bar(x + (k - 0.5) * w, vals, width=w, color=c, label=nm)
        zeros[k] = [int((d[f"tpr_{s}"] < 1e-9).sum()) for s in SKEY]
    # TPR=0 的档数排成固定高度的一行，避免贴着柱顶时相邻标签相撞
    ax.axhline(0.655, color="#dddddd", lw=0.8)
    ax.text(-0.48, 0.688, "TPR=0 档数（route / official）", fontsize=7.4,
            color="#555555", ha="left", va="bottom")
    for xi, s in enumerate(SKEY):
        ax.text(xi, 0.655, f"{zeros[0][xi]} / {zeros[1][xi]}", ha="center", va="bottom",
                fontsize=8.4, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels([SNAME[s] for s in SKEY], fontsize=7.8)
    ax.set_ylabel("代价风险（FN:FP = 2:1，每张）")
    ax.set_ylim(0, 0.755)
    ax.set_title("(a) 只有「重定位操作点」的 F 在两个域上都降低代价，\n且从不把任何档位压成 TPR=0",
                 fontsize=10.4, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    style(ax)

    # (b) ECE–灵敏度平面（只画 A/B/C/F：D≡A、E≡B 是同一分类器，见正文）
    ax = axes[1]
    LEV = ["A", "B", "C", "F"]
    for nm, d, c, mk in (("域内 route", da, OKABE_ITO["blue"], "o"),
                         ("偏移域 official", do, OKABE_ITO["vermillion"], "^")):
        e = [d[f"ece_{s}"].mean() for s in LEV]
        t = [d[f"tpr_{s}"].mean() for s in LEV]
        ax.scatter(e, t, s=86, color=c, marker=mk, edgecolor="white", linewidth=1.0,
                   zorder=4, label=nm)
        for ei, ti, s in zip(e, t, LEV):
            ax.annotate(s, (ei, ti), textcoords="offset points", xytext=(9, -3),
                        fontsize=9.5, fontweight="bold", color=c)
    ax.annotate("", xy=(0.098, 0.195), xytext=(0.205, 0.335),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.4,
                                connectionstyle="arc3,rad=0.10"))
    ax.text(0.148, 0.300, "温度类修正：\nECE 降，灵敏度也降", fontsize=8.2,
            color="#444444", ha="left", va="center")
    ax.set_xlabel("平均 ECE（越低越「校准好」）"); ax.set_ylabel("平均灵敏度 TPR")
    ax.set_xlim(0.0, 0.28); ax.set_ylim(0.08, 0.70)
    ax.set_title("(b) 校准指标与临床灵敏度方向相反（D≡A、E≡B 故略去）",
                 fontsize=10.4, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    style(ax, "both")

    # (c) 稳健性
    ax = axes[2]
    rb = load("table_calibfix_robust_A_seed42_official.csv").copy()
    rb["_o"] = rb["cond"].map({c: i for i, c in enumerate(ORDER)})
    rb = rb.sort_values("_o").reset_index(drop=True)
    xs = np.arange(len(rb))
    ax.plot(xs, rb["cost_A_fixed0.5"], "-o", color=OKABE_ITO["grey"], ms=4.2, lw=1.5,
            label="固定阈值 0.5（不修正）")
    ax.plot(xs, rb["cost_F_oracle"], "-s", color=OKABE_ITO["vermillion"], ms=4.2, lw=1.5,
            label="先验匹配阈值（已知 pi_t）")
    ax.plot(xs, rb["cost_F_est"], "--^", color=OKABE_ITO["blue"], ms=5, lw=1.7,
            label="先验匹配阈值（pi_t 由半数样本估计）")
    ax.set_xticks(xs)
    ax.set_xticklabels([CN[c] for c in rb["cond"]], fontsize=6.8, rotation=52, ha="right")
    ax.set_ylabel("平均代价风险（400 次两折）")
    ax.set_title("(c) 偏移域稳健性：pi_t 只能估计时优势几乎不损\n（0.538 → 0.427 已知 / 0.434 估计）",
                 fontsize=10.4, loc="left")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    ax.set_ylim(0.28, 0.90)
    style(ax)

    fig.tight_layout()
    p = os.path.join(OUT, "fig4_校准修正对照.png")
    fig.savefig(p, dpi=210); print("saved", p)


# --------------------------------------------------------------------------
# fig5 —— 操作性代价
# --------------------------------------------------------------------------
def fig5():
    plt = setup_matplotlib()
    sets = [("设计 A · 域内 route", "table_operational_A_seed42_route.csv", OKABE_ITO["blue"]),
            ("设计 A · 偏移域 official", "table_operational_A_seed42_official.csv", OKABE_ITO["vermillion"]),
            ("设计 B · 偏移域 official", "table_operational_B_seed42_official.csv", OKABE_ITO["grey"])]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(2); w = 0.26
    for k, (nm, f, c) in enumerate(sets):
        d = load(f)
        vals = d["delta_cost"].to_numpy()
        lo = d["ci_lo"].to_numpy(); hi = d["ci_hi"].to_numpy()
        ax.errorbar(x + (k - 1) * w, vals, yerr=[vals - lo, hi - vals], fmt="o",
                    color=c, ecolor=c, capsize=3.5, ms=7, lw=1.5, label=nm, zorder=3)
    ax.axhline(0, color="k", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(["覆盖率 50%", "覆盖率 75%"], fontsize=9.6)
    ax.set_ylabel("Δ代价风险（置信度+质量 − 纯置信度）")
    ax.set_title("固定复核预算下，加入质量信号的代价收益很小且不稳定\n（6 个点中 5 个区间跨 0）",
                 fontsize=10.6, loc="left")
    ax.legend(frameon=False, fontsize=8.4, loc="lower left")
    style(ax, "both")
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_操作性代价.png")
    fig.savefig(p, dpi=210); print("saved", p)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
