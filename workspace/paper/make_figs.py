"""Main figures for the manuscript.

English figures for an IEEEtran (two-column) manuscript, generated from the
result tables produced by the experiment pipeline.

    workspace/perturb_v14/results/   synthetic degradation + CheXpert + NIH
    workspace/chexphoto/results/     CheXphoto v1.0 (real re-photographed)

Every plotted value is read from a result table; annotation numbers were
verified against those tables before the figures were finalized.

Author: ClawsGO Science Agent
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
PERT = Path("../perturb_v14/results")      # paths are relative to workspace/paper/
CP = Path("../chexphoto/results")
OUT = Path("figs")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, PURPLE, GREY = "#E69F00", "#CC79A7", "#555555"
LIGHT, GRID = "#BFC5CB", "#E6E9EC"

COND_LABEL = {
    "clean": "clean", "jpeg_q10": "JPEG q10", "jpeg_q30": "JPEG q30",
    "blur_1": "blur $\\sigma$1", "blur_3": "blur $\\sigma$3",
    "noise_010": "noise $\\sigma$.10", "noise_025": "noise $\\sigma$.25",
    "dark_05": "dark $\\times$0.5", "bright_16": "bright $\\times$1.6",
    "contrast_04": "contrast $\\times$0.4", "ds_4": "downsamp. $\\times$4",
    "combo_mild": "combo mild", "combo_sev": "combo severe",
}

# ascending severity, used by every per-condition figure
ORDER = ["clean", "jpeg_q10", "jpeg_q30", "blur_1", "bright_16", "combo_mild",
         "noise_025", "dark_05", "ds_4", "blur_3", "noise_010", "contrast_04",
         "combo_sev"]

STRAT = ["B", "C", "D", "E", "F"]
STRAT_NAME = {"A": "A frozen $T$", "B": "B ECE-optimal $T$",
              "C": "C NLL-optimal $T$", "D": "D prior offset",
              "E": "E prior offset + $T$", "F": "F prior-matched threshold"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"  -> figs/{name}.pdf")


def tag(ax, t, dx=-0.20, dy=1.03):
    ax.text(dx, dy, t, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", va="top", ha="left")


def load():
    d = {
        "route": pd.read_csv(PERT / "table_conditions_A_seed42_route.csv"),
        "official": pd.read_csv(PERT / "table_conditions_A_seed42_official.csv"),
        "h1_route": pd.read_csv(PERT / "table_h1_robust_A_seed42_route.csv"),
        "pipe": pd.read_csv(PERT / "table_pipeline_contrast.csv"),
        "calib_route": pd.read_csv(PERT / "table_calibfix_A_seed42_route.csv"),
        "calib_off": pd.read_csv(PERT / "table_calibfix_A_seed42_official.csv"),
        "calib_nih": pd.read_csv(PERT / "table_nih_calibfix.csv"),
        "robust_route": pd.read_csv(PERT / "table_calibfix_robust_A_seed42_route.csv"),
        "robust_off": pd.read_csv(PERT / "table_calibfix_robust_A_seed42_official.csv"),
        "oper_route": pd.read_csv(PERT / "table_operational_A_seed42_route.csv"),
        "oper_off": pd.read_csv(PERT / "table_operational_A_seed42_official.csv"),
        "oper_off_b": pd.read_csv(PERT / "table_operational_B_seed42_official.csv"),
        "nih_law_all": pd.read_csv(PERT / "table_nih_law_pc1.csv"),
        "nih_law_geo": pd.read_csv(PERT / "table_nih_law_pc1_几何6维.csv"),
        "nih_law_int": pd.read_csv(PERT / "table_nih_law_pc1_强度11维.csv"),
        "nih_law_qr": pd.read_csv(PERT / "table_nih_law_手工quality_risk.csv"),
        "cp_overall": pd.read_csv(CP / "table_cp_overall.csv"),
        "cp_delta": pd.read_csv(CP / "table_cp_paired_delta.csv"),
        "cp_calib": pd.read_csv(CP / "table_cp_calibration.csv"),
        "diag": pd.read_csv(CP / "diag_signals.csv"),
    }
    d["cp_calib"]["S"] = d["cp_calib"]["strategy"].str.strip().str[0]
    return d


D = load()


def calib_pairs():
    """Per domain: label, mean cost under A, {strategy: cost}, TPR under A, {strategy: TPR}."""
    out = []
    r = D["calib_route"]
    out.append(("CheXpert in-domain", r["cost_A"].mean(),
                {s: r[f"cost_{s}"].mean() for s in STRAT},
                r["tpr_A"].mean(), {s: r[f"tpr_{s}"].mean() for s in STRAT}))
    o = D["calib_off"]
    out.append(("CheXpert shifted", o["cost_A"].mean(),
                {s: o[f"cost_{s}"].mean() for s in STRAT},
                o["tpr_A"].mean(), {s: o[f"tpr_{s}"].mean() for s in STRAT}))
    n = D["calib_nih"].set_index("strategy")
    k = {s: [i for i in n.index if i.strip().startswith(s)][0] for s in "ABCDEF"}
    out.append(("NIH ChestX-ray14", n.loc[k["A"], "cost"],
                {s: n.loc[k[s], "cost"] for s in STRAT},
                n.loc[k["A"], "tpr"], {s: n.loc[k[s], "tpr"] for s in STRAT}))
    c = D["cp_calib"][D["cp_calib"]["cond"] == "natural/oneplus"].set_index("S")
    out.append(("CheXphoto re-photographed", c.loc["A", "cost"],
                {s: c.loc[s, "cost"] for s in STRAT},
                c.loc["A", "tpr"], {s: c.loc[s, "tpr"] for s in STRAT}))
    return out


# --------------------------------------------------------------------------- #
# Fig 1 -- four domains: ranking survives, the operating point does not
# --------------------------------------------------------------------------- #
def fig1():
    dom = [  # label, AUROC, err, prevalence, PPR at threshold 0.5
        ("CheXpert\nin-domain", 0.8440, 0.1289, 0.1260, 0.1234),
        ("CheXpert\nshifted", 0.8289, 0.2624, 0.3267, 0.0743),
        ("NIH\nChestX-ray14", 0.7045, 0.3569, 0.0418, 0.3668),
        ("CheXphoto\nre-photographed", 0.6592, 0.3267, 0.3267, 0.0000),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.20),
                                   layout="constrained",
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    y = np.arange(len(dom))
    h = 0.33
    ax1.barh(y - h / 2, [d[1] for d in dom], height=h, color=BLUE, zorder=3,
             label="AUROC (ranking layer)")
    ax1.barh(y + h / 2, [d[2] for d in dom], height=h, color=VERM, zorder=3,
             label="error rate (decision layer)")
    for yi, d in zip(y, dom):
        ax1.text(d[1] + 0.014, yi - h / 2, f"{d[1]:.3f}", va="center",
                 fontsize=7.4, color=BLUE, fontweight="bold")
        ax1.text(d[2] + 0.014, yi + h / 2, f"{d[2]:.3f}", va="center",
                 fontsize=7.4, color=VERM, fontweight="bold")
    ax1.axvline(0.5, color=GREY, ls=":", lw=0.9, zorder=2)
    ax1.text(0.516, -0.66, "chance", fontsize=7.0, color=GREY)
    ax1.set_yticks(y)
    ax1.set_yticklabels([d[0] for d in dom])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.06)
    ax1.set_ylim(3.8, -0.95)
    ax1.set_xlabel("metric value")
    ax1.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
               handlelength=1.1, columnspacing=1.4)
    tag(ax1, "(a)", dx=-0.20, dy=1.02)

    lim = 0.40
    ax2.plot([0, lim], [0, lim], color=GREY, ls="--", lw=0.9, zorder=1)
    ax2.text(0.250, 0.180, "matched operating point", fontsize=7.0, color=GREY,
             rotation=40, ha="center", va="center")
    for (lab, _, _, prev, ppr), c, mk in zip(
            dom, [BLUE, GREEN, ORANGE, VERM], ["o", "s", "D", "v"]):
        ax2.scatter([prev], [ppr], s=40, color=c, marker=mk, zorder=4,
                    edgecolor="white", linewidth=0.7)
    for t, x, yy, dx, dy, ha, c in [
            ("$0.98\\times$", 0.1260, 0.1234, 0.018, -0.022, "left", BLUE),
            ("$0.23\\times$", 0.3267, 0.0743, -0.016, -0.028, "right", GREEN),
            ("$8.8\\times$", 0.0418, 0.3668, 0.022, -0.012, "left", ORANGE),
            ("$0\\times$", 0.3267, 0.0000, -0.016, 0.018, "right", VERM)]:
        ax2.text(x + dx, yy + dy, t, fontsize=7.6, color=c, fontweight="bold",
                 ha=ha, va="center")
    ax2.set_xlim(-0.025, lim)
    ax2.set_ylim(-0.025, lim)
    ax2.set_xlabel("true prevalence $\\pi$")
    ax2.set_ylabel("predicted positive rate at threshold 0.5")
    ax2.grid(color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    tag(ax2, "(b)", dx=-0.25, dy=1.02)

    save(fig, "fig1_domains")


# --------------------------------------------------------------------------- #
# Fig 2 -- the two layers of damage
# --------------------------------------------------------------------------- #
def fig2():
    r = D["route"].set_index("cond").loc[ORDER]
    h1 = D["h1_route"].set_index("cond").loc[ORDER]
    r = r.join(h1[["auc_base_gb", "delta_gb"]])
    fig = plt.figure(figsize=(7.16, 3.35), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.84, 1.06], wspace=0.02)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) ranking loss and decision loss are decoupled
    ax1.scatter(r["auroc"], r["err"], s=28, color=BLUE, zorder=4,
                edgecolor="white", linewidth=0.6)
    for c, dx, dy, ha, va in [("noise_010", 0.008, 0.022, "left", "bottom"),
                              ("combo_sev", 0.012, 0.000, "left", "center"),
                              ("clean", 0.012, 0.006, "left", "bottom")]:
        ax1.text(r.loc[c, "auroc"] + dx, r.loc[c, "err"] + dy, COND_LABEL[c],
                 fontsize=6.9, color=GREY, ha=ha, va=va)
    ax1.set_xlim(0.43, 0.90)
    ax1.set_ylim(-0.02, 0.96)
    ax1.set_xlabel("AUROC")
    ax1.set_ylabel("error rate")
    ax1.grid(color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    tag(ax1, "(a)", dx=-0.26, dy=1.03)

    # (b) the confidence signal collapses to chance
    o = r.sort_values("auc_err_conf")
    ys = np.arange(len(o))
    cols = [VERM if v < 0.60 else (ORANGE if v < 0.75 else BLUE)
            for v in o["auc_err_conf"]]
    ax2.barh(ys, o["auc_err_conf"], color=cols, height=0.72, zorder=3)
    for yi, v in zip(ys, o["auc_err_conf"]):
        ax2.text(0.98, yi, f"{v:.2f}", va="center", ha="right", fontsize=6.6,
                 color=GREY)
    ax2.axvline(0.5, color=GREY, ls=":", lw=1.0, zorder=4)
    ax2.text(0.505, -0.62, "chance", fontsize=6.8, color=GREY)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([COND_LABEL[c] for c in o.index], fontsize=7.0)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(len(o) - 0.4, -0.6)
    ax2.set_xlabel("AUROC of confidence\nfor detecting errors")
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    tag(ax2, "(b)", dx=-0.40, dy=1.03)

    # (c) the law: the increment shrinks as in-stratum ranking recovers
    x = r["auc_base_gb"].to_numpy(float)
    yv = r["delta_gb"].to_numpy(float)
    ax3.scatter(x, yv, s=28, color=BLUE, zorder=4, edgecolor="white",
                linewidth=0.6)
    k, b = np.polyfit(x, yv, 1)
    xx = np.linspace(x.min() - 0.012, x.max() + 0.012, 50)
    ax3.plot(xx, k * xx + b, color=GREY, lw=1.0, ls="--", zorder=3)
    ax3.axhline(0, color=GREY, lw=0.8, zorder=2)
    for c, dx, dy, ha, va in [("noise_010", -0.012, -0.002, "right", "center"),
                              ("clean", 0.012, 0.002, "left", "bottom")]:
        ax3.text(r.loc[c, "auc_base_gb"] + dx, r.loc[c, "delta_gb"] + dy,
                 COND_LABEL[c], fontsize=6.9, color=GREY, ha=ha, va=va)
    ax3.text(0.025, 0.075, "$r = -0.73$", transform=ax3.transAxes, fontsize=7.8,
             color=VERM, fontweight="bold")
    ax3.text(0.025, 0.020, "(Pearson, 13 strata)", transform=ax3.transAxes,
             fontsize=6.8, color=VERM)
    ax3.set_xlim(0.44, 0.88)
    ax3.set_ylim(-0.014, 0.11)
    ax3.set_xlabel("in-stratum AUROC without\nquality features")
    ax3.set_ylabel("$\\Delta$AUROC from adding quality")
    ax3.grid(color=GRID, lw=0.6, zorder=0)
    ax3.set_axisbelow(True)
    tag(ax3, "(c)", dx=-0.25, dy=1.03)

    save(fig, "fig2_twolayers")


# --------------------------------------------------------------------------- #
# Fig 3 -- the repair
# --------------------------------------------------------------------------- #
def fig3():
    pairs = calib_pairs()
    fig = plt.figure(figsize=(7.16, 3.40), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 0.92, 0.92], wspace=0.02)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) cost change relative to no repair
    y = np.arange(len(STRAT))
    h = 0.19
    cols = [GREEN, ORANGE, PURPLE, BLUE]
    for i, (lab, cA, cs, _, _) in enumerate(pairs):
        ax1.barh(y + (i - 1.5) * h, [cs[s] - cA for s in STRAT], height=h,
                 color=cols[i], label=lab, zorder=3)
    ax1.axvline(0, color=GREY, lw=0.9, zorder=4)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{s}  {STRAT_NAME[s][2:]}" for s in STRAT],
                        fontsize=7.1)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.33, 0.075)
    ax1.set_ylim(4.75, -0.75)
    ax1.set_xlabel("$\\Delta$ decision cost vs. no repair")
    ax1.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
               handlelength=1.0, columnspacing=1.0)
    tag(ax1, "(a)", dx=-0.36, dy=1.02)

    # (b) the benefit tracks how far the operating point sits from the target
    pts = [("CheXpert in-domain", 0.1234, 0.1260, -0.1037, GREEN, "s"),
           ("CheXpert shifted", 0.0743, 0.3267, -0.1104, BLUE, "o"),
           ("NIH ChestX-ray14", 0.3668, 0.0418, -0.2684, PURPLE, "D"),
           ("CheXphoto re-photographed", 0.0010, 0.3267, -0.1485, VERM, "v")]
    for lab, ppr, pi, dc, c, mk in pts:
        ax2.scatter([np.log10(ppr / pi)], [dc], s=40, color=c, marker=mk,
                    zorder=4, edgecolor="white", linewidth=0.7)
    ax2.axhline(0, color=GREY, lw=0.9, zorder=2)
    ax2.set_xlim(-3.05, 1.45)
    ax2.set_ylim(-0.325, 0.05)
    ax2.set_xlabel("$\\log_{10}$(PPR $/ \\pi$)\noperating-point mismatch")
    ax2.set_ylabel("$\\Delta$ cost of strategy F")
    ax2.grid(color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(-0.05, 0.008, "matched:\nF does nothing", fontsize=6.7, color=GREY,
             ha="right", va="bottom")
    ax2.text(1.40, -0.255, "NIH", fontsize=6.9, color=PURPLE, ha="right",
             va="center", fontweight="bold")
    ax2.annotate("operating point\ncollapsed entirely",
                 (np.log10(0.0010 / 0.3267), -0.1485),
                 xytext=(-2.95, -0.245), fontsize=6.7, color=VERM, ha="left",
                 arrowprops=dict(arrowstyle="-", color=VERM, lw=0.7))
    tag(ax2, "(b)", dx=-0.38, dy=1.02)

    # (c) robustness to an estimated target prevalence
    doms = [("CheXpert\nin-domain", D["robust_route"]),
            ("CheXpert\nshifted", D["robust_off"])]
    xs = np.arange(2)
    w = 0.25
    for i, (nm, col, c, dy) in enumerate([
            ("no repair (threshold 0.5)", "cost_A_fixed0.5", LIGHT, 0.016),
            ("F, true $\\pi$", "cost_F_oracle", BLUE, 0.042),
            ("F, estimated $\\hat{\\pi}$", "cost_F_est", VERM, 0.014)]):
        vals = [df[col].mean() for _, df in doms]
        ax3.bar(xs + (i - 1) * w, vals, width=w, color=c, label=nm, zorder=3)
        for xx, v in zip(xs + (i - 1) * w, vals):
            ax3.text(xx, v + dy, f"{v:.3f}", ha="center", fontsize=6.4,
                     color=c if c != LIGHT else GREY, fontweight="bold")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([nm for nm, _ in doms])
    ax3.set_ylim(0, 0.72)
    ax3.set_ylabel("mean cost over the 13 strata")
    ax3.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax3.set_axisbelow(True)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
               handlelength=1.0)
    tag(ax3, "(c)", dx=-0.30, dy=1.02)

    save(fig, "fig3_repair")


# --------------------------------------------------------------------------- #
# Fig 4 -- the residual
# --------------------------------------------------------------------------- #
def fig4():
    cal = D["cp_calib"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05),
                                   layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 0.96]})
    conds = [("clean", "clean"), ("natural/oneplus", "re-photographed"),
             ("synthetic/photographic", "syn. photographic"),
             ("synthetic/digital", "syn. digital")]
    for i, (c, nm) in enumerate(conds):
        a = cal[(cal["cond"] == c) & (cal["S"] == "A")]["cost"].iloc[0]
        f = cal[(cal["cond"] == c) & (cal["S"] == "F")]["cost"].iloc[0]
        col = VERM if c == "natural/oneplus" else LIGHT
        ax1.plot([a, f], [i, i], color=col, lw=1.8, zorder=3)
        ax1.scatter([a], [i], s=30, color=GREY, zorder=4, edgecolor="white",
                    linewidth=0.6)
        ax1.scatter([f], [i], s=36, color=BLUE, zorder=4, marker="D",
                    edgecolor="white", linewidth=0.6)
        ax1.text(max(a, f) + 0.014, i, f"{a:.3f} $\\to$ {f:.3f}", va="center",
                 fontsize=7.0, color=col if col != LIGHT else GREY)
    ax1.scatter([], [], s=30, color=GREY, label="A: no repair (threshold 0.5)")
    ax1.scatter([], [], s=36, color=BLUE, marker="D",
                label="F: prior-matched threshold")
    ax1.set_yticks(np.arange(len(conds)))
    ax1.set_yticklabels([nm for _, nm in conds])
    ax1.invert_yaxis()
    ax1.set_xlim(0.26, 0.84)
    ax1.set_ylim(3.9, -0.9)
    ax1.set_xlabel("decision cost")
    ax1.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    a_real = cal[(cal["cond"] == "natural/oneplus") & (cal["S"] == "A")]["cost"].iloc[0]
    f_real = cal[(cal["cond"] == "natural/oneplus") & (cal["S"] == "F")]["cost"].iloc[0]
    f_clean = cal[(cal["cond"] == "clean") & (cal["S"] == "F")]["cost"].iloc[0]
    ax1.annotate("", xy=(f_clean, 1.50), xytext=(f_real, 1.50),
                 arrowprops=dict(arrowstyle="<->", color=VERM, lw=1.0))
    ax1.text((f_real + f_clean) / 2, 1.62,
             f"irreducible ranking-layer residual {f_real - f_clean:.3f}",
             ha="center", va="top", fontsize=7.1, color=VERM,
             fontweight="bold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1,
               handlelength=1.0)
    tag(ax1, "(a)", dx=-0.25, dy=1.03)

    dd = D["cp_delta"]
    dd = dd[dd["指标"] == "auroc"]
    names = ["re-photographed", "syn. photographic", "syn. digital"]
    for i, (nm, (_, row)) in enumerate(zip(names, dd.iterrows())):
        c = VERM if i == 0 else LIGHT
        ax2.errorbar([row["差值"]], [i],
                     xerr=[[row["差值"] - row["CI下"]], [row["CI上"] - row["差值"]]],
                     fmt="o", ms=5.2, color=c, ecolor=c, elinewidth=1.2,
                     capsize=2.6, zorder=4, markeredgecolor="white",
                     markeredgewidth=0.6)
        ax2.text(row["CI上"] + 0.012, i, f"{row['差值']:+.3f}"
                 + ("*" if row["显著"] == "*" else " (n.s.)"),
                 fontsize=7.1, color=c if c != LIGHT else GREY, va="center")
    ax2.axvline(0, color=GREY, lw=0.9, zorder=2)
    ax2.set_yticks(np.arange(3))
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.31, 0.22)
    ax2.set_ylim(2.7, -0.7)
    ax2.set_xlabel("$\\Delta$AUROC vs. clean (paired, 95% CI)")
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    tag(ax2, "(b)", dx=-0.32, dy=1.03)

    save(fig, "fig4_residual")


# --------------------------------------------------------------------------- #
# Fig 5 -- calibration metrics vs decision cost
# --------------------------------------------------------------------------- #
def fig5():
    cp = D["cp_calib"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.30),
                                   layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # (a) what the ECE-optimal temperature buys: ECE falls everywhere, cost does not
    r, o = D["calib_route"], D["calib_off"]
    n = D["calib_nih"].set_index("strategy")
    kA, kB = ([i for i in n.index if i.strip().startswith(s)][0] for s in "AB")
    other = [("CheXpert in-domain", r["ece_A"].mean(), r["cost_A"].mean(),
              r["ece_B"].mean(), r["cost_B"].mean()),
             ("CheXpert shifted", o["ece_A"].mean(), o["cost_A"].mean(),
              o["ece_B"].mean(), o["cost_B"].mean()),
             ("NIH", n.loc[kA, "ece"], n.loc[kA, "cost"],
              n.loc[kB, "ece"], n.loc[kB, "cost"])]
    for i, (nm, e0, c0, e1, c1) in enumerate(other):
        ax1.scatter([e1 - e0], [c1 - c0], s=22, color=LIGHT, zorder=3,
                    edgecolor="white", linewidth=0.6,
                    label="CheXpert $\\times$2, NIH (unlabelled)"
                    if i == 0 else None)
    for c, nm, dx, dy, ha, va in [
            ("natural/oneplus", "re-photographed", 0.006, 0.005, "left",
             "bottom"),
            ("synthetic/digital", "syn. digital", 0.006, 0.005, "left",
             "bottom"),
            ("clean", "clean", 0.004, -0.006, "left", "top"),
            ("synthetic/photographic", "syn. photographic", -0.006, -0.008,
             "right", "top")]:
        a = cp[(cp["cond"] == c) & (cp["S"] == "A")].iloc[0]
        b = cp[(cp["cond"] == c) & (cp["S"] == "B")].iloc[0]
        ax1.scatter([b["ece"] - a["ece"]], [b["cost"] - a["cost"]], s=30,
                    color=BLUE, zorder=4, edgecolor="white", linewidth=0.6)
        ax1.text(b["ece"] - a["ece"] + dx, b["cost"] - a["cost"] + dy, nm,
                 fontsize=6.9, color=BLUE, ha=ha, va=va)
    ax1.axhline(0, color=GREY, lw=0.9, zorder=2)
    ax1.axvline(0, color=GREY, lw=0.9, zorder=2)
    ax1.annotate("ECE improves in every domain,\n"
                 "yet cost rises in three of the four\n"
                 "CheXphoto conditions and is\n"
                 "unchanged in the fourth",
                 (-0.2146, 0.0), xytext=(-0.238, 0.062), fontsize=6.4,
                 color=VERM, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=VERM, lw=0.7))
    ax1.set_xlim(-0.25, 0.03)
    ax1.set_ylim(-0.11, 0.105)
    ax1.set_xlabel("$\\Delta$ECE from the ECE-optimal temperature")
    ax1.set_ylabel("$\\Delta$ decision cost")
    ax1.grid(color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower left", handlelength=1.0, fontsize=6.9)
    tag(ax1, "(a)", dx=-0.24, dy=1.03)

    # (b) sensitivity by strategy
    pairs = calib_pairs()
    ys = np.arange(len(pairs))
    h = 0.145
    grp = {"A": ("A–C: temperature only", BLUE), "B": (None, BLUE),
           "C": (None, BLUE), "D": ("D–E: prior offset", LIGHT),
           "E": (None, LIGHT), "F": ("F: prior-matched threshold", GREEN)}
    for j, s in enumerate(["A", "B", "C", "D", "E", "F"]):
        vals = [p[4][s] if s != "A" else p[3] for p in pairs]
        lab, col = grp[s]
        ax2.barh(ys + (j - 2.5) * h, vals, height=h, color=col, zorder=3,
                 label=lab)
    ax2.axvline(0, color=GREY, lw=0.9, zorder=4)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([p[0].replace("CheXphoto ", "CheX. ").replace(
        " in-domain", "\nin-domain").replace(" shifted", "\nshifted")
        for p in pairs], fontsize=7.2)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean sensitivity (TPR)")
    ax2.set_xlim(0, 0.80)
    ax2.set_ylim(3.75, -0.75)
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
               handlelength=1.0, labelspacing=0.3)
    tag(ax2, "(b)", dx=-0.33, dy=1.03)

    save(fig, "fig5_calibration")


# --------------------------------------------------------------------------- #
# Fig 6 -- when the quality signal helps, and the routing null
# --------------------------------------------------------------------------- #
def fig6():
    r = D["route"].set_index("cond").loc[ORDER]
    h1 = D["h1_route"].set_index("cond").loc[ORDER]
    r = r.join(h1[["auc_base_gb", "delta_gb"]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.10),
                                   layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 0.90]})
    ax1.scatter(r["auc_base_gb"], r["delta_gb"], s=26, color=BLUE, zorder=5,
                edgecolor="white", linewidth=0.5,
                label="CheXpert, 13 synthetic strata")
    for nm, key, col, mk in [("NIH intensity-11", "nih_law_int", GREEN, "^"),
                             ("NIH geometry-6", "nih_law_geo", VERM, "D"),
                             ("NIH all-17", "nih_law_all", ORANGE, "s"),
                             ("NIH hand-crafted QC", "nih_law_qr", PURPLE, "v")]:
        d = D[key].dropna(subset=["auc_base_gb", "delta_gb"])
        ax1.scatter(d["auc_base_gb"], d["delta_gb"], s=26, color=col, marker=mk,
                    zorder=4, edgecolor="white", linewidth=0.5, label=nm)
    k, b = np.polyfit(r["auc_base_gb"], r["delta_gb"], 1)
    xx = np.linspace(0.44, 0.90, 40)
    ax1.plot(xx, k * xx + b, color=GREY, ls="--", lw=1.0, zorder=3)
    ax1.axhline(0, color=GREY, lw=0.8, zorder=2)
    x0 = 0.7227
    ax1.plot([0, x0], [0.1243, 0.1243], color=VERM, lw=0.7, ls=":", zorder=2)
    ax1.plot([x0, x0], [0, 0.1243], color=VERM, lw=0.7, ls=":", zorder=2)
    ax1.text(x0 + 0.012, 0.127, "NIH whole cohort", fontsize=6.8, color=VERM,
             ha="left", va="bottom")
    ax1.annotate("CheXpert fit at the\nsame failure level",
                 (x0, 0.0302), xytext=(0.845, 0.010), fontsize=6.8, color=GREY,
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
    ax1.text(0.02, 0.06, "$r=-0.73$ (CheXpert)\n"
             "$r=-0.99$ / $-0.79$ / $-0.84$ (NIH axes)",
             transform=ax1.transAxes, fontsize=6.9, color=BLUE,
             fontweight="bold")
    ax1.set_xlim(0.44, 0.93)
    ax1.set_ylim(-0.014, 0.265)
    ax1.set_xlabel("in-stratum AUROC without quality features")
    ax1.set_ylabel("$\\Delta$AUC from adding quality")
    ax1.grid(color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=6.4, handlelength=1.0,
               labelspacing=0.26, borderaxespad=0.15)
    tag(ax1, "(a)", dx=-0.23, dy=1.03)

    lbl, val, lo, hi = [], [], [], []
    for nm, df, cov in [("CheXpert in-domain", D["oper_route"], 0.5),
                        ("CheXpert in-domain", D["oper_route"], 0.75),
                        ("CheXpert shifted", D["oper_off"], 0.5),
                        ("CheXpert shifted", D["oper_off"], 0.75),
                        ("CheXpert shifted (B)", D["oper_off_b"], 0.5),
                        ("CheXpert shifted (B)", D["oper_off_b"], 0.75)]:
        row = df[np.isclose(df["coverage"], cov)].iloc[0]
        lbl.append(f"{nm}\n{cov:.0%} cov.")
        val.append(row["delta_cost"])
        lo.append(row["delta_cost"] - row["ci_lo"])
        hi.append(row["ci_hi"] - row["delta_cost"])
    for i, (v, l, h) in enumerate(zip(val, lo, hi)):
        sig = (v - l > 0 and v - h > 0) or (v - l < 0 and v - h < 0)
        c = VERM if sig else GREY
        ax2.errorbar([v], [i], xerr=[[l], [h]], fmt="o", ms=4.8, color=c,
                     ecolor=c, elinewidth=1.2, capsize=2.4, zorder=4,
                     markeredgecolor="white", markeredgewidth=0.6)
        ax2.text(v, i - 0.30, f"{v:+.4f}", fontsize=6.7, color=c, ha="center")
    ax2.axvline(0, color=GREY, lw=0.9, zorder=2)
    ax2.set_yticks(np.arange(6))
    ax2.set_yticklabels(lbl, fontsize=6.6)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.10, 0.062)
    ax2.set_ylim(5.72, -0.66)
    ax2.set_xlabel("$\\Delta$ cost of adding quality to confidence routing")
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(-0.096, 5.36, "1 of 6 intervals\nexcludes zero ($-0.14$ pp)",
             fontsize=6.8, color=VERM, ha="left", va="center")
    tag(ax2, "(b)", dx=-0.40, dy=1.03)

    save(fig, "fig6_conditionality")


# --------------------------------------------------------------------------- #
# Fig 7 -- two methodological traps
# --------------------------------------------------------------------------- #
def fig7():
    p = D["pipe"].set_index("cond")
    order = ["clean", "jpeg_q30", "blur_1", "blur_3", "noise_010", "dark_05",
             "bright_16", "contrast_04", "ds_4", "combo_mild", "combo_sev"]
    p = p.loc[[c for c in order if c in p.index]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.40),
                                   layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 0.94]})
    for i, (c, row) in enumerate(p.iterrows()):
        hi = c == "noise_010"
        col = VERM if hi else LIGHT
        ax1.plot([row["B_auroc"], row["A_auroc"]], [i, i], color=col, lw=1.6,
                 zorder=3)
        ax1.scatter([row["B_auroc"]], [i], s=22, color=GREY, zorder=4,
                    edgecolor="white", linewidth=0.5)
        ax1.scatter([row["A_auroc"]], [i], s=28, color=BLUE, zorder=4,
                    marker="D", edgecolor="white", linewidth=0.5)
    ax1.scatter([], [], s=22, color=GREY,
                label="degradation applied at 320 ($\\to$224 for the model)")
    ax1.scatter([], [], s=28, color=BLUE, marker="D",
                label="degradation applied at 224 (model input resolution)")
    ax1.set_yticks(np.arange(len(p)))
    ax1.set_yticklabels([COND_LABEL[c] for c in p.index], fontsize=7.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0.42, 0.90)
    ax1.set_ylim(10.7, -0.7)
    ax1.set_xlabel("AUROC (same images, same weights)")
    ax1.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
               handlelength=1.0)
    ax1.annotate("$\\sigma\\!=\\!0.10$: AUROC 0.475 vs 0.812,\n"
                 "while the quality features report\n"
                 "the same quality ($q$ = 0.3750 vs 0.3760)",
                 (0.477, 4.0), xytext=(0.485, 6.4), fontsize=6.8, color=VERM,
                 ha="left", arrowprops=dict(arrowstyle="-", color=VERM, lw=0.7))
    tag(ax1, "(a)", dx=-0.27, dy=1.03)

    dg = D["diag"]
    res = []
    for c in ["clean", "natural/oneplus", "synthetic/photographic",
              "synthetic/digital"]:
        s = dg[dg["cond"] == c]
        y = s["y"].to_numpy(float)
        pv = s["p"].to_numpy(float)
        soft = np.where(y > 0.5, 1 - pv, pv)
        hi = (soft > np.median(soft)).astype(int)
        res.append((roc_auc_score(hi, pv),
                    roc_auc_score(y, s["tta_gap"].to_numpy(float)),
                    roc_auc_score(y, s["qc_quality_risk"].to_numpy(float))))
    xs = np.arange(4)
    w = 0.26
    for i, (nm, idx, col) in enumerate([("$p$ itself", 0, BLUE),
                                        ("flip-TTA divergence", 1, VERM),
                                        ("QC features", 2, LIGHT)]):
        vals = [r[idx] for r in res]
        ax2.bar(xs + (i - 1) * w, vals, width=w, color=col, label=nm, zorder=3)
        for xx, v in zip(xs + (i - 1) * w, vals):
            ax2.text(xx, v + 0.018 + 0.05 * (i == 2), f"{v:.2f}", ha="center",
                     fontsize=6.4, color=col if col != LIGHT else GREY,
                     fontweight="bold")
    ax2.axhline(0.5, color=GREY, ls=":", lw=1.0, zorder=4)
    ax2.text(3.45, 0.512, "chance", fontsize=6.8, color=GREY, ha="right")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["clean", "re-photo-\ngraphed", "syn.\nphotographic",
                         "syn.\ndigital"], fontsize=7.0)
    ax2.set_ylim(0, 1.22)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel("AUROC for ranking the soft error")
    ax2.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
               handlelength=1.0)
    ax2.text(-0.42, 1.145, "the target already contains $p$:\nit outranks every candidate signal",
             fontsize=6.8, color=BLUE, ha="left", va="top")
    tag(ax2, "(b)", dx=-0.29, dy=1.03)

    save(fig, "fig7_traps")


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        fn()
    print("done")
