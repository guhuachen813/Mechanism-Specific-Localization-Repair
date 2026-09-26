r"""思路A 第2步主图：CAM 定位鲁棒性基准（raw CAM）。

Fig A：各条件的配对 Δbest-IoU（条件 − clean，95% CI，整群 Bootstrap 400 次），
       按算子家族着色，\* = CI 不含零。
Fig B：分类-定位解耦。x = Δprob（正类概率变化），y = Δbest-IoU；空心大点 =
       条件均值，小点 = 逐图（抽 clean+显著条件）；右上无失效，左下双失效，
       左上 = 定位单独失效（prob 不掉甚至上升）。标注解耦代表条件。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
})

FAMILY = {
    "clean": "clean",
    "jpeg_q50": "JPEG", "jpeg_q30": "JPEG", "jpeg_q10": "JPEG",
    "blur_1": "Blur", "blur_3": "Blur",
    "noise_003": "Noise", "noise_006": "Noise", "noise_010": "Noise",
    "noise_025": "Noise",
    "dark_05": "Exposure", "bright_16": "Exposure", "contrast_04": "Exposure",
    "ds_2": "Resolution", "ds_4": "Resolution",
    "combo_mild": "Combo", "combo_sev": "Combo",
}
COLOR = {"clean": "#888888", "JPEG": "#0072B2", "Blur": "#56B4E9",
         "Noise": "#D55E00", "Exposure": "#E69F00", "Resolution": "#009E73",
         "Combo": "#CC79A7"}


def main() -> int:
    paired = pd.read_csv(RES / "cam_bench_paired.csv")
    summ = pd.read_csv(RES / "cam_bench_summary.csv")
    raw = pd.read_csv(RES / "cam_raw_nih.csv")
    files = sorted(raw["nih_file"].unique())
    clean = raw[raw["condition"] == "clean"].set_index("nih_file").loc[files]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------- Fig A ----------------
    ax = axes[0]
    p = paired.sort_values("d_best_iou")
    x = np.arange(len(p))
    for i, (_, r) in enumerate(p.iterrows()):
        fam = FAMILY[r["condition"]]
        sig = r["sig"] == "*"
        ax.errorbar(i, r["d_best_iou"],
                    yerr=[[r["d_best_iou"] - r["ci_lo"]], [r["ci_hi"] - r["d_best_iou"]]],
                    fmt="o", ms=4.5, color=COLOR[fam],
                    alpha=1.0 if sig else 0.35,
                    markeredgecolor="black" if sig else "none", markeredgewidth=0.5,
                    capsize=2, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(p["condition"], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel(r"$\Delta$ best-IoU vs. clean (paired)")
    ax.set_title("(a)  Localization degradation ladder (raw CAM)", loc="left", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[f], label=f)
               for f in ["JPEG", "Blur", "Noise", "Exposure", "Resolution", "Combo"]]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right")

    # ---------------- Fig B ----------------
    ax = axes[1]
    conds = [c for c in summ["condition"] if c != "clean"]
    for c in conds:
        d = raw[raw["condition"] == c].set_index("nih_file").loc[files]
        dp = (d["prob"].values - clean["prob"].values)
        di = (d["best_iou"].values - clean["best_iou"].values)
        sig = paired.set_index("condition").loc[c, "sig"] == "*"
        ax.scatter(dp, di, s=5, color=COLOR[FAMILY[c]],
                   alpha=0.12 if c in ("noise_006",) else 0.10, lw=0)
        ax.scatter(dp.mean(), di.mean(), s=90, facecolor="white",
                   edgecolor=COLOR[FAMILY[c]], lw=1.8, zorder=5)
        offs = {"contrast_04": (8, -12), "combo_sev": (8, 6), "noise_006": (-30, 10),
                "combo_mild": (8, -4), "blur_3": (6, 6), "bright_16": (-52, -14),
                "noise_025": (6, 4)}
        if c in offs:
            ax.annotate(c, (dp.mean(), di.mean()), textcoords="offset points",
                        xytext=offs[c], fontsize=7.2, color=COLOR[FAMILY[c]])
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1.05, 0.55)
    ax.set_ylim(-0.75, 0.45)
    ax.set_xlabel(r"$\Delta$ positive-class probability")
    ax.set_ylabel(r"$\Delta$ best-IoU")
    ax.set_title("(b)  Classification vs. localization failure", loc="left", fontsize=10)
    ax.text(0.03, 0.93, "cls-only\nfailure", transform=ax.transAxes, fontsize=7.5,
            color="#888888", va="top")
    ax.text(0.03, 0.03, "both fail", transform=ax.transAxes, fontsize=7.5,
            color="#888888")
    ax.text(0.97, 0.03, "loc-only\nfailure", transform=ax.transAxes, fontsize=7.5,
            color="#888888", ha="right")
    ax.text(0.97, 0.93, "no failure", transform=ax.transAxes, fontsize=7.5,
            color="#888888", ha="right", va="top")

    fig.tight_layout()
    out = RES / "cam_bench_fig.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
