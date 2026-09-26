r"""思路A 第3步主图：先验约束 CAM vs raw CAM。

Fig A：16 个降质条件下的 best-IoU（raw / prior_oracle_hard / prior_real_hard
       三条序列），按 raw 退化幅度排序——先验约束在对比度/复合轴的救援一目了然。
Fig B：配对 Δ(prior_real_hard − raw) 及 95% CI（整群 Bootstrap 400 次），
       正值 = 约束救回的定位；clean 上的增益也在其中。

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
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
})

FAMILY = {
    "jpeg_q50": "JPEG", "jpeg_q30": "JPEG", "jpeg_q10": "JPEG",
    "blur_1": "Blur", "blur_3": "Blur",
    "noise_003": "Noise", "noise_006": "Noise", "noise_010": "Noise",
    "noise_025": "Noise",
    "dark_05": "Exposure", "bright_16": "Exposure", "contrast_04": "Exposure",
    "ds_2": "Resolution", "ds_4": "Resolution",
    "combo_mild": "Combo", "combo_sev": "Combo",
}
COLOR = {"JPEG": "#0072B2", "Blur": "#56B4E9", "Noise": "#D55E00",
         "Exposure": "#E69F00", "Resolution": "#009E73", "Combo": "#CC79A7"}


def main() -> int:
    df = pd.read_csv(RES / "cam_prior_nih.csv")
    files = sorted(df["nih_file"].unique())
    n = len(files)

    def get(v, c, col="best_iou"):
        d = df[(df["variant"] == v) & (df["condition"] == c)]
        return d.set_index("nih_file").loc[files][col].values

    conds = sorted(FAMILY, key=lambda c: get("raw", c).mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})

    # ---------------- Fig A ----------------
    ax = axes[0]
    x = np.arange(len(conds))
    ax.plot(x, [get("raw", c).mean() for c in conds], "o-", color="#444444",
            ms=4, lw=1.4, label="raw CAM")
    ax.plot(x, [get("prior_oracle_hard", c).mean() for c in conds], "s-",
            color="#0072B2", ms=4, lw=1.4, label="prior (oracle)")
    ax.plot(x, [get("prior_real_hard", c).mean() for c in conds], "^--",
            color="#009E73", ms=4.5, lw=1.4, label="prior (realistic)")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("best-IoU")
    ax.set_title("(a)  Prior-constrained vs. raw CAM", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # ---------------- Fig B ----------------
    ax = axes[1]
    diffs = []
    for c in conds + ["clean"]:
        a = get("prior_real_hard", c)
        b = get("raw", c)
        d = a - b
        bs = [d[RNG.integers(0, n, n)].mean() for _ in range(400)]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        diffs.append((c, d.mean(), lo, hi))
    order = sorted(diffs, key=lambda t: t[1])
    for i, (c, m, lo, hi) in enumerate(order):
        fam = FAMILY.get(c, "clean")
        sig = (lo > 0) or (hi < 0)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4.5,
                    color=COLOR.get(fam, "#888888"), alpha=1.0 if sig else 0.35,
                    markeredgecolor="black" if sig else "none", markeredgewidth=0.5,
                    capsize=2, lw=1.2)
        if c == "clean":
            ax.annotate("clean", (i, m), textcoords="offset points",
                        xytext=(-4, 8), fontsize=7.2, color="#666666", ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([t[0] for t in order], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel(r"$\Delta$ best-IoU (prior $-$ raw)")
    ax.set_title("(b)  Rescue effect (paired, 95% CI)", loc="left", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[f], label=f)
               for f in ["JPEG", "Blur", "Noise", "Exposure", "Resolution", "Combo"]]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    out = RES / "cam_prior_fig.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
