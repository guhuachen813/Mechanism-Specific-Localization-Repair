r"""思路A 第4步汇总图：真实翻拍一致性 + 随机区域消融。

Fig A：CheXphoto 三条件 × 三变体的定位一致性 agree_iou（与各自 clean CAM 的
       固定阈值 IoU），真实翻拍 vs 合成条件的落差一目了然。
Fig B：随机区域消融——raw / 随机矩形(面积匹配) / 解剖先验在 4 条件上的
       best-IoU 配对差（95% CI）。

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
COND_LAB = {"natural/oneplus": "natural\n(real re-photograph)",
            "synthetic/photographic": "synthetic\nphotographic",
            "synthetic/digital": "synthetic\ndigital"}


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---------------- Fig A：真实翻拍一致性 ----------------
    ax = axes[0]
    df = pd.read_csv(RES / "cam_real_photo.csv")
    files = sorted(df["base_key"].unique())
    n = len(files)
    conds = ["synthetic/digital", "synthetic/photographic", "natural/oneplus"]
    variants = [("raw", "#444444"), ("prior_oracle_hard", "#0072B2"),
                ("prior_real_hard", "#009E73")]
    w = 0.26
    for i, c in enumerate(conds):
        d = df[df["condition"] == c]
        for j, (v, col) in enumerate(variants):
            a = d[d["variant"] == v].set_index("base_key").loc[files]["agree_iou"].values
            m = a.mean()
            bs = [a[RNG.integers(0, n, n)].mean() for _ in range(400)]
            lo, hi = np.quantile(bs, [0.025, 0.975])
            ax.errorbar(i + (j - 1) * w, m, yerr=[[m - lo], [hi - m]], fmt="o",
                        ms=5, color=col, capsize=2, lw=1.2,
                        label=v if i == 0 else None)
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LAB[c] for c in conds], fontsize=8)
    ax.set_ylabel("CAM agreement with clean (IoU@0.45)")
    ax.set_title("(a)  Real re-photograph vs. synthetic (CheXphoto, n=202)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, 0.85)

    # ---------------- Fig B：随机区域消融 ----------------
    ax = axes[1]
    df = pd.read_csv(RES / "cam_random_nih.csv")
    files = sorted(df["nih_file"].unique())
    n = len(files)
    conds = ["clean", "contrast_04", "combo_sev", "noise_006"]
    groups = [("raw", "#888888", "raw"), ("random", "#E69F00", "matched random box"),
              ("prior", "#009E73", "anatomy prior")]
    x = np.arange(len(conds))
    for gi, (gname, col, lab) in enumerate(groups):
        vals, los, his = [], [], []
        for c in conds:
            d = df[df["condition"] == c].set_index("nih_file")
            b = d[d["variant"] == "raw"].loc[files]["best_iou"].values
            if gname == "raw":
                a = b
            elif gname == "prior":
                a = d[d["variant"] == "prior_real_hard"].loc[files]["best_iou"].values
            else:
                r = df[(df["condition"] == c)
                       & df["variant"].str.startswith("random_")]
                a = r.groupby("nih_file")["best_iou"].mean().loc[files].values
            diff = a - b
            if gname == "raw":
                vals.append(0.0); los.append(0.0); his.append(0.0)
                continue
            bs = [diff[RNG.integers(0, n, n)].mean() for _ in range(400)]
            lo, hi = np.quantile(bs, [0.025, 0.975])
            vals.append(diff.mean()); los.append(lo); his.append(hi)
        ax.errorbar(x + (gi - 1) * 0.2, vals,
                    yerr=[np.array(vals) - np.array(los),
                          np.array(his) - np.array(vals)],
                    fmt="o", ms=5, color=col, capsize=2, lw=1.2, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel(r"$\Delta$ best-IoU vs. raw (paired)")
    ax.set_title("(b)  Anatomy prior vs. matched random box (n=146)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out = RES / "cam_real_ablation_fig.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
