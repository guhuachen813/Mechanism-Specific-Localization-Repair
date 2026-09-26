r"""思路A 第5步主图：Atelectasis 定位地板归因（2×2）。

两面板（Atelectasis / Cardiomegaly 对照），各三柱：
DenseNet CAM 读出（分类驱动）· DenseNet 特征+框监督线性头 · MedSAM 特征+框监督线性头。
探针为 5 折 CV held-out；误差棒 = 按图整群 Bootstrap 95% CI。

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
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
})


def ci(a):
    bs = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(400)]
    return np.quantile(bs, [0.025, 0.975])


def main() -> int:
    probe = pd.read_csv(RES / "probe_all.csv")
    raw_c = pd.read_csv(RES / "cam_raw_nih.csv")
    raw_a = pd.read_csv(RES / "cam_atelectasis_nih.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, finding, rawdf in [
            (axes[0], "Atelectasis", raw_a), (axes[1], "Cardiomegaly", raw_c)]:
        vals, los, his, cols, labs = [], [], [], [], []
        # CAM（分类驱动读出，clean）
        m_raw = (rawdf.condition == "clean")
        if "variant" in rawdf.columns:      # 第4c步 CSV 有 variant 列
            m_raw &= rawdf.variant == "raw"
        rw = rawdf[m_raw]
        files = sorted(rw["nih_file"].unique())
        a = rw.set_index("nih_file").loc[files]["best_iou"].values
        lo, hi = ci(a)
        vals.append(a.mean()); los.append(lo); his.append(hi)
        cols.append("#888888"); labs.append("DenseNet CAM\n(class-driven)")
        for bb, col, lab in [("densenet", "#0072B2", "DenseNet feat.\n+ box-supervised head"),
                             ("medsam", "#009E73", "MedSAM feat.\n+ box-supervised head")]:
            d = probe[(probe.finding == finding) & (probe.backbone == bb)]
            files_p = sorted(d["nih_file"].unique())
            a = d.set_index("nih_file").loc[files_p]["best_iou"].values
            lo, hi = ci(a)
            vals.append(a.mean()); los.append(lo); his.append(hi)
            cols.append(col); labs.append(lab)
        x = np.arange(3)
        ax.bar(x, vals, color=cols, width=0.62, zorder=3)
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los),
                                   np.array(his) - np.array(vals)],
                    fmt="none", ecolor="#333333", capsize=3, lw=1.1, zorder=4)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=7.6)
        ax.set_ylim(0, max(his) * 1.22)
        ax.set_title(f"{finding} (box-GT best-IoU)", loc="left", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("best-IoU (held-out / clean)")
    fig.tight_layout()
    out = RES / "probe_attribution_fig.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
