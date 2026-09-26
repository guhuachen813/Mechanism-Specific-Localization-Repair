r"""第四阶段 2：先验失败风险建模（门控输入特征基础）。

数据：cam_prior_nih.csv（NIH 146 图 × 17 条件 × 7 变体，含 cov_oracle/cov_real/
field_ok_real）与 cam_real_photo.csv（CheXphoto 202 研究 × 4 条件 × 3 变体，
含 field_ok/frac/components）。

产出：
  results/prior_risk_summary.csv   —— 每条件风险表（两层分开）
  results/prior_risk_fig.png       —— 两面板图（coverage 退化曲线 + rescue-cov 关系）

作者：ClawsGO Science Agent
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RNG = np.random.default_rng(42)

LADDER = ["clean", "jpeg_q50", "jpeg_q30", "jpeg_q10", "blur_1", "blur_3",
          "dark_05", "bright_16", "contrast_04", "ds_2", "ds_4",
          "noise_003", "noise_006", "noise_010", "noise_025",
          "combo_mild", "combo_sev"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#dddddd",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9,
})


def boot_ci(a, n=400):
    a = np.asarray(a, float)
    bs = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return np.quantile(bs, [0.025, 0.975])


def paired_rescue(prior: pd.DataFrame, raw: pd.DataFrame):
    """按图配对 rescue = prior best_iou - raw best_iou，返回 (per-image, mean, ci)."""
    m = prior[["nih_file", "best_iou"]].rename(columns={"best_iou": "p"})
    r = raw[["nih_file", "best_iou"]].rename(columns={"best_iou": "r"})
    j = m.merge(r, on="nih_file")
    d = (j.p - j.r).values
    lo, hi = boot_ci(d)
    return d, d.mean(), (lo, hi)


def main() -> int:
    cp = pd.read_csv(RES / "cam_prior_nih.csv")
    pr = cp[cp.variant == "prior_real_hard"]
    raw = cp[cp.variant == "raw"]

    rows = []
    for cond in LADDER:
        p, r = pr[pr.condition == cond], raw[raw.condition == cond]
        d, resc, (lo, hi) = paired_rescue(p, r)
        cov = p.cov_real.values
        rows.append({
            "condition": cond,
            "field_ok": p.field_ok_real.mean(),
            "cov_mean": cov.mean(), "cov_median": np.median(cov),
            "cov_std": cov.std(), "cov_lt05": (cov < 0.5).mean(),
            "cov_gt09": (cov > 0.9).mean(),
            "rescue": resc, "rescue_lo": lo, "rescue_hi": hi,
            "rescue_star": "*" if (lo > 0 or hi < 0) else "",
        })
    s = pd.DataFrame(rows)
    s.to_csv(RES / "prior_risk_summary.csv", index=False)
    print("== NIH 每条件先验风险表（层1 field_ok / 层2 mask 覆盖率 / rescue）==")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # rescue vs cov_real 分箱（跨条件汇总，hard 变体）
    pj = pr[pr.condition != "clean"][["nih_file", "condition", "best_iou", "cov_real"]]
    rj = raw[raw.condition != "clean"][["nih_file", "condition", "best_iou"]]
    j = pj.merge(rj, on=["nih_file", "condition"], suffixes=("_p", "_r"))
    j["rescue"] = j.best_iou_p - j.best_iou_r
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    j["covbin"] = pd.cut(j.cov_real, bins, include_lowest=True)
    binstat = j.groupby("covbin", observed=True).agg(
        n=("rescue", "size"), rescue=("rescue", "mean")).reset_index()
    print("\n== rescue 按 per-image cov_real 分箱（clean 外全部条件，hard）==")
    print(binstat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # CheXphoto：真实翻拍层的 mask 统计
    cr = pd.read_csv(RES / "cam_real_photo.csv")
    praw = cr[cr.variant == "prior_real_hard"]
    rows2 = []
    for cond in ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]:
        d = praw[praw.condition == cond]
        rr = cr[(cr.variant == "raw") & (cr.condition == cond)]
        rj = rr.set_index("base_key").add_suffix("_r")[["agree_iou_r"]]
        m = d.set_index("base_key").join(rj)
        agree = (m.agree_iou - m.agree_iou_r).dropna()
        rows2.append({
            "condition": cond, "field_ok": d.field_ok.mean(),
            "frac_mean": d.frac.mean(), "components_mean": d.components.mean(),
            "prior_agree": d.agree_iou.mean(), "raw_agree": rr.agree_iou.mean(),
            "agree_gain": agree.mean(),
        })
    s2 = pd.DataFrame(rows2)
    print("\n== CheXphoto 每条件先验统计（真实翻拍层）==")
    print(s2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    s2.to_csv(RES / "prior_risk_chexphoto.csv", index=False)

    # ------------------------------------------------------------- 主图 --
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax = axes[0]
    x = np.arange(len(LADDER))
    ax.plot(x, s.cov_mean, "-o", color="#0072B2", lw=1.4, ms=3.5,
            label="cov_real (mean)")
    ax.plot(x, s.cov_median, "-s", color="#56B4E9", lw=1.2, ms=3.2,
            label="cov_real (median)")
    ax.plot(x, s.cov_lt05 * s.cov_mean.max(), "-^", color="#D55E00", lw=1.2,
            ms=3.2, label="P(cov<0.5) [scaled]")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([LADDER[i] for i in range(0, len(LADDER), 2)],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("realistic mask GT coverage")
    ax.set_title("(a) mask-layer degradation across ladder", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    labels = [str(b) for b in binstat.covbin]
    cols = ["#D55E00" if v < 0 else "#009E73" for v in binstat.rescue]
    ax.bar(range(len(binstat)), binstat.rescue, color=cols, width=0.6, zorder=3)
    vmin, vmax = binstat.rescue.min(), binstat.rescue.max()
    for i, (v, n) in enumerate(zip(binstat.rescue, binstat.n)):
        ax.text(i, v + (0.006 if v >= 0 else -0.006), f"{v:.3f}\n(n={n})",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.2)
    ax.set_ylim(vmin - 0.06, vmax + 0.035)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(len(binstat)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("per-image cov_real bin")
    ax.set_ylabel("paired rescue (best-IoU)")
    ax.set_title("(b) rescue vs mask quality (hard variant, 16 shifted conds)",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    out = RES / "prior_risk_fig.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
