"""聚合 cam_prior_nih.csv：先验约束 vs raw CAM 的 clean 代价与退化收益。

口径
----
* clean 代价：clean 条件下 prior_* − raw（best-Iou 差，负 = 约束伤了干净定位）。
* 退化收益：各条件下 prior_* − raw 的配对差（正 = 约束救回了定位），
  按图整群 Bootstrap 400 次给 95% CI。
* 两层先验鲁棒性：oracle（先验来自 clean 图）vs realistic（先验来自降质图）；
  field_ok_real = realistic 口径下场检测未触发全图回退的比例。
* 固定阈值口径（τ = raw clean 的 argmax τ）作交叉核对。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
RNG = np.random.default_rng(42)
NBOOT = 400
T_FIXED = 0.45  # 第2步 raw clean 的 MaxBoxAcc 最优阈值


def main() -> int:
    df = pd.read_csv(HERE / "results" / "cam_prior_nih.csv")
    df["iou_vec"] = df["iou_curve"].map(lambda s: np.fromstring(s, sep=";"))
    files = sorted(df["nih_file"].unique())
    conds = [c for c in df["condition"].unique()]
    variants = ["raw", "prior_oracle_hard", "prior_oracle_soft", "prior_oracle_dil",
                "prior_real_hard", "prior_real_soft", "prior_real_dil"]

    # ---- 表 1：各变体 × 条件 best-IoU 均值 + field 检测成功率 ----
    rows = []
    for c in conds:
        d = df[df["condition"] == c]
        rec = {"condition": c,
               "field_ok": d[d["variant"] == "raw"]["field_ok_real"].mean()}
        for v in variants:
            rec[v] = d[d["variant"] == v]["best_iou"].mean()
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(HERE / "results" / "cam_prior_summary.csv", index=False)
    show = tab.copy()
    for v in variants:
        show[v] = (show[v] - show["raw"]).round(3) if v != "raw" else show[v].round(3)
    print("best-IoU（raw 列）与相对 raw 的增量（其余列，×100 省略即 IoU 差）：")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 表 2：关键对比的配对 Bootstrap ----
    wide = {v: df[df["variant"] == v].set_index(["nih_file", "condition"])["best_iou"]
            for v in variants}
    vecs = {v: df[df["variant"] == v].set_index(["nih_file", "condition"])["iou_vec"]
            for v in variants}
    n = len(files)
    print("\n配对差（95% CI，整群 Bootstrap 400 次）：")
    print(f"{'对比':34s} {'clean':>18s} {'contrast_04':>18s} "
          f"{'noise_006':>18s} {'combo_sev':>18s}")
    comps = [("prior_oracle_hard", "raw"), ("prior_oracle_dil", "raw"),
             ("prior_real_hard", "raw"), ("prior_real_dil", "raw"),
             ("prior_real_hard", "prior_oracle_hard"),
             ("prior_real_dil", "prior_oracle_dil")]
    out = []
    for v1, v0 in comps:
        cells = []
        for c in ["clean", "contrast_04", "noise_006", "combo_sev"]:
            a = np.array([wide[v1].loc[(f, c)] for f in files])
            b = np.array([wide[v0].loc[(f, c)] for f in files])
            diff = a - b
            bs = [diff[RNG.integers(0, n, n)].mean() for _ in range(NBOOT)]
            lo, hi = np.quantile(bs, [0.025, 0.975])
            star = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{diff.mean():+.3f}{star}[{lo:+.3f},{hi:+.3f}]")
            out.append({"variant": v1, "ref": v0, "condition": c,
                        "diff": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                        "sig": star})
        print(f"{v1+' − '+v0:34s} " + " ".join(f"{x:>18s}" for x in cells))
    pd.DataFrame(out).to_csv(HERE / "results" / "cam_prior_paired.csv", index=False)

    # ---- 固定阈值口径交叉核对（contrast_04）----
    print(f"\n固定 τ={T_FIXED} 口径（contrast_04，IoU 均值）：")
    for v in variants:
        m = df[(df["variant"] == v) & (df["condition"] == "contrast_04")]
        vals = [x[list(THRESHOLDS).index(T_FIXED)] for x in m["iou_vec"]]
        print(f"  {v:22s} {np.mean(vals):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
