"""聚合 cam_raw_nih.csv：MaxBoxAcc、BoxAcc@τ、配对 IoU 退化与分类-定位解耦。

指标定义
--------
* MaxBoxAcc(cond) = max_τ mean_i IoU_i(cond, τ)（IoU-τ 曲线在图像总体上的峰值）。
* BestIoU(cond)   = mean_i max_τ IoU_i(cond, τ)（每图取最优阈值的均值，上界性质）。
* BoxAcc@k        = 在 MaxBoxAcc 的最优阈值 τ* 处，IoU≥k 的图像比例。
* 配对退化        = 同一图在同一 τ 网格下 best-IoU 的配对差（条件 − clean），
  整群 Bootstrap（按图重采样 400 次）给 95% CI。
* 解耦分析        = 分类概率掉没掉 vs 定位 IoU 掉没掉的 2×2 计数
  （prob 降幅 ≥0.1、best-IoU 降幅 ≥0.1 为阈值）。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
DROP = 0.1
RNG = np.random.default_rng(42)
NBOOT = 400


def main() -> int:
    df = pd.read_csv(HERE / "results" / "cam_raw_nih.csv")
    df["iou_vec"] = df["iou_curve"].map(lambda s: np.fromstring(s, sep=";"))
    conds = df["condition"].unique()
    files = sorted(df["nih_file"].unique())
    print(f"{len(files)} 图 × {len(conds)} 条件")

    # 主表
    rows = []
    for c in conds:
        d = df[df["condition"] == c]
        M = np.stack(d["iou_vec"])                 # (N, T)
        mean_curve = M.mean(axis=0)
        t_star = int(mean_curve.argmax())
        rows.append({
            "condition": c,
            "maxboxacc": mean_curve[t_star],
            "t_star": THRESHOLDS[t_star],
            "best_iou": M.max(axis=1).mean(),
            "boxacc_010": (M[:, t_star] >= 0.10).mean(),
            "boxacc_025": (M[:, t_star] >= 0.25).mean(),
            "boxacc_050": (M[:, t_star] >= 0.50).mean(),
            "mean_prob": d["prob"].mean(),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(HERE / "results" / "cam_bench_summary.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 配对退化（相对 clean，best-IoU 与同 τ*=clean 最优的曲线值）
    clean = df[df["condition"] == "clean"].set_index("nih_file")
    prows = []
    for c in conds:
        if c == "clean":
            continue
        d = df[df["condition"] == c].set_index("nih_file").loc[files]
        cb = clean.loc[files]
        diff_best = d["best_iou"].values - cb["best_iou"].values
        # 同阈值口径：用 clean 全体最优 τ 的曲线列
        Mc = np.stack(cb["iou_vec"]); Md = np.stack(d["iou_vec"])
        ts = int(np.stack(cb["iou_vec"]).mean(axis=0).argmax())
        diff_fix = Md[:, ts] - Mc[:, ts]
        # 整群 Bootstrap（按图）
        bs_best, bs_fix = [], []
        n = len(files)
        for _ in range(NBOOT):
            idx = RNG.integers(0, n, n)
            bs_best.append(diff_best[idx].mean())
            bs_fix.append(diff_fix[idx].mean())
        lo_b, hi_b = np.quantile(bs_best, [0.025, 0.975])
        lo_f, hi_f = np.quantile(bs_fix, [0.025, 0.975])
        star_b = "*" if lo_b > 0 or hi_b < 0 else " "
        star_f = "*" if lo_f > 0 or hi_f < 0 else " "
        prows.append({"condition": c,
                      "d_best_iou": diff_best.mean(), "ci_lo": lo_b, "ci_hi": hi_b,
                      "sig": star_b,
                      "d_fixed_t": diff_fix.mean(), "ci_lo_f": lo_f, "ci_hi_f": hi_f,
                      "sig_f": star_f})
    ptab = pd.DataFrame(prows)
    ptab.to_csv(HERE / "results" / "cam_bench_paired.csv", index=False)
    print("\n", ptab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 解耦分析：分类掉 vs 定位掉
    print("\n解耦（prob 降幅≥0.1 × best-IoU 降幅≥0.1，逐图逐条件）：")
    dec = []
    for c in conds:
        if c == "clean":
            continue
        d = df[df["condition"] == c].set_index("nih_file").loc[files]
        dp = d["prob"].values - clean.loc[files]["prob"].values
        di = d["best_iou"].values - clean.loc[files]["best_iou"].values
        dec.append({"condition": c,
                    "both_drop": int(((dp <= -DROP) & (di <= -DROP)).sum()),
                    "cls_only": int(((dp <= -DROP) & (di > -DROP)).sum()),
                    "loc_only": int(((dp > -DROP) & (di <= -DROP)).sum()),
                    "neither": int(((dp > -DROP) & (di > -DROP)).sum()),
                    "cls_loc_corr": float(np.corrcoef(dp, di)[0, 1])})
    dtab = pd.DataFrame(dec)
    dtab.to_csv(HERE / "results" / "cam_bench_decouple.csv", index=False)
    print(dtab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
