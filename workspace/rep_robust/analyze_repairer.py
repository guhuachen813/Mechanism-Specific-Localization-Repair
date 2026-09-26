r"""第四阶段 4 分析：修复器逐条件分解 + Bootstrap CI + 三假设检验。

输入：results/repairer_eval.csv（repairer.py --stage train 折外输出）
输出：控制台表 + results/repairer_by_cond.csv + results/repairer_fig.png
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


def ci(a, n=400):
    a = np.asarray(a, float)
    bs = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return np.quantile(bs, [0.025, 0.975])


def main() -> int:
    df = pd.read_csv(RES / "repairer_eval.csv")
    rows = []
    for cond in LADDER:
        d = df[df.condition == cond]
        r = {"condition": cond, "n": len(d),
             "gate_rate": d.repaired.mean()}
        for m in ["45", "best"]:
            for col in ["raw", "prior", "center", "cand", "gated"]:
                r[f"{col}_{m}"] = d[f"{col}_{m}"].mean()
        # 配对差：cand vs raw、gated vs raw（IoU45）
        for name, col in [("cand", "cand_45"), ("gated", "gated_45"),
                          ("prior", "prior_45")]:
            dd = (d[col] - d.raw_45).values
            lo, hi = ci(dd)
            r[f"d_{name}"] = dd.mean()
            r[f"d_{name}_lo"], r[f"d_{name}_hi"] = lo, hi
            r[f"d_{name}_star"] = "*" if (lo > 0 or hi < 0) else ""
        # 安全性
        hurt = (d.cand_45 < d.raw_45 - 0.1).mean()
        r["hurt_rate"] = hurt
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(RES / "repairer_by_cond.csv", index=False)
    show = ["condition", "n", "gate_rate", "raw_45", "prior_45", "cand_45",
            "gated_45", "d_cand", "d_cand_star", "d_gated", "d_gated_star",
            "hurt_rate"]
    print(s[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 三假设检验（IoU45 口径，配对差 vs raw）
    print("\n== 三假设（cand 配对 ΔIoU45 vs raw）==")
    for name, conds in [("H1 激活漂移可修复", ["contrast_04", "combo_sev"]),
                        ("H2 区域内重写不可修复", ["noise_006", "noise_010"]),
                        ("clean 保持", ["clean"])]:
        for c_ in conds:
            r = s[s.condition == c_].iloc[0]
            print(f"  {c_:12s} Δ={r.d_cand:+.3f}{r.d_cand_star} "
                  f"[{r.d_cand_lo:+.3f},{r.d_cand_hi:+.3f}] "
                  f"gate={r.gate_rate:.2f} hurt={r.hurt_rate:.3f}")

    # risk-coverage：按 gate 分位保留
    print("\n== risk-coverage（gated，按 gate 分数排序，越高越先保留修复）==")
    d = df[~df.condition.isin(["noise_006", "noise_010", "noise_025"])]
    order = d.sort_values("gate", ascending=False)
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = max(1, int(len(order) * frac))
        sel = order.head(k)
        print(f"  top{frac:.0%}: n={k} gated45={sel.gated_45.mean():.3f} "
              f"cand45={sel.cand_45.mean():.3f} raw45={sel.raw_45.mean():.3f}")

    # 主图
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    x = np.arange(len(LADDER))
    for col, c_, lab in [("raw_45", "#888888", "raw"),
                         ("prior_45", "#009E73", "prior hard"),
                         ("cand_45", "#D55E00", "repairer"),
                         ("gated_45", "#0072B2", "repairer+gate")]:
        ax.plot(x, s[col], "-o", color=c_, ms=3.0, lw=1.3, label=lab)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([LADDER[i] for i in range(0, len(LADDER), 2)],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("IoU at fixed τ=0.45 (held-out)")
    ax.set_title("(a) per-condition held-out IoU45", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[1]
    dd = s[s.condition != "clean"]
    ax.scatter(dd.gate_rate, dd.d_cand, c="#D55E00", s=22, label="repairer (cand)")
    ax.scatter(dd.gate_rate, dd.d_gated, c="#0072B2", s=22, label="repairer+gate")
    ax.axhline(0, color="#333", lw=0.8)
    for _, r in dd.iterrows():
        if abs(r.d_cand) > 0.1 or abs(r.d_gated) > 0.1:
            ax.annotate(r.condition, (r.gate_rate, r.d_gated), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("gate trigger rate")
    ax.set_ylabel("paired ΔIoU45 vs raw")
    ax.set_title("(b) benefit vs gate triggering (shifted conds)",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    out = RES / "repairer_fig.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
