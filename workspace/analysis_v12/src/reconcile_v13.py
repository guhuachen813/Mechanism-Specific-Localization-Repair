"""
reconcile_v13.py — 收口计算

1. 高精度 AUROC，核对两个 seed 的 route AUROC 是否真的相同
2. 用「同覆盖率」重算风险落差的来源分解
3. 更新分解图

作者：ClawsGO Science Agent
"""
from __future__ import annotations
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from rc_lib import setup_matplotlib, OKABE_ITO
from analyze_v13_full import binary, risk_at_coverage, ece, logit, sigm
from risk_benchmark_v13 import calibrated_risk_curve, risk_at, bayes_risk, majority_risk

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    print("=" * 88)
    print("1. 高精度 AUROC —— 核对两个 seed 的 route 值是否真的相同")
    print("=" * 88)
    for seed in (42, 7):
        for split, tag in (("route_validation_predictions", "route"),
                           ("official_valid_predictions", "official")):
            _, y, p, _ = binary(seed, split)
            print(f"  seed{seed:<3} {tag:<9} n={len(y):>6}  "
                  f"AUROC(raw)={roc_auc_score(y, p):.10f}")
    print("\n  两个 seed 的 route 预测来自不同划分（17816 vs 18388 行），")
    print("  由独立文件分别计算，非复制粘贴。")

    print("\n" + "=" * 88)
    print("2. 同覆盖率口径下的风险落差分解")
    print("=" * 88)
    rows = []
    for seed in (42, 7):
        _, yr, pr, _ = binary(seed, "route_validation_predictions")
        _, yo, po, _ = binary(seed, "official_valid_predictions")
        _, risk_r = risk_at_coverage(yr, pr, 0.75)
        _, risk_o = risk_at_coverage(yo, po, 0.75)
        auc_r, auc_o = roc_auc_score(yr, pr), roc_auc_score(yo, po)
        rows.append(dict(seed=seed, pi_r=yr.mean(), pi_o=yo.mean(),
                         auc_r=auc_r, auc_o=auc_o,
                         obs_r=risk_r, obs_o=risk_o))

    print(f"{'seed':<6}{'π_route':>9}{'π_off':>8}{'AUROC_r':>9}{'AUROC_o':>9}"
          f"{'route@75':>10}{'off@75':>9}")
    print("-" * 70)
    for r in rows:
        print(f"{r['seed']:<6}{r['pi_r']:>9.4f}{r['pi_o']:>8.4f}{r['auc_r']:>9.4f}"
              f"{r['auc_o']:>9.4f}{r['obs_r']:>10.4f}{r['obs_o']:>9.4f}")

    print("\n逐 seed 分解（全部在 75% 覆盖率下，同口径）：")
    comp_all = []
    for r in rows:
        cov_r, risk_r_cal = calibrated_risk_curve(r["pi_r"], r["auc_r"])
        cov_o, risk_o_cal = calibrated_risk_curve(r["pi_o"], r["auc_o"])
        b_r = risk_at(cov_r, risk_r_cal, 0.75)
        b_o = risk_at(cov_o, risk_o_cal, 0.75)
        gap = r["obs_o"] - r["obs_r"]
        model_short = b_r - r["obs_r"]           # 负 = 模型优于基准
        prev_eff = b_o - b_r
        excess = r["obs_o"] - b_o
        comp_all.append((model_short, prev_eff, excess))
        print(f"\n  seed{r['seed']}: route 观测 {r['obs_r']:.4f}  →  official 观测 {r['obs_o']:.4f}"
              f"   落差 {gap:+.4f}")
        print(f"    模型自身（route 观测 vs 基准 {b_r:.4f}）  : {model_short:+.4f}")
        print(f"    患病率效应（{r['pi_r']*100:.1f}%→{r['pi_o']*100:.1f}%，"
              f"基准 {b_r:.4f}→{b_o:.4f}）: {prev_eff:+.4f}")
        print(f"    真实超额（official 观测 vs 基准 {b_o:.4f}）: {excess:+.4f}")
        print(f"    合计 {model_short+prev_eff+excess:+.4f}")

    m = np.mean([c[0] for c in comp_all])
    p = np.mean([c[1] for c in comp_all])
    e = np.mean([c[2] for c in comp_all])
    gap_mean = np.mean([r["obs_o"] - r["obs_r"] for r in rows])
    print(f"\n  两 seed 平均：落差 {gap_mean:+.4f}")
    print(f"    模型自身 {m:+.4f} ({100*m/gap_mean:+.0f}%)   "
          f"患病率 {p:+.4f} ({100*p/gap_mean:+.0f}%)   "
          f"真实超额 {e:+.4f} ({100*e/gap_mean:+.0f}%)")

    print("\n" + "=" * 88)
    print("3. 校准落差在「同分箱」下的真实大小")
    print("=" * 88)
    print(f"{'seed':<6}{'route ECE15':>13}{'official ECE15':>16}{'落差':>9}"
          f"{'先验修正后':>12}{'最优偏移后':>12}")
    print("-" * 72)
    for seed in (42, 7):
        _, yr, pr, _ = binary(seed, "route_validation_predictions")
        _, yo, po, _ = binary(seed, "official_valid_predictions")
        e_r = ece(yr, pr, 15)
        e_o = ece(yo, po, 15)
        shift = logit(yo.mean()) - logit(yr.mean())
        e_shift = ece(yo, sigm(logit(po) + shift), 15)
        best = min(ece(yo, sigm(logit(po) + a), 15) for a in np.linspace(-3, 3, 601))
        print(f"{seed:<6}{e_r:>13.4f}{e_o:>16.4f}{e_o-e_r:>9.4f}"
              f"{e_shift:>12.4f}{best:>12.4f}")
    print("\n  注：『先验修正后』是只用两个集合的患病率算出的理论偏移；")
    print("      『最优偏移后』是用 official 标签事后拟合的上界，真实部署不可用。")

    # ---------------- 图 ----------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    ax = axes[0]
    r0 = rows[0]
    cov_r, rk_r = calibrated_risk_curve(r0["pi_r"], r0["auc_r"])
    cov_o, rk_o = calibrated_risk_curve(r0["pi_o"], r0["auc_o"])
    mr = cov_r <= 0.95
    mo = cov_o <= 0.95
    ax.plot(cov_r[mr], rk_r[mr], color=OKABE_ITO["blue"], lw=2,
            label=f"route 完美校准基准 (π={r0['pi_r']*100:.1f}%)")
    ax.plot(cov_o[mo], rk_o[mo], color=OKABE_ITO["vermillion"], lw=2,
            label=f"official 完美校准基准 (π={r0['pi_o']*100:.1f}%)")
    ax.scatter([0.75], [r0["obs_r"]], s=60, color=OKABE_ITO["blue"],
               edgecolor="k", zorder=5)
    ax.scatter([0.75], [r0["obs_o"]], s=60, color=OKABE_ITO["vermillion"],
               edgecolor="k", zorder=5)
    ax.annotate(f"route 观测 {r0['obs_r']*100:.1f}%", (0.75, r0["obs_r"]),
                textcoords="offset points", xytext=(10, -14), fontsize=9)
    ax.annotate(f"official 观测 {r0['obs_o']*100:.1f}%", (0.75, r0["obs_o"]),
                textcoords="offset points", xytext=(10, 4), fontsize=9)
    ax.axvline(0.75, color="grey", ls=":", lw=1)
    ax.set_xlabel("覆盖率"); ax.set_ylabel("选择性风险")
    ax.set_title("同覆盖率（75%）下的两个工作点", fontsize=11)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    ax.grid(alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    obs_r_bar = float(np.mean([r["obs_r"] for r in rows]))
    comps = [("route 观测", obs_r_bar, OKABE_ITO["grey"]),
             ("模型自身不足", m, OKABE_ITO["sky"]),
             ("患病率效应", p, OKABE_ITO["blue"]),
             ("真实超额", e, OKABE_ITO["vermillion"])]
    bottom = 0.0
    for lab, v, col in comps:
        ax.bar(0, v, bottom=bottom, color=col, width=0.5,
               label=f"{lab} ({v*100:+.1f}pp)")
        bottom += v
    ax.bar(0, obs_r_bar, color=comps[0][2], width=0.5)
    ax.set_xticks([]); ax.set_ylabel("选择性风险 @75% 覆盖率")
    ax.set_title(f"落差 {gap_mean*100:.1f}pp 的来源（同口径，2 seed 平均）", fontsize=11)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_risk_decomposition_v13_matched.png"), dpi=200)
    print("\n图已保存：out/fig_risk_decomposition_v13_matched.png")
