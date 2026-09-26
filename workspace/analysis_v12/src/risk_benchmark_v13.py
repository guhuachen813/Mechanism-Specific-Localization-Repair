"""
risk_benchmark_v13.py — 修复后（v1.3）风险迁移的定量分解

输入（用户 2026-09-12 报告）
---------------------------
    T          : seed42 0.4943 / seed7 0.4427
    ECE route  : 0.0201 / 0.0335      ECE official : 0.1809 / 0.1838
    AUROC route: 0.8440 / 0.8440      AUROC official: 0.8289 / 0.8307
    75% 覆盖率工作点：cutoff 0.8338 / 0.8408
    route selective risk 6.09% / 6.55%    official coverage 80.69% / 81.19%
    official selective risk 22.09% / 20.73%
    DenseNet 全覆盖错误率：route 12.89% / 13.75%；official 26.24% / 26.24%

目的
----
把 "route 6.3% → official 21.4%" 这 15.1 个百分点的差距分解为：
    (1) 患病率效应   —— 纯粹由 π 12.9% → 32.7% 造成（机械、可解释）
    (2) 覆盖率失配   —— route@75% vs official@81% 的 Apples-to-oranges
    (3) 真实超额     —— 扣除前两项后剩余的、需要归因的部分

作者：ClawsGO Science Agent
"""
from __future__ import annotations
import os
import sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from rc_lib import setup_matplotlib, OKABE_ITO  # noqa: E402


def d_from_auroc(auc):
    return float(np.sqrt(2.0) * norm.ppf(auc))


def scores(prevalence, auroc, n=600_000, seed=0):
    rng = np.random.default_rng(seed)
    d = d_from_auroc(auroc)
    y = (rng.random(n) < prevalence).astype(int)
    x = rng.normal(d * y, 1.0)
    return y, x, d


def calibrated_risk_curve(prevalence, auroc, seed=0):
    """
    完美校准 + 达到给定 AUROC 的模型，其 risk-coverage 曲线。
    最优决策规则：p(y=1|x) > 0.5 ⟺ x > 阈值；按 |logit p| 排序取最自信的。
    """
    y, x, d = scores(prevalence, auroc, seed=seed)
    p = 1.0 / (1.0 + np.exp(-(d * x - 0.5 * d * d + np.log(prevalence / (1 - prevalence)))))
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf)
    y_ord = y[order]
    n = len(y_ord)
    ks = np.arange(1, n + 1)
    cum_err = np.cumsum((y_ord == 0).astype(float))
    # 注意：选择性风险定义为"被接受样本中预测错误的比例"
    pred = (p[order] >= 0.5).astype(int)
    cum_pred_err = np.cumsum(pred != y_ord)
    cov = ks / n
    risk = cum_pred_err / ks
    return cov, risk


def risk_at(cov, risk, target):
    i = np.searchsorted(cov, target)
    return float(risk[min(i, len(risk) - 1)])


def majority_risk(prevalence):
    return prevalence            # 全判阴性 → 错误率 = 阳性率


def bayes_risk(prevalence, auroc, seed=0):
    y, x, d = scores(prevalence, auroc, seed=seed)
    p = 1.0 / (1.0 + np.exp(-(d * x - 0.5 * d * d + np.log(prevalence / (1 - prevalence)))))
    return float(((p >= 0.5).astype(int) != y).mean())


CASES = [
    dict(tag="route-validation (seed42)", pi=0.1230, auc=0.8440),
    dict(tag="route-validation (seed7)",  pi=0.1289, auc=0.8440),
    dict(tag="official-valid (seed42)",   pi=0.3270, auc=0.8289),
    dict(tag="official-valid (seed7)",    pi=0.3270, auc=0.8307),
]

if __name__ == "__main__":
    print("=" * 82)
    print("完美校准模型的理论风险基准（v1.3 修复后 AUROC）")
    print("=" * 82)
    res = {}
    for c in CASES:
        cov, risk = calibrated_risk_curve(c["pi"], c["auc"])
        mb = majority_risk(c["pi"])
        br = bayes_risk(c["pi"], c["auc"])
        r75 = risk_at(cov, risk, 0.75)
        r80 = risk_at(cov, risk, 0.80)
        r81 = risk_at(cov, risk, 0.81)
        res[c["tag"]] = dict(pi=c["pi"], auc=c["auc"], maj=mb, bayes=br,
                             r75=r75, r80=r80, r81=r81)
        print(f"\n【{c['tag']}】π={c['pi']:.3f}  AUROC={c['auc']:.4f}")
        print(f"  多数类基线（全判阴性）错误率 : {mb:.4f}")
        print(f"  Bayes 最优错误率（全覆盖）   : {br:.4f}")
        print(f"  完美校准选择性风险 @75%      : {r75:.4f}")
        print(f"  完美校准选择性风险 @80%      : {r80:.4f}")
        print(f"  完美校准选择性风险 @81%      : {r81:.4f}")

    # ---------------- 观测值 vs 基准 ----------------
    obs_route_risk, obs_route_cov = 0.0632, 0.75
    obs_off_risk, obs_off_cov = 0.2141, 0.8094

    print("\n" + "=" * 82)
    print("观测 vs 基准：15.1 个百分点落差的分解")
    print("=" * 82)
    br75 = res["route-validation (seed42)"]["r75"]
    bo75 = res["official-valid (seed42)"]["r75"]
    bo81 = res["official-valid (seed42)"]["r81"]

    print(f"  观测 route    @75.0% : {obs_route_risk:.4f}")
    print(f"  观测 official @81.0% : {obs_off_risk:.4f}   （注意覆盖率不匹配）")
    print(f"  观测落差             : {obs_off_risk - obs_route_risk:+.4f}")

    print(f"\n  分解（逐步扣除）：")
    step1 = br75
    print(f"  (0) route 观测                                   : {obs_route_risk:.4f}")
    print(f"  (1) → 换到 route 的完美校准基准（消除模型自身不足）: {br75:.4f}"
          f"   [{br75 - obs_route_risk:+.4f}]")
    print(f"  (2) → 换到 official 的完美校准基准@75%（纯患病率效应）: {bo75:.4f}"
          f"   [{bo75 - br75:+.4f}]  ← 患病率效应")
    print(f"  (3) → 换到 81% 覆盖率（覆盖率失配效应）           : {bo81:.4f}"
          f"   [{bo81 - bo75:+.4f}]  ← 覆盖率失配")
    print(f"  (4) → official 观测                              : {obs_off_risk:.4f}"
          f"   [{obs_off_risk - bo81:+.4f}]  ← 真实超额（需归因）")

    tot = (br75 - obs_route_risk) + (bo75 - br75) + (bo81 - bo75) + (obs_off_risk - bo81)
    print(f"\n  四项合计 = {tot:+.4f}（应等于观测落差 {obs_off_risk - obs_route_risk:+.4f}）")
    prev_eff = bo75 - br75
    cov_eff = bo81 - bo75
    excess = obs_off_risk - bo81
    denom = obs_off_risk - obs_route_risk
    print(f"\n  占比：患病率 {100*prev_eff/denom:.0f}%   覆盖率失配 {100*cov_eff/denom:.0f}%   "
          f"真实超额 {100*excess/denom:.0f}%")

    # ---------------- 相对风险下降 ----------------
    print("\n" + "=" * 82)
    print("跨患病率可比的量：相对风险下降（vs 多数类基线）")
    print("=" * 82)
    for tag, obs_r, pi in [("route @75%", obs_route_risk, 0.1230),
                           ("official @81%", obs_off_risk, 0.3270)]:
        base = majority_risk(pi)
        print(f"  {tag:16s}  基线 {base:.4f} → 观测 {obs_r:.4f}   "
              f"相对下降 {100*(1-obs_r/base):.1f}%")

    # ---------------- 图 ----------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    ax = axes[0]
    for c, col, lab in zip(CASES,
                           [OKABE_ITO["blue"], OKABE_ITO["sky"],
                            OKABE_ITO["vermillion"], OKABE_ITO["orange"]],
                           [c["tag"] for c in CASES]):
        cov, risk = calibrated_risk_curve(c["pi"], c["auc"])
        m = cov <= 0.95
        ax.plot(cov[m], risk[m], color=col, lw=1.8, label=lab)
    ax.scatter([0.75], [obs_route_risk], color=OKABE_ITO["black"], zorder=5, s=42,
               marker="o", label="route 观测 @75%")
    ax.scatter([obs_off_cov], [obs_off_risk], color=OKABE_ITO["black"], zorder=5, s=52,
               marker="D", label="official 观测 @81%")
    ax.axvline(0.75, color="grey", ls=":", lw=1)
    ax.axvline(obs_off_cov, color="grey", ls=":", lw=1)
    ax.set_xlabel("覆盖率"); ax.set_ylabel("选择性风险")
    ax.set_title("完美校准基准 vs 观测工作点", fontsize=11)
    ax.legend(frameon=False, fontsize=7.6)
    ax.grid(alpha=0.22, lw=0.6)

    ax = axes[1]
    comps = [("route 观测", obs_route_risk, OKABE_ITO["grey"]),
             ("模型自身不足", br75 - obs_route_risk, OKABE_ITO["sky"]),
             ("患病率效应", prev_eff, OKABE_ITO["blue"]),
             ("覆盖率失配", cov_eff, OKABE_ITO["orange"]),
             ("真实超额", excess, OKABE_ITO["vermillion"])]
    bottom = 0.0
    for lab, v, col in comps:
        ax.bar(0, v, bottom=bottom, color=col, width=0.5,
               label=f"{lab} ({v*100:+.1f}pp)")
        bottom += v
    ax.bar(0, obs_route_risk, color=comps[0][2], width=0.5)
    ax.set_xticks([]); ax.set_ylabel("选择性风险")
    ax.set_title("15.1pp 落差的来源分解", fontsize=11)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        axes[0].spines[s].set_visible(False)
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "out")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_risk_decomposition_v13.png"), dpi=200)
    print("\n图已保存：out/fig_risk_decomposition_v13.png")
