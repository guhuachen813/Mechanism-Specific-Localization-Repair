"""
diag_prereq_shift.py — 校准不迁移的机制诊断：先验漂移 vs 划分缺陷

背景
----
修复划分后，用户报告：
    T(seed42) = 0.4943,  T(seed7) = 0.4427        （修复前 0.638）
    route-validation  ECE = 0.0201 / 0.0335       （修复前 0.0041）
    official-valid    ECE = 0.1809 / 0.1838       （修复前 0.1639）
    AUROC route 0.8440；official 0.8289 / 0.8307

问题：official-valid 上 0.18 的 ECE，能否**单独由先验漂移**解释？
若能，则"校准不迁移"是一个可解释、可修正的机制问题，而不是模型退化。

方法
----
在双正态潜变量模型下构造一个**在源域上完美校准**的二分类器，
把它原样应用到目标患病率的测试集上，测量 ECE。
对照三种修正：
    (a) 无修正
    (b) 仅温度缩放（在源域上拟合 T——在理想模型上 T=1，故等价于无修正）
    (c) 先验偏移修正（logit 上加 logit(π_t) − logit(π_s)）
    (d) 先验修正 + 温度缩放

作者：ClawsGO Science Agent
"""
from __future__ import annotations
import os
import sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from rc_lib import ece, brier_decomposition, setup_matplotlib  # noqa: E402

RNG = np.random.default_rng(20260912)
N = 400_000


def d_from_auroc(auc: float) -> float:
    """双正态等方差模型下，AUROC = Φ(d/√2)。"""
    return float(np.sqrt(2.0) * norm.ppf(auc))


def make_scores(prevalence: float, auroc: float, n: int = N, rng=RNG):
    """返回 (y, x)：y∈{0,1} 真实标签，x 为潜变量分数（越大越像阳性）。"""
    d = d_from_auroc(auroc)
    y = (rng.random(n) < prevalence).astype(int)
    x = rng.normal(loc=d * y, scale=1.0)
    return y, x, d


def posterior(x, prevalence: float, d: float) -> np.ndarray:
    """双正态模型下的真后验 P(y=1|x)。"""
    log_lr = d * x - 0.5 * d * d          # log f1(x) - log f0(x)
    return 1.0 / (1.0 + np.exp(-(log_lr + np.log(prevalence / (1 - prevalence)))))


def bin_ece(y, p_pos, n_bins: int = 15) -> float:
    """
    严格二分类 ECE：置信度 = max(p_pos, 1-p_pos)，正确性 = argmax 是否命中。
    与用户报告口径一致。
    """
    conf = np.maximum(p_pos, 1.0 - p_pos)
    pred = (p_pos >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(y)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def analyse(tag: str, pi_s: float, pi_t: float, auroc_s: float, auroc_t: float,
            obs_ece_t: float | None = None):
    print("=" * 74)
    print(f"【{tag}】源域 π_s={pi_s:.3f}  AUROC_s={auroc_s:.4f}"
          f"   →   目标域 π_t={pi_t:.3f}  AUROC_t={auroc_t:.4f}")

    # --- 源域：拟合/校验 ---
    y_s, x_s, d_s = make_scores(pi_s, auroc_s)
    p_s = posterior(x_s, pi_s, d_s)                 # 在源域上完美校准
    ece_s = bin_ece(y_s, p_s)

    # --- 目标域：用源域模型原样打分 ---
    y_t, x_t, d_t = make_scores(pi_t, auroc_t)
    p_t_raw = posterior(x_t, pi_s, d_t)             # 仍用源域先验
    ece_t_raw = bin_ece(y_t, p_t_raw)

    # --- 修正 c：先验偏移 ---
    shift = np.log(pi_t / (1 - pi_t)) - np.log(pi_s / (1 - pi_s))
    def logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
    p_t_prior = 1.0 / (1.0 + np.exp(-(logit(p_t_raw) + shift)))
    ece_t_prior = bin_ece(y_t, p_t_prior)

    # --- 修正 d：先验偏移 + 温度（温度在目标域上最优拟合） ---
    best = (1e9, 1.0)
    for T in np.linspace(0.3, 3.0, 271):
        p = 1.0 / (1.0 + np.exp(-(logit(p_t_raw) + shift) / T))
        v = bin_ece(y_t, p)
        if v < best[0]:
            best = (v, T)
    ece_t_both, T_opt = best

    print(f"  源域 ECE（完美校准，应≈0）      : {ece_s:.4f}")
    print(f"  目标域 ECE，无修正              : {ece_t_raw:.4f}")
    print(f"  目标域 ECE，仅先验偏移修正      : {ece_t_prior:.4f}")
    print(f"  目标域 ECE，先验+温度修正       : {ece_t_both:.4f}  (最优 T={T_opt:.2f})")
    if obs_ece_t is not None:
        print(f"  —— 实测目标域 ECE               : {obs_ece_t:.4f}"
              f"   →  先验漂移解释比例 ≈ "
              f"{100 * min(1.0, ece_t_raw / obs_ece_t):.0f}%")
    return dict(ece_t_raw=ece_t_raw, ece_t_prior=ece_t_prior,
                ece_t_both=ece_t_both, ece_s=ece_s)


if __name__ == "__main__":
    print(f"模拟样本量 N = {N:,}\n")
    r1 = analyse("路线 A：修复前的报告口径", 0.031, 0.327, 0.8091, 0.8500, obs_ece_t=0.1639)
    print()
    r2 = analyse("路线 B：修复后（seed42）", 0.1230, 0.327, 0.8440, 0.8289, obs_ece_t=0.1809)
    print()
    r3 = analyse("路线 C：修复后（seed7）", 0.1289, 0.327, 0.8440, 0.8307, obs_ece_t=0.1838)
