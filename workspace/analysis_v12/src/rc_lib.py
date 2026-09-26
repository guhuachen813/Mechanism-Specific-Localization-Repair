"""
rc_lib.py — 选择性分类 / 风险-覆盖 / 代价敏感 / 校准 核心库

设计原则
--------
1. 只依赖预测表（每行 = 一张图），不需要图像、不需要 GPU。
2. 所有区间估计一律用 **患者级整群 Bootstrap**：图像按患者聚集，
   忽略聚类会系统性低估方差（同一个患者的多张图高度相关）。
3. 所有工作点选择都在 **覆盖率空间**，因为经验证据表明置信度的
   排序可迁移、绝对值不可迁移。
4. 代价敏感风险与错误率分开报告：两者的最优策略不同。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Sequence, Iterable

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

CJK_FONT = "Noto Sans CJK SC"

# Okabe–Ito 色盲友好调色板
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}


def setup_matplotlib():
    """必须在任何绘图之前调用：显式设置中文字体，避免静默出空框。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [CJK_FONT, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return plt


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class CostModel:
    """
    非对称代价模型。默认占位值来自项目既定方案（Cardiomegaly FN≈2×FP）。
    上线前必须由 1–2 位合作医生确认；本库把 phi 作为可扫描参数。
    """
    fn_cost: float = 2.0   # 漏诊代价（阳性判为阴性）
    fp_cost: float = 1.0   # 误报代价（阴性判为阳性）

    @property
    def ratio(self) -> float:
        return self.fn_cost / self.fp_cost


@dataclass
class PredictionTable:
    """
    规范化的预测表。列含义：
      patient_id    患者标识（Bootstrap 聚类单位）
      split         子集名（route_validation / official_valid / ...）
      label         金标准二分类标签 0/1
      p1_pos        Model 1 的 P(阳性)（已温度校准）
      p2_pos        Model 2 的 P(阳性)，可缺省
      p1_unc        U-MultiClass 的 p_uncertain，可缺省
      quality_risk  软质控风险分 [0,1]
      qc_*          其余软质控特征（可选，前缀 qc_）
    """
    df: pd.DataFrame
    qc_cols: list[str] = field(default_factory=list)

    def split(self, name: str) -> "PredictionTable":
        sub = self.df[self.df["split"] == name].reset_index(drop=True)
        return PredictionTable(sub, self.qc_cols)

    @property
    def splits(self) -> list[str]:
        return sorted(self.df["split"].unique().tolist())

    def __len__(self) -> int:
        return len(self.df)


REQUIRED_COLS = ["patient_id", "split", "label", "p1_pos"]


def load_predictions(path: str) -> PredictionTable:
    """读入并校验预测表。缺列直接报错，避免静默跑出错误结论。"""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"预测表缺少必需列 {missing}；必需列为 {REQUIRED_COLS}。"
            f"实际列：{list(df.columns)}"
        )
    if df["label"].isna().any():
        raise ValueError("label 列存在缺失值，需在导出阶段剔除未知标签样本。")
    df["label"] = df["label"].astype(int)
    if "quality_risk" not in df.columns:
        df["quality_risk"] = np.nan
    qc_cols = [c for c in df.columns if c.startswith("qc_")]
    return PredictionTable(df, qc_cols)


# --------------------------------------------------------------------------
# 基本派生量
# --------------------------------------------------------------------------

def confidence(p_pos: np.ndarray) -> np.ndarray:
    """二分类置信度 = max(p, 1-p)。与 v1 报告的 confidence 定义保持一致。"""
    p = np.asarray(p_pos, dtype=float)
    return np.maximum(p, 1.0 - p)


def pred_positive(p_pos: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(p_pos, dtype=float) >= threshold).astype(int)


def binary_errors(p_pos: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """0/1 错误指示。"""
    return (pred_positive(p_pos, threshold) != np.asarray(y, dtype=int)).astype(int)


def confusion(p_pos: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = pred_positive(p_pos, threshold)
    y = np.asarray(y, dtype=int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    n = len(y)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "prevalence": float(y.mean()) if n else float("nan"),
        "error_rate": float((fp + fn) / n) if n else float("nan"),
        "fnr": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
    }


# --------------------------------------------------------------------------
# 风险-覆盖
# --------------------------------------------------------------------------

def risk_coverage_curve(
    scores: np.ndarray,
    errors: np.ndarray,
    n_points: int = 100,
    high_score_is_confident: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    标准 risk–coverage 曲线（选择性预测）。
      scores: 越大越"可信"（例如 confidence）
      errors: 0/1 错误指示
    返回 (coverages, risks)；coverage 为接受比例，risk 为接受子集错误率。
    """
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-scores if high_score_is_confident else scores, kind="mergesort")
    e_sorted = errors[order]
    cum_err = np.cumsum(e_sorted)
    ks = np.unique(np.clip((np.linspace(1, n, n_points)).astype(int), 1, n))
    coverages = ks / n
    risks = cum_err[ks - 1] / ks
    return coverages, risks


def aurc(scores: np.ndarray, errors: np.ndarray) -> float:
    """AURC：risk–coverage 曲线下面积（越低越好）。"""
    cov, risk = risk_coverage_curve(scores, errors, n_points=len(errors))
    if len(cov) == 0:
        return float("nan")
    return float(np.trapezoid(risk, cov))


def cost_weighted_risk_coverage(
    scores: np.ndarray,
    y: np.ndarray,
    p_pos: np.ndarray,
    cost: CostModel,
    n_points: int = 100,
    high_score_is_confident: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    代价加权 risk–coverage：接受子集上的**每样本期望代价**。
    与错误率曲线的关键区别：FN 与 FP 权重不同，且最优决策阈值不再是 0.5。
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_pos, dtype=float)
    n = len(scores)
    order = np.argsort(-scores if high_score_is_confident else scores, kind="mergesort")
    y_s, p_s = y[order], p[order]
    # 逐样本代价：阳性漏诊 = fn_cost，阴性误报 = fp_cost
    per_sample = np.where(y_s == 1, cost.fn_cost * (1 - p_s), cost.fp_cost * p_s)
    cum = np.cumsum(per_sample)
    ks = np.unique(np.clip(np.linspace(1, n, n_points).astype(int), 1, n))
    return ks / n, cum[ks - 1] / ks


def risk_at_coverage(
    scores: np.ndarray, errors: np.ndarray, target_coverage: float,
    high_score_is_confident: bool = True,
) -> tuple[float, float, float]:
    """
    在给定覆盖率处取接受子集的风险。
    返回 (实际覆盖率, 风险, 该覆盖率处的分数 cutoff)。
    这是本方案的主力工作点形式——取代固定概率阈值。
    """
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    n = len(scores)
    k = max(1, int(round(target_coverage * n)))
    order = np.argsort(-scores if high_score_is_confident else scores, kind="mergesort")
    accepted = order[:k]
    cutoff = float(scores[order[k - 1]])
    risk = float(errors[accepted].mean())
    return k / n, risk, cutoff


# --------------------------------------------------------------------------
# 患者级整群 Bootstrap
# --------------------------------------------------------------------------

def cluster_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    seed: int = 42,
    cluster_col: str = "patient_id",
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    患者级整群 Bootstrap。返回 (点估计, 下界, 上界)。
    点估计用全样本计算；区间由重抽患者构成。
    """
    point = stat_fn(df)
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster_col, sort=False)]
    n_g = len(groups)
    if n_g < 2:
        return point, float("nan"), float("nan")
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_g, n_g)
        boot = pd.concat([groups[i] for i in idx], ignore_index=True)
        stats[b] = stat_fn(boot)
    lo = float(np.nanpercentile(stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    seed: int = 42,
    cluster_col: str = "patient_id",
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    配对差值的整群 Bootstrap（用于 A−B 的配对比较，如 ΔAUC、Δ风险）。
    返回 (差值点估计, 下界, 上界, 双侧 p 值近似)。
    同一批重抽患者同时施加于两个方法，消除样本构成带来的共同方差。
    """
    point = stat_fn(df)
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster_col, sort=False)]
    n_g = len(groups)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_g, n_g)
        boot = pd.concat([groups[i] for i in idx], ignore_index=True)
        stats[b] = stat_fn(boot)
    lo = float(np.nanpercentile(stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    # 近似双侧 p：零假设下差值分布关于 0 对称
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return point, lo, hi, float(min(p, 1.0))


# --------------------------------------------------------------------------
# 校准与 Brier 分解
# --------------------------------------------------------------------------

def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """期望校准误差（等宽分箱）。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(p)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """
    Murphy 分解：Brier = 可靠性 − 分辨率 + 不确定性。
      可靠性 (reliability)  越低越好 —— 校准误差
      分辨率 (resolution)   越高越好 —— 区分能力
      不确定性 (uncertainty) 数据固有难度，与模型无关
    这个分解是回答"校准迁移到底是先验漂移还是分箱噪声"的关键工具：
    若 official_valid 上 uncertainty 项显著不同，则是先验漂移而非模型退化。
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(p)
    ybar = y.mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        w = m.sum() / n
        pk, yk = p[m].mean(), y[m].mean()
        rel += w * (pk - yk) ** 2
        res += w * (yk - ybar) ** 2
    unc = ybar * (1 - ybar)
    brier = float(np.mean((p - y) ** 2))
    return {
        "brier": brier, "reliability": float(rel),
        "resolution": float(res), "uncertainty": float(unc),
        "check_sum": float(rel - res + unc),
    }


def calibration_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """可靠性图数据：(分箱平均预测, 分箱实际频率, 样本数)。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


# --------------------------------------------------------------------------
# 质量分层
# --------------------------------------------------------------------------

def stratify_by_quality(df: pd.DataFrame, col: str = "quality_risk",
                        n_strata: int = 3, labels: Sequence[str] | None = None):
    """按质量风险分位分箱，返回带 stratum 列的副本。"""
    d = df.copy()
    try:
        d["stratum"] = pd.qcut(d[col], n_strata, labels=labels, duplicates="drop")
    except ValueError:
        d["stratum"] = pd.qcut(d[col].rank(method="first"), n_strata,
                               labels=labels, duplicates="drop")
    return d


def stratified_coverage_policy(
    df: pd.DataFrame,
    quality_col: str = "quality_risk",
    score_col: str = "p1_pos",
    label_col: str = "label",
    n_strata: int = 3,
    cost: CostModel = CostModel(),
    coverage_grid: np.ndarray | None = None,
) -> dict:
    """
    QGDR 式质量分层工作点：每个质量层单独选覆盖率目标。
      - 在每一层内用**代价加权风险**在 coverage_grid 上选最优目标覆盖率；
      - 目标是所有层合计覆盖率等于 target_global（预算约束）。
    这实现了"质量差 → 少自动接受"的状态依赖路由，且完全不依赖绝对概率阈值。
    """
    if coverage_grid is None:
        coverage_grid = np.arange(0.30, 0.96, 0.05)
    d = stratify_by_quality(df, quality_col, n_strata)
    out = {}
    for s, g in d.groupby("stratum", observed=True):
        p = g[score_col].to_numpy(float)
        y = g[label_col].to_numpy(int)
        scores = confidence(p)
        rows = []
        for c in coverage_grid:
            cov, _, _ = risk_at_coverage(scores, (pred_positive(p) != y).astype(int), c)
            _, cw = cost_weighted_risk_coverage(scores, y, p, cost, n_points=max(2, len(g)))
            # 取对应覆盖率处的代价
            k = max(1, int(round(c * len(g))))
            order = np.argsort(-scores, kind="mergesort")
            ys, ps = y[order][:k], p[order][:k]
            per = np.where(ys == 1, cost.fn_cost * (1 - ps), cost.fp_cost * ps)
            rows.append({"coverage": c, "cost": float(per.mean())})
        out[str(s)] = pd.DataFrame(rows)
    return out


def default_cost_fn(y: np.ndarray, p: np.ndarray, cost: CostModel) -> np.ndarray:
    """逐样本代价（用软概率的近似的期望代价）。"""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return np.where(y == 1, cost.fn_cost * (1 - p), cost.fp_cost * p)
