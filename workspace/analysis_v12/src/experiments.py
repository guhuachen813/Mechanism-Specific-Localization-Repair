"""
experiments.py — v1.2 四项离线分析实验

  exp1  风险-覆盖（错误率 + 代价加权）与工作点迁移
  exp2  质量→错误预测器 R=P(E=1|C,Q) —— H1 的正式检验
  exp3  质量分层覆盖策略（QGDR 迁移）—— 状态依赖路由的实证
  exp4  校准迁移诊断（可靠性图 + Brier 分解）

用法：
    python3 src/experiments.py exp1 [--pred out/xxx.csv]
    python3 src/experiments.py all  [--pred out/xxx.csv]

作者：ClawsGO Science Agent
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from rc_lib import (  # noqa: E402
    CostModel, load_predictions, confidence, pred_positive, binary_errors,
    confusion, risk_coverage_curve, cost_weighted_risk_coverage,
    risk_at_coverage, aurc, cluster_bootstrap, paired_cluster_bootstrap,
    ece, brier_decomposition, calibration_curve, stratify_by_quality,
    setup_matplotlib, OKABE_ITO,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

TARGET_COVERAGE = 0.75
COST = CostModel(fn_cost=2.0, fp_cost=1.0)   # Cardiomegaly 占位值，需医生确认


# ==========================================================================
# exp1 — 风险-覆盖与工作点迁移
# ==========================================================================

def exp1(pt, tag="synthetic"):
    rows, curves = [], {}
    for split in pt.splits:
        g = pt.split(split).df
        y = g["label"].to_numpy(int)
        for name, pcol in [("Model1", "p1_pos"), ("Model2", "p2_pos")]:
            if pcol not in g.columns:
                continue
            p = g[pcol].to_numpy(float)
            err = binary_errors(p, y)
            s = confidence(p)
            cov, risk = risk_coverage_curve(s, err)
            curves[(split, name)] = (cov, risk)
            _, r, cut = risk_at_coverage(s, err, TARGET_COVERAGE)
            cm = confusion(p, y)
            rows.append({
                "split": split, "model": name, "n": len(g),
                "prevalence": cm["prevalence"], "error_rate": cm["error_rate"],
                "fn": cm["fn"], "fp": cm["fp"], "fnr": cm["fnr"], "fpr": cm["fpr"],
                "auroc_auc_auc": np.nan,
                "coverage@work": TARGET_COVERAGE, "risk@work": r, "cutoff": cut,
                "AURC": aurc(s, err),
            })
        # Fusion
        if "p2_pos" in g.columns:
            pf = 0.5 * (g["p1_pos"].to_numpy(float) + g["p2_pos"].to_numpy(float))
            err = binary_errors(pf, y)
            s = confidence(pf)
            cov, risk = risk_coverage_curve(s, err)
            curves[(split, "Fusion")] = (cov, risk)
            _, r, cut = risk_at_coverage(s, err, TARGET_COVERAGE)
            cm = confusion(pf, y)
            rows.append({
                "split": split, "model": "Fusion", "n": len(g),
                "prevalence": cm["prevalence"], "error_rate": cm["error_rate"],
                "fn": cm["fn"], "fp": cm["fp"], "fnr": cm["fnr"], "fpr": cm["fpr"],
                "auroc_auc_auc": np.nan,
                "coverage@work": TARGET_COVERAGE, "risk@work": r, "cutoff": cut,
                "AURC": aurc(s, err),
            })
    tab = pd.DataFrame(rows)

    # ---- Bootstrap：工作点选择性风险的 CI ----
    ci_rows = []
    for split in pt.splits:
        g = pt.split(split).df
        for name, pcol in [("Model1", "p1_pos"), ("Model2", "p2_pos")]:
            if pcol not in g.columns:
                continue
            y = g["label"].to_numpy(int)
            p = g[pcol].to_numpy(float)
            err = binary_errors(p, y)
            s = confidence(p)

            def stat(d, y=y, p=p, s=s):
                yy = d["label"].to_numpy(int)
                pp = d[pcol].to_numpy(float)
                ss = confidence(pp)
                ee = (pred_positive(pp) != yy).astype(int)
                return risk_at_coverage(ss, ee, TARGET_COVERAGE)[1]

            pt_est, lo, hi = cluster_bootstrap(g, stat, n_boot=1000)
            ci_rows.append({"split": split, "model": name,
                            "risk": pt_est, "ci_lo": lo, "ci_hi": hi})
    ci = pd.DataFrame(ci_rows)

    # ---- 代价加权曲线 ----
    cost_rows = []
    for split in pt.splits:
        g = pt.split(split).df
        y = g["label"].to_numpy(int)
        p = g["p1_pos"].to_numpy(float)
        s = confidence(p)
        cov, cw = cost_weighted_risk_coverage(s, y, p, COST)
        k = max(1, int(round(TARGET_COVERAGE * len(g))))
        order = np.argsort(-s, kind="mergesort")
        ys, ps = y[order][:k], p[order][:k]
        cost_rows.append({
            "split": split, "coverage": TARGET_COVERAGE,
            "cost_per_sample": float(np.mean(np.where(ys == 1,
                                       COST.fn_cost * (1 - ps), COST.fp_cost * ps))),
            "error_rate@work": float((pred_positive(ps) != ys).mean()),
        })
    cost_tab = pd.DataFrame(cost_rows)

    _plot_exp1(curves, ci, tag)
    tab.to_csv(f"{OUT}/exp1_workpoints_{tag}.csv", index=False)
    ci.to_csv(f"{OUT}/exp1_risk_ci_{tag}.csv", index=False)
    cost_tab.to_csv(f"{OUT}/exp1_cost_{tag}.csv", index=False)
    return {"workpoints": tab, "risk_ci": ci, "cost": cost_tab}


def _plot_exp1(curves, ci, tag):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"Model1": OKABE_ITO["blue"], "Model2": OKABE_ITO["orange"],
              "Fusion": OKABE_ITO["green"]}
    splits = sorted({k[0] for k in curves})
    for ax, split in zip(axes, splits):
        for name in ["Model1", "Model2", "Fusion"]:
            key = (split, name)
            if key not in curves:
                continue
            cov, risk = curves[key]
            ax.plot(cov, risk, color=colors[name], lw=1.8, label=name)
        ax.axvline(TARGET_COVERAGE, color=OKABE_ITO["grey"], ls="--", lw=1)
        ax.set_xlabel("覆盖率 (接受比例)")
        ax.set_ylabel("选择性风险 (接受子集错误率)")
        ax.set_title(split)
        ax.set_xlim(0.2, 1.0)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"风险-覆盖曲线（工作点 {TARGET_COVERAGE:.0%}）", fontsize=12)
    fig.savefig(f"{OUT}/fig_exp1_risk_coverage_{tag}.png")
    plt.close(fig)


# ==========================================================================
# exp2 — 质量→错误预测器（H1 正式检验）
# ==========================================================================

def _auc(model, X, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def exp2(pt, tag="synthetic"):
    """
    三种特征集在同一目标（Model 1 是否犯错）上比较：
      A. 仅置信度                  [confidence]
      B. 置信度 + p_uncertain      [+ p1_unc]
      C. 置信度 + p_unc + 全部软质控特征
    在 route_validation 上做 5 折交叉验证，再整体重训后迁移到 official_valid。
    ΔAUC 用患者级配对 Bootstrap 给区间。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    qc = pt.qc_cols
    route = pt.split("route_validation").df.copy()
    off = pt.split("official_valid").df.copy() if "official_valid" in pt.splits else None

    def featsets(g):
        base = pd.DataFrame({"conf": confidence(g["p1_pos"].to_numpy(float))})
        if "p1_unc" in g.columns:
            base["p_unc"] = g["p1_unc"].to_numpy(float)
        A = base[["conf"]]
        B = base[[c for c in ["conf", "p_unc"] if c in base.columns]]
        C = pd.concat([B, g[qc].reset_index(drop=True)], axis=1) if qc else B
        return {"A_conf": A, "B_conf+unc": B, "C_conf+unc+QC": C}

    def targets(g):
        y = g["label"].to_numpy(int)
        p = g["p1_pos"].to_numpy(float)
        return (pred_positive(p) != y).astype(int)

    fs_route, fs_off = featsets(route), featsets(off) if off is not None else None
    e_route = targets(route)
    e_off = targets(off) if off is not None else None

    def mk():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    # ---- 5 折交叉验证（route_validation 内）----
    from sklearn.model_selection import StratifiedKFold
    cv_rows = []
    for name, X in fs_route.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(e_route))
        for tr, te in skf.split(X, e_route):
            m = mk().fit(X.iloc[tr], e_route[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        cv_rows.append({"featureset": name, "n_features": X.shape[1],
                        "auc_cv_route": float(roc_auc_score(e_route, oof))})
    cv = pd.DataFrame(cv_rows)

    # ---- 迁移评估：route 训练 → official 测试 ----
    trans_rows = []
    for name, X in fs_route.items():
        m = mk().fit(X, e_route)
        row = {"featureset": name, "n_features": X.shape[1],
               "auc_refit_route": _auc(m, X, e_route)}
        if fs_off is not None:
            row["auc_transfer_official"] = _auc(m, fs_off[name], e_off)
        trans_rows.append(row)
    trans = pd.DataFrame(trans_rows)

    # ---- ΔAUC 的配对整群 Bootstrap（C − A），route 上 ----
    delta_ci = {}
    Xa, Xc = fs_route["A_conf"], fs_route["C_conf+unc+QC"]
    route_r = route.reset_index(drop=True)

    def make_stat(Xa, Xc):
        def stat(d):
            idx = d.index.to_numpy()
            ya = e_route[idx]
            if len(np.unique(ya)) < 2:
                return 0.0
            ma = mk().fit(Xa.iloc[idx], ya)
            mc = mk().fit(Xc.iloc[idx], ya)
            return _auc(mc, Xc.iloc[idx], ya) - _auc(ma, Xa.iloc[idx], ya)
        return stat

    pt_est, lo, hi, p = paired_cluster_bootstrap(
        route_r, make_stat(Xa, Xc), n_boot=300, seed=7)
    delta_ci["route_delta_auc_C_minus_A"] = {
        "delta": pt_est, "ci_lo": lo, "ci_hi": hi, "p_approx": p}

    # ---- 单变量：quality_risk 与错误的关联（分箱）----
    bins = stratify_by_quality(route_r, "quality_risk", 5)
    qbin = bins.groupby("stratum", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "quality_risk_mean": g["quality_risk"].mean(),
            "error_rate": (pred_positive(g["p1_pos"].to_numpy(float))
                           != g["label"].to_numpy(int)).mean(),
        }), include_groups=False).reset_index()
    qbin.to_csv(f"{OUT}/exp2_quality_bins_{tag}.csv", index=False)

    _plot_exp2(cv, trans, qbin, tag)
    cv.to_csv(f"{OUT}/exp2_cv_{tag}.csv", index=False)
    trans.to_csv(f"{OUT}/exp2_transfer_{tag}.csv", index=False)
    with open(f"{OUT}/exp2_delta_auc_{tag}.json", "w") as f:
        json.dump(delta_ci, f, indent=2, ensure_ascii=False)
    return {"cv": cv, "transfer": trans, "delta_auc": delta_ci, "quality_bins": qbin}


def _plot_exp2(cv, trans, qbin, tag):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    # 1) CV AUC
    ax = axes[0]
    ax.barh(cv["featureset"], cv["auc_cv_route"], color=OKABE_ITO["blue"], height=0.5)
    for i, v in enumerate(cv["auc_cv_route"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("错误预测 AUC（route_validation 5 折 CV）")
    ax.set_title("A. 质量特征是否提升错误预测")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    # 2) 迁移 AUC
    ax = axes[1]
    if "auc_transfer_official" in trans.columns:
        ax.barh(trans["featureset"], trans["auc_transfer_official"],
                color=OKABE_ITO["vermillion"], height=0.5)
        for i, v in enumerate(trans["auc_transfer_official"]):
            ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("错误预测 AUC（迁移到 official_valid）")
    ax.set_title("B. 迁移性")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    # 3) 质量分箱错误率
    ax = axes[2]
    ax.plot(qbin["quality_risk_mean"], qbin["error_rate"],
            marker="o", color=OKABE_ITO["green"], lw=1.8)
    ax.set_xlabel("平均 quality_risk（分位箱）")
    ax.set_ylabel("错误率")
    ax.set_title("C. 质量风险 vs 错误率")
    ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle("质量信号对错误的预测能力（H1 检验）", fontsize=12)
    fig.savefig(f"{OUT}/fig_exp2_quality_error_{tag}.png")
    plt.close(fig)


# ==========================================================================
# exp3 — 质量分层覆盖策略
# ==========================================================================

def _marginal_cost_curve(g, score_col, cost, grid):
    """给定层内样本，返回 {coverage: 该覆盖率下的每样本期望代价}。"""
    y = g["label"].to_numpy(int)
    p = g[score_col].to_numpy(float)
    s = confidence(p)
    order = np.argsort(-s, kind="mergesort")
    ys, ps = y[order], p[order]
    per = np.where(ys == 1, cost.fn_cost * (1 - ps), cost.fp_cost * ps)
    cum = np.cumsum(per)
    n = len(g)
    out = {}
    for c in grid:
        k = max(1, int(round(c * n)))
        out[c] = float(cum[k - 1] / k)
    return out


def exp3(pt, tag="synthetic", score_col="p1_pos", n_strata=3):
    """
    在固定全局覆盖率(75%)下，比较：
      均匀策略：每层同一覆盖率
      分层策略：按各层边际代价贪心分配覆盖率（把自动接受额度让给模型更可靠的层）
    指标：全局每样本期望代价。差值用患者级配对 Bootstrap 给区间。
    """
    grid = np.round(np.arange(0.30, 0.96, 0.02), 4)
    results = {}
    for split in pt.splits:
        g0 = pt.split(split).df.copy()
        g0 = stratify_by_quality(g0, "quality_risk", n_strata)
        strata = [s for s in g0["stratum"].cat.categories]
        per_stratum = {s: g0[g0["stratum"] == s] for s in strata}
        mc = {s: _marginal_cost_curve(per_stratum[s], score_col, COST, grid)
              for s in strata}
        ns = {s: len(per_stratum[s]) for s in strata}
        N = sum(ns.values())

        # ---- 均匀策略 ----
        target = TARGET_COVERAGE
        uni_cov = {s: target for s in strata}

        # ---- 分层策略：贪心下移边际代价最低的层 ----
        cur = {s: float(grid[0]) for s in strata}
        def total_cov():
            return sum(ns[s] * cur[s] for s in strata) / N
        # 若起点总覆盖率已高于目标，则改为从高端下撤
        if total_cov() > target:
            cur = {s: float(grid[-1]) for s in strata}
            while total_cov() > target:
                best, best_gain = None, None
                for s in strata:
                    j = int(np.where(grid == cur[s])[0][0])
                    if j == 0:
                        continue
                    gain = ns[s] * (cur[s] - grid[j - 1]) / N
                    if total_cov() - gain >= target - 1e-9:
                        if best_gain is None or gain > best_gain:
                            best, best_gain = s, gain
                if best is None:
                    break
                j = int(np.where(grid == cur[best])[0][0])
                cur[best] = float(grid[j - 1])
        else:
            while True:
                best, best_cost = None, None
                for s in strata:
                    j = int(np.where(grid == cur[s])[0][0])
                    if j >= len(grid) - 1:
                        continue
                    if total_cov() + ns[s] * (grid[j + 1] - cur[s]) / N > target + 1e-9:
                        continue
                    c = mc[s][float(grid[j + 1])] - mc[s][cur[s]]
                    if best_cost is None or c < best_cost:
                        best, best_cost = s, c
                if best is None:
                    break
                j = int(np.where(grid == cur[best])[0][0])
                cur[best] = float(grid[j + 1])
        strat_cov = cur

        def global_cost(cov_map):
            tot = 0.0
            for s in strata:
                y = per_stratum[s]["label"].to_numpy(int)
                p = per_stratum[s][score_col].to_numpy(float)
                sc = confidence(p)
                k = max(1, int(round(cov_map[s] * len(y))))
                order = np.argsort(-sc, kind="mergesort")
                ys, ps = y[order][:k], p[order][:k]
                per = np.where(ys == 1, COST.fn_cost * (1 - ps), COST.fp_cost * ps)
                tot += per.sum()
            return tot / N

        results[split] = {
            "uniform": {"cost": global_cost(uni_cov),
                        "coverage": dict(uni_cov)},
            "stratified": {"cost": global_cost(strat_cov),
                           "coverage": {s: round(float(v), 3) for s, v in strat_cov.items()}},
            "n_strata": {s: ns[s] for s in strata},
            "stratum_mean_quality": {s: float(per_stratum[s]["quality_risk"].mean())
                                     for s in strata},
        }

    _plot_exp3(results, tag)
    with open(f"{OUT}/exp3_stratified_{tag}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def _plot_exp3(results, tag):
    plt = setup_matplotlib()
    splits = list(results.keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.0),
                             squeeze=False)
    for ax, split in zip(axes[0], splits):
        r = results[split]
        strata = list(r["uniform"]["coverage"].keys())
        x = np.arange(len(strata))
        u = [r["uniform"]["coverage"][s] for s in strata]
        st = [r["stratified"]["coverage"][s] for s in strata]
        ax.bar(x - 0.2, u, 0.4, label="均匀策略", color=OKABE_ITO["grey"])
        ax.bar(x + 0.2, st, 0.4, label="质量分层策略", color=OKABE_ITO["blue"])
        for i, s in enumerate(strata):
            ax.text(i, -0.045, f"q̄={r['stratum_mean_quality'][s]:.2f}",
                    ha="center", fontsize=8, color=OKABE_ITO["grey"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"层{i+1}" for i in range(len(strata))])
        ax.set_ylabel("该层目标覆盖率")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{split}\n均匀代价={r['uniform']['cost']:.4f}  "
                     f"分层代价={r['stratified']['cost']:.4f}")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.suptitle(f"质量分层覆盖率分配（全局目标 {TARGET_COVERAGE:.0%}）", fontsize=12)
    fig.savefig(f"{OUT}/fig_exp3_stratified_{tag}.png")
    plt.close(fig)


# ==========================================================================
# exp4 — 校准迁移诊断
# ==========================================================================

def exp4(pt, tag="synthetic"):
    rows = []
    for split in pt.splits:
        g = pt.split(split).df
        y = g["label"].to_numpy(int)
        p = g["p1_pos"].to_numpy(float)
        bd = brier_decomposition(p, y)
        rows.append({
            "split": split, "n": len(g), "prevalence": float(y.mean()),
            "mean_p": float(p.mean()),
            "ECE": ece(p, y),
            "brier": bd["brier"], "reliability": bd["reliability"],
            "resolution": bd["resolution"], "uncertainty": bd["uncertainty"],
        })
    tab = pd.DataFrame(rows)

    _plot_exp4(pt, tab, tag)
    tab.to_csv(f"{OUT}/exp4_calibration_{tag}.csv", index=False)
    return tab


def _plot_exp4(pt, tab, tag):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # 1) 可靠性图
    ax = axes[0]
    cols = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"]]
    for i, split in enumerate(pt.splits):
        g = pt.split(split).df
        y = g["label"].to_numpy(int)
        p = g["p1_pos"].to_numpy(float)
        xs, ys, ns = calibration_curve(p, y)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.5, color=cols[i % 3],
                label=f"{split} (n={len(g)})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=OKABE_ITO["grey"], label="完美校准")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际阳性频率")
    ax.set_title("A. 可靠性图")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    # 2) Brier 分解堆叠
    ax = axes[1]
    x = np.arange(len(tab))
    ax.bar(x, tab["reliability"], 0.5, label="可靠性(越低越好)",
           color=OKABE_ITO["vermillion"])
    ax.bar(x, -tab["resolution"], 0.5, label="−分辨率(越高越好)",
           color=OKABE_ITO["blue"])
    ax.plot(x, tab["brier"], "kD", ms=6, label="Brier")
    ax.set_xticks(x)
    ax.set_xticklabels(tab["split"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("B. Brier 分解")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    # 3) 患病率 vs 平均预测概率
    ax = axes[2]
    ax.bar(x - 0.2, tab["prevalence"], 0.4, label="真实患病率",
           color=OKABE_ITO["green"])
    ax.bar(x + 0.2, tab["mean_p"], 0.4, label="平均预测概率",
           color=OKABE_ITO["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(tab["split"])
    ax.set_ylabel("比例")
    ax.set_title("C. 先验漂移检查")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.suptitle("校准迁移诊断", fontsize=12)
    fig.savefig(f"{OUT}/fig_exp4_calibration_{tag}.png")
    plt.close(fig)


# ==========================================================================
# 主入口
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["exp1", "exp2", "exp3", "exp4", "all"])
    ap.add_argument("--pred", default=f"{OUT}/synthetic_predictions.csv")
    ap.add_argument("--tag", default="synthetic")
    a = ap.parse_args()

    pt = load_predictions(a.pred)
    print(f"载入预测表：{a.pred}  子集={pt.splits}  总行数={len(pt)}")

    if a.which in ("exp1", "all"):
        print("\n[exp1] 风险-覆盖与工作点迁移 …")
        r = exp1(pt, a.tag)
        print(r["workpoints"].to_string(index=False))
        print("\n风险 CI：")
        print(r["risk_ci"].to_string(index=False))
    if a.which in ("exp2", "all"):
        print("\n[exp2] 质量→错误预测器（H1 检验）…")
        r = exp2(pt, a.tag)
        print(r["cv"].to_string(index=False))
        print(r["transfer"].to_string(index=False))
        print("ΔAUC(C−A) =", json.dumps(r["delta_auc"], ensure_ascii=False))
    if a.which in ("exp3", "all"):
        print("\n[exp3] 质量分层覆盖策略 …")
        r = exp3(pt, a.tag)
        for s, v in r.items():
            print(f"  {s}: 均匀={v['uniform']['cost']:.4f}  "
                  f"分层={v['stratified']['cost']:.4f}")
            print(f"     分层覆盖率={v['stratified']['coverage']}")
    if a.which in ("exp4", "all"):
        print("\n[exp4] 校准迁移诊断 …")
        r = exp4(pt, a.tag)
        print(r.to_string(index=False))
    print(f"\n输出目录：{os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
