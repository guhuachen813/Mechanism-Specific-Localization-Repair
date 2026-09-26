"""扰动实验分析：降质条件下的质量信号、置信度与校准。

核心问题（论文"质量感知"支柱的存亡检验）
----------------------------------------
Q1 操纵检验：降质是否真的改变了 quality_risk？
Q2 降质是否真的损害模型？（AUROC / 错误率 / 校准）
Q3 **质量信号在降质条件下能否预测错误？相对置信度是否有增量？**（H1 正式检验）
Q4 手工 quality_risk 与「学习式质量分类器」的差距有多大？
Q5 最优温度是否随图像质量移动？（把「校准不迁移」从现象升级为机制）

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 仓库的 3 类顺序 = [0:negative, 1:positive, 2:uncertain]
C1 = ["l1_0", "l1_1", "l1_2"]
C2 = ["l2_0", "l2_1", "l2_2"]
C1F = ["l1f_0", "l1f_1", "l1f_2"]

QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]

DISPLAY_ORDER = ["clean",
                 "jpeg_q50", "jpeg_q30", "jpeg_q10",
                 "blur_1", "blur_3",
                 "noise_003", "noise_006", "noise_010", "noise_025",
                 "dark_05", "bright_16",
                 "contrast_04", "ds_2", "ds_4",
                 "combo_mild", "combo_sev"]


# ---------------------------------------------------------------- 基础工具

def softmax_np(L, T):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64))
    return torch.softmax(X / T, dim=1).numpy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def ece(y, p, n_bins=15):
    """与 v1.3 复核报告完全一致的口径：conf=max(p,1-p)，15 箱 [0.5,1]。"""
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == np.asarray(y)).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(y) * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def nll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.asarray(y) * np.log(p) + (1 - np.asarray(y)) * np.log(1 - p)))


def fit_temperature(y, L, logger="nll"):
    """在给定集合上重拟合最优温度。

    必须作用在**原始三分类 logits**上：softmax(L/T)[:,1]。
    早期版本把 3 类后验压成标量再 sigmoid(z/T)，丢掉了另两类的竞争，
    会系统性高估 T*（曾出现 T*=8.0 的伪影）。此处为唯一正确口径。
    """
    L = np.asarray(L, dtype=np.float64)
    grid = np.exp(np.linspace(np.log(0.05), np.log(8.0), 400))
    best = (np.inf, 1.0)
    for T in grid:
        p = softmax_np(L, T)[:, 1]
        v = ece(y, p) if logger == "ece" else nll(y, p)
        if v < best[0]:
            best = (v, float(T))
    return best[1]


def p_pos_at(L, T):
    return softmax_np(np.asarray(L, dtype=np.float64), T)[:, 1]


def risk_at_coverage(y, score, coverage):
    """score 越大 = 风险越高。接受风险最低的 coverage 比例。"""
    y = np.asarray(y)
    order = np.argsort(score)              # 升序：低风险在前
    k = int(round(coverage * len(y)))
    ys = y[order][:k]
    return float((1 - ys).mean())          # 接受子集里的错误率


# 非对称代价（占位值取自项目既定方案：Cardiomegaly FN ≈ 2×FP）
FN_COST, FP_COST = 2.0, 1.0


def accepted_mask(score, coverage):
    """按风险升序接受前 coverage 比例，返回布尔掩码。"""
    n = len(score)
    k = int(round(coverage * n))
    keep = np.zeros(n, dtype=bool)
    keep[np.argsort(np.asarray(score))[:k]] = True
    return keep


def cost_risk(y, p, mask):
    """接受子集上按非对称代价归一的风险（每接受一张的期望代价）。"""
    y = np.asarray(y)
    pred = (np.asarray(p) >= 0.5).astype(int)
    fn = int(((pred == 0) & (y == 1) & mask).sum())
    fp = int(((pred == 1) & (y == 0) & mask).sum())
    return float((FN_COST * fn + FP_COST * fp) / max(int(mask.sum()), 1))


def fn_rate(y, p, mask):
    y = np.asarray(y)
    pred = (np.asarray(p) >= 0.5).astype(int)
    pos = int(((y == 1) & mask).sum())
    return float(((pred == 0) & (y == 1) & mask).sum() / pos) if pos else np.nan


def cluster_bootstrap_delta(y, p, score_a, score_b, groups, coverage,
                            n_boot=400, seed=0):
    """患者级整群 Bootstrap：两种路由在固定覆盖率下的代价风险之差。"""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    point = (cost_risk(y, p, accepted_mask(score_b, coverage))
             - cost_risk(y, p, accepted_mask(score_a, coverage)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        a = accepted_mask(np.asarray(score_a)[idx], coverage)
        b = accepted_mask(np.asarray(score_b)[idx], coverage)
        vals.append(cost_risk(y[idx], p[idx], b) - cost_risk(y[idx], p[idx], a))
    if len(vals) < 20:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------- 数据加载

def load_condition(outdir, cond, t1, t2):
    f = os.path.join(outdir, f"{cond}.csv")
    d = pd.read_csv(f)
    p1 = softmax_np(d[C1].to_numpy(), t1)
    p2 = softmax_np(d[C2].to_numpy(), t2)
    d["p1_pos"] = p1[:, 1]
    d["p1_neg"] = p1[:, 0]
    d["p1_unc"] = p1[:, 2]
    d["p2_pos"] = p2[:, 1]
    if all(c in d.columns for c in C1F):
        d["p1f_pos"] = softmax_np(d[C1F].to_numpy(), t1)[:, 1]
    d["cond"] = cond
    return d


def make_signals(d):
    """构造四路风险评分（越大 = 越可能出错）。"""
    y = (d["label"].to_numpy() == 1).astype(int)
    p = d["p1_pos"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    s = {
        "conf": 1.0 - conf,                                    # 置信度不足
        "quality": d["qc_quality_risk"].to_numpy(float),       # 手工质量风险
        "tta": np.abs(p - d["p1f_pos"].to_numpy(float)),       # TTA 不一致
        "disagree": np.abs(p - d["p2_pos"].to_numpy(float)),   # 模型分歧
    }
    return y, p, conf, s


# ---------------------------------------------------------------- 交叉验证

def cv_score(X, y, groups=None, n_splits=5, seed=0):
    """交叉验证的样本外风险评分（logistic）。groups 非空则按条件分组划分。"""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    out = np.full(len(y), np.nan)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y))
    else:
        g = np.asarray(groups)
        n = min(n_splits, len(np.unique(g)))
        folds = [(np.where(g != gg)[0], np.where(g == gg)[0]) for gg in np.unique(g)][:n] \
            if n < n_splits else list(GroupKFold(n_splits=n_splits).split(X, y, g))
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(sc.transform(X[tr]), y[tr])
        out[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def bootstrap_delta_auc(y, s_base, s_new, n_boot=400, seed=0):
    """配对 Bootstrap：AUC(new) - AUC(base) 的分布。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    a0, a1 = safe_auc(y, s_base), safe_auc(y, s_new)
    if not np.isfinite(a0) or not np.isfinite(a1):
        return np.nan, np.nan, np.nan
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v0, v1 = safe_auc(y[i], s_base[i]), safe_auc(y[i], s_new[i])
        if np.isfinite(v0) and np.isfinite(v1):
            deltas.append(v1 - v0)
    if len(deltas) < 20:
        return a1 - a0, np.nan, np.nan
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return a1 - a0, float(lo), float(hi)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split-tag", required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--t2", type=float, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--conds", default="")
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    conds = [c for c in (args.conds.split(",") if args.conds else DISPLAY_ORDER)
             if os.path.exists(os.path.join(args.outdir, f"{c}.csv"))]
    print(f"载入 {len(conds)} 个条件：{conds}\n")

    frames, rows = {}, []
    for cond in conds:
        d = load_condition(args.outdir, cond, args.t1, args.t2)
        d = d[d["label"] != 2].reset_index(drop=True)   # 排除不确定标签，与既有口径一致
        frames[cond] = d
        y, p, conf, s = make_signals(d)
        err = ((p >= 0.5).astype(int) != y).astype(int)

        row = dict(
            cond=cond, n=len(y), prevalence=float(y.mean()),
            auroc=safe_auc(y, p), acc=float(1 - err.mean()), err=float(err.mean()),
            fn=float(((p < 0.5) & (y == 1)).mean()), fp=float(((p >= 0.5) & (y == 0)).mean()),
            mean_conf=float(conf.mean()),
            ece_Tfrozen=ece(y, p, 15),
            mean_quality_risk=float(d["qc_quality_risk"].mean()),
            max_quality_risk=float(d["qc_quality_risk"].max()),
            n_quality_levels=int(d["qc_quality_risk"].nunique()),
            hard_fail=int(d["qc_hard_fail"].sum()),
            corr_q_err=safe_corr(s["quality"], err),
            corr_tta_err=safe_corr(s["tta"], err),
            corr_dis_err=safe_corr(s["disagree"], err),
            auc_err_conf=safe_auc(err, s["conf"]),
            auc_err_quality=safe_auc(err, s["quality"]),
            auc_err_tta=safe_auc(err, s["tta"]),
            auc_err_disagree=safe_auc(err, s["disagree"]),
            risk75_conf=risk_at_coverage(y, s["conf"], 0.75),
            risk75_all4=risk_at_coverage(y, cv_score(
                np.c_[s["conf"], s["quality"], s["tta"], s["disagree"]], err), 0.75),
        )
        # 温度重拟合：直接作用在原始三分类 logits 上（NLL 与 ECE 两种准则）
        Lraw = d[C1].to_numpy(float)
        row["T_star_nll"] = fit_temperature(y, Lraw, "nll")
        row["T_star_ece"] = fit_temperature(y, Lraw, "ece")
        p_best = p_pos_at(Lraw, row["T_star_ece"])
        row["ece_Trefit"] = ece(y, p_best, 15)
        # 退化监测：若 ECE 最优把预测阳性率压到接近 0，则 ECE 改善是假象
        row["ppr_Tfrozen"] = float((p >= 0.5).mean())
        row["ppr_Tece"] = float((p_best >= 0.5).mean())
        row["tpr_Tece"] = float(((p_best >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1))
        rows.append(row)
        print(f"  [{cond:12s}] n={len(y):>5} AUROC={row['auroc']:.4f} "
              f"err={row['err']:.4f} qr={row['mean_quality_risk']:.4f} "
              f"AUCerr(conf)={row['auc_err_conf']:.4f} (qr)={row['auc_err_quality']:.4f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.results, f"table_conditions_{args.split_tag}.csv"), index=False)

    # ---------------------------------------------------------- 池化增量化
    print("\n" + "=" * 96)
    print("H1 池化检验：质量信号相对置信度的增量（5 折交叉验证，样本外评分）")
    print("=" * 96)
    pool = pd.concat(frames.values(), ignore_index=True)
    y = (pool["label"].to_numpy() == 1).astype(int)
    p = pool["p1_pos"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    err = ((p >= 0.5).astype(int) != y).astype(int)
    s = {
        "conf": 1.0 - conf,
        "quality": pool["qc_quality_risk"].to_numpy(float),
        "tta": np.abs(p - pool["p1f_pos"].to_numpy(float)),
        "disagree": np.abs(p - pool["p2_pos"].to_numpy(float)),
    }
    groups_cond = pool["cond"].to_numpy()

    inc_rows = []
    for proto, gv in (("随机 5 折（同分布内）", None), ("按条件分折（跨降质泛化）", groups_cond)):
        base = cv_score(s["conf"][:, None], err, groups=gv)
        for name, cols in (("+quality", ["conf", "quality"]),
                           ("+tta", ["conf", "tta"]),
                           ("+disagree", ["conf", "disagree"]),
                           ("+quality+tta", ["conf", "quality", "tta"]),
                           ("+全部四路", ["conf", "quality", "tta", "disagree"])):
            sc = cv_score(np.c_[*[s[c] for c in cols]], err, groups=gv)
            d, lo, hi = bootstrap_delta_auc(err, base, sc)
            inc_rows.append(dict(protocol=proto, model=name,
                                 auc_conf=safe_auc(err, base), auc_model=safe_auc(err, sc),
                                 delta=d, ci_lo=lo, ci_hi=hi))
            print(f"  {proto:<22}{name:<16} AUC {safe_auc(err, base):.4f} → "
                  f"{safe_auc(err, sc):.4f}  Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    pd.DataFrame(inc_rows).to_csv(
        os.path.join(args.results, f"table_incremental_{args.split_tag}.csv"), index=False)

    # ---------------------------------------------------------- 操作性：固定复核预算
    print("\n" + "=" * 96)
    print("操作性检验：固定复核（拒诊）预算下，质量感知路由 vs 纯置信度路由")
    print(f"代价权重 FN:FP = {FN_COST:g}:{FP_COST:g}；区间为患者级整群 Bootstrap 95% CI")
    print("=" * 96)
    pat = pool["Patient"].astype(str).to_numpy()
    s_conf_q = cv_score(np.c_[s["conf"], s["quality"]], err, groups=None)
    op_rows = []
    for cov in (0.5, 0.75):
        mA = accepted_mask(s["conf"], cov)
        mB = accepted_mask(s_conf_q, cov)
        pa, la, ha = cluster_bootstrap_delta(y, p, s["conf"], s_conf_q, pat, cov)
        row = dict(coverage=cov,
                   cost_conf=cost_risk(y, p, mA), cost_conf_quality=cost_risk(y, p, mB),
                   delta_cost=pa, ci_lo=la, ci_hi=ha,
                   err_conf=1 - y[mA].mean() if mA.sum() else np.nan,
                   err_conf_quality=1 - y[mB].mean() if mB.sum() else np.nan,
                   fn_conf=fn_rate(y, p, mA), fn_conf_quality=fn_rate(y, p, mB))
        op_rows.append(row)
        print(f"  覆盖率 {cov:.0%}：代价风险 {row['cost_conf']:.4f} → {row['cost_conf_quality']:.4f}"
              f"  Δ={pa:+.4f} [{la:+.4f},{ha:+.4f}]   "
              f"FN {row['fn_conf']:.4f} → {row['fn_conf_quality']:.4f}")
    pd.DataFrame(op_rows).to_csv(
        os.path.join(args.results, f"table_operational_{args.split_tag}.csv"), index=False)

    # ---------------------------------------------------------- 学习式质量分类器
    print("\n" + "=" * 96)
    print("Q4 学习式质量分类器：14 个原始 QC 特征（而非布尔均值）能预测错误吗")
    print("=" * 96)
    qcols = [f"qc_{c}" for c in QC_FEATS if f"qc_{c}" in pool.columns]
    Xq = pool[qcols].to_numpy(float)
    Xq = np.nan_to_num(Xq, nan=0.0, posinf=0.0, neginf=0.0)
    learn_rows = []
    for proto, gv in (("随机 5 折（同分布内）", None), ("按条件分折（跨降质泛化）", groups_cond)):
        sc_q = cv_score(Xq, err, groups=gv)
        sc_all = cv_score(np.c_[Xq, s["conf"]], err, groups=gv)
        base = cv_score(s["conf"][:, None], err, groups=gv)
        learn_rows.append(dict(protocol=proto,
                               auc_conf=safe_auc(err, base),
                               auc_qc_feats=safe_auc(err, sc_q),
                               auc_qc_feats_plus_conf=safe_auc(err, sc_all)))
        print(f"  {proto:<22} 置信度 {safe_auc(err, base):.4f} | "
              f"14 原始 QC 特征 {safe_auc(err, sc_q):.4f} | "
              f"QC+置信度 {safe_auc(err, sc_all):.4f}")
    pd.DataFrame(learn_rows).to_csv(
        os.path.join(args.results, f"table_learned_qc_{args.split_tag}.csv"), index=False)

    # ---------------------------------------------------------- 条件内检验（决定性）
    print("\n" + "=" * 96)
    print("Q4b 条件内检验（决定性）：在每个降质条件**内部**单独做")
    print("    跨条件池化时，质量特征可以靠「认出这是哪个降质条件」间接预测错误率；")
    print("    条件内检验堵死这条捷径，测的是「这张图本身有多难」。")
    print("=" * 96)
    print(f"{'条件':<12}{'n':>6}{'错误率':>8} | {'手工 quality_risk':>17}{'14 特征':>9}"
          f"{'置信度':>9}{'特征+置信度':>12}")
    print("-" * 96)
    within_rows = []
    for cond, d in frames.items():
        yy = (d["label"].to_numpy() == 1).astype(int)
        pp = d["p1_pos"].to_numpy(float)
        ee = ((pp >= 0.5).astype(int) != yy).astype(int)
        if len(np.unique(ee)) < 2 or ee.sum() < 20:
            continue
        qcols_c = [f"qc_{c}" for c in QC_FEATS if f"qc_{c}" in d.columns]
        Xq = np.nan_to_num(d[qcols_c].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        cfg = 1.0 - np.maximum(pp, 1 - pp)
        a_hq = safe_auc(ee, d["qc_quality_risk"].to_numpy(float))
        a_qc = safe_auc(ee, cv_score(Xq, ee))
        a_cf = safe_auc(ee, cv_score(cfg[:, None], ee))
        a_bo = safe_auc(ee, cv_score(np.c_[Xq, cfg], ee))
        within_rows.append(dict(cond=cond, n=len(yy), err=float(ee.mean()),
                                auc_quality_risk=a_hq, auc_qc14=a_qc,
                                auc_conf=a_cf, auc_qc14_conf=a_bo,
                                delta_vs_conf=(a_bo - a_cf) if np.isfinite(a_bo) and np.isfinite(a_cf) else np.nan))
        print(f"{cond:<12}{len(yy):>6}{ee.mean():>8.4f} | {a_hq:>17.4f}{a_qc:>9.4f}"
              f"{a_cf:>9.4f}{a_bo:>12.4f}")
    wdf = pd.DataFrame(within_rows)
    if len(wdf):
        print("-" * 96)
        print(f"{'平均':<12}{'':>6}{'':>8} | {wdf['auc_quality_risk'].mean():>17.4f}"
              f"{wdf['auc_qc14'].mean():>9.4f}{wdf['auc_conf'].mean():>9.4f}"
              f"{wdf['auc_qc14_conf'].mean():>12.4f}")
        print(f"   14 特征相对置信度的平均增量 ΔAUC = {wdf['delta_vs_conf'].mean():+.4f}  "
              f"（跨 {len(wdf)} 个条件）")
        wdf.to_csv(os.path.join(args.results, f"table_within_condition_{args.split_tag}.csv"),
                   index=False)

    # ---------------------------------------------------------- 2×2 各信号
    print("\n" + "=" * 96)
    print("各风险信号的错误检测能力（池化，全条件）")
    print("=" * 96)
    sig_rows = []
    for k, v in s.items():
        sig_rows.append(dict(signal=k, auc=safe_auc(err, v), corr=safe_corr(v, err)))
        print(f"  {k:<10} AUC={safe_auc(err, v):.4f}  corr={safe_corr(v, err):+.4f}")
    sig_rows.append(dict(signal="n(错误)", auc=np.nan, corr=np.nan))
    pd.DataFrame(sig_rows).to_csv(
        os.path.join(args.results, f"table_signals_{args.split_tag}.csv"), index=False)

    print("\n输出目录：", args.results)
    print(json.dumps({"conditions": conds, "pooled_n": int(len(pool)),
                      "pooled_err": float(err.mean())}, indent=2))


if __name__ == "__main__":
    main()
