"""NIH ChestX-ray14 外部验证：把 v1.5 的全部结论搬到第三个域上复算。

三个域的先验坐标
----------------
  route-validation（域内）   12.6%   ← 模型训练所在域
  official-valid（偏移域）   32.7%   ← CheXpert 官方划分，同源不同先验
  **NIH ChestX-ray14        4.18%** ← 本脚本：跨机构、跨标注体系、跨设备

NIH 比 official-valid 更远：它既不是 CheXpert 的抽样，标注由 NLP 从报告抽取
而非放射科医生判读，且图像来自多个机构的不同设备。因此它是「能证伪就证伪」
的最强外部检验点。

分析分四块
----------
1 迁移基线：AUROC / ECE / NLL / 操作点，图像级与**患者级整群** Bootstrap 区间
2 校准修补 A–F：在第三章域上，F（先验匹配阈值）是否仍然既降代价又不买掉灵敏度
3 「何时有用」定律：按真实质量分层，跨层检验「置信度越失效、质量特征越有用」
4 风险–覆盖率与固定复核预算下的代价

本镜像自带患者 ID（由 nih_attach_ids.py 从官方清单恢复），故 Bootstrap 可做
患者级整群重抽样。官方 test 划分 2,797 名患者、均 9.15 张、最多 184 张，
聚类效应不可忽略。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calib_fix_compare import (C1, ece, metrics, nll,  # noqa: E402
                               softmax_pos, thr_for_prevalence, binary_logit)

T1 = 0.4943421483039856   # 源域冻结温度（densenet121 seed42）
PI_SOURCE = 0.126         # 源域患病率
SEEDS = 5

QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def cv_oof(X, y, learner, seed):
    """样本外评分，与 h1_robust.py 完全一致，保证两域可比。"""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        if learner == "lr":
            sz = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=3000, C=1.0)
            m.fit(sz.transform(X[tr]), y[tr])
            oof[te] = m.predict_proba(sz.transform(X[te]))[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                               max_leaf_nodes=15, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=seed)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def pos_prob(L, T=T1, delta=0.0):
    """三分类 softmax 的正类后验（先验偏移加在正类 logit 上）。"""
    return softmax_pos(L, T, delta)


def cluster_bootstrap(df, stat_fn, n_boot=400, seed=0):
    """患者级整群 Bootstrap：以患者为单位有放回重抽，整簇进整簇出。

    用 bincount + repeat 向量化实现：先抽每个患者被重抽到的次数，
    再把该次数广播到该患者的每一张图。得到的是按行号排序的重复索引，
    与「逐簇拼接」等价——所有统计量都是置换不变的。
    """
    rng = np.random.default_rng(seed)
    gid = df.groupby("Patient", sort=False).ngroup().to_numpy()
    n_g = int(gid.max()) + 1
    n = len(gid)
    ar = np.arange(n)
    vals = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_g, n_g)
        cnt = np.bincount(drawn, minlength=n_g)
        idx = np.repeat(ar, cnt[gid])
        if len(idx) == 0:
            vals.append(np.nan)
            continue
        vals.append(stat_fn(df.iloc[idx]))
    return np.asarray(vals, float)


def ci(v, lo=2.5, hi=97.5):
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.nan, np.nan
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="nih_attach_ids.py 的输出")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--chexpert-qc", default="", help="CheXpert 干净档 QC 列，用于分布对照")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    d = pd.read_csv(args.csv)
    d = d[d["label"] != 2].reset_index(drop=True)
    y = (d["label"].to_numpy() == 1).astype(int)
    L = d[C1].to_numpy(np.float64)
    pi_t = float(y.mean())
    n_pat = d["Patient"].nunique()

    print("=" * 100)
    print("NIH ChestX-ray14 外部验证")
    print("=" * 100)
    print(f"图像 {len(d)}，患者 {n_pat}，均 {len(d)/n_pat:.2f} 张/人")
    print(f"Cardiomegaly 患病率 {pi_t:.4f}（route 12.6% / official 32.7%）")

    pA = pos_prob(L, T1)
    conf = np.maximum(pA, 1 - pA)
    pred = (pA >= 0.5).astype(int)
    err = (pred != y).astype(int)

    # ---------------------------------------------------------------- 1 基线
    print("\n" + "-" * 100)
    print("1 迁移基线（冻结温度 T=0.4943，阈值 0.5）")
    print("-" * 100)

    def base_stats(g):
        yy = (g["label"].to_numpy() == 1).astype(int)
        pp = pos_prob(g[C1].to_numpy(np.float64), T1)
        e = ((pp >= 0.5).astype(int) != yy).astype(int)
        m = metrics(yy, pp)
        return [safe_auc(yy, pp), safe_auc(e, -np.maximum(pp, 1 - pp)),
                m["ece"], m["nll"], m["acc"], m["tpr"], m["ppr"], m["err"], m["cost"]]

    point = base_stats(d)
    boot = cluster_bootstrap(d, base_stats, n_boot=args.boot)
    labels = ["AUROC（排序）", "置信度检测错误的 AUROC", "ECE", "NLL",
              "准确率", "灵敏度 TPR", "预测阳性率 PPR", "错误率", "代价风险 FN:FP=2:1"]
    base_rows = []
    print(f"{'指标':<26}{'点估计':>10}{'患者级 95% CI':>22}")
    for i, nm in enumerate(labels):
        lo, hi = ci(boot[:, i])
        star = "" if (lo <= point[i] <= hi) else "  !!"
        print(f"{nm:<26}{point[i]:>10.4f}   [{lo:>7.4f}, {hi:>7.4f}]{star}")
        base_rows.append(dict(metric=nm, point=point[i], ci_lo=lo, ci_hi=hi,
                              boot_sd=float(np.nanstd(boot[:, i]))))
    pd.DataFrame(base_rows).to_csv(
        os.path.join(args.results, "table_nih_baseline.csv"), index=False)

    # 对照：图像级 Bootstrap（用于量化聚类效应把区间放大了多少）
    img_lo, img_hi = ci(cluster_bootstrap(
        d.assign(Patient=lambda x: np.arange(len(x))), base_stats, n_boot=args.boot)[:, 0])
    print(f"\n聚类效应对照：AUROC 图像级 CI [{img_lo:.4f}, {img_hi:.4f}] "
          f"vs 患者级 CI [{ci(boot[:,0])[0]:.4f}, {ci(boot[:,0])[1]:.4f}]")

    # ---------------------------------------------------------- 2 校准修补 A–F
    print("\n" + "-" * 100)
    print("2 校准修补 A–F（与 CheXpert 两域同一套定义）")
    print("-" * 100)
    grid = np.exp(np.linspace(np.log(0.1), np.log(8.0), 160))
    delta_prior = float(binary_logit(pi_t) - binary_logit(PI_SOURCE))
    TB = min(((ece(y, pos_prob(L, T)), T) for T in grid), key=lambda t: t[0])[1]
    TC = min(((nll(y, pos_prob(L, T)), T) for T in grid), key=lambda t: t[0])[1]
    bestE = (np.inf, 1.0)
    for T in grid:
        v = ece(y, pos_prob(L, T, delta_prior))
        if v < bestE[0]:
            bestE = (v, T)
    thrF = thr_for_prevalence(pA, pi_t)

    strategies = [("A 冻结温度", pA, 0.5),
                  ("B ECE最优温度", pos_prob(L, TB), 0.5),
                  ("C NLL最优温度", pos_prob(L, TC), 0.5),
                  ("D 仅先验偏移", pos_prob(L, T1, delta_prior), 0.5),
                  ("E 先验偏移+温度", pos_prob(L, bestE[1], delta_prior), 0.5),
                  ("F 冻结+先验匹配阈值", pA, thrF)]
    print(f"π_t={pi_t:.4f}  Δ_prior={delta_prior:+.4f}  T_B={TB:.3f}  T_C={TC:.3f}  "
          f"thr_F={thrF:.4f}")
    print(f"\n{'策略':<22}{'ECE':>8}{'NLL':>8}{'TPR':>8}{'PPR':>8}{'错误率':>9}{'代价':>9}{'代价 95% CI':>20}")

    def cost_of(g, thr):
        yy = (g["label"].to_numpy() == 1).astype(int)
        return metrics(yy, pos_prob(g[C1].to_numpy(np.float64), T1), thr)["cost"]

    bootA = cluster_bootstrap(d, lambda g: [cost_of(g, 0.5)], n_boot=args.boot)[:, 0]
    bootF = cluster_bootstrap(d, lambda g: [cost_of(g, thrF)], n_boot=args.boot)[:, 0]

    calib_rows = []
    for nm, p, thr in strategies:
        m = metrics(y, p, thr)
        if nm.startswith("A"):
            lo, hi = ci(bootA)
        elif nm.startswith("F"):
            lo, hi = ci(bootF)
        else:
            lo = hi = np.nan
        cis = f"[{lo:.4f}, {hi:.4f}]" if np.isfinite(lo) else "—"
        print(f"{nm:<22}{m['ece']:>8.3f}{m['nll']:>8.3f}{m['tpr']:>8.3f}"
              f"{m['ppr']:>8.3f}{m['err']:>9.3f}{m['cost']:>9.4f}{cis:>20}")
        calib_rows.append(dict(strategy=nm, thr=thr, ece=m["ece"], nll=m["nll"],
                               acc=m["acc"], tpr=m["tpr"], ppr=m["ppr"], err=m["err"],
                               cost=m["cost"], cost_lo=lo, cost_hi=hi))
    pd.DataFrame(calib_rows).to_csv(
        os.path.join(args.results, "table_nih_calibfix.csv"), index=False)

    # 代价的配对差（同一批患者上的 F − A）
    pair = bootF - bootA
    lo, hi = ci(pair)
    print(f"\n配对差 F − A 的代价风险：{np.nanmean(pair):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          f"  → {'显著' if lo * hi > 0 else '跨 0，不显著'}")

    # ------------------------------------------------------ 3 「何时有用」定律
    print("\n" + "-" * 100)
    print("3 「何时有用」定律的外部检验：按真实质量分层")
    print("-" * 100)
    print("CheXpert 上的律：跨降质档，条件内置信度检测错误的 AUROC 越低，")
    print("质量特征的增量 ΔAUC 越大（r = −0.73 ~ −0.81）。")
    print("NIH 上没有合成档位，改用**真实质量**分层构造「档位」。\n")

    # 特征分组：用于「分层变量与预测量不相交」的对照
    GEOM = ["width", "height", "aspect_ratio", "center_offset",
            "projection_ap", "projection_unknown"]
    NONGEOM = [c for c in QC_FEATS if c not in GEOM]

    def make_axis(cols, name):
        X = np.nan_to_num(d[[f"qc_{c}" for c in cols]].to_numpy(float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Xs = StandardScaler().fit_transform(X)
        p = PCA(n_components=1, random_state=0).fit(Xs)
        return p.transform(Xs)[:, 0], name

    def strat_stats(g, pred_cols):
        yy = (g["label"].to_numpy() == 1).astype(int)
        pp = pos_prob(g[C1].to_numpy(np.float64), T1)
        e = ((pp >= 0.5).astype(int) != yy).astype(int)
        if e.sum() < 20 or len(np.unique(e)) < 2:
            return None
        cf = np.maximum(pp, 1 - pp)
        Xq = np.nan_to_num(g[[f"qc_{c}" for c in pred_cols]].to_numpy(float),
                           nan=0.0, posinf=0.0, neginf=0.0)
        rec = dict(n=len(yy), pos=int(yy.sum()), err=float(e.mean()),
                   auc_err_conf=safe_auc(e, -cf),
                   auc_err_qr=safe_auc(e, g["qc_quality_risk"].to_numpy(float)))
        for lr in ("lr", "gb"):
            b, bo = [], []
            for s in range(SEEDS):
                b.append(safe_auc(e, cv_oof(cf[:, None], e, lr, s)))
                bo.append(safe_auc(e, cv_oof(np.c_[cf, Xq], e, lr, s)))
            rec[f"auc_base_{lr}"] = float(np.nanmean(b))
            rec[f"auc_both_{lr}"] = float(np.nanmean(bo))
            rec[f"delta_{lr}"] = rec[f"auc_both_{lr}"] - rec[f"auc_base_{lr}"]
        return rec

    law_tables = {}
    axes = []
    axes.append((*make_axis(QC_FEATS, "pc1_全部17维"), QC_FEATS,
                 "分层与预测用同一批特征"))
    axes.append((*make_axis(GEOM, "pc1_几何6维"), NONGEOM,
                 "分层用几何/格式 6 维，预测用其余 11 维（不相交）"))
    axes.append((*make_axis(NONGEOM, "pc1_强度11维"), GEOM,
                 "分层用强度/伪影 11 维，预测用几何 6 维（不相交）"))
    # 手工聚合轴：与 CheXpert 的「降质档位」最可比的一种分层
    axes.append((d["qc_quality_risk"].to_numpy(float), "手工quality_risk", QC_FEATS,
                 "9 个布尔判据取均值（与 7 个预测量重叠，仅供与档位框架对照）"))

    for vals, key, pred_cols, desc in axes:
        d = d.assign(_axis=vals)
        try:
            bins = pd.qcut(d["_axis"], 5, labels=False, duplicates="drop")
        except ValueError:
            print(f"[跳过] {key} 分箱失败\n")
            continue
        rows = []
        print(f"—— 质量轴：{key} ——  {desc}")
        print(f"   （预测量：{len(pred_cols)} 维）")
        print(f"{'层':<4}{'n':>7}{'阳性':>7}{'错误率':>8}{'置信度AUC':>11}"
              f"{'ΔAUC(GB)':>11}{'ΔAUC(LR)':>11}")
        for s in sorted(pd.Series(bins).dropna().unique()):
            g = d[bins == s]
            r = strat_stats(g, pred_cols)
            if r is None:
                continue
            r["stratum"] = int(s)
            rows.append(r)
            print(f"{int(s):<4}{r['n']:>7}{r['pos']:>7}{r['err']:>8.3f}"
                  f"{r['auc_base_gb']:>11.4f}{r['delta_gb']:>+11.4f}{r['delta_lr']:>+11.4f}")
        t = pd.DataFrame(rows)
        law_tables[key] = t
        if len(t) >= 3:
            for lr in ("gb", "lr"):
                r_ = np.corrcoef(t[f"auc_base_{lr}"], t[f"delta_{lr}"])[0, 1]
                print(f"   跨层相关 r(置信度AUC, ΔAUC) = {r_:+.3f}  [{lr}]")
        print()
    for k, t in law_tables.items():
        t.to_csv(os.path.join(args.results, f"table_nih_law_{k}.csv"), index=False)

    # 不分层的整库增量：这是与 CheXpert「整档」唯一真正可比的口径
    # （分层是在质量轴上选出极端子集，与「对全部图像施加同一算子」不是同一种分组）
    whole = strat_stats(d, QC_FEATS)
    pd.DataFrame([whole]).to_csv(
        os.path.join(args.results, "table_nih_law_whole.csv"), index=False)
    print("—— 不分层的整库口径（与 CheXpert 整档可比）——")
    print(f"   n={whole['n']}  错误率={whole['err']:.3f}  "
          f"置信度AUC={whole['auc_base_gb']:.4f}  "
          f"ΔAUC(GB)={whole['delta_gb']:+.4f}  ΔAUC(LR)={whole['delta_lr']:+.4f}")
    print(f"   CheXpert 干净档同口径：置信度AUC=0.8078  ΔAUC(GB)=+0.0018\n")

    # ------------------------------------------------- 4 风险–覆盖率
    print("-" * 100)
    print("4 风险–覆盖率（按置信度复核）与全量质量分布")
    print("-" * 100)
    order = np.argsort(-conf)          # 置信度降序：最自信的在前
    rc_rows = []
    for cov in (0.25, 0.50, 0.75):
        k = int(round(len(d) * cov))   # 复核掉最不确定的 k 张
        keep = order[:len(d) - k]      # 剩下的是最自信的 1−cov
        rev = order[len(d) - k:]
        rc_rows.append(dict(reviewed=cov, coverage=1 - cov, n_kept=len(keep),
                            err_all=float(err.mean()),
                            err_kept=float(err[keep].mean()),
                            ppr_kept=float(pred[keep].mean()),
                            err_captured=float(err[rev].sum() / max(err.sum(), 1))))
        print(f"  复核最不确定的 {cov:.0%}：剩余错误率 {err[keep].mean():.4f}"
              f"（不复核 {err.mean():.4f}），该批次捕获了全部错误的 "
              f"{err[rev].sum()/max(err.sum(),1):.1%}")
    pd.DataFrame(rc_rows).to_csv(
        os.path.join(args.results, "table_nih_riskcov.csv"), index=False)

    if args.chexpert_qc and os.path.exists(args.chexpert_qc):
        cq = pd.read_csv(args.chexpert_qc, usecols=["qc_quality_risk"])
        cmp_rows = []
        for nm, s in (("CheXpert route 干净档（224）", cq["qc_quality_risk"]),
                      ("NIH ChestX-ray14（224）", d["qc_quality_risk"])):
            cmp_rows.append(dict(dataset=nm, n=len(s), mean=float(s.mean()),
                                 sd=float(s.std()), p05=float(s.quantile(.05)),
                                 p50=float(s.quantile(.50)), p95=float(s.quantile(.95)),
                                 nunique=int(s.nunique())))
        cdf = pd.DataFrame(cmp_rows)
        cdf.to_csv(os.path.join(args.results, "table_nih_qc_dist.csv"), index=False)
        print("\n质量分布对照（quality_risk，两域均在 224 上计算）")
        print(cdf.to_string(index=False))

    print("\n完成，结果写入", args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
