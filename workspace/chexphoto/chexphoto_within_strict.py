"""严格版：真实降质内部，与分数正交的零标注信号还有多少路由价值？

对前一版（chexphoto_within.py）的两处修正
----------------------------------------
修正 1 — 目标不能是「硬错误」。
    真实翻拍条件下模型的 202 张全部被判为阴性，于是「硬错误」恰好等于
    「阳性」。用任何分数去预测硬错误，就退化成用该分数预测阳性，是循环的
    （证据：该条件下 auroc(1−maxprob → 硬错误) = 0.6592，与 auroc(p → 标签)
    完全相同）。本脚本改用**软错误**作为目标：

        soft_err = 1 − p(真实标签)
                 = 1 − p  若 y = 1
                 = p      若 y = 0

    它不依赖任何决策阈值，在所有条件下都有定义，也不与操作点塌缩耦合。

修正 2 — 特征必须与分数正交。
    预测熵与最大概率都是 p 的单调函数，不是独立信号。真正与 p 正交的
    零标注信号只有两类：
        (a) 翻转 TTA 分歧 |p − p_flip|  —— 自洽性
        (b) 17 维 QC 特征               —— 图像本身的物理质量
    本脚本只用这两组，逐条件评估其对软错误的排序能力。

不确定性用**患者级整群 Bootstrap**（重采样患者，而非图像），
在已固定的 OOF 预测上计算，因此反映的是采样不确定性，不含重训练方差。

用法：
    python chexphoto_within_strict.py --preds /root/chexphoto/out/preds.csv --out results/

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T1 = 0.4943421483039856
QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]
COLS = [f"qc_{k}" for k in QC_FEATS]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}


def softmax_pos(L, T=T1, delta=0.0):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T).clone()
    X[:, 1] = X[:, 1] + delta
    return torch.softmax(X, dim=1).numpy()[:, 1]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def spearman(a, b):
    """秩相关，用于连续目标（软错误）。与 AUROC 同族但不需要二值化。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def split_bin(y_cont):
    """软错误中位数切分，得到接近平衡的二分类目标，便于沿用 AUROC 口径。"""
    med = float(np.median(y_cont))
    return (np.asarray(y_cont) > med).astype(int)


def cv_oof(X, y, seed, learner="gb", n_splits=5):
    """患者分组交叉验证由调用方负责（此处按行分层）。"""
    oof = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return oof
    n_splits = min(n_splits, int(np.bincount(y.astype(int)).min()))
    if n_splits < 2:
        return oof
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                           max_leaf_nodes=15, min_samples_leaf=20,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def cluster_boot_auc(y, s, groups, n_boot=800, seed=0):
    """患者级整群 Bootstrap，在固定预测上重算 AUROC。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, s[idx]))
    if len(out) < 50:
        return np.nan, np.nan
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preds)
    fr = df[df["orient"] == "frontal"].copy().reset_index(drop=True)
    fr["y"] = fr["label"].astype(int)
    L = fr[["l1_0", "l1_1", "l1_2"]].to_numpy(float)
    fr["p"] = softmax_pos(L)
    fr["p_flip"] = softmax_pos(fr[["l1f_0", "l1f_1", "l1f_2"]].to_numpy(float))
    fr["tta_gap"] = (fr["p"] - fr["p_flip"]).abs()
    # 软错误：模型对真实标签的概率亏损，不依赖阈值
    fr["soft_err"] = np.where(fr["y"] == 1, 1.0 - fr["p"], fr["p"])
    fr["hard_err"] = ((fr["p"] >= 0.5).astype(int) != fr["y"]).astype(int)

    print("=" * 100)
    print("0  软错误 vs 硬错误：验证「全判阴」条件下硬错误退化的问题")
    print("=" * 100)
    print(f"{'条件':<14}{'n':>5}{'阳性':>6}{'判正数':>7}{'硬错误':>8}{'软错误均值':>12}"
          f"{'硬错误≡阳性?':>14}")
    for c in SHORT:
        g = fr[fr["cond"] == c]
        same = "是" if (g["hard_err"] == g["y"]).all() else "否"
        print(f"{SHORT[c]:<14}{len(g):>5}{int(g['y'].sum()):>6}"
              f"{int((g['p']>=0.5).sum()):>7}{int(g['hard_err'].sum()):>8}"
              f"{g['soft_err'].mean():>12.4f}{same:>14}")

    # ------------------------------------------------------------ 主分析
    print("\n" + "=" * 100)
    print("1  主分析：零标注信号对**软错误**的排序能力（逐条件，患者级聚类 Bootstrap 95% CI）")
    print("=" * 100)
    print("   AUROC 用软错误的中位数切分（≈平衡），口径与前述表格一致；")
    print("   ρ 为对连续软错误的 Spearman 秩相关（无需切分）。")
    print("   特征组： TTA = 翻转分歧 |p−p_flip|（单值，无需训练）；"
          "QC = 17 维质量特征（5 折 OOF）；TTA+QC = 拼接（5 折 OOF）")
    print(f"\n{'条件':<14}{'n':>5}   {'TTA 单值':>24}{'QC 17 维':>24}{'TTA+QC':>24}")
    print(f"{'':<14}{'':>5}   " + "".join(f"{'AUROC        ρ':>24}" for _ in range(3)))

    rows = []
    for c in SHORT:
        g = fr[fr["cond"] == c].reset_index(drop=True)
        yc = g["soft_err"].to_numpy(float)      # 连续
        y = split_bin(yc)                       # 中位数切分
        grp = g["Patient"].to_numpy()

        # TTA 单值（无需训练，直接可用）
        s_tta = g["tta_gap"].to_numpy(float)
        a_tta = auroc(y, s_tta)
        r_tta = spearman(s_tta, yc)
        lo_t, hi_t = cluster_boot_auc(y, s_tta, grp)

        # QC 17 维 OOF（5 个种子取均值）
        oof_q = np.nanmean([cv_oof(g[COLS].to_numpy(float), y, s) for s in range(5)], axis=0)
        a_q = auroc(y, oof_q)
        r_q = spearman(oof_q, yc)
        lo_q, hi_q = cluster_boot_auc(y, oof_q, grp)

        # TTA + QC
        Xb = np.c_[g[COLS].to_numpy(float), s_tta]
        oof_b = np.nanmean([cv_oof(Xb, y, s) for s in range(5)], axis=0)
        a_b = auroc(y, oof_b)
        r_b = spearman(oof_b, yc)
        lo_b, hi_b = cluster_boot_auc(y, oof_b, grp)

        def cell(a, r, lo, hi):
            star = "" if not np.isfinite(lo) else ("*" if (lo > 0.5 or hi < 0.5) else " ")
            return f"{a:.3f}[{lo:.2f},{hi:.2f}]{star}{r:+.3f}"

        print(f"{SHORT[c]:<14}{len(g):>5}   {cell(a_tta,r_tta,lo_t,hi_t):>24}"
              f"{cell(a_q,r_q,lo_q,hi_q):>24}{cell(a_b,r_b,lo_b,hi_b):>24}")
        rows.append({"cond": c, "n": len(g), "soft_err_mean": float(yc.mean()),
                     "tta_auc": a_tta, "tta_rho": r_tta, "tta_lo": lo_t, "tta_hi": hi_t,
                     "qc_auc": a_q, "qc_rho": r_q, "qc_lo": lo_q, "qc_hi": hi_q,
                     "both_auc": a_b, "both_rho": r_b, "both_lo": lo_b, "both_hi": hi_b})
        pd.DataFrame(rows).to_csv(args.out / "table_cp_within_strict.csv", index=False)

    print("\n  * = AUROC 的 95% CI 不含 0.5（显著优于随机）")

    # ------------------------------------------------------------ 对照
    print("\n" + "=" * 100)
    print("2  对照：同一目标、同一信号，在四个条件间的落差")
    print("=" * 100)
    t = pd.DataFrame(rows).set_index("cond")
    print(f"  {'条件':<14}{'TTA 单值':>10}{'QC 17 维':>10}{'TTA+QC':>10}{'相对干净落差(TTA+QC)':>22}")
    for c in SHORT:
        d = t.loc[c, "both_auc"] - t.loc["clean", "both_auc"]
        print(f"  {SHORT[c]:<14}{t.loc[c,'tta_auc']:>10.4f}{t.loc[c,'qc_auc']:>10.4f}"
              f"{t.loc[c,'both_auc']:>10.4f}{d:>+22.4f}")

    # ------------------------------------------------------------ 分片
    print("\n" + "=" * 100)
    print("3  真实翻拍分片：按 quality_risk 与 TTA 分歧各切两半，看软错误是否真的分层")
    print("=" * 100)
    g = fr[fr["cond"] == "natural/oneplus"].reset_index(drop=True)
    for nm, col, hi_is_bad in (("quality_risk", "qc_quality_risk", True),
                               ("TTA 分歧", "tta_gap", True)):
        med = g[col].median()
        lo_g = g[g[col] <= med]; hi_g = g[g[col] > med]
        print(f"  {nm:<14} 低半(n={len(lo_g)}) 软错误 {lo_g['soft_err'].mean():.4f}  |  "
              f"高半(n={len(hi_g)}) 软错误 {hi_g['soft_err'].mean():.4f}  |  "
              f"差 {hi_g['soft_err'].mean()-lo_g['soft_err'].mean():+.4f}")

    print(f"\n  -> {args.out}/table_cp_within_strict.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
