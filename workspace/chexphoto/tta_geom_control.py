"""方向二补充（控制实验）：几何 TTA 的增益是否只是「变相重新使用了 p」？

动机
----
实验 A（tta_geom_analyze.py）给出：真实翻拍条件下，TTA 分歧对软错误的 AUROC
从单次翻转的 0.682 升到几何视图 K=12 的 0.787（+0.105）。

但同一批数据上的正交性检查显示，分歧与主分数 p 的 Spearman 秩相关**同时**
从 0.510 升到 0.815。这带出一个必须排除的解释：增益来自「分歧变得更像 p」，
而非新增了与 p 无关的独立信息。

CheXphoto 报告 §3.4 的修正二把「与 p 正交」列为零标注信号入选的前提，
因此本节是结论能否成立的关键检验。三件事：

1. 基线：AUROC(p → 软错误)。若它本身已接近 0.787，几何 TTA 就没有贡献。
   同时报告两个常见的「不确定度」信号作对照。
2. 增量：梯度提升下 {p} 与 {p, div} 的 5 折 OOF AUROC 之差。这是「控制了 p
   之后，div 还能贡献多少」的直接度量。
3. 层内：把样本按 p 分五层，在每层内部算 div 的 AUROC 再加权平均。
   层内 AUROC 对 p 的严格单调变换不敏感，因此是「在同等的 p 条件下」比较，
   不受 ρ_p 高低的干扰。

用法
----
    python tta_geom_control.py --tta results/tta_geom.csv --out results/

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
CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}
VIEWS = [f"v{i:02d}" for i in range(16)]
K_USE = 12                      # 各条件在 K=12 均接近其峰值
BASELINE_FLIP = {"clean": 0.899, "natural/oneplus": 0.682,
                 "synthetic/photographic": 0.876, "synthetic/digital": 0.898}


def softmax_pos(L, T=T1):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T)
    return torch.softmax(X, dim=1).numpy()[:, 1]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def split_bin(y_cont):
    return (np.asarray(y_cont, float) > float(np.median(y_cont))).astype(int)


def cv_oof(X, y, seed, n_splits=5):
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


def within_strata_auc(p, s, y, n_strata=5):
    """按 p 分层的层内 AUROC（层样本量加权）。只保留层内两类都存在的层。"""
    edges = np.quantile(p, np.linspace(0, 1, n_strata + 1))
    idx = np.digitize(p, edges[1:-1])
    num, den = 0.0, 0.0
    for b in range(n_strata):
        m = idx == b
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        num += roc_auc_score(y[m], s[m]) * m.sum()
        den += m.sum()
    return (num / den if den > 0 else np.nan), int(den)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tta)
    df["y"] = df["label"].astype(int)
    P = {v: softmax_pos(df[[f"{v}_0", f"{v}_1", f"{v}_2"]].to_numpy(float))
         for v in VIEWS}
    df["p"] = P["v00"]
    df["soft_err"] = np.where(df["y"] == 1, 1.0 - df["p"], df["p"])

    print("=" * 104)
    print(f"控制实验：在控制了主分数 p 之后，几何 TTA 分歧（K={K_USE}）还剩多少独立信息？")
    print("=" * 104)
    print("  目标：软错误 = 1 − p(真实标签)，中位数切分（口径同 §3.4）\n")

    rows = []
    hdr = (f"{'条件':<14}{'AUROC(p)':>10}{'AUROC(−p)':>11}{'AUROC(不确定度)':>16}"
           f"{'AUROC(div)':>12}{'层内(div|p)':>13}{'{p}':>8}{'{p,div}':>9}{'增量':>8}")
    print(hdr)
    print("-" * 104)

    for c in CONDS:
        mm = (df["cond"] == c).to_numpy()
        yc = df.loc[mm, "soft_err"].to_numpy(float)
        y = split_bin(yc)
        p = df.loc[mm, "p"].to_numpy(float)
        Pm = np.stack([P[v][mm] for v in VIEWS[:K_USE]], axis=1)
        div = Pm.std(axis=1)

        ap_pos = auroc(y, p)
        ap_neg = auroc(y, -p)
        a_unc = auroc(y, 1.0 - np.maximum(p, 1.0 - p))
        a_div = auroc(y, div)
        a_strata, n_used = within_strata_auc(p, div, y, n_strata=5)

        oof_p = np.nanmean([cv_oof(p[:, None], y, s) for s in range(5)], axis=0)
        oof_b = np.nanmean([cv_oof(np.c_[p, div], y, s) for s in range(5)], axis=0)
        a_p_l = auroc(y, oof_p)
        a_b_l = auroc(y, oof_b)
        gain = a_b_l - a_p_l

        print(f"{SHORT[c]:<14}{ap_pos:>10.3f}{ap_neg:>11.3f}{a_unc:>16.3f}"
              f"{a_div:>12.3f}{a_strata:>13.3f}{a_p_l:>8.3f}{a_b_l:>9.3f}{gain:>+8.3f}")
        rows.append({"cond": c, "auroc_p": ap_pos, "auroc_negp": ap_neg,
                     "auroc_unc": a_unc, "auroc_div": a_div,
                     "auroc_div_within_p": a_strata, "n_strata": n_used,
                     "auroc_gb_p": a_p_l, "auroc_gb_p_div": a_b_l,
                     "gain_gb": gain,
                     "baseline_flip": BASELINE_FLIP[c],
                     "gain_vs_flip": a_div - BASELINE_FLIP[c]})

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "table_tta_geom_control.csv", index=False)

    print("\n" + "=" * 104)
    print("读法")
    print("=" * 104)
    for _, r in t.iterrows():
        verdict = []
        if np.isfinite(r["auroc_p"]) and r["auroc_p"] > 0.75:
            verdict.append("p 本身就能排序软错误")
        else:
            verdict.append("p 单独几乎无排序能力")
        if np.isfinite(r["auroc_div_within_p"]) and r["auroc_div_within_p"] > 0.60:
            verdict.append("层内仍有信息")
        else:
            verdict.append("层内退化到接近随机")
        print(f"  {r['cond']:<24} div={r['auroc_div']:.3f} vs 层内={r['auroc_div_within_p']:.3f}"
              f"  |  " + "；".join(verdict))

    print(f"\n  -> {args.out}/table_tta_geom_control.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
