"""H1 负结果的稳健性：条件内增量 ≈ 0 是不是「学习器太弱」造成的？

审稿人最可能的攻击：你用一个 logistic 回归去看质量特征有没有增量，
线性模型当然抓不到非线性的质量→错误关系。若换成强学习器增量就出来了，
你的 H1 负结论就是假的。

本脚本把学习器换成梯度提升（HistGradientBoosting），并把 5 折 CV 重复多次
以给出 ΔAUC 的区间；若强学习器下条件内增量仍≈0，负结论才站得住。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def cv_oof(X, y, learner, seed):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for cond in args.conds.split(","):
        f = os.path.join(args.outdir, f"{cond}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d["label"] != 2].reset_index(drop=True)
        yy = (d["label"].to_numpy() == 1).astype(int)
        L = d[["l1_0", "l1_1", "l1_2"]].to_numpy(float) / args.t1
        e = np.exp(L - L.max(1, keepdims=True))
        p = e[:, 1] / e.sum(1)
        err = ((p >= 0.5).astype(int) != yy).astype(int)
        if err.sum() < 20 or len(np.unique(err)) < 2:
            continue
        conf = np.maximum(p, 1 - p)
        qcols = [f"qc_{c}" for c in QC_FEATS if f"qc_{c}" in d.columns]
        Xq = np.nan_to_num(d[qcols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)

        rec = dict(cond=cond, n=len(yy), err=float(err.mean()))
        for learner in ("lr", "gb"):
            base, both = [], []
            for s in range(args.seeds):
                a = cv_oof(conf[:, None], err, learner, s)
                b = cv_oof(np.c_[conf, Xq], err, learner, s)
                base.append(safe_auc(err, a)); both.append(safe_auc(err, b))
            rec[f"auc_base_{learner}"] = float(np.nanmean(base))
            rec[f"auc_both_{learner}"] = float(np.nanmean(both))
            dlt = np.array(both) - np.array(base)
            rec[f"delta_{learner}"] = float(np.nanmean(dlt))
            rec[f"delta_lo_{learner}"] = float(np.nanpercentile(dlt, 2.5))
            rec[f"delta_hi_{learner}"] = float(np.nanpercentile(dlt, 97.5))
        rows.append(rec)
        print(f"{cond:<12} n={len(yy):>5} err={err.mean():.3f} | "
              f"LR  Δ={rec['delta_lr']:+.4f}[{rec['delta_lo_lr']:+.4f},{rec['delta_hi_lr']:+.4f}]  "
              f"GB  Δ={rec['delta_gb']:+.4f}[{rec['delta_lo_gb']:+.4f},{rec['delta_hi_gb']:+.4f}]",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, f"table_h1_robust_{args.tag}.csv"), index=False)
    print(f"\n汇总 {args.tag}（{len(out)} 个条件，条件内 5 折 CV × {args.seeds} seed）")
    print(f"  置信度单独 AUC：LR {out['auc_base_lr'].mean():.4f} | GB {out['auc_base_gb'].mean():.4f}")
    print(f"  置信度+质量  AUC：LR {out['auc_both_lr'].mean():.4f} | GB {out['auc_both_gb'].mean():.4f}")
    print(f"  条件内增量 ΔAUC： LR {out['delta_lr'].mean():+.4f} | GB {out['delta_gb'].mean():+.4f}")
    print(f"  Δ>0 的条件数：   LR {int((out['delta_lr'] > 0).sum())}/{len(out)} | "
          f"GB {int((out['delta_gb'] > 0).sum())}/{len(out)}")


if __name__ == "__main__":
    main()
