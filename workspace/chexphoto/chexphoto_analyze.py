"""CheXphoto v1.0 配对真实降质分析。

设计
----
202 张 frontal 胸片（200 患者），每张 4 个条件：
    clean                  本地 CheXpert-v1.0-small/valid 原图
    natural/oneplus        真实手机翻拍   ← 真实物理降质
    synthetic/photographic 合成「翻拍」降质
    synthetic/digital      合成「数字」降质
标签在四条件间完全相同（已验证），因此可做**配对**比较：
每张图自己当自己的对照，图像难度这个混杂因素被消掉。

口径（与 v1.5 设计 A / NIH 外部验证完全一致）
------------------------------------------
* 类别顺序 [0: negative, 1: positive, 2: uncertain]，正类 = logit 索引 1
* 冻结温度 T1 = 0.4943421483039856，操作点阈值 0.5
* ECE 15 箱；代价 = (2*FN + FP)/N（FN:FP = 2:1）
* 患者级整群 Bootstrap

用法：
    python chexphoto_analyze.py --preds /tmp/chexphoto_preds.csv --out results/

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

T1 = 0.4943421483039856
PI_SOURCE = 0.126
QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]

CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}


# --------------------------------------------------------------------------
def softmax_pos(L, T=T1, delta=0.0):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T).clone()
    X[:, 1] = X[:, 1] + delta
    return torch.softmax(X, dim=1).numpy()[:, 1]


def ece(y, p, n_bins=15):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    n = len(y)
    return {
        "n": n, "auroc": safe_auc(y, p), "err": float((pred != y).mean()),
        "ece": ece(y, p), "cost": (2 * fn + fp) / n,
        "ppr": float(pred.mean()), "tpr": tp / max(int((y == 1).sum()), 1),
        "fn": fn, "fp": fp,
    }


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


def paired_bootstrap(df, stat_fn, n_boot=600, seed=0):
    """患者级配对 Bootstrap：抽患者，整簇（含四个条件）一起进出。"""
    rng = np.random.default_rng(seed)
    gid = df.groupby("Patient", sort=False).ngroup().to_numpy()
    n_g = gid.max() + 1
    # 每个患者 -> 行号
    rows_by_g = [np.flatnonzero(gid == g) for g in range(n_g)]
    out = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
        idx = np.concatenate([np.repeat(rows_by_g[g], cnt[g])
                              for g in range(n_g) if cnt[g] > 0])
        out.append(stat_fn(df.iloc[idx]))
    return np.asarray(out, float)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--boot", type=int, default=600)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preds)
    fr = df[df["orient"] == "frontal"].copy()
    fr["y"] = fr["label"].astype(int)
    fr["p"] = softmax_pos(fr[["l1_0", "l1_1", "l1_2"]].to_numpy(float))
    print(f"frontal 子集: {len(fr)} 行 / {fr['base_key'].nunique()} 张图 / "
          f"{fr['Patient'].nunique()} 患者，阳性率 {fr['y'].mean():.4f}\n")

    # ---------------------------------------------------------------- 表 1
    rows = []
    for c in CONDS:
        g = fr[fr["cond"] == c]
        m = metrics(g["y"].to_numpy(int), g["p"].to_numpy())
        m["cond"] = c
        m["qr"] = float(g["qc_quality_risk"].mean())
        rows.append(m)
    tab1 = pd.DataFrame(rows)[
        ["cond", "n", "auroc", "err", "ece", "cost", "ppr", "tpr", "fn", "fp", "qr"]]
    print("=" * 88)
    print("表 1  四条件总体表现（T=0.4943，阈值 0.5，全 202 张）")
    print("=" * 88)
    print(tab1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------- 配对差
    print("\n" + "=" * 88)
    print("表 2  配对差（相对干净原图）与患者级配对 Bootstrap 95% CI")
    print("=" * 88)

    def stat(dfb):
        o = []
        for c in CONDS:
            g = dfb[dfb["cond"] == c]
            y = g["y"].to_numpy(int); p = g["p"].to_numpy()
            o += [metrics(y, p)["auroc"], metrics(y, p)["err"], metrics(y, p)["cost"]]
        return o

    boot = paired_bootstrap(fr, stat, n_boot=args.boot)
    base = np.asarray(stat(fr))
    prow = []
    for ci, c in enumerate(CONDS):
        for ki, kn in enumerate(["auroc", "err", "cost"]):
            v = base[ci * 3 + ki]
            if c == "clean":
                continue
            d = v - base[ki]           # clean 是该指标的第 0 组
            bd = boot[:, ci * 3 + ki] - boot[:, ki]
            lo, hi = np.percentile(bd, [2.5, 97.5])
            prow.append({"对比": f"{SHORT[c]} − 干净", "指标": kn,
                         "值": v, "干净": base[ki], "差值": d,
                         "CI下": lo, "CI上": hi,
                         "显著": "*" if (lo > 0 or hi < 0) else ""})
    tab2 = pd.DataFrame(prow)
    print(tab2.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    tab2.to_csv(args.out / "table_cp_paired_delta.csv", index=False)

    # ---------------------------------------------------------------- 表 3
    print("\n" + "=" * 88)
    print("表 3  操作点塌缩（预测阳性率 / 真实患病率）")
    print("=" * 88)
    c0 = tab1.set_index("cond")
    for c in CONDS:
        t = fr.loc[fr["cond"] == c, "y"].mean()
        p_ = c0.loc[c, "ppr"]
        print(f"  {SHORT[c]:10s}  真实 {t:.4f}  预测 {p_:.4f}  ×{p_/max(t,1e-9):.2f}")

    # ---------------------------------------------------------------- 表 4：定律
    print("\n" + "=" * 88)
    print("表 4  「何时有用」定律：各条件下的置信度失效程度与质量特征增量")
    print("=" * 88)
    law = []
    for c in CONDS:
        g = fr[fr["cond"] == c]
        y = g["y"].to_numpy(int)
        p = g["p"].to_numpy()
        e = ((p >= 0.5).astype(int) != y).astype(int)
        rec = {"cond": c, "auc_base": safe_auc(e, np.maximum(p, 1 - p))}
        Xq = g[[f"qc_{k}" for k in QC_FEATS]].to_numpy(float)
        for lr in ("gb", "lr"):
            b, bo = [], []
            for s in range(5):
                b.append(safe_auc(e, cv_oof(np.maximum(p, 1 - p)[:, None], e, lr, s)))
                bo.append(safe_auc(e, cv_oof(np.c_[np.maximum(p, 1 - p), Xq], e, lr, s)))
            rec[f"base_{lr}"] = float(np.nanmean(b))
            rec[f"both_{lr}"] = float(np.nanmean(bo))
            rec[f"delta_{lr}"] = rec[f"both_{lr}"] - rec[f"base_{lr}"]
        law.append(rec)
    tab4 = pd.DataFrame(law)
    print(tab4.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tab4.to_csv(args.out / "table_cp_law.csv", index=False)

    # ---------------------------------------------------------------- 表 5
    print("\n" + "=" * 88)
    print("表 5  QC 特征能否区分「真实降质」与「合成降质」")
    print("=" * 88)
    deg = fr[fr["cond"] != "clean"].copy()
    deg["is_real"] = (deg["cond"] == "natural/oneplus").astype(int)
    qcols = [f"qc_{k}" for k in QC_FEATS]
    ccols = [c for c in deg.columns if c.startswith("col_")]
    for nm, cols in (("仅 17 维 QC 特征", qcols),
                     ("仅 5 维色彩特征", ccols),
                     ("QC + 色彩", qcols + ccols)):
        X = deg[cols].to_numpy(float); y = deg["is_real"].to_numpy()
        aucs = []
        for s in range(5):
            oof = cv_oof(X, y, "gb", s)
            aucs.append(safe_auc(y, oof))
        print(f"  {nm:18s} 区分真实/合成的 AUROC = {np.nanmean(aucs):.4f}"
              f"  (n={len(y)}, 真实 {y.sum()} / 合成 {(1-y).sum()})")
    print("\n  各条件 quality_risk 与色彩特征均值：")
    agg = fr.groupby("cond")[["qc_quality_risk", "qc_noise_score", "qc_contrast",
                              "qc_blur_score", "col_sat_mean", "col_rb_gap"]].mean()
    agg.index = [SHORT[c] for c in agg.index]
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))

    tab1.to_csv(args.out / "table_cp_overall.csv", index=False)
    print(f"\n结果已写入 {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
