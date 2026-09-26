"""诊断：现有质量信号能否检测「真实物理降质」？

动机
----
chexphoto_analyze.py 的表 1 显示：真实翻拍（natural/oneplus）的
`qc_quality_risk` 均值 = 0.2041，**低于**干净原图的 0.2431。
也就是说，当前这套为合成降质校准出来的质量轴，在真实降质上给出的是
最低风险值之一 —— 如果直接拿它做路由，会把最需要人工复核的图像放行。

本脚本回答三个问题：
  Q1  17 维 QC 特征能否把「干净原图」与「真实翻拍」分开？（逐个特征 AUROC）
  Q2  翻转 TTA 分歧 |p − p_flip| 是不是一个更好的零标注风险信号？
  Q3  分数本身的分布统计（熵、最大概率）有没有信号？

口径：患者级；AUROC 一律以「真实降质=1」为正类。

用法：
    python chexphoto_qc_diag.py --preds /root/chexphoto/out/preds.csv --out results/

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
    a = float(roc_auc_score(y[ok], s[ok]))
    return max(a, 1 - a)          # 方向无关的判别力


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
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preds)
    fr = df[df["orient"] == "frontal"].copy()
    fr["y"] = fr["label"].astype(int)
    fr["p"] = softmax_pos(fr[["l1_0", "l1_1", "l1_2"]].to_numpy(float))
    fr["p_flip"] = softmax_pos(fr[["l1f_0", "l1f_1", "l1f_2"]].to_numpy(float))
    fr["tta_gap"] = (fr["p"] - fr["p_flip"]).abs()
    ent = -(fr["p"] * np.log(fr["p"].clip(1e-12)) +
            (1 - fr["p"]) * np.log((1 - fr["p"]).clip(1e-12)))
    fr["entropy"] = -(
        (lambda P: (P * np.log(P.clip(1e-12))).sum(axis=1))(
            torch.softmax(torch.from_numpy(fr[["l1_0", "l1_1", "l1_2"]]
                                           .to_numpy(float) / T1), dim=1).numpy()))
    fr["maxprob"] = np.maximum(fr["p"], 1 - fr["p"])

    # ------------------------------------------------------------- 分布
    print("=" * 92)
    print("Q0  各条件风险信号的分布（均值 / 中位 / p90）")
    print("=" * 92)
    sig = ["qc_quality_risk", "tti_placeholder"] if False else ["qc_quality_risk"]
    cols = ["qc_quality_risk", "tta_gap", "entropy", "p"]
    print(f"{'条件':<14}{'n':>5}" + "".join(f"{c:>26}" for c in cols))
    print(f"{'':<14}{'':>5}" + "".join(f"{'均值      中位      p90':>26}" for _ in cols))
    for c in SHORT:
        g = fr[fr["cond"] == c]
        line = f"{SHORT[c]:<14}{len(g):>5}"
        for col in cols:
            v = g[col].to_numpy()
            line += f"{v.mean():>10.4f}{np.median(v):>9.4f}{np.percentile(v,90):>7.4f}"
        print(line)

    # ------------------------------------------------------------- Q1
    print("\n" + "=" * 92)
    print("Q1  检测「真实降质」的能力：干净原图(=0) vs 真实翻拍(=1)，n=404")
    print("=" * 92)
    d = fr[fr["cond"].isin(["clean", "natural/oneplus"])].copy()
    y = (d["cond"] == "natural/oneplus").astype(int).to_numpy()

    print(f"\n  单个 QC 特征（方向无关判别力，>0.5 即有效）：")
    single = []
    for col in COLS:
        a = auroc(y, d[col].to_numpy(float))
        single.append((col.replace("qc_", ""), a))
    for nm, a in sorted(single, key=lambda t: -(t[1] if np.isfinite(t[1]) else 0)):
        if np.isfinite(a):
            print(f"    {nm:<22} {a:.4f}")

    print(f"\n  多特征组合（5 折 OOF）：")
    X = d[COLS].to_numpy(float)
    for lr in ("gb", "lr"):
        aucs = [auroc(y, cv_oof(X, y, lr, s)) for s in range(5)]
        print(f"    17 维 QC 特征（{lr.upper()}）        {np.nanmean(aucs):.4f}")

    # ------------------------------------------------------------- Q2
    print("\n" + "=" * 92)
    print("Q2  零标注替代信号：翻转 TTA 分歧")
    print("=" * 92)
    for lr in ("gb", "lr"):
        aucs = [auroc(y, cv_oof(d[["tta_gap"]].to_numpy(float), y, lr, s)) for s in range(5)]
        print(f"    仅 tta_gap（{lr.upper()}）            {np.nanmean(aucs):.4f}")
    print(f"    仅 tta_gap 单值（方向无关）        {auroc(y, d['tta_gap'].to_numpy(float)):.4f}")

    aucs = [auroc(y, cv_oof(d[["tta_gap", "entropy", "maxprob"]].to_numpy(float), y, lr, s))
            for lr in ("gb",) for s in range(5)]
    print(f"    tta_gap+熵+最大概率（GB）          {np.nanmean(aucs):.4f}")

    aucs = [auroc(y, cv_oof(np.c_[X, d[["tta_gap", "entropy", "maxprob"]].to_numpy(float)],
                            y, "gb", s)) for s in range(5)]
    print(f"    QC + 全部不确定度信号（GB）        {np.nanmean(aucs):.4f}")

    # ------------------------------------------------------------- Q3
    print("\n" + "=" * 92)
    print("Q3  tta_gap 与「模型判错」的关系（全院 808 行 frontal）")
    print("=" * 92)
    err = ((fr["p"] >= 0.5).astype(int) != fr["y"]).astype(int).to_numpy()
    Xe = np.c_[fr["tta_gap"].to_numpy(), fr["entropy"].to_numpy(),
               fr["maxprob"].to_numpy(), fr["qc_quality_risk"].to_numpy()]
    for nm, sel in (("tta_gap 单值", Xe[:, [0]]),
                    ("tta_gap+熵+最大概率", Xe[:, :3]),
                    ("QC + 不确定度", Xe)):
        aucs = [auroc(err, cv_oof(sel, err, "gb", s)) for s in range(5)]
        print(f"    {nm:<26} 预测错误的 AUROC = {np.nanmean(aucs):.4f}")

    # ------------------------------------------------------------- 汇总
    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    qr_r = fr.groupby("cond")["qc_quality_risk"].mean()
    tg_r = fr.groupby("cond")["tta_gap"].mean()
    print(f"  {'条件':<14}{'quality_risk':>14}{'tta_gap':>12}")
    for c in SHORT:
        print(f"  {SHORT[c]:<14}{qr_r[c]:>14.4f}{tg_r[c]:>12.4f}")
    fr[["base_key", "cond", "Patient", "y", "p", "p_flip", "tta_gap",
        "qc_quality_risk"]].to_csv(args.out / "diag_signals.csv", index=False)
    print(f"\n  -> {args.out}/diag_signals.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
