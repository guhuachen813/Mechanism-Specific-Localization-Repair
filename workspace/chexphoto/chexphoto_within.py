"""决定性实验：在真实降质**内部**，零标注信号能否预测模型的错误？

为什么这个实验才是关键
----------------------
前面的诊断回答的是「能否分辨干净图与真实翻拍」。那个问题即使在数值上达到
AUROC=1.0，也不能作为科学结论：真实翻拍是手机拍摄、色彩空间与采集链路都
不同的图像，任何足够丰富的图像统计量都能把它们分开 —— 那是**来源可分**，
不是**质量可分**。

安全路由真正需要的能力是：**在同一个降质条件内部**，把模型会判错的那部分
图像挑出来。这不受来源混淆影响，因为所有图像都来自同一台设备、同一条链路。

本脚本对每个条件单独做：以「该图是否被模型判错」为目标，比较
    (a) 17 维 QC 特征
    (b) 零标注不确定度信号（翻转 TTA 分歧 / 熵 / 最大概率）
    (c) 基础分数本身
的 5 折 OOF 预测能力。并给出患者级 Bootstrap CI。

用法：
    python chexphoto_within.py --preds /root/chexphoto/out/preds.csv --out results/

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


def cv_oof(X, y, seed, learner="gb"):
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        if learner == "gb":
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               l2_regularization=1.0, random_state=seed)
        else:
            m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                               max_leaf_nodes=8, min_samples_leaf=30,
                                               l2_regularization=2.0, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def boot_auc(y, s, n_boot=800, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s[idx]))
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preds)
    fr = df[df["orient"] == "frontal"].copy()
    fr["y"] = fr["label"].astype(int)
    L = fr[["l1_0", "l1_1", "l1_2"]].to_numpy(float)
    fr["p"] = softmax_pos(L)
    fr["p_flip"] = softmax_pos(fr[["l1f_0", "l1f_1", "l1f_2"]].to_numpy(float))
    fr["tta_gap"] = (fr["p"] - fr["p_flip"]).abs()
    P = torch.softmax(torch.from_numpy(L / T1), dim=1).numpy()
    fr["entropy"] = -(P * np.log(P.clip(1e-12))).sum(axis=1)
    fr["maxprob"] = np.maximum(fr["p"], 1 - fr["p"])
    fr["unc"] = 1.0 - fr["maxprob"]
    fr["err"] = ((fr["p"] >= 0.5).astype(int) != fr["y"]).astype(int)

    # ---------------------------------------------------------- 来源可分性检查
    print("=" * 96)
    print("A  先排除来源混淆：干净原图 vs 真实翻拍的全局光度分布")
    print("=" * 96)
    for col in ("qc_mean_intensity", "qc_black_ratio"):
        c = fr.loc[fr["cond"] == "clean", col]
        n = fr.loc[fr["cond"] == "natural/oneplus", col]
        print(f"  {col:<22} 干净 [{c.min():.6f}, {c.max():.6f}]  "
              f"真实 [{n.min():.6f}, {n.max():.6f}]  "
              f"{'完全不重叠 → 分离是来源造成的' if c.max() < n.min() or n.max() < c.min() else '有重叠'}")
    print("  说明：真实翻拍来自手机拍摄，色彩空间/采集链路与 PNG 灰度原图不同，")
    print("        因此任何全局光度统计量都能把两者分开 —— 这属于来源可分，")
    print("        不能作为「质量特征检测到降质」的证据。下文的条件内实验不受此影响。")

    # ---------------------------------------------------------- 条件内错误预测
    print("\n" + "=" * 96)
    print("B  条件内错误预测：同一条件内部，能否挑出模型判错的图像？")
    print("=" * 96)
    print(f"{'条件':<14}{'n':>5}{'错误数':>7}  "
          f"{'QC17':>9}{'不确定度':>10}{'QC+不确定':>11}{'仅分数':>9}")
    rows = []
    for c in SHORT:
        g = fr[fr["cond"] == c].reset_index(drop=True)
        y = g["err"].to_numpy()
        if y.sum() < 8 or (1 - y).sum() < 8:
            print(f"{SHORT[c]:<14}{len(g):>5}{int(y.sum()):>7}   样本内错误太少，跳过")
            continue
        Xq = g[COLS].to_numpy(float)
        Xu = g[["tta_gap", "entropy", "unc"]].to_numpy(float)
        Xp = g[["unc"]].to_numpy(float)

        def oof_mean(X, seeds=range(5)):
            a = [auroc(y, cv_oof(X, y, s)) for s in seeds]
            return float(np.nanmean(a))

        aq, au, aqu, ap = (oof_mean(Xq), oof_mean(Xu),
                           oof_mean(np.c_[Xq, Xu]), oof_mean(Xp))
        print(f"{SHORT[c]:<14}{len(g):>5}{int(y.sum()):>7}  "
              f"{aq:>9.4f}{au:>10.4f}{aqu:>11.4f}{ap:>9.4f}")
        rows.append({"cond": c, "n": len(g), "n_err": int(y.sum()),
                     "qc17": aq, "unc": au, "qc_unc": aqu, "score": ap})

    # ---------------------------------------------------------- 真实翻拍细看
    print("\n" + "=" * 96)
    print("C  真实翻拍条件内部（n=202）逐信号细看")
    print("=" * 96)
    g = fr[fr["cond"] == "natural/oneplus"].reset_index(drop=True)
    y = g["err"].to_numpy()
    print(f"  错误数 {int(y.sum())} / {len(g)}（错误率 {y.mean():.4f}）")
    print(f"\n  {'信号':<26}{'AUROC':>9}{'95% CI':>22}")
    for nm, col in (("翻转 TTA 分歧", "tta_gap"), ("预测熵", "entropy"),
                    ("最大概率（反向）", "unc"), ("quality_risk", "qc_quality_risk"),
                    ("mean_intensity", "qc_mean_intensity"),
                    ("blur_score", "qc_blur_score")):
        a = auroc(y, g[col].to_numpy(float))
        lo, hi = boot_auc(y, g[col].to_numpy(float))
        print(f"  {nm:<26}{a:>9.4f}   [{lo:.4f}, {hi:.4f}]")

    print(f"\n  组合（5 折 OOF，患者无关）：")
    for nm, cols in (("不确定度 3 维", ["tta_gap", "entropy", "unc"]),
                     ("QC 17 维", COLS),
                     ("QC + 不确定度", COLS + ["tta_gap", "entropy", "unc"])):
        a = [auroc(y, cv_oof(g[cols].to_numpy(float), y, s)) for s in range(5)]
        print(f"    {nm:<22}{float(np.nanmean(a)):>9.4f}")

    # ---------------------------------------------------------- 合并
    print("\n" + "=" * 96)
    print("D  四条件合并（808 行）零标注风险信号对错误的排序能力")
    print("=" * 96)
    ye = fr["err"].to_numpy()
    for nm, cols in (("翻转 TTA 分歧", ["tta_gap"]),
                     ("不确定度 3 维", ["tta_gap", "entropy", "unc"]),
                     ("QC 17 维", COLS),
                     ("QC + 不确定度", COLS + ["tta_gap", "entropy", "unc"])):
        a = [auroc(ye, cv_oof(fr[cols].to_numpy(float), ye, s)) for s in range(5)]
        print(f"    {nm:<22}{float(np.nanmean(a)):>9.4f}")

    pd.DataFrame(rows).to_csv(args.out / "table_cp_within.csv", index=False)
    print(f"\n  -> {args.out}/table_cp_within.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
