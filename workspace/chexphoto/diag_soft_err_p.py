"""诊断：软错误目标的 p 耦合——AUROC(p → 软错误) 为何高达 0.84~0.95？

背景
----
控制实验（tta_geom_control.py）发现一个此前未报告的基线：
把软错误 = 1 − p(真实标签) 作为目标时，**主分数 p 本身**的 AUROC 就达到
0.844（真实翻拍）～0.951（干净原图），高于几何 TTA 分歧的 0.787（真实翻拍）。

这需要解释，因为它决定了 CheXphoto 报告 §3.4 表 5 的读法：
若任何与 p 相关的信号都能在软错误上取得 0.7+，那么该表把 0.682 / 0.605
与"随机 0.5"比较就是错的参照系。

本脚本给出机制证据：
  1. 各条件下 y=1 与 y=0 的 p 区间、软错误区间是否重叠
  2. 软错误中位数切分后，高组中 y=1 的占比
  3. 按 p 五分位，各层 y 比例与软错误均值

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

T1 = 0.4943421483039856
CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}


def softmax_pos(L, T=T1):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T)
    return torch.softmax(X, dim=1).numpy()[:, 1]


df = pd.read_csv("results/tta_geom.csv")
df["y"] = df["label"].astype(int)
df["p"] = softmax_pos(df[["v00_0", "v00_1", "v00_2"]].to_numpy(float))
df["soft_err"] = np.where(df["y"] == 1, 1.0 - df["p"], df["p"])

for c in CONDS:
    g = df[df["cond"] == c]
    g1 = g[g["y"] == 1]
    g0 = g[g["y"] == 0]
    print("=" * 92)
    print(f"{SHORT[c]}（n={len(g)}，阳性 {len(g1)}）")
    print("=" * 92)
    print(f"  y=1: p ∈ [{g1['p'].min():.4f}, {g1['p'].max():.4f}]   "
          f"软错误 ∈ [{g1['soft_err'].min():.4f}, {g1['soft_err'].max():.4f}]")
    print(f"  y=0: p ∈ [{g0['p'].min():.4f}, {g0['p'].max():.4f}]   "
          f"软错误 ∈ [{g0['soft_err'].min():.4f}, {g0['soft_err'].max():.4f}]")
    overlap = (g1["soft_err"].min() <= g0["soft_err"].max())
    print(f"  两组软错误区间{'有' if overlap else '无'}重叠")
    med = float(g["soft_err"].median())
    hi = g[g["soft_err"] > med]
    print(f"  中位数 {med:.4f} → 高组 n={len(hi)}，其中阳性占 {hi['y'].mean():.3f}")

    a_y = roc_auc_score(g["y"], g["p"])
    a_se = roc_auc_score((g["soft_err"] > med).astype(int), g["p"])
    print(f"  AUROC(p → 标签) = {a_y:.4f}    AUROC(p → 软错误中位切分) = {a_se:.4f}")

    print(f"\n  按 p 五分位分层：")
    print(f"    {'层':<6}{'n':>5}{'p 均值':>10}{'阳性占比':>10}{'软错误均值':>12}")
    q = pd.qcut(g["p"], 5, labels=False, duplicates="drop")
    for b in sorted(q.unique()):
        s = g[q == b]
        print(f"    {int(b):<6}{len(s):>5}{s['p'].mean():>10.4f}"
              f"{s['y'].mean():>10.3f}{s['soft_err'].mean():>12.4f}")
    print()
