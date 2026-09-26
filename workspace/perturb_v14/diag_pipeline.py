"""诊断：设计A（224/224）与设计B（320→224）在模型输入侧到底差在哪。

关键线索：两设计下 QC 特征几乎完全相同（noise_010 的 qc_noise_score 0.0938 vs 0.0957），
但模型置信度的错误预测能力差 0.3 AUC。若模型输入本身不同，就能解释这个矛盾。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

T1 = 0.4943421483039856


def ppos(df):
    L = torch.from_numpy(df[["l1_0", "l1_1", "l1_2"]].to_numpy(np.float64) / T1)
    return torch.softmax(L, dim=1).numpy()[:, 1]


def main():
    base = sys.argv[1]
    for cond in ["clean", "noise_010", "blur_3", "dark_05", "ds_4"]:
        A = pd.read_csv(f"{base}/out_seed42_route/{cond}.csv")
        B = pd.read_csv(f"{base}/outB_seed42_route/{cond}.csv")
        pa, pb = ppos(A), ppos(B)
        ya = (A["label"].to_numpy() == 1).astype(int)
        print(f"--- {cond}  (患病率 {ya.mean():.4f}) ---")
        print(f"    A(224/224)  p_pos mean={pa.mean():.4f} sd={pa.std():.4f} "
              f"分位[5,50,95]={np.round(np.percentile(pa, [5, 50, 95]), 4)} "
              f"预测阳性率={(pa >= 0.5).mean():.4f}")
        print(f"    B(320→224)  p_pos mean={pb.mean():.4f} sd={pb.std():.4f} "
              f"分位[5,50,95]={np.round(np.percentile(pb, [5, 50, 95]), 4)} "
              f"预测阳性率={(pb >= 0.5).mean():.4f}")
        for nm, d, p in (("A", A, pa), ("B", B, pb)):
            y = (d["label"].to_numpy() == 1).astype(int)
            err = ((p >= 0.5).astype(int) != y).astype(int)
            print(f"    {nm} 错误率={err.mean():.4f}  mean|logit|={np.abs(d[['l1_0','l1_1','l1_2']].to_numpy()).mean():.3f}")


if __name__ == "__main__":
    main()
