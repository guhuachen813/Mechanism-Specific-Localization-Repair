"""管线对照：同一批图、同一模型、同一 QC 代码，只改工作分辨率。
输出每档的 (a) 模型输入侧统计 (b) QC 特征均值，用于定位混淆来源。

预测：双线性 320→224 会平均掉高频成分——
  高斯噪声 σ 约降为 0.70 倍，σ=3 的模糊在 224 尺度上只剩 σ≈2.1，
  而亮度/对比度这类逐像素强度算子是尺度不变的。
因此「高频类」降质（noise/blur/ds/jpeg）在两管线下差异应显著大于「强度类」（dark/bright/contrast）。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd, torch

T1 = 0.4943421483039856

def ppos(df):
    L = torch.from_numpy(df[["l1_0", "l1_1", "l1_2"]].to_numpy(np.float64) / T1)
    return torch.softmax(L, dim=1).numpy()[:, 1]

def main():
    base = sys.argv[1]
    conds = sys.argv[2].split(",")
    rows = []
    for cond in conds:
        A = pd.read_csv(f"{base}/out_seed42_route/{cond}.csv")
        B = pd.read_csv(f"{base}/outB_seed42_route/{cond}.csv")
        pa, pb = ppos(A), ppos(B)
        ya = (A["label"].to_numpy() == 1).astype(int)
        yb = (B["label"].to_numpy() == 1).astype(int)
        rows.append(dict(
            cond=cond,
            A_ppr=float((pa >= .5).mean()), B_ppr=float((pb >= .5).mean()),
            A_sd=float(pa.std()), B_sd=float(pb.std()),
            A_err=float(((pa >= .5).astype(int) != ya).mean()),
            B_err=float(((pb >= .5).astype(int) != yb).mean()),
            A_qr=float(A["qc_quality_risk"].mean()), B_qr=float(B["qc_quality_risk"].mean()),
            A_noise=float(A["qc_noise_score"].mean()), B_noise=float(B["qc_noise_score"].mean()),
            A_blur=float(A["qc_blur_score"].mean()), B_blur=float(B["qc_blur_score"].mean()),
        ))
    d = pd.DataFrame(rows)
    d["d_sd"] = d.B_sd / d.A_sd.replace(0, np.nan)
    d["d_err"] = d.B_err - d.A_err
    print(d.round(4).to_string(index=False))
    d.to_csv(f"{sys.argv[3]}/table_pipeline_contrast.csv", index=False)

if __name__ == "__main__":
    main()
