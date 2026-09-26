"""同批同档对照的完整版：模型输入侧 + QC 侧，A(224/224) vs B(320→224)。"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

T1 = 0.4943421483039856

def ppos(df):
    L = torch.from_numpy(df[["l1_0", "l1_1", "l1_2"]].to_numpy(np.float64) / T1)
    return torch.softmax(L, dim=1).numpy()[:, 1]

def stat(d, p, name):
    y = (d["label"].to_numpy() == 1).astype(int)
    err = ((p >= .5).astype(int) != y).mean()
    auroc = roc_auc_score(y, p)
    return dict(**{f"{name}_auroc": auroc, f"{name}_err": err,
                   f"{name}_ppr": float((p >= .5).mean()), f"{name}_sd": float(p.std()),
                   f"{name}_qr": float(d["qc_quality_risk"].mean())})

def main():
    base, conds, outdir = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    rows = []
    for c in conds:
        A = pd.read_csv(f"{base}/out_seed42_route/{c}.csv")
        B = pd.read_csv(f"{base}/outB_seed42_route/{c}.csv")
        rows.append(dict(cond=c, **stat(A, ppos(A), "A"), **stat(B, ppos(B), "B")))
    d = pd.DataFrame(rows)
    d["d_auroc"] = d.B_auroc - d.A_auroc
    d["d_err"] = d.B_err - d.A_err
    d["sd_ratio"] = d.B_sd / d.A_sd.replace(0, np.nan)
    print(d.round(4).to_string(index=False))
    print(f"\n平均: A AUROC {d.A_auroc.mean():.4f} err {d.A_err.mean():.4f} | B AUROC {d.B_auroc.mean():.4f} err {d.B_err.mean():.4f}")
    print(f"B 错误率低于 A 的档数: {int((d.d_err < 0).sum())}/{len(d)}")
    d.to_csv(f"{outdir}/table_pipeline_contrast.csv", index=False)

if __name__ == "__main__":
    main()
