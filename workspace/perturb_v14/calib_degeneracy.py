"""校准退化的反面证据：ECE 最优温度是否退化成「多数类预测器」？

动机
----
ECE 的一个已知病理：把预测推向基率（多数类）会最小化 ECE，
却让灵敏度归零。若重度降质下 T*_ece 很大，须检查它是不是在做这件事。
报告 ECE 必须同时报告 AUROC / 灵敏度 / 预测阳性率（PPR），否则会误导。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

C1 = ["l1_0", "l1_1", "l1_2"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece(y, p, n_bins=15):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).sum() / len(y) * abs(correct[idx == b].mean() - conf[idx == b].mean())
                     for b in range(n_bins) if (idx == b).sum()))


def nll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def stats(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return dict(acc=float((pred == y).mean()), err=float((pred != y).mean()),
                ece=ece(y, p), tpr=tp / max(tp + fn, 1), ppr=float(pred.mean()),
                mean_p=float(p.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t-frozen", type=float, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", required=True)
    args = ap.parse_args()

    order = args.conds.split(",")
    rows = []
    for cond in order:
        f = os.path.join(args.outdir, f"{cond}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d["label"] != 2].reset_index(drop=True)
        y = (d["label"].to_numpy() == 1).astype(int)
        L = torch.from_numpy(d[C1].to_numpy(np.float64))

        def p_at(T):
            return torch.softmax(L / T, dim=1).numpy()[:, 1]

        grid = np.exp(np.linspace(np.log(0.05), np.log(8.0), 400))
        rec = dict(cond=cond, n=len(y), prevalence=float(y.mean()))
        for name, T in (("frozen", args.t_frozen), ("T=1", 1.0)):
            s = stats(y, p_at(T)); rec.update({f"{k}_{name}": v for k, v in s.items()})
        for crit, fn in (("ece", ece), ("nll", nll)):
            best = min(((fn(y, p_at(T)), T) for T in grid), key=lambda t: t[0])
            s = stats(y, p_at(best[1]))
            rec[f"T_{crit}"] = best[1]
            rec.update({f"{k}_{crit}": v for k, v in s.items()})
        rows.append(rec)
        print(f"{cond:<12} π={y.mean():.3f} | 冻结: acc={rec['acc_frozen']:.3f} "
              f"ECE={rec['ece_frozen']:.3f} TPR={rec['tpr_frozen']:.3f} PPR={rec['ppr_frozen']:.3f} "
              f"| T*_ece={rec['T_ece']:.2f}: acc={rec['acc_ece']:.3f} ECE={rec['ece_ece']:.3f} "
              f"TPR={rec['tpr_ece']:.3f} PPR={rec['ppr_ece']:.3f} "
              f"| T*_nll={rec['T_nll']:.2f}: acc={rec['acc_nll']:.3f} TPR={rec['tpr_nll']:.3f}",
              flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(args.results, exist_ok=True)
    out.to_csv(os.path.join(args.results, f"table_degeneracy_{args.tag}.csv"), index=False)
    print("\n多数类基线错误率（全判阴性）= 患病率 =", out["prevalence"].mean())


if __name__ == "__main__":
    main()
