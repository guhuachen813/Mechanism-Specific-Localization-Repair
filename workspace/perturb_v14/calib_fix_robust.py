"""稳健性检验：策略 D/F 需要目标患病率 π_t，靠「估计」而非「已知」还成立吗？

calib_fix_compare 里 D（先验偏移）与 F（先验匹配阈值）都直接用了真实 π_t，
这是 oracle 信息。审稿人会问：这点优势是不是白来的？
本脚本用两折重复随机划分：一半估计 π̂_t，另一半评估，与 oracle 版和固定阈值 0.5 对比。
若优势在 π̂ 有噪声时仍成立，结论才站得住。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

C1 = ["l1_0", "l1_1", "l1_2"]
FN_COST, FP_COST = 2.0, 1.0


def softmax_pos(L, T, delta=0.0):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T).clone()
    X[:, 1] = X[:, 1] + delta
    return torch.softmax(X, dim=1).numpy()[:, 1]


def binary_logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def cost_at(y, p, thr):
    pred = (p >= thr).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    pos = int((y == 1).sum())
    return float((FN_COST * fn + FP_COST * fp) / len(y)), tp / max(pos, 1), float(pred.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t-frozen", type=float, required=True)
    ap.add_argument("--pi-source", type=float, default=0.126)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for cond in args.conds.split(","):
        f = os.path.join(args.outdir, f"{cond}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d["label"] != 2].reset_index(drop=True)
        y_all = (d["label"].to_numpy() == 1).astype(int)
        L_all = d[C1].to_numpy(np.float64)
        p_all = softmax_pos(L_all, args.t_frozen)
        pi_true = float(y_all.mean())
        n = len(y_all)
        half = n // 2

        acc = {k: [] for k in ("A_fixed0.5", "F_oracle", "F_est", "D_est")}
        for _ in range(args.reps):
            idx = rng.permutation(n)
            a, b = idx[:half], idx[half:]
            if len(np.unique(y_all[b])) < 2 or len(np.unique(y_all[a])) < 2:
                continue
            yb, pb = y_all[b], p_all[b]
            pi_hat = float(y_all[a].mean())           # 只用一半样本估计

            thr_or = float(np.quantile(pb, 1.0 - pi_true))
            thr_es = float(np.quantile(pb, 1.0 - pi_hat))
            dlt = binary_logit(pi_hat) - binary_logit(args.pi_source)
            pD = softmax_pos(L_all, args.t_frozen, dlt)[b]
            # D 也要选阈值：用同一半估计的 pi_hat 做阈值匹配，才是可比口径
            thr_d = float(np.quantile(pD, 1.0 - pi_hat))

            acc["A_fixed0.5"].append(cost_at(yb, pb, 0.5))
            acc["F_oracle"].append(cost_at(yb, pb, thr_or))
            acc["F_est"].append(cost_at(yb, pb, thr_es))
            acc["D_est"].append(cost_at(yb, pD, thr_d))

        rec = dict(cond=cond, n=n, pi_true=pi_true)
        for k, v in acc.items():
            if not v:
                continue
            arr = np.array(v)
            rec[f"cost_{k}"] = float(arr[:, 0].mean())
            rec[f"tpr_{k}"] = float(arr[:, 1].mean())
            rec[f"ppr_{k}"] = float(arr[:, 2].mean())
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(args.results, exist_ok=True)
    out.to_csv(os.path.join(args.results, f"table_calibfix_robust_{args.tag}.csv"), index=False)

    print(f"\n{args.tag}：π_t 由一半样本估计（{args.reps} 次重复两折）")
    print(f"{'条件':<12}{'A 固定0.5':>11}{'F oracle':>10}{'F 估计π̂':>11}{'D 估计π̂':>10}")
    print("-" * 58)
    for _, r in out.iterrows():
        print(f"{r['cond']:<12}{r['cost_A_fixed0.5']:>11.3f}{r['cost_F_oracle']:>10.3f}"
              f"{r['cost_F_est']:>11.3f}{r['cost_D_est']:>10.3f}")
    print("-" * 58)
    print(f"{'均值':<12}{out['cost_A_fixed0.5'].mean():>11.3f}{out['cost_F_oracle'].mean():>10.3f}"
          f"{out['cost_F_est'].mean():>11.3f}{out['cost_D_est'].mean():>10.3f}")
    print(f"{'平均 TPR':<12}{out['tpr_A_fixed0.5'].mean():>11.3f}{out['tpr_F_oracle'].mean():>10.3f}"
          f"{out['tpr_F_est'].mean():>11.3f}{out['tpr_D_est'].mean():>10.3f}")


if __name__ == "__main__":
    main()
