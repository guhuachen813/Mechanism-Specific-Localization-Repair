"""校准修补对照：在偏移域与降质下，哪一种「修正」既降 ECE 又不买掉灵敏度？

背景
----
前面的 calib_degeneracy 发现：ECE 最优温度在偏移域/重度降质档会退化成
「全判阴性」（TPR=0，准确率恰等于多数类基线），ECE 看起来从 0.28 降到 0.10，
但模型已经死了。NLL 最优温度退化得更早。
所以「校准改善」本身不构成证据；必须同时看灵敏度。

本脚本在同一批原始 logits 上并列比较五种修正，全部离线可算：
  A 冻结温度（源域拟合 T=0.4943，未修正）
  B ECE 最优温度
  C NLL 最优温度
  D 仅先验偏移：正类 logit 加 logit(π_t) − logit(π_s)，温度不动
  E 先验偏移 + 温度网格（二维搜索 ECE 最优）
  F 冻结温度 + 先验匹配阈值（把阈值移到使 PPR = π_t，即工作点重定位）

判据（同时报告，缺一不可）：ECE、TPR、PPR、错误率、代价风险（FN:FP=2:1）。

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
    """三分类 softmax 的正类后验；delta 加到正类 logit（先验偏移）。"""
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T)
    X = X.clone()
    X[:, 1] = X[:, 1] + delta
    return torch.softmax(X, dim=1).numpy()[:, 1]


def ece(y, p, n_bins=15):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(y) * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def nll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return dict(ece=ece(y, p), nll=nll(y, p), acc=float((pred == y).mean()),
                err=float((pred != y).mean()), tpr=tp / max(tp + fn, 1),
                ppr=float(pred.mean()),
                cost=float((FN_COST * fn + FP_COST * fp) / len(y)))


def thr_for_prevalence(p, target_ppr):
    """选阈值使预测阳性率 ≈ 目标值（工作点重定位到目标先验）。"""
    return float(np.quantile(p, 1.0 - target_ppr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t-frozen", type=float, required=True)
    ap.add_argument("--pi-source", type=float, default=0.126,
                    help="源域（route 训练集）患病率，用于先验偏移")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", required=True)
    args = ap.parse_args()

    grid = np.exp(np.linspace(np.log(0.1), np.log(8.0), 160))
    rows = []
    for cond in args.conds.split(","):
        f = os.path.join(args.outdir, f"{cond}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d["label"] != 2].reset_index(drop=True)
        y = (d["label"].to_numpy() == 1).astype(int)
        L = d[C1].to_numpy(np.float64)
        pi_t = float(y.mean())

        rec = dict(cond=cond, n=len(y), pi_source=args.pi_source, pi_target=pi_t)
        # A 冻结
        pA = softmax_pos(L, args.t_frozen)
        # B ECE 最优温度
        TB = min(((ece(y, softmax_pos(L, T)), T) for T in grid), key=lambda t: t[0])[1]
        # C NLL 最优温度
        TC = min(((nll(y, softmax_pos(L, T)), T) for T in grid), key=lambda t: t[0])[1]
        # D 仅先验偏移（温度冻结）
        delta_prior = float(binary_logit(pi_t) - binary_logit(args.pi_source))
        pD = softmax_pos(L, args.t_frozen, delta_prior)
        # E 先验偏移 + 温度二维搜索（ECE 最优）
        bestE = (np.inf, 1.0)
        for T in grid:
            v = ece(y, softmax_pos(L, T, delta_prior))
            if v < bestE[0]:
                bestE = (v, T)
        pE = softmax_pos(L, bestE[1], delta_prior)
        # F 冻结温度 + 先验匹配阈值
        thrF = thr_for_prevalence(pA, pi_t)

        for name, p, extra in (("A 冻结温度", pA, {}),
                               ("B ECE最优温度", softmax_pos(L, TB), dict(T=TB)),
                               ("C NLL最优温度", softmax_pos(L, TC), dict(T=TC)),
                               ("D 仅先验偏移", pD, dict(delta=delta_prior)),
                               ("E 先验偏移+温度", pE, dict(T=bestE[1], delta=delta_prior)),
                               ("F 冻结+先验匹配阈值", pA, dict(thr=thrF))):
            thr = extra.get("thr", 0.5)
            rec.update({f"{k}_{name[0]}": v for k, v in metrics(y, p, thr).items()})
            if "T" in extra:
                rec[f"T_{name[0]}"] = extra["T"]
        rows.append(rec)

        print(f"\n### {cond}  π_t={pi_t:.3f}  Δ_prior={delta_prior:+.3f}")
        print(f"{'策略':<22}{'ECE':>7}{'NLL':>7}{'TPR':>7}{'PPR':>7}{'错误率':>9}{'代价':>8}")
        for nm in ["A 冻结温度", "B ECE最优温度", "C NLL最优温度",
                   "D 仅先验偏移", "E 先验偏移+温度", "F 冻结+先验匹配阈值"]:
            k = nm[0]
            print(f"{nm:<22}{rec[f'ece_{k}']:>7.3f}{rec[f'nll_{k}']:>7.3f}"
                  f"{rec[f'tpr_{k}']:>7.3f}{rec[f'ppr_{k}']:>7.3f}"
                  f"{rec[f'err_{k}']:>9.3f}{rec[f'cost_{k}']:>8.3f}")

    out = pd.DataFrame(rows)
    os.makedirs(args.results, exist_ok=True)
    out.to_csv(os.path.join(args.results, f"table_calibfix_{args.tag}.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"跨条件汇总（{args.tag}）：各策略的均值")
    print("=" * 100)
    print(f"{'策略':<22}{'ECE':>8}{'TPR':>8}{'PPR':>8}{'错误率':>9}{'代价':>8}{'TPR=0 档数':>12}")
    for nm in ["A 冻结温度", "B ECE最优温度", "C NLL最优温度",
               "D 仅先验偏移", "E 先验偏移+温度", "F 冻结+先验匹配阈值"]:
        k = nm[0]
        zero = int((out[f"tpr_{k}"] <= 1e-9).sum())
        print(f"{nm:<22}{out[f'ece_{k}'].mean():>8.3f}{out[f'tpr_{k}'].mean():>8.3f}"
              f"{out[f'ppr_{k}'].mean():>8.3f}{out[f'err_{k}'].mean():>9.3f}"
              f"{out[f'cost_{k}'].mean():>8.3f}{zero:>12d}")


if __name__ == "__main__":
    main()
