"""CheXphoto 真实降质的校准修补分析：A–F 六策略能否救回塌缩的操作点？

背景
----
真实翻拍（natural/oneplus）条件下，模型在阈值 0.5 处把**全部 202 张**判为阴性
（TPR = PPR = 0.0000），尽管排序信号尚存（AUROC 0.6592）。这是典型的
**操作点塌缩**：分数整体被压低，排序没全丢，但决策阈值失效。

我们的方法主张的正是这一类失效可由**无需标注**的校准修补救回。本脚本把
与 CheXpert 两域 / NIH 外部验证完全相同的 A–F 六策略施加到 CheXphoto 的
四个条件上，检验补救效果。

六策略（全部在部署时可用，不需要目标域标签）
------------------------------------------
A  冻结温度 T1 + 阈值 0.5                  —— 现状基线
B  在目标域上按 ECE 最优选温度 + 阈值 0.5   —— 需要少量标注
C  在目标域上按 NLL 最优选温度 + 阈值 0.5   —— 需要少量标注
D  仅先验偏移：logit 加 Δ_prior + 阈值 0.5  —— 只需知道源/目标患病率
E  先验偏移 + ECE 最优温度 + 阈值 0.5       —— 需少量标注
F  冻结温度 + 先验匹配阈值                  —— 只需知道目标患病率

口径与 chexphoto_analyze.py / nih_analyze.py 一致。

用法：
    python chexphoto_calib.py --preds /root/chexphoto/out/preds.csv --out results/

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

T1 = 0.4943421483039856
PI_SOURCE = 0.126
PI_TARGET = 0.3267          # CheXphoto valid 202 张 frontal 的实测患病率（66/202）

CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}


def softmax_pos(L, T=T1, delta=0.0):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T).clone()
    X[:, 1] = X[:, 1] + delta
    return torch.softmax(X, dim=1).numpy()[:, 1]


def binary_logit(p):
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return np.log(p / (1 - p))


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


def nll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def safe_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def thr_for_prevalence(p, pi_t):
    """取阈值使预测阳性率最接近 pi_t。"""
    cand = np.unique(np.concatenate([[0.0], np.asarray(p, float), [1.0]]))
    best, bt = np.inf, 0.5
    for t in cand:
        v = abs(float((p >= t).mean()) - pi_t)
        if v < best:
            best, bt = v, float(t)
    return bt


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "auroc": safe_auc(y, p), "ece": ece(y, p), "nll": nll(y, p),
        "err": float((pred != y).mean()), "cost": (2 * fn + fp) / len(y),
        "ppr": float(pred.mean()), "tpr": tp / max(int((y == 1).sum()), 1),
        "fn": fn, "fp": fp, "tp": tp,
    }


def cluster_bootstrap(df, stat_fn, n_boot=600, seed=0):
    rng = np.random.default_rng(seed)
    gid = df.groupby("Patient", sort=False).ngroup().to_numpy()
    n_g = gid.max() + 1
    rows_by_g = [np.flatnonzero(gid == g) for g in range(n_g)]
    out = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
        idx = np.concatenate([np.repeat(rows_by_g[g], cnt[g])
                              for g in range(n_g) if cnt[g] > 0])
        out.append(stat_fn(df.iloc[idx]))
    return np.asarray(out, float)


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


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
    L = fr[["l1_0", "l1_1", "l1_2"]].to_numpy(float)
    fr["pA"] = softmax_pos(L)                      # 冻结温度 T1

    delta_prior = float(binary_logit(PI_TARGET) - binary_logit(PI_SOURCE))
    print(f"源患病率 π_s={PI_SOURCE}  目标患病率 π_t={PI_TARGET}")
    print(f"Δ_prior = logit(π_t) − logit(π_s) = {delta_prior:+.4f}\n")

    # ------------------------------------------------ 分数分布诊断
    print("=" * 96)
    print("0  各条件分数分布（冻结温度 T1，阈值 0.5）——诊断操作点塌缩")
    print("=" * 96)
    print(f"{'条件':<14}{'n':>5}{'阳性':>5}{'p均值':>9}{'p中位':>9}{'p90':>9}"
          f"{'p最大':>9}{'≥0.5张数':>10}{'AUROC':>9}")
    for c in CONDS:
        g = fr[fr["cond"] == c]
        p = g["pA"].to_numpy()
        print(f"{SHORT[c]:<14}{len(g):>5}{int(g['y'].sum()):>5}{p.mean():>9.4f}"
              f"{np.median(p):>9.4f}{np.percentile(p,90):>9.4f}{p.max():>9.4f}"
              f"{int((p>=0.5).sum()):>10}{safe_auc(g['y'], p):>9.4f}")

    # ------------------------------------------------ A–F 全条件
    print("\n" + "=" * 96)
    print("1  A–F 校准修补在各条件上的效果")
    print("=" * 96)
    grid = np.exp(np.linspace(np.log(0.1), np.log(8.0), 160))
    all_rows = []

    for c in CONDS:
        g = fr[fr["cond"] == c].copy()
        y = g["y"].to_numpy(int)
        Lc = g[["l1_0", "l1_1", "l1_2"]].to_numpy(float)
        pA = softmax_pos(Lc)

        TB = min(((ece(y, softmax_pos(Lc, T)), T) for T in grid), key=lambda t: t[0])[1]
        TC = min(((nll(y, softmax_pos(Lc, T)), T) for T in grid), key=lambda t: t[0])[1]
        bestE = (np.inf, 1.0)
        for T in grid:
            v = ece(y, softmax_pos(Lc, T, delta_prior))
            if v < bestE[0]:
                bestE = (v, T)
        thrF = thr_for_prevalence(pA, PI_TARGET)

        strategies = [
            ("A 冻结温度",            pA,                              0.5),
            ("B ECE最优温度",         softmax_pos(Lc, TB),             0.5),
            ("C NLL最优温度",         softmax_pos(Lc, TC),             0.5),
            ("D 仅先验偏移",          softmax_pos(Lc, T1, delta_prior), 0.5),
            ("E 先验偏移+温度",       softmax_pos(Lc, bestE[1], delta_prior), 0.5),
            ("F 冻结+先验匹配阈值",   pA,                              thrF),
        ]
        print(f"\n--- {SHORT[c]}（n={len(g)}，阳性 {int(y.sum())}）---")
        print(f"    T_B={TB:.3f}  T_C={TC:.3f}  T_E={bestE[1]:.3f}  thr_F={thrF:.4f}")
        print(f"    {'策略':<20}{'ECE':>8}{'NLL':>8}{'TPR':>8}{'PPR':>8}"
              f"{'错误率':>9}{'代价':>9}{'代价95%CI':>20}")

        def cost_of(gg, thr):
            yy = (gg["y"] == 1).astype(int).to_numpy()
            return metrics(yy, softmax_pos(gg[["l1_0", "l1_1", "l1_2"]].to_numpy(float)), thr)["cost"]

        bootA = cluster_bootstrap(g, lambda gg: [cost_of(gg, 0.5)], n_boot=args.boot)[:, 0]
        bootF = cluster_bootstrap(g, lambda gg: [cost_of(gg, thrF)], n_boot=args.boot)[:, 0]

        for nm, p, thr in strategies:
            m = metrics(y, p, thr)
            if nm.startswith("A"):
                lo, hi = ci(bootA)
            elif nm.startswith("F"):
                lo, hi = ci(bootF)
            else:
                lo = hi = np.nan
            cis = f"[{lo:.4f}, {hi:.4f}]" if np.isfinite(lo) else "—"
            print(f"    {nm:<20}{m['ece']:>8.4f}{m['nll']:>8.4f}{m['tpr']:>8.4f}"
                  f"{m['ppr']:>8.4f}{m['err']:>9.4f}{m['cost']:>9.4f}{cis:>20}")
            all_rows.append({"cond": c, "strategy": nm, "thr": thr,
                             "T": T1 if nm.startswith(("A", "D", "F")) else
                                  (TB if nm.startswith("B") else
                                   (TC if nm.startswith("C") else bestE[1])),
                             "cost_lo": lo, "cost_hi": hi, **m})

    tab = pd.DataFrame(all_rows)
    tab.to_csv(args.out / "table_cp_calibration.csv", index=False)

    # ------------------------------------------------ 关键对照
    print("\n" + "=" * 96)
    print("2  关键问题：修补能否救回真实翻拍的操作点？")
    print("=" * 96)
    key = ["A 冻结温度", "D 仅先验偏移", "F 冻结+先验匹配阈值", "B ECE最优温度"]
    print(f"{'条件':<14}{'策略':<22}{'TPR':>8}{'PPR':>8}{'代价':>9}{'AUROC':>9}")
    for c in CONDS:
        for s in key:
            r = tab[(tab["cond"] == c) & (tab["strategy"] == s)].iloc[0]
            print(f"{SHORT[c]:<14}{s:<22}{r['tpr']:>8.4f}{r['ppr']:>8.4f}"
                  f"{r['cost']:>9.4f}{r['auroc']:>9.4f}")
        print()

    print(f"结果已写入 {args.out}/table_cp_calibration.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
