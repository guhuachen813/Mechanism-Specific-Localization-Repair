"""思路C 前哨实验 第3步：SAM/MedSAM 线性探针 vs DenseNet-121 在 CheXphoto 四条件上的配对比较。

口径
----
* DenseNet 正类概率：softmax([l0,l1,l2]/T)[1]，T = 0.4943421483039856（与
  chexphoto_analyze.py 完全一致）。
* 三个模型在**同一批 202×4 正位图**上评估；不确定性区间用患者级整群
  Bootstrap（400 次，seed 42）。
* 只比较排序层（AUROC）——探针概率未经温度校准，操作点层不可比。

用法（本地）：
    python3 analyze_rep_robust.py --results results/
作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

T1 = 0.4943421483039856
SEED = 42
N_BOOT = 400
CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "clean", "natural/oneplus": "real",
         "synthetic/photographic": "syn.photo", "synthetic/digital": "syn.digital"}


def densenet_p(df: pd.DataFrame) -> np.ndarray:
    L = df[["l1_0", "l1_1", "l1_2"]].to_numpy(dtype=np.float64)
    X = torch.from_numpy(L / T1)
    return torch.softmax(X, dim=1).numpy()[:, 1]


def cluster_boot_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    point = roc_auc_score(y, s)
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss))
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return point, lo, hi


def paired_delta(df: pd.DataFrame, model_p: str, cond: str,
                 n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """同一 base_key 在 cond 与 clean 上的配对 AUROC 差（患者级整群 bootstrap）。"""
    clean = df[df["cond"] == "clean"][["base_key", "label", model_p]]
    other = df[df["cond"] == cond][["base_key", "label", model_p]]
    m = clean.merge(other, on="base_key", suffixes=("_c", "_o")).dropna()
    rng = np.random.RandomState(seed)
    groups = m["base_key"].to_numpy()
    ug = np.unique(groups)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        a = m.iloc[idx]
        try:
            deltas.append(roc_auc_score(a["label_o"], a[model_p + "_o"])
                          - roc_auc_score(a["label_c"], a[model_p + "_c"]))
        except ValueError:
            continue
    point = (roc_auc_score(m["label_o"], m[model_p + "_o"])
             - roc_auc_score(m["label_c"], m[model_p + "_c"]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path,
                    default=Path("../chexphoto/results/chexphoto_preds.csv"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.results.mkdir(exist_ok=True)

    preds = pd.read_csv(args.preds)
    preds = preds[preds["orient"] == "frontal"].copy()
    preds["p_densenet"] = densenet_p(preds)

    frames = [preds[["base_key", "cond", "orient", "Patient", "label", "p_densenet"]]]
    for tag in ("sam", "medsam"):
        fp = args.results / f"eval_p_{tag}.csv"
        if not fp.exists():
            print(f"[跳过] 缺 {fp}")
            continue
        t = pd.read_csv(fp)
        frames.append(t[["base_key", "cond", "label", "p"]]
                      .rename(columns={"p": f"p_{tag}"}))
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["base_key", "cond"], how="inner")
    print(f"合并后 {len(df)} 行，{df['base_key'].nunique()} 张基础图")

    models = [c[2:] for c in df.columns if c.startswith("p_")]
    print("模型:", models)

    # ---- 每条件 AUROC（患者级整群 bootstrap） -----------------------------
    rows = []
    for cond in CONDS:
        g = df[df["cond"] == cond]
        g = g[g["label"].notna()]
        for m in models:
            a, lo, hi = cluster_boot_auc(g["label"].to_numpy(),
                                         g[f"p_{m}"].to_numpy(),
                                         g["Patient"].to_numpy())
            rows.append({"cond": SHORT[cond], "model": m, "n": len(g),
                         "auroc": a, "ci_lo": lo, "ci_hi": hi})
    tab = pd.DataFrame(rows)
    tab.to_csv(args.results / "rep_robust_auroc.csv", index=False)
    print("\n=== 每条件 AUROC [95% CI] ===")
    for cond in tab["cond"].unique():
        print(f"--- {cond} ---")
        for _, r in tab[tab["cond"] == cond].iterrows():
            print(f"  {r['model']:9s} {r['auroc']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

    # ---- 配对 ΔAUROC（各条件 vs clean） -----------------------------------
    rows = []
    for cond in CONDS[1:]:
        for m in models:
            a, lo, hi = paired_delta(df, f"p_{m}", cond)
            rows.append({"cond": SHORT[cond], "model": m,
                         "delta": a, "ci_lo": lo, "ci_hi": hi})
    dt = pd.DataFrame(rows)
    dt.to_csv(args.results / "rep_robust_paired.csv", index=False)
    print("\n=== 配对 ΔAUROC（条件 - clean）===")
    for cond in dt["cond"].unique():
        print(f"--- {cond} ---")
        for _, r in dt[dt["cond"] == cond].iterrows():
            sig = "*" if r["ci_hi"] < 0 or r["ci_lo"] > 0 else ""
            print(f"  {r['model']:9s} {r['delta']:+.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
