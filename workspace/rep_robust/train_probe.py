"""思路C 前哨实验 第2步：冻结表征 + 线性探针，并在 CheXphoto 四条件上出预测。

协议
----
* 患者级 80/20 划分选 C（内部验证 AUROC），选好后全量重拟合。
* 逻辑回归（max_iter 2000）；输出 eval_p_{tag}.csv（key, cond, orient, base_key,
  label, p）供与 DenseNet 预测做配对比较。

用法：
    /root/miniconda3/bin/python train_probe.py --tag sam
    /root/miniconda3/bin/python train_probe.py --tag medsam

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUTDIR = Path("/root/autodl-tmp/experiments/rep_robust")


def patient_of(path: str) -> str:
    m = re.search(r"(patient\d+)", path)
    return m.group(1) if m else path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tr_emb = np.load(OUTDIR / f"emb_{args.tag}_train_list.npy")
    tr_key = pd.read_csv(OUTDIR / "train_list.csv")
    ev_emb = np.load(OUTDIR / f"emb_{args.tag}_eval_list.npy")
    ev_key = pd.read_csv(OUTDIR / "eval_list.csv")
    assert len(tr_emb) == len(tr_key) and len(ev_emb) == len(ev_key)

    y = tr_key["label"].to_numpy()
    pat = tr_key["disk_path"].map(patient_of).to_numpy()
    rng = np.random.RandomState(args.seed)
    patients = np.unique(pat)
    rng.shuffle(patients)
    hold = set(patients[: max(1, int(len(patients) * 0.2))])
    is_hold = np.array([p in hold for p in pat])
    print(f"train {len(y)}（阳性 {y.sum()}），holdout 患者 {len(hold)}"
          f"（图像 {is_hold.sum()}）")

    best = (None, -1.0)
    for c in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(max_iter=2000, C=c)
        clf.fit(tr_emb[~is_hold], y[~is_hold])
        auc = roc_auc_score(y[is_hold], clf.predict_proba(tr_emb[is_hold])[:, 1])
        print(f"  C={c:<5} holdout AUROC {auc:.4f}")
        if auc > best[1]:
            best = (c, auc)
    c = best[0]
    print(f"选定 C={c}（holdout AUROC {best[1]:.4f}）")

    clf = LogisticRegression(max_iter=2000, C=c)
    clf.fit(tr_emb, y)
    p = clf.predict_proba(ev_emb)[:, 1]

    out = ev_key.copy()
    out["p"] = p
    out.to_csv(OUTDIR / f"eval_p_{args.tag}.csv", index=False)
    print(f"-> eval_p_{args.tag}.csv")
    for cond, g in out.groupby("cond"):
        m = g["label"].notna()
        print(f"  {cond:24s} n={int(m.sum()):4d}  "
              f"AUROC={roc_auc_score(g.loc[m, 'label'], g.loc[m, 'p']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
