"""思路C 前哨实验 第0步：构建嵌入抽取的图像清单。

产出两个清单（都在 GPU06 上运行）：
* eval_list.csv   —— CheXphoto 四条件 × 正位 202 张/条件（来自 manifest.csv，与
                     chexphoto_infer.py 的口径一致：orient == "frontal"）。
* train_list.csv  —— CheXpert train 正位、Cardiomegaly 标签为确定值(0/1)的图像，
                     全部阳性 + 按 seed42 抽取的 2 万张阴性，用于训练线性探针。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("/root/chexphoto/manifest.csv"))
    ap.add_argument("--train-csv", type=Path,
                    default=Path("/root/autodl-tmp/CheXpert-v1.0-small/train.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("/root/autodl-tmp/experiments/rep_robust"))
    ap.add_argument("--neg-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- eval list：CheXphoto 四条件（正位） -------------------------------
    man = pd.read_csv(args.manifest)
    ev = man[man["orient"] == "frontal"].copy()
    ev["key"] = ev["base_key"] + "|" + ev["cond"]
    ev_list = ev[["key", "cond", "orient", "base_key", "label", "disk_path"]].copy()
    ev_list.to_csv(args.outdir / "eval_list.csv", index=False)
    print(f"eval_list: {len(ev_list)} 行")
    print(ev_list.groupby("cond").size().to_string())
    print(f"eval 阳性率 {ev_list['label'].mean():.4f}")

    # ---- train list：CheXpert train 正位，Cardiomegaly 确定 0/1 ------------
    tr = pd.read_csv(args.train_csv)
    tr = tr[tr["Frontal/Lateral"] == "Frontal"].copy()
    tr = tr[tr["Cardiomegaly"].isin([0.0, 1.0])].copy()
    tr["label"] = tr["Cardiomegaly"].astype(int)
    pos = tr[tr["label"] == 1]
    neg = tr[tr["label"] == 0]
    rng = np.random.RandomState(args.seed)
    neg_s = neg.sample(n=min(args.neg_n, len(neg)), random_state=args.seed)
    use = pd.concat([pos, neg_s], ignore_index=True)
    use = use.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    root = "/root/autodl-tmp"
    tr_list = pd.DataFrame({
        "key": use["Path"],
        "label": use["label"],
        "disk_path": use["Path"].map(
            lambda p: p if str(p).startswith("/") else f"{root}/{p}"),
    })
    tr_list.to_csv(args.outdir / "train_list.csv", index=False)
    print(f"train_list: {len(tr_list)} 行（阳性 {int(tr_list['label'].sum())}，"
          f"阴性 {int((tr_list['label'] == 0).sum())}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
