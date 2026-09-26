"""分歧定位：设计A（全量224/224）与设计B（2000子样本320→224）的 Δ 相反，
是「样本量」造成的，还是「降质工作分辨率」造成的？

做法
----
直接从设计A 已跑好的全量 CSV 里，按设计B 的分层抽样规则再抽 2000 行，
用完全相同的 CV 设置重算条件内 ΔAUC。
  - 若 Δ 仍≈+0.03（接近设计A 全量值）→ 差异来自**管线/分辨率**，与样本量无关
  - 若 Δ 掉到≈0（接近设计B 值）      → 差异来自**统计功效**，设计A 的 +0.03 是低功效造成的假象
这一步不重跑 GPU，只做抽样与 CV，几分钟出结果。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from h1_robust import QC_FEATS, cv_oof, safe_auc

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for cond in args.conds.split(","):
        f = os.path.join(args.outdir, f"{cond}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if len(d) > args.max_rows:
            parts = []
            for lab, g in d.groupby("label"):
                k = max(1, int(round(args.max_rows * len(g) / len(d))))
                parts.append(g.sample(n=min(k, len(g)), random_state=args.seed))
            d = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        d = d[d["label"] != 2].reset_index(drop=True)
        yy = (d["label"].to_numpy() == 1).astype(int)
        L = d[["l1_0", "l1_1", "l1_2"]].to_numpy(float) / args.t1
        e = np.exp(L - L.max(1, keepdims=True))
        p = e[:, 1] / e.sum(1)
        err = ((p >= 0.5).astype(int) != yy).astype(int)
        if err.sum() < 15 or len(np.unique(err)) < 2:
            continue
        conf = np.maximum(p, 1 - p)
        qcols = [f"qc_{c}" for c in QC_FEATS if f"qc_{c}" in d.columns]
        Xq = np.nan_to_num(d[qcols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)

        rec = dict(cond=cond, n=len(yy), err=float(err.mean()))
        for learner in ("lr", "gb"):
            base, both = [], []
            for s in range(args.seeds):
                base.append(safe_auc(err, cv_oof(conf[:, None], err, learner, s)))
                both.append(safe_auc(err, cv_oof(np.c_[conf, Xq], err, learner, s)))
            rec[f"auc_base_{learner}"] = float(np.nanmean(base))
            rec[f"delta_{learner}"] = float(np.nanmean(np.array(both) - np.array(base)))
        rows.append(rec)
        print(f"{cond:<12} n={len(yy):>5} err={err.mean():.3f} | "
              f"LR base={rec['auc_base_lr']:.4f} Δ={rec['delta_lr']:+.4f} | "
              f"GB base={rec['auc_base_gb']:.4f} Δ={rec['delta_gb']:+.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, f"table_subsample_{args.tag}.csv"), index=False)
    print(f"\n汇总 {args.tag}（{len(out)} 档，每档降至 {args.max_rows} 张）")
    print(f"  LR 平均 Δ={out['delta_lr'].mean():+.4f}   GB 平均 Δ={out['delta_gb'].mean():+.4f}")
    print(f"  GB Δ>0 档数 {int((out['delta_gb'] > 0).sum())}/{len(out)}")


if __name__ == "__main__":
    main()
