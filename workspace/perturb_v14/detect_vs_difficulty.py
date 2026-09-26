"""质量特征的「能」与「不能」：降质识别 vs 单图错误预测。

动机
----
H1 的条件内检验已表明：手工质量特征在**同一降质条件内部**几乎不携带
逐图难度信息（相对置信度的增量 ≈ 0）。但若就此说「质量特征没用」，
是过度否定——它们可能仍能可靠地识别**图像处于哪个质量regime**。
本脚本把两件事分开量化：

  T1 降质识别：qc_quality_risk（及最佳单特征）能否把降质图与干净图分开？
               配对比较，同图同患者，AUROC ≈ 1 表示 regime 检测极准。
  T2 难度预测：在降质条件**内部**，同一批特征能否预测这张图模型会判错？
               这正是路由真正需要的能力，已知接近随机。

两者并列，结论才精确：**质量特征是 regime 检测器，不是逐图难度估计器。**

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", required=True)
    args = ap.parse_args()

    d0 = pd.read_csv(os.path.join(args.outdir, "clean.csv"))
    frame = pd.read_csv(os.path.join(args.outdir, "clean.csv"))
    # 按行号对齐：所有条件的 CSV 行序一致（同一 frame，同一 idx）
    key = "__idx"
    d0 = d0.assign(**{key: np.arange(len(d0))})

    rows = []
    for cond in args.conds.split(","):
        if cond == "clean":
            continue
        f = os.path.join(args.outdir, f"{cond}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        n = min(len(d), len(d0))
        d, dc = d.iloc[:n].reset_index(drop=True), d0.iloc[:n].reset_index(drop=True)

        # --- T1 降质识别：配对（同一张图的两个版本），用秩和 AUROC
        y_det = np.r_[np.zeros(n), np.ones(n)]
        auc_qr = safe_auc(y_det, np.r_[dc["qc_quality_risk"].to_numpy(float),
                                       d["qc_quality_risk"].to_numpy(float)])
        best_f, best_a = None, -np.inf
        for c in QC_FEATS:
            col = f"qc_{c}"
            if col not in d.columns:
                continue
            s = np.r_[dc[col].to_numpy(float), d[col].to_numpy(float)]
            a = safe_auc(y_det, np.abs(s - np.nanmedian(dc[col].to_numpy(float))))
            if np.isfinite(a) and a > best_a:
                best_f, best_a = c, a

        # 公平口径：17 维特征的多变量检测器（5 折 CV，样本外评分）
        # 「最佳单特征」是逐条件挑出来的 oracle 上界，不能当作可达性能。
        qcols = [f"qc_{c}" for c in QC_FEATS if f"qc_{c}" in d.columns]
        Xa = np.nan_to_num(dc[qcols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        Xb = np.nan_to_num(d[qcols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.r_[Xa, Xb]
        sc = np.zeros(len(X))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(X, y_det):
            sz = StandardScaler().fit(X[tr])
            lr = LogisticRegression(max_iter=2000, C=1.0)
            lr.fit(sz.transform(X[tr]), y_det[tr])
            sc[te] = lr.predict_proba(sz.transform(X[te]))[:, 1]
        auc_mv = safe_auc(y_det, sc)

        # --- T2 条件内难度预测：只用该条件内部，预测模型是否判错
        lab = d["label"].to_numpy()
        keep = lab != 2
        dd = d[keep].reset_index(drop=True)
        yy = (dd["label"].to_numpy() == 1).astype(int)
        L = dd[["l1_0", "l1_1", "l1_2"]].to_numpy(float) / args.t1
        p = np.exp(L - L.max(1, keepdims=True)); p = p[:, 1] / p.sum(1)
        err = ((p >= 0.5).astype(int) != yy).astype(int)
        auc_err_qr = safe_auc(err, dd["qc_quality_risk"].to_numpy(float))
        conf = np.maximum(p, 1 - p)
        auc_err_cf = safe_auc(err, -conf)

        rows.append(dict(cond=cond, n=n, err=float(err.mean()),
                         auc_detect_qr=auc_qr, auc_detect_mv=auc_mv, auc_detect_best_feat=best_a,
                         detect_feat=best_f, auc_err_qr=auc_err_qr,
                         auc_err_conf=auc_err_cf))
        print(f"{cond:<12} 降质识别 quality_risk={auc_qr:.4f} 多变量={auc_mv:.4f}（最佳单特征 {best_f} {best_a:.4f}）  "
              f"| 条件内错误预测 AUC={auc_err_qr:.4f}（置信度 {auc_err_cf:.4f}）", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, f"table_detect_vs_difficulty_{args.tag}.csv"), index=False)
    print(f"\n汇总 {args.tag}：降质识别 平均 AUC={out['auc_detect_qr'].mean():.4f}（多变量 {out['auc_detect_mv'].mean():.4f}）  "
          f"| 条件内错误预测 平均 AUC(质量)={out['auc_err_qr'].mean():.4f}  "
          f"(置信度)={out['auc_err_conf'].mean():.4f}")


if __name__ == "__main__":
    main()
