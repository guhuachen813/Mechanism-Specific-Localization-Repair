"""方向二补充：把几何 TTA 与配对分歧放进**代价 / 选择性风险**框架。

动机
----
前两步用「软错误」作目标，而控制实验（tta_geom_control.py）表明该目标与 p 存在
定义性耦合——AUROC(p → 软错误) 高达 0.844~0.951，任何与 p 相关的信号都会被
系统性高估。本脚本改用**不依赖 p 的评估目标**：固定复核预算下的决策代价。

口径（对齐 v1.5 §7 的 F7，使结论可直接与既有结果对照）
--------------------------------------------------
* 代价 = (2FN + FP) / N，FN:FP = 2:1，决策阈值 0.5
* 覆盖率 c = 送人工复核（拒绝）的比例，按信号**从大到小**拒绝
* 在**保留集**上计算代价，并折算到保留集规模
* AURC = 风险–覆盖率曲线下面积（梯形积分，c ∈ [0, 0.9]）
* 不确定性用患者级整群 Bootstrap

信号
----
random     随机（固定 seed）
unc        1 − max(p, 1−p)                    标准不确定度
costrisk   代价加权期望风险（p<0.5 取 2p，否则取 1−p）——「纯置信度路由」
div2       单次翻转分歧 |p − p_flip|            既有口径
div12      几何 TTA 分歧（K=12 视图间 std）      本方向的新信号
pairdelta  跨条件配对分歧 |p_c − p_clean|        诊断性，需配对结构
qc17       17 维 QC 特征的 5 折 OOF 预测         既有的质量通道
oracle     真实错判代价（知道标签的上界，不可实现）

副表还给出 AUROC(信号 → 阳性标签 y)。在操作点塌缩条件下全部判阴、FP=0，
代价路由实质上退化为「能否把阳性挑出来」，因此这一列是代价结果的机制解释。

用法
----
    python tta_geom_selective.py --tta results/tta_geom.csv \
        --preds results/preds.csv --out results/

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T1 = 0.4943421483039856
CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}
VIEWS = [f"v{i:02d}" for i in range(16)]
K_USE = 12
COV_GRID = np.arange(0.0, 1.0, 0.1)
COV_POINTS = [0.50, 0.75]
QC_FEATS = ["width", "height", "aspect_ratio", "foreground_ratio", "dynamic_range",
            "mean_intensity", "intensity_std", "contrast", "blur_score", "noise_score",
            "black_ratio", "white_ratio", "border_crop_score", "left_right_symmetry",
            "center_offset", "projection_ap", "projection_unknown"]


def softmax_pos(L, T=T1):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T)
    return torch.softmax(X, dim=1).numpy()[:, 1]


def costs(y, pred, weights=(2, 1)):
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return weights[0] * fn + weights[1] * fp


def risk_at(y, p, s, cov, thr=0.5):
    """按信号 s 降序拒绝 cov 比例，返回保留集上的代价（每样本）。"""
    n = len(y)
    k = int(round(cov * n))
    k = min(max(k, 0), n - 1)
    order = np.argsort(-np.asarray(s, float), kind="stable")
    keep = order[k:]
    if len(keep) == 0:
        return np.nan
    pred = (p[keep] >= thr).astype(int)
    return costs(y[keep], pred) / len(keep)


def _trapz(r, x):
    """梯形积分（numpy 2.x 已移除 np.trapz）。"""
    r = np.asarray(r, float)
    x = np.asarray(x, float)
    return float(np.sum((r[1:] + r[:-1]) / 2.0 * np.diff(x)))


def aurc(y, p, s, thr=0.5):
    r = np.array([risk_at(y, p, s, c, thr) for c in COV_GRID], float)
    return _trapz(r, COV_GRID) / (COV_GRID[-1] - COV_GRID[0])


def cluster_boot(fn, groups, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        out.append(fn(idx))
    out = np.asarray([o for o in out if np.isfinite(o)], float)
    if len(out) < 50:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def qc_oof(X, y, seed, n_splits=5):
    oof = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return oof
    k = min(n_splits, int(np.bincount(y.astype(int)).min()))
    if k < 2:
        return oof
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                           max_leaf_nodes=15, min_samples_leaf=20,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tta)
    pr = pd.read_csv(args.preds)
    pr = pr[pr["orient"] == "frontal"].reset_index(drop=True)
    qcols = [f"qc_{k}" for k in QC_FEATS]
    df = df.merge(pr[["Path"] + qcols], left_on="disk_path", right_on="Path",
                  how="left", suffixes=("", "_ref"))
    df["y"] = df["label"].astype(int)
    P = {v: softmax_pos(df[[f"{v}_0", f"{v}_1", f"{v}_2"]].to_numpy(float))
         for v in VIEWS}
    df["p"] = P["v00"]
    df["p_flip"] = P["v01"]
    df["pred"] = (df["p"] >= 0.5).astype(int)
    df["err_cost"] = np.where((df["pred"] == 0) & (df["y"] == 1), 2.0,
                              np.where((df["pred"] == 1) & (df["y"] == 0), 1.0, 0.0))
    clean_p = df[df["cond"] == "clean"].set_index("base_key")["p"]

    print("=" * 106)
    print("代价 / 选择性风险框架：固定复核预算下，各信号能否压低保留集的决策代价")
    print("=" * 106)
    print(f"  口径：代价=(2FN+FP)/N，阈值 0.5，按信号降序拒绝；AURC 为 c∈[0,0.9] 的梯形积分")
    print(f"  Bootstrap {args.boot} 次，患者级整群\n")

    rows, rows50, rows75, rows_auc = [], [], [], []
    for c in CONDS:
        g = df[df["cond"] == c].reset_index(drop=True)
        y = g["y"].to_numpy(int)
        p = g["p"].to_numpy(float)
        grp = g["Patient"].to_numpy()
        Pm = np.stack([P[v][(df["cond"] == c).to_numpy()] for v in VIEWS[:K_USE]], axis=1)
        div12 = Pm.std(axis=1)
        div2 = np.abs(p - g["p_flip"].to_numpy(float))
        unc = 1.0 - np.maximum(p, 1.0 - p)
        costrisk = np.where(p < 0.5, 2.0 * p, 1.0 - p)
        # QC 17 维 OOF（目标 = 是否被错判，口径同 v1.5 §5.2）
        ymiss = (g["err_cost"].to_numpy() > 0).astype(int)
        Xq = g[qcols].to_numpy(float)
        qc = np.nanmean([qc_oof(Xq, ymiss, s) for s in range(5)], axis=0)
        # 跨条件配对分歧
        if c == "clean":
            paird = np.full(len(g), np.nan)
        else:
            idx = g.set_index("base_key").index.intersection(clean_p.index)
            s_map = pd.Series(
                (g.set_index("base_key").loc[idx, "p"] - clean_p.loc[idx]).abs(),
                index=idx)
            paird = g["base_key"].map(s_map).to_numpy(float)

        rng = np.random.default_rng(0)
        # 组合信号：代价加权置信度 与 几何 TTA 分歧 的秩平均。
        # 两者若提供互补信息，组合的 AURC 应低于任一单路。
        _rk = lambda a: pd.Series(np.asarray(a, float)).rank().to_numpy()
        combo = (_rk(costrisk) + _rk(div12)) / 2.0
        signals = [
            ("random", rng.random(len(g))),
            ("unc", unc),
            ("costrisk", costrisk),
            ("div2", div2),
            ("div12", div12),
            ("pairdelta", paird),
            ("qc17", qc),
            ("combo", combo),
            ("oracle", g["err_cost"].to_numpy(float)),
        ]

        print(f"--- {SHORT[c]}（n={len(g)}，阳性 {int(y.sum())}）---")
        print(f"   {'信号':<12}{'AURC':>9}{'95% CI':>20}{'c=50%':>9}{'c=75%':>9}"
              f"{'AUROC(→y)':>11}")
        for nm, s in signals:
            s = np.asarray(s, float)
            if not np.isfinite(s).all():
                print(f"   {nm:<12}{'—':>9}{'（不适用）':>20}{'—':>9}{'—':>9}{'—':>11}")
                continue
            a = aurc(y, p, s)
            lo, hi = cluster_boot(
                lambda idx, s=s: aurc(y[idx], p[idx], s[idx]), grp, n_boot=args.boot)
            r50 = risk_at(y, p, s, 0.50)
            r75 = risk_at(y, p, s, 0.75)
            rng2 = np.random.default_rng(1)
            a_y = (float(roc_auc_score(y, s)) if len(np.unique(s)) > 1
                   else np.nan)
            star = "*" if (np.isfinite(lo) and np.isfinite(hi)) else " "
            print(f"   {nm:<12}{a:>9.4f}{f'[{lo:.4f},{hi:.4f}]':>20}{r50:>9.4f}"
                  f"{r75:>9.4f}{a_y:>11.4f}")
            rows.append({"cond": c, "signal": nm, "aurc": a, "aurc_lo": lo,
                         "aurc_hi": hi, "risk50": r50, "risk75": r75, "auroc_y": a_y})
        print()

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "table_tta_selective.csv", index=False)

    # ---------------------------------------------------------------- 关键对照
    print("=" * 106)
    print("关键对照：真实翻拍上，几何 TTA（div12）相对既有信号是否带来实质改善")
    print("=" * 106)
    sub = t[t["cond"] == "natural/oneplus"].set_index("signal")
    print(f"   {'信号':<12}{'AURC':>9}{'c=50% 风险':>12}{'c=75% 风险':>12}")
    for nm in ["random", "unc", "costrisk", "div2", "qc17", "div12", "pairdelta",
               "combo", "oracle"]:
        r = sub.loc[nm]
        print(f"   {nm:<12}{r['aurc']:>9.4f}{r['risk50']:>12.4f}{r['risk75']:>12.4f}")
    d = sub.loc["div2", "aurc"] - sub.loc["div12", "aurc"]
    print(f"\n   div12 相对 div2 的 AURC 降幅：{d:+.4f}"
          f"（{'几何 TTA 更优' if d > 0 else '几何 TTA 并未更优'}）")
    d2 = sub.loc["div12", "aurc"] - sub.loc["costrisk", "aurc"]
    print(f"   div12 相对 costrisk 的 AURC 降幅：{d2:+.4f}"
          f"（{'几何 TTA 更优' if d2 > 0 else '几何 TTA 并未更优'}）")

    # ---------------------------------------------------------------- 无条件汇总
    print("\n" + "=" * 106)
    print("四条件汇总：AURC 均值（越低越好）")
    print("=" * 106)
    piv = t.pivot_table(index="signal", columns="cond", values="aurc")
    piv = piv[[c for c in CONDS if c in piv.columns]]
    piv["均值"] = piv.mean(axis=1)
    piv = piv.sort_values("均值")
    print(piv.round(4).to_string())
    piv.to_csv(args.out / "table_tta_selective_pivot.csv")

    print(f"\n  -> {args.out}/table_tta_selective.csv, table_tta_selective_pivot.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
