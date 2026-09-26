"""方向二（实验 A + B）分析：几何 TTA 分歧的信号强度，以及跨条件配对分歧。

实验 A —— 视图预算 K 扫描
    对每个条件，用前 K 个几何视图（K ∈ {2,4,6,8,12,16}）计算**视图间分歧**
    （各视图 p 的标准差），评估它对软错误的排序能力。
    K=2 时 std ≡ |p₀ − p₁| / 2，与既有报告的「翻转 TTA 分歧」单调等价，
    因此 K=2 那一列应当精确复现真实翻拍上的 0.682 —— 这是一个管线哨兵。

实验 B —— 跨条件配对分歧（诊断性）
    CheXphoto 的四条件共享同一批 202 张基础图像，因此可以逐图构造
        Δ_c = p_c − p_clean
    在真实翻拍条件内部检验 |Δ| 能否预测软错误。它需要配对结构、不能用于单张
    推理，但直接回答机制问题：模型的失效是否集中在「被降质改变最多」的图上。

口径
----
* 软错误 = 1 − p(真实标签)，阈值无关（见 CheXphoto 报告 §3.4 的修正一）
* AUROC 用软错误中位数切分；CI 为患者级整群 Bootstrap
* 冻结温度 T1 = 0.4943421483039856
* 正交性检查：分歧与主分数 p 的 Spearman 秩相关（要求接近 0 才算独立信号）

用法
----
    python tta_geom_analyze.py \
        --tta results/tta_geom.csv --preds results/preds.csv --out results/

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
CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
SHORT = {"clean": "干净原图", "natural/oneplus": "真实翻拍",
         "synthetic/photographic": "合成·翻拍", "synthetic/digital": "合成·数字"}
VIEWS = [f"v{i:02d}" for i in range(16)]
K_PREFIX = [2, 4, 6, 8, 12, 16]
# 已有的单次翻转基线（CheXphoto 报告 §3.4 表 5）
BASELINE_FLIP = {"clean": 0.899, "natural/oneplus": 0.682,
                 "synthetic/photographic": 0.876, "synthetic/digital": 0.898}


def softmax_pos(L, T=T1):
    X = torch.from_numpy(np.asarray(L, dtype=np.float64) / T)
    return torch.softmax(X, dim=1).numpy()[:, 1]


def view_p(df: pd.DataFrame, v: str) -> np.ndarray:
    return softmax_pos(df[[f"{v}_0", f"{v}_1", f"{v}_2"]].to_numpy(float))


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or np.unique(s[ok]).size < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def split_bin(y_cont):
    return (np.asarray(y_cont, float) > float(np.median(y_cont))).astype(int)


def cluster_boot_auc(y, s, groups, n_boot=800, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, s[idx]))
    if len(out) < 50:
        return np.nan, np.nan
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tta)
    df["y"] = df["label"].astype(int)
    P = {v: view_p(df, v) for v in VIEWS}
    df["p_main"] = P["v00"]
    df["soft_err"] = np.where(df["y"] == 1, 1.0 - df["p_main"], df["p_main"])

    # ------------------------------------------------------------ 哨兵
    print("=" * 100)
    print("0  哨兵校验：v00 是否为既有管线的主前向")
    print("=" * 100)
    pr = pd.read_csv(args.preds)
    pr = pr[pr["orient"] == "frontal"].reset_index(drop=True)
    m = df.merge(pr[["Path", "l1_0", "l1_1", "l1_2"]],
                 left_on="disk_path", right_on="Path", how="inner")
    dmax = np.abs(m[["v00_0", "v00_1", "v00_2"]].to_numpy(float)
                   - m[["l1_0", "l1_1", "l1_2"]].to_numpy(float)).max()
    print(f"  匹配行数 {len(m)}（命中 {len(m)/len(df):.1%}）  "
          f"v00 与既有 l1 logits 最大偏差 {dmax:.3e}")
    print("  " + ("一致，管线忠实。" if dmax < 1e-3 else "!! 偏差过大，需排查 !!"))

    # ------------------------------------------------------------ 分歧的机制诊断
    print("\n" + "=" * 100)
    print("1  机制诊断：各条件的视图分歧水平（若「分布压缩」假说成立，真实翻拍应最小）")
    print("=" * 100)
    print(f"{'条件':<14}{'K=2':>10}{'K=4':>10}{'K=8':>10}{'K=16':>10}{'p 均值':>10}{'p 最大':>10}")
    for c in CONDS:
        mm = (df["cond"] == c).to_numpy()
        cells = []
        for K in (2, 4, 8, 16):
            Pm = np.stack([P[v][mm] for v in VIEWS[:K]], axis=1)
            cells.append(Pm.std(axis=1).mean())
        p = df.loc[mm, "p_main"].to_numpy()
        print(f"{SHORT[c]:<14}" + "".join(f"{v:>10.4f}" for v in cells)
              + f"{p.mean():>10.4f}{p.max():>10.4f}")

    # ------------------------------------------------------------ 实验 A
    print("\n" + "=" * 100)
    print("2  实验 A：视图预算 K 扫描——分歧对软错误的 AUROC（患者级聚类 95% CI）")
    print("=" * 100)
    print(f"  参照：单次翻转的既有数值 "
          f"{ {SHORT[c]: BASELINE_FLIP[c] for c in CONDS} }")
    print(f"\n{'条件':<14}" + "".join(f"{'K=' + str(K):>20}" for K in K_PREFIX))
    rows = []
    for c in CONDS:
        mm = (df["cond"] == c).to_numpy()
        yc = df.loc[mm, "soft_err"].to_numpy(float)
        y = split_bin(yc)
        grp = df.loc[mm, "Patient"].to_numpy()
        pv = df.loc[mm, "p_main"].to_numpy(float)
        line = f"{SHORT[c]:<14}"
        for K in K_PREFIX:
            Pm = np.stack([P[v][mm] for v in VIEWS[:K]], axis=1)
            div = Pm.std(axis=1)
            a = auroc(y, div)
            lo, hi = cluster_boot_auc(y, div, grp)
            rho_p = spearman(div, pv)
            star = "*" if (np.isfinite(lo) and (lo > 0.5 or hi < 0.5)) else " "
            line += f"{a:>9.3f}[{lo:.2f},{hi:.2f}]{star}".rjust(20)
            rows.append({"cond": c, "K": K, "auroc": a, "lo": lo, "hi": hi,
                         "rho_soft": spearman(div, yc), "rho_p": rho_p,
                         "div_mean": float(div.mean())})
        print(line)
    tA = pd.DataFrame(rows)
    tA.to_csv(args.out / "table_tta_geom_K.csv", index=False)
    print("\n  * = 95% CI 不含 0.5")

    # ------------------------------------------------------------ 正交性
    print("\n" + "=" * 100)
    print("3  正交性检查：分歧与主分数 p 的秩相关（|ρ| 越小越独立）")
    print("=" * 100)
    print(f"{'条件':<14}" + "".join(f"{'K=' + str(K):>12}" for K in K_PREFIX))
    for c in CONDS:
        line = f"{SHORT[c]:<14}"
        sub = tA[tA["cond"] == c].set_index("K")
        for K in K_PREFIX:
            line += f"{sub.loc[K, 'rho_p']:>12.3f}"
        print(line)

    # ------------------------------------------------------------ 实验 B
    print("\n" + "=" * 100)
    print("4  实验 B：跨条件配对分歧 |p_c − p_clean| 在条件内部预测软错误")
    print("=" * 100)
    spec = df.set_index("base_key")["p_main"]
    clean_p = df[df["cond"] == "clean"].set_index("base_key")["p_main"]
    rowsB = []
    print(f"{'条件':<14}{'n':>5}{'|Δ| 均值':>11}{'AUROC':>9}{'95% CI':>20}{'ρ(Δ,软错误)':>14}")
    for c in CONDS:
        g = df[df["cond"] == c].set_index("base_key")
        common = g.index.intersection(clean_p.index)
        delta = (g.loc[common, "p_main"] - clean_p.loc[common]).abs()
        yc = g.loc[common, "soft_err"].to_numpy(float)
        y = split_bin(yc)
        grp = g.loc[common, "Patient"].to_numpy()
        a = auroc(y, delta.to_numpy())
        lo, hi = cluster_boot_auc(y, delta.to_numpy(), grp)
        rho = spearman(delta.to_numpy(), yc)
        star = "*" if (np.isfinite(lo) and (lo > 0.5 or hi < 0.5)) else " "
        print(f"{SHORT[c]:<14}{len(common):>5}{delta.mean():>11.4f}{a:>9.3f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>20}{star:>2}{rho:>12.3f}")
        rowsB.append({"cond": c, "n": len(common), "delta_mean": float(delta.mean()),
                      "auroc": a, "lo": lo, "hi": hi, "rho_soft": rho})
    pd.DataFrame(rowsB).to_csv(args.out / "table_tta_pairdelta.csv", index=False)

    # ------------------------------------------------------------ 关键对照
    print("\n" + "=" * 100)
    print("5  关键对照：真实翻拍上，「几何 TTA」相对「单次翻转 + QC」是否有实质提升")
    print("=" * 100)
    best = tA[tA["cond"] == "natural/oneplus"].sort_values("auroc", ascending=False).iloc[0]
    print(f"  单次翻转（既有）           0.682 [0.611, 0.750]")
    print(f"  QC 17 维（既有）           0.605 [0.528, 0.685]")
    print(f"  几何 TTA 最优 K={int(best['K'])}          "
          f"{best['auroc']:.3f} [{best['lo']:.3f}, {best['hi']:.3f}]")
    print(f"  提升                       {best['auroc'] - 0.682:+.3f}")
    print(f"\n  -> {args.out}/table_tta_geom_K.csv, {args.out}/table_tta_pairdelta.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
