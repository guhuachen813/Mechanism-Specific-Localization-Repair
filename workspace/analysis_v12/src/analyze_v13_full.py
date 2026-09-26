"""
analyze_v13_full.py — 修复后预测表的完整离线分析

覆盖四件事：
    A. 口径重算（二进制 AUROC / 两种置信度定义 / 两种分箱下的 ECE）
    B. 同覆盖率比较（official 内部排序取前 75%）
    C. 先验偏移修正的实测检验
    D. 质量信号与错误的关系（H1）

作者：ClawsGO Science Agent
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data_from_gpu02")
CLIP = 1e-12


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def load(seed: int, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, f"s{seed}_{split}.csv"))


def load_qc(seed: int, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, f"s{seed}_qc_{split}.csv"))


def binary(seed: int, split: str):
    """返回 (df, y, p_raw, p_renorm)，已排除 label==2。"""
    d = load(seed, split)
    d = d[d["label"] != 2].reset_index(drop=True)
    y = (d["label"] == 1).astype(int).to_numpy()
    p_raw = d["p1_positive"].to_numpy(float)
    p_re = p_raw / np.clip(p_raw + d["p1_negative"].to_numpy(float), CLIP, None)
    return d, y, p_raw, p_re


def ece(y, p, n_bins):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    if n_bins == 5:
        edges = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    else:
        edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges) - 2)
    return float(sum((idx == b).sum() / len(y) *
                     abs(correct[idx == b].mean() - conf[idx == b].mean())
                     for b in range(len(edges) - 1) if (idx == b).sum()))


def risk_at_coverage(y, p, target):
    """按置信度降序取前 target 比例，返回 (实际覆盖率, 选择性风险)。"""
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf)
    k = int(round(target * len(y)))
    ys, ps = y[order][:k], p[order][:k]
    pred = (ps >= 0.5).astype(int)
    return k / len(y), float((pred != ys).mean())


def full_coverage_risk(y, p, thr=0.5):
    return float(((p >= thr).astype(int) != y).mean())


# ============================== A ==============================
def part_A():
    print("=" * 100)
    print("A. 指标重算（两种置信度定义 × 两种分箱）")
    print("=" * 100)
    print(f"{'集合':<22}{'n':>6}{'阳性率':>8} | {'AUROC':>7}{'AUPRC':>7}"
          f" | {'ECE5':>7}{'ECE15':>7}{'ECE30':>7} | {'acc@0.5':>8}{'acc@0.38':>9}")
    print("-" * 100)
    out = {}
    for seed in (42, 7):
        for split, tag in (("route_validation_predictions", "route"),
                           ("official_valid_predictions", "official")):
            _, y, p_raw, p_re = binary(seed, split)
            for cname, p in (("raw", p_raw), ("renorm", p_re)):
                auc = roc_auc_score(y, p)
                ap = average_precision_score(y, p)
                e5, e15, e30 = ece(y, p, 5), ece(y, p, 15), ece(y, p, 30)
                a5 = 1 - full_coverage_risk(y, p, 0.5)
                a38 = 1 - full_coverage_risk(y, p, 0.38)
                key = f"seed{seed}/{tag}/{cname}"
                out[key] = dict(auc=auc, ap=ap, e5=e5, e15=e15, e30=e30, a5=a5)
                print(f"{key:<22}{len(y):>6}{y.mean():>8.4f} | {auc:>7.4f}{ap:>7.4f}"
                      f" | {e5:>7.4f}{e15:>7.4f}{e30:>7.4f} | {a5:>8.4f}{a38:>9.4f}")
        print()
    return out


# ============================== B ==============================
def part_B():
    print("=" * 100)
    print("B. 同覆盖率比较：official 内部按置信度排序取前 75%")
    print("=" * 100)
    print(f"{'':<10}{'route@75%':>12}{'official 自身@75%':>18}{'official 绝对cutoff':>21}")
    print("-" * 100)
    for seed in (42, 7):
        _, yr, pr, _ = binary(seed, "route_validation_predictions")
        _, yo, po, _ = binary(seed, "official_valid_predictions")
        cov_r, risk_r = risk_at_coverage(yr, pr, 0.75)
        cov_o, risk_o = risk_at_coverage(yo, po, 0.75)
        # 绝对 cutoff：用 route 选出的 75% 分位置信度
        conf_r = np.maximum(pr, 1 - pr)
        cutoff = np.quantile(conf_r, 0.25)
        conf_o = np.maximum(po, 1 - po)
        m = conf_o >= cutoff
        cov_abs = m.mean()
        risk_abs = float(((po[m] >= 0.5).astype(int) != yo[m]).mean())
        print(f"seed{seed:<6}{risk_r:>12.4f}{risk_o:>18.4f}{risk_abs:>21.4f}")
        print(f"{'':<10}{'(cov 75.0%)':>12}{f'(cov {cov_o*100:.1f}%)':>18}"
              f"{f'(cov {cov_abs*100:.1f}%)':>21}   cutoff={cutoff:.4f}")
        print(f"{'':<10}{'':>12}{'← 同口径':>18}{'← 现报告口径':>21}")
    print("\n解读：第三列是现在报告用的口径（把 route 的绝对 cutoff 搬到 official），")
    print("      第二列才是与 route 真正可比的同覆盖率数字。")


# ============================== C ==============================
def part_C():
    print("\n" + "=" * 100)
    print("C. 先验偏移修正的实测检验")
    print("=" * 100)
    for seed in (42, 7):
        _, yr, pr, _ = binary(seed, "route_validation_predictions")
        _, yo, po, _ = binary(seed, "official_valid_predictions")
        pi_s, pi_t = yr.mean(), yo.mean()
        shift = logit(pi_t) - logit(pi_s)
        print(f"\nseed{seed}:  π_s(route)={pi_s:.4f}  π_t(official)={pi_t:.4f}  "
              f"先验偏移 Δ={shift:+.4f}")
        base = ece(yo, po, 15)
        print(f"  official ECE15  未修正               : {base:.4f}")
        best = None
        for a in np.linspace(-2.5, 2.5, 501):
            v = ece(yo, sigm(logit(po) + a), 15)
            if best is None or v < best[0]:
                best = (v, a)
        print(f"  official ECE15  施加理论先验偏移 Δ   : {ece(yo, sigm(logit(po)+shift), 15):.4f}")
        print(f"  official ECE15  最优偏移（事后上界） : {best[0]:.4f}  (最优 Δ*={best[1]:+.3f})")
        # 准确率影响
        print(f"  准确率@0.5  未修正 {1-full_coverage_risk(yo,po):.4f} → "
              f"先验修正后 {1-full_coverage_risk(yo, sigm(logit(po)+shift)):.4f}")
        # 覆盖率会发生什么
        c0 = np.maximum(po, 1-po)
        c1 = np.maximum(sigm(logit(po)+shift), 1-sigm(logit(po)+shift))
        print(f"  平均置信度 {c0.mean():.4f} → {c1.mean():.4f}")


# ============================== D ==============================
def part_D():
    print("\n" + "=" * 100)
    print("D. 质量信号与错误的关系（H1）")
    print("=" * 100)
    for seed in (42, 7):
        for split, tag in (("route_validation", "route"), ("official_valid", "official")):
            pred = load(seed, f"{split}_predictions")
            qc = load_qc(seed, split)
            m = pred.merge(qc[["Path", "quality_risk", "hard_fail", "foreground_ratio",
                               "dynamic_range", "contrast", "blur_score", "noise_score"]],
                           on="Path", how="inner", suffixes=("", "_qc"))
            m = m[m["label"] != 2].reset_index(drop=True)
            y = (m["label"] == 1).astype(int).to_numpy()
            p = m["p1_positive"].to_numpy(float)
            err = ((p >= 0.5).astype(int) != y).astype(int)
            conf = np.maximum(p, 1 - p)
            qr = m["quality_risk"].to_numpy(float)
            print(f"\nseed{seed} / {tag}  (n={len(m)}, 错误率={err.mean():.4f})")
            print(f"  quality_risk: 均值={qr.mean():.4f} 标准差={qr.std():.4f} "
                  f"唯一值数={len(np.unique(qr))}  范围=[{qr.min():.3f},{qr.max():.3f}]")
            corr = np.corrcoef(qr, err)[0, 1] if qr.std() > 0 else np.nan
            corrc = np.corrcoef(qr, conf)[0, 1] if qr.std() > 0 else np.nan
            print(f"  corr(quality_risk, error)      = {corr:+.4f}")
            print(f"  corr(quality_risk, confidence) = {corrc:+.4f}")
            print(f"  hard_fail 触发数 = {int(m['hard_fail'].sum()) if 'hard_fail' in m else 'NA'}")
            if qr.std() > 0:
                q = pd.qcut(qr, min(4, len(np.unique(qr))), duplicates="drop")
                g = pd.DataFrame({"q": q, "err": err, "conf": conf}).groupby("q", observed=True)
                print("  按 quality_risk 分位：")
                for k, v in g:
                    print(f"    {str(k):<22} n={int(v['err'].count()):>5} "
                          f"错误率={v['err'].mean():.4f} 平均置信度={v['conf'].mean():.4f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    part_A()
    part_B()
    part_C()
    part_D()
