"""
verify_metrics_v13.py — 从预测表直接重算指标，钉死口径

背景：用户报告 route AUROC = 0.8440 / official = 0.8289，
但机器上 repaired_seed42/densenet121/*_metrics.json 报 0.6220 / 0.6065。
两者相差巨大，必须先确定哪一个是对的、以及各自是什么口径。

已知指纹（repaired_binary_ece.json，seed42_official，5 个 bin）：
    [0.9,1.0)  rows=143  mean_confidence=0.9754759471196359  accuracy=0.8321678321678322
用它反推置信度定义。

作者：ClawsGO Science Agent
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data_from_gpu02")
EDGES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def load(seed: int, split: str) -> pd.DataFrame:
    f = os.path.join(DATA, f"s{seed}_{split}.csv")
    df = pd.read_csv(f)
    # 二元化标签：-1/2 视为不确定，排除；NaN→0 已在 manifest 处理
    df["label"] = df["label"].astype(int)
    return df


def p_pos_variants(df: pd.DataFrame) -> dict:
    raw = df["p1_positive"].to_numpy(float)
    neg = df["p1_negative"].to_numpy(float)
    renorm = raw / np.clip(raw + neg, 1e-12, None)
    return {"raw": raw, "renorm": renorm}


def ece_5bin(y, p):
    """复刻 repaired_binary_ece.json 的 5 分箱（0.5→1.0，步长 0.1）口径。"""
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    idx = np.clip(np.digitize(conf, EDGES[1:-1]), 0, len(EDGES) - 2)
    tot, rows = 0.0, []
    for b in range(len(EDGES) - 1):
        m = idx == b
        if m.sum() == 0:
            rows.append((f"[{EDGES[b]},{EDGES[b+1]})", 0, np.nan, np.nan, np.nan))
            continue
        mc, ac = conf[m].mean(), correct[m].mean()
        tot += m.sum() / len(y) * abs(ac - mc)
        rows.append((f"[{EDGES[b]},{EDGES[b+1]})", int(m.sum()), mc, ac, abs(ac - mc)))
    return float(tot), rows


def ece_15bin(y, p, n_bins=15):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).sum() / len(y) *
                     abs(correct[idx == b].mean() - conf[idx == b].mean())
                     for b in range(n_bins) if (idx == b).sum()))


if __name__ == "__main__":
    print("=" * 92)
    print("一、口径反推：seed42 official 的 143 行高置信 bin")
    print("   已知指纹 mean_confidence = 0.9754759471196359, accuracy = 0.8321678321678322")
    print("=" * 92)
    d = load(42, "official_valid_predictions")
    y = (d["label"] == 1).astype(int).to_numpy()
    print(f"行数={len(d)}  label 分布={dict(d['label'].value_counts().sort_index())}")
    print(f"阳性率={y.mean():.6f}   （报告 positive_rate=0.32673267326732675）\n")

    for name, p in p_pos_variants(d).items():
        conf = np.maximum(p, 1 - p)
        top = np.argsort(-conf)[:143]
        print(f"  p_pos 取 {name:7s}: 最高 143 行的 mean_conf={conf[top].mean():.16f} "
              f"acc={( (p[top]>=0.5).astype(int)==y[top] ).mean():.16f}")
        e5, rows = ece_5bin(y, p)
        print(f"      5-bin ECE={e5:.16f}   （目标 0.18086152190949403）")
        for r in rows:
            print(f"        {r[0]:12s} n={r[1]:4d} conf={r[2]:.16f} "
                  f"acc={r[3]:.16f} gap={r[4]:.16f}" if r[1] else f"        {r[0]} 空")
        print()

    print("=" * 92)
    print("二、四个集合的完整重算（排除 label==2 的不确定行）")
    print("=" * 92)
    hdr = (f"{'seed/split':<26}{'n':>6}{'已知n':>7}{'阳性率':>9}{'AUROC':>9}"
           f"{'AUPRC':>8}{'acc':>8}{'ECE5':>8}{'ECE15':>8}{'T':>7}")
    print(hdr)
    print("-" * len(hdr))
    for seed in (42, 7):
        for split in ("route_validation_predictions", "official_valid_predictions"):
            d = load(seed, split)
            known = d[d["label"] != 2]
            y = (known["label"] == 1).astype(int).to_numpy()
            for vname, p_all in p_pos_variants(known).items():
                if vname != "renorm":
                    continue
                auc = roc_auc_score(y, p_all)
                ap = average_precision_score(y, p_all)
                pred = (p_all >= 0.5).astype(int)
                acc = (pred == y).mean()
                e5, _ = ece_5bin(y, p_all)
                e15 = ece_15bin(y, p_all)
                short = "route" if "route" in split else "official"
                print(f"seed{seed:<3}/{short:<16}{len(d):>6}{len(known):>7}"
                      f"{y.mean():>9.4f}{auc:>9.4f}{ap:>8.4f}{acc:>8.4f}"
                      f"{e5:>8.4f}{e15:>8.4f}")
