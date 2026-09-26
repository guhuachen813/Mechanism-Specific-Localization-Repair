r"""修复器门控 v2：三选一 {raw, prior_hard, cand} + 训练折内阈值校准。

v1 复盘：gate 从 {raw,cand} 二选一，切掉近半收益（gated 0.362 < cand 0.382）。
v2 改动：
  1. 选择集扩为 {raw, prior, cand}（利用先验/修复器互补性）。
  2. 阈值校准：训练折内按患者 80/20 划 train_g/calib，δ 在 calib 上网格搜索
     （决策：a* = argmax P(a)；非 raw 选项需 P(a*) − P(raw) ≥ δ，否则 no-op）。
  3. 评估仍为患者级 5 折折外：fold f 的门控只在其余折上训练/校准。

标签：y = argmax{0, prior_45−raw_45, cand_45−raw_45}（训练折内，v1 同口径）。
报告 oracle（逐样本 argmax，GT 上界）作参照。

输出：repairer/gate_v2_eval.csv + 控制台汇总。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPAIR_DIR = "/root/autodl-tmp/experiments/rep_robust/repairer"
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GateV2(nn.Module):
    def __init__(self, n_in: int = 9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def patient_of(fname: str) -> str:
    return fname.split(".")[0].split("_")[0]


def train_gate(X: np.ndarray, y: np.ndarray, seed: int) -> GateV2:
    torch.manual_seed(seed)
    g = GateV2(X.shape[1])
    opt = torch.optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    for _ in range(300):
        opt.zero_grad()
        loss = lossf(g(xt), yt)
        loss.backward()
        opt.step()
    return g.eval()


def apply_gate(g: GateV2, X: np.ndarray, out: dict[str, np.ndarray],
               deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (iou45, choice, margin)。out: {'raw','prior','cand'} -> 45 值。"""
    with torch.no_grad():
        p = torch.softmax(g(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
    choice = p.argmax(1)                    # 0 raw / 1 prior / 2 cand
    margin = p[np.arange(len(p)), choice] - p[:, 0]
    pick = choice.copy()
    pick[(choice > 0) & (margin < deltas)] = 0
    iou = out["raw"].copy()
    iou[pick == 1] = out["prior"][pick == 1]
    iou[pick == 2] = out["cand"][pick == 2]
    return iou, pick, margin


def main() -> int:
    df = pd.read_csv(f"{REPAIR_DIR}/repairer_eval.csv")
    z = np.load(f"{REPAIR_DIR}/cache.npz", allow_pickle=True)
    files, conds = list(z["files"]), list(z["conds"])
    qs = np.concatenate([z["qs"], z["cs"]], axis=-1)      # (146,17,9)
    floc = {f: i for i, f in enumerate(files)}
    cloc = {c: j for j, c in enumerate(conds)}
    X = np.stack([qs[floc[r.nih_file], cloc[r.condition]]
                  for r in df.itertuples()]).astype(np.float32)
    raw = df.raw_45.to_numpy()
    prior = df.prior_45.to_numpy()
    cand = df.cand_45.to_numpy()
    folds = df.fold.to_numpy()
    patients = df.nih_file.map(patient_of).to_numpy()
    log(f"样本 {len(df)}，特征 {X.shape}")

    # 折外门控
    g2 = np.full(len(df), np.nan)
    ch = np.full(len(df), -1)
    mg = np.full(len(df), np.nan)
    for f in sorted(set(folds)):
        te = folds == f
        tr = ~te
        # 训练折内按患者 80/20 划 train_g / calib
        rng = np.random.default_rng(SEED + int(f))
        pts = np.array(sorted(set(patients[tr])))
        calib_p = set(pts[rng.permutation(len(pts))[: max(1, len(pts) // 5)]])
        is_calib = np.array([p in calib_p for p in patients[tr]])
        Xtr, y_all = X[tr], np.stack([np.zeros(tr.sum()),
                                      prior[tr] - raw[tr],
                                      cand[tr] - raw[tr]]).argmax(0)
        g = train_gate(Xtr[~is_calib], y_all[~is_calib], SEED + int(f))
        out_tr = {"raw": raw[tr], "prior": prior[tr], "cand": cand[tr]}
        # δ 校准（calib 子集）
        best = (-1e9, 0.0)
        for d in np.arange(0.0, 0.55, 0.05):
            iou_c, _, _ = apply_gate(g, Xtr[is_calib],
                                     {k: v[is_calib] for k, v in out_tr.items()},
                                     np.full(is_calib.sum(), d))
            if iou_c.mean() > best[0]:
                best = (iou_c.mean(), d)
        d_star = best[1]
        iou_te, pick_te, marg_te = apply_gate(g, X[te],
                                              {"raw": raw[te], "prior": prior[te],
                                               "cand": cand[te]},
                                              np.full(te.sum(), d_star))
        g2[te], ch[te], mg[te] = iou_te, pick_te, marg_te
        cnt = np.bincount(pick_te, minlength=3) / te.sum()
        log(f"fold {f}: δ*={d_star:.2f}  gated2 {iou_te.mean():.3f}  "
            f"choice raw/prior/cand {cnt[0]:.2f}/{cnt[1]:.2f}/{cnt[2]:.2f}")

    df["gated2_45"] = g2
    df["gated2_choice"] = ch
    df["gated2_margin"] = mg
    oracle = np.maximum(raw, np.maximum(prior, cand))
    df["oracle_45"] = oracle
    df.to_csv(f"{REPAIR_DIR}/gate_v2_eval.csv", index=False)

    # 汇总 + 按图整群 Bootstrap
    rng = np.random.default_rng(SEED)
    uniq = df.nih_file.unique()
    fmap = {f: i for i, f in enumerate(uniq)}
    file_ord = df.nih_file.map(fmap).to_numpy()

    def boot(a, b):
        d = a - b
        vals = []
        for _ in range(400):
            sel = rng.choice(len(uniq), len(uniq))
            vals.append(d[np.isin(file_ord, sel)].mean())
        return np.mean(d), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    print("\n== 患者级 5 折折外（τ45 聚合） ==")
    for name, v in [("raw", raw), ("prior", prior), ("cand", cand),
                    ("gated_v1", df.gated_45.to_numpy()),
                    ("gated_v2", g2), ("oracle", oracle)]:
        print(f"  {name:10s} {v.mean():.3f}")
    for name, v in [("prior", prior), ("cand", cand), ("gated_v2", g2)]:
        m, lo, hi = boot(v, raw)
        star = "*" if lo > 0 else ""
        print(f"  {name:10s} vs raw: {m:+.3f} CI95 [{lo:+.3f},{hi:+.3f}]{star}")
    m, lo, hi = boot(g2, df.gated_45.to_numpy())
    print(f"  gated_v2 vs gated_v1: {m:+.3f} CI95 [{lo:+.3f},{hi:+.3f}]")

    print("\n== 逐条件 gated_v2 ==")
    piv = df.groupby("condition")[["raw_45", "prior_45", "cand_45",
                                   "gated_45", "gated2_45", "oracle_45"]].mean()
    print(piv.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
