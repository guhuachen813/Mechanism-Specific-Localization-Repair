r"""5.1a v2：适配协议修正版——内层校准池 early-stop + δ 门控校准。

v1 发现：固定 60 epoch 适配全规模劣于 raw（-0.10），8 epoch 冒烟却接近 raw
→ 过拟合银标签。v2 给适配公平协议：
  1. 每 5 epoch 在内层校准患者（fit 池外 20%）上评 IoU45，保留最优 epoch 权重
  2. 报告最优 epoch 分布（诊断过拟合程度）
  3. 附加 selective 变体：校准池上学全局规则 c∈{always_raw, always_cand, per-img gate}
     per-img gate = 线性逻辑回归（q9 特征）预测 cand 相对 raw 是否为正，
     阈值取校准池上使 IoU45 均值最大的点
用法：/root/miniconda3/bin/python p5a_adapt_cv2.py
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
ART = "/root/autodl-tmp/experiments/rep_robust/repairer_v2_artifacts"
ALPHA = 0.5
N_OUTER = 5
EPOCHS_MAX = 60
BS = 16
LR = 1e-3
LAM_OUT, LAM_SM = 0.25, 0.05
TAU = 0.45
EVAL_EVERY = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RepairHead(torch.nn.Module):
    def __init__(self, nq: int = 9):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3 + nq, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, hraw, mask, q):
        B, _, H, W = hraw.shape
        qb = q[:, :, None, None].expand(B, q.shape[1], H, W)
        x = torch.cat([hraw, mask, hraw * mask, qb], dim=1)
        return torch.tanh(self.net(x))


def iou_curve(cam: np.ndarray, box) -> np.ndarray:
    taus = np.arange(0.05, 1.0, 0.05)
    out = np.zeros(len(taus))
    x1, y1, x2, y2 = box
    for i, t in enumerate(taus):
        m = cam >= t
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ix = max(0, min(x2, bx2) - max(x1, bx1))
        iy = max(0, min(y2, by2) - max(y1, by1))
        inter = ix * iy
        union = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter
        out[i] = inter / max(union, 1)
    return out


def iou_at(cam, box, tau=TAU):
    return float(iou_curve(cam, box)[int(round((tau - 0.05) / 0.05))])


def box_of(m: np.ndarray):
    ys, xs = np.where(m > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def train_head_es(idxs, cal_idx, D, device):
    """带校准池 early-stop 的适配头训练。返回 (head, best_epoch)。"""
    head = RepairHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    boxes = [box_of(D["gts"][i]) for i in idxs]
    Y = np.zeros((len(idxs), 1, 224, 224), np.float32)
    for j, (i, b) in enumerate(zip(idxs, boxes)):
        Y[j, 0, int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1.0
    rng = np.random.default_rng(0)
    best = (-1e9, None, 0)
    for ep in range(EPOCHS_MAX):
        perm = rng.permutation(len(idxs))
        for s0 in range(0, len(perm), BS):
            sel = perm[s0:s0 + BS]
            ii = np.array([idxs[s] for s in sel])
            h = torch.from_numpy(D["cams_p"][ii]).unsqueeze(1).to(device)
            m = torch.from_numpy(D["masks_p"][ii].astype(np.float32)).unsqueeze(1).to(device)
            y = torch.from_numpy(Y[sel]).to(device)
            q = torch.from_numpy(D["q9"][ii]).to(device)
            delta = head(h, m, q)
            cand = torch.clamp(h + ALPHA * delta, 0, 1)
            loss = F.binary_cross_entropy(cand, y) \
                + LAM_OUT * (cand * (1 - m)).mean() \
                + LAM_SM * ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                            + (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % EVAL_EVERY == 0 or ep == EPOCHS_MAX - 1:
            head.eval()
            with torch.no_grad():
                sc = []
                for i in cal_idx:
                    h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
                    m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
                    q = torch.from_numpy(D["q9"][i][None]).to(device)
                    cand = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[0, 0].cpu().numpy()
                    sc.append(iou_at(cand, box_of(D["gts"][i])))
            sc = float(np.mean(sc))
            if sc > best[0]:
                best = (sc, {k: v.clone() for k, v in head.state_dict().items()}, ep + 1)
            head.train()
    head.load_state_dict(best[1])
    head.eval()
    return head, best[2]


def apply_head(head, idxs, D, device):
    out = []
    with torch.no_grad():
        for i in idxs:
            h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
            m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
            q = torch.from_numpy(D["q9"][i][None]).to(device)
            cand = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[0, 0].cpu().numpy()
            out.append(cand)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,2024")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda")

    D = dict(np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True))
    keys = list(D["keys"])
    patients = np.array([k.split("_")[0] for k in keys])
    uniq_p = sorted(set(patients))
    N = len(keys)
    log(f"缓存 {N} 图 / {len(uniq_p)} 患者")

    from repairer import RepairHead as V2Head
    heads_v2 = []
    for kf in range(5):
        h = V2Head().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)
    cand_zs = []
    with torch.no_grad():
        for i in range(N):
            h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
            m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
            q = torch.from_numpy(D["q9"][i][None]).to(device)
            cs = [torch.clamp(h + ALPHA * hh(h, m, q), 0, 1)[0, 0].cpu().numpy()
                  for hh in heads_v2]
            cand_zs.append(np.median(np.stack(cs), 0))

    boxes = [box_of(D["gts"][i]) for i in range(N)]
    raw45 = np.array([iou_at(D["cams_p"][i], boxes[i]) for i in range(N)])
    zs45 = np.array([iou_at(cand_zs[i], boxes[i]) for i in range(N)])

    scales = [0, 5, 10, 20, 40, 52]
    rows = []
    for seed in seeds:
        rng0 = np.random.default_rng(seed)
        perm = rng0.permutation(len(uniq_p))
        folds = [sorted(uniq_p[j] for j in perm[k::N_OUTER]) for k in range(N_OUTER)]
        for k in range(N_OUTER):
            te_p = set(folds[k])
            tr_p = [p for j, f in enumerate(folds) if j != k for p in f]
            te_idx = [i for i in range(N) if patients[i] in te_p]
            rng1 = np.random.default_rng(seed * 10 + k)
            tr_shuf = list(rng1.permutation(len(tr_p)))
            n_cal = max(4, len(tr_p) // 5)
            cal_p = set(tr_p[j] for j in tr_shuf[:n_cal])
            fit_p = [p for p in tr_p if p not in cal_p]
            cal_idx = [i for i in range(N) if patients[i] in cal_p]

            for s in scales:
                if s == 0:
                    for sysname, cands in [("cand_zs", [cand_zs[i] for i in te_idx]),]:
                        d45 = np.array([iou_at(c, boxes[i]) for c, i in zip(cands, te_idx)]) - raw45[te_idx]
                        rows.append({"seed": seed, "fold": k, "scale": s, "sys": sysname,
                                     "raw45": float(raw45[te_idx].mean()),
                                     "delta45": float(d45.mean()),
                                     "win": float((d45 > 0).mean()),
                                     "harm": float((d45 < -0.05).mean()),
                                     "best_ep": 0})
                    continue
                take = fit_p if s >= len(fit_p) else list(rng1.choice(fit_p, s, replace=False))
                tidx = [i for i in range(N) if patients[i] in set(take)]
                head, bep = train_head_es(tidx, cal_idx, D, device)
                cands = apply_head(head, te_idx, D, device)
                d45 = np.array([iou_at(c, boxes[i]) for c, i in zip(cands, te_idx)]) - raw45[te_idx]
                rows.append({"seed": seed, "fold": k, "scale": s, "sys": "cand_es",
                             "raw45": float(raw45[te_idx].mean()),
                             "delta45": float(d45.mean()),
                             "win": float((d45 > 0).mean()),
                             "harm": float((d45 < -0.05).mean()),
                             "best_ep": bep})
            log(f"seed{seed} fold{k} 完成")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p5a_adapt_curve_v2.csv", index=False)
    print("\n== 标注量曲线 v2（early-stop 协议） ==")
    g = df.groupby("scale").agg(d45_mean=("delta45", "mean"), d45_std=("delta45", "std"),
                                win=("win", "mean"), harm=("harm", "mean"),
                                best_ep_med=("best_ep", "median"))
    print(g.round(3).to_string())
    piv = df.pivot_table(index=["seed", "fold"], columns="scale", values="delta45")
    for s in scales:
        if s == 0:
            continue
        d = piv[s] - piv[0]
        print(f"adapt{s:02d}(es) − zero-shot: {d.mean():+.3f} ± {d.std():.3f} (>0: {(d>0).sum()}/{len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
