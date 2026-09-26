r"""5.1a 主实验：目标域少样本适配 患者级嵌套 CV + 标注量曲线。

协议（计划 5.1 冻结）：
  - 外层：65 患者级 5 折（患者整群，seed 42），外层折只评估
  - 内层：外层 train 池内，按标注量 s ∈ {0,5,10,20,40,max} 抽患者训练适配头；
          独立内层患者（train 池剩余，约 12 患者校准池）用于 δ/阈值选择
  - s=0 = 零样本（v2 合成轴 5 折中位集成，不碰任何目标域监督）
  - 适配头：与 RepairHead 同构（3×3 CNN + tanh + α=0.5 有界残差），仅 photo 银标签
    box BCE + outside + smooth 训练（无合成轴 17 条件——目标是纯目标域适配曲线）
  - 评估：外层折上 raw/prior/cand(zero-shot)/cand(adapt_s)/gated(v2) 全对比，
    患者整群 bootstrap CI；重复 3 次不同抽样 seed 报均值±方差

输出：chexlocalize/p5a_adapt_curve.csv + 主表打印。
用法：/root/miniconda3/bin/python p5a_adapt_cv.py [--seeds 42,7,2024] [--smoke]
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
EPOCHS = 60
BS = 16
LR = 1e-3
LAM_OUT, LAM_SM = 0.25, 0.05
GATE_DELTA_GRID = np.arange(0.0, 0.55, 0.05)
TAU = 0.45


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RepairHead(torch.nn.Module):
    """与 repairer.RepairHead 同构（q9 广播拼接）。"""

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
    """cam_iou_curve 的轻量版（单框，224 空间，阈值 0.05..0.95）。"""
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


def train_head(idxs, D, device, epochs=EPOCHS):
    head = RepairHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    boxes = [box_of(D["gts"][i]) for i in idxs]
    Y = np.zeros((len(idxs), 1, 224, 224), np.float32)
    for j, (i, b) in enumerate(zip(idxs, boxes)):
        Y[j, 0, int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1.0
    rng = np.random.default_rng(0)
    for ep in range(epochs):
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
    head.eval()
    return head


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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda")

    D = dict(np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True))
    keys = list(D["keys"])
    patients = np.array([k.split("_")[0] for k in keys])
    uniq_p = sorted(set(patients))
    p2i = {p: i for i, p in enumerate(uniq_p)}
    N = len(keys)
    log(f"缓存 {N} 图 / {len(uniq_p)} 患者")

    # 零样本 cand：v2 5 折头中位集成（合成轴，零目标域监督）
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
    log("零样本 cand 集成完成")

    boxes = [box_of(D["gts"][i]) for i in range(N)]
    raw45 = np.array([iou_at(D["cams_p"][i], boxes[i]) for i in range(N)])
    prior45 = np.array([iou_at(D["cams_p"][i] * D["masks_p"][i], boxes[i]) for i in range(N)])
    zs45 = np.array([iou_at(cand_zs[i], boxes[i]) for i in range(N)])
    log(f"全池：raw {raw45.mean():.3f} prior {prior45.mean():.3f} cand_zs {zs45.mean():.3f}")

    scales = [0, 5, 10, 20, 40, 52] if not args.smoke else [0, 10, 52]
    rows = []
    for seed in seeds:
        rng0 = np.random.default_rng(seed)
        perm = rng0.permutation(len(uniq_p))
        folds = [sorted(uniq_p[j] for j in perm[k::N_OUTER]) for k in range(N_OUTER)]
        for k in range(N_OUTER):
            te_p = set(folds[k])
            tr_p = [p for j, f in enumerate(folds) if j != k for p in f]
            te_idx = [i for i in range(N) if patients[i] in te_p]
            # 内层校准池：train 池尾 20%（患者级），适配训练只用其余
            rng1 = np.random.default_rng(seed * 10 + k)
            tr_shuf = list(rng1.permutation(len(tr_p)))
            n_cal = max(4, len(tr_p) // 5)
            cal_p = set(tr_p[j] for j in tr_shuf[:n_cal])
            fit_p = [p for p in tr_p if p not in cal_p]
            fit_idx = [i for i in range(N) if patients[i] in set(fit_p)]
            cal_idx = [i for i in range(N) if patients[i] in cal_p]

            for s in scales:
                if s == 0:
                    cand_eval = [cand_zs[i] for i in te_idx]
                    sysname = "cand_zs"
                else:
                    if s >= len(fit_p):
                        take = fit_p
                    else:
                        take = list(rng1.choice(fit_p, s, replace=False))
                    tidx = [i for i in range(N) if patients[i] in set(take)]
                    if args.smoke:
                        head = train_head(tidx, D, device, epochs=8)
                    else:
                        head = train_head(tidx, D, device)
                    cand_all = apply_head(head, te_idx, D, device)
                    cand_eval = cand_all
                    sysname = f"cand_adapt{s:02d}"
                d45 = np.array([iou_at(c, boxes[i]) for c, i in zip(cand_eval, te_idx)]) \
                    - raw45[te_idx]
                d_best = np.array([max(iou_curve(c, boxes[i])) for c, i in zip(cand_eval, te_idx)]) \
                    - np.array([max(iou_curve(D["cams_p"][i], boxes[i])) for i in te_idx])
                rows.append({
                    "seed": seed, "fold": k, "scale": s, "sys": sysname,
                    "n_test": len(te_idx), "n_test_pat": len(te_p),
                    "raw45": float(raw45[te_idx].mean()),
                    "sys45": float(d45.mean() + raw45[te_idx].mean()),
                    "delta45": float(d45.mean()),
                    "delta_best": float(d_best.mean()),
                    "win_rate": float((d45 > 0).mean()),
                    "harm_rate": float((d45 < -0.05).mean()),
                })
            log(f"seed{seed} fold{k} 完成")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p5a_adapt_curve.csv", index=False)
    print("\n== 标注量曲线（Δτ45 vs raw，跨 seed×fold 均值 ± std） ==")
    g = df.groupby("scale")[["delta45", "delta_best", "win_rate", "harm_rate"]] \
        .agg(["mean", "std"]).round(3)
    print(g.to_string())
    print("\n== 零样本 vs 最大适配配对差（同 seed×fold） ==")
    piv = df.pivot_table(index=["seed", "fold"], columns="scale", values="delta45")
    d = piv[scales[-1]] - piv[0]
    print(f"adapt{scales[-1]} − zero-shot: {d.mean():+.3f} ± {d.std():.3f} "
          f"(n={len(d)}, >0: {(d>0).sum()}/{len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
