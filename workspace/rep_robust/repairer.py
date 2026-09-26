r"""第四阶段 3：采集偏移感知的选择性定位修复器（bounded residual + no-op gate）。

模型（全部冻结 DenseNet CAM 生成与 MedSAM mask 提取，只训修复头与门控头）
  输入  ：H_raw（raw CAM,224）· M（realistic 解剖 mask）· F（场框）· q（GT-free 质量特征）
  修复头：3×3 小 CNN，Δ = tanh(...) ∈ [-1,1]
          A_cand = clip(H_raw + α·Δ, 0, 1)，α = 0.5（有界残差）
  门控  ：两阶段。第二阶段用训练折内的定位收益标签
          y = 1[ IoU45(A_cand) - IoU45(H_raw) > δ ]（δ=0.05，仅训练集可用 GT）
          g = MLP(q)；A_out = g·A_cand + (1-g)·H_raw，评估用 g≥0.5（验证折校准）

损失  L = L_box + λ1·L_identity + λ2·L_outside + λ3·L_smooth
  L_box     BCE(A_cand, 框填充 GT mask)（监督）
  L_identity clean 样本上 mean(Δ²)（clean 保持）
  L_outside mean(A_cand · (1-M))（解剖外抑制）
  L_smooth  TV(Δ)

q 特征（GT-free，测试期可用）：field_ok · mask/场面积 frac · 连通域数 ·
  亮度/对比度/噪声估计/Laplacian 锐度 · CAM 在 mask 内质量占比 ·
  CAM 质心与 mask 质心距离

协议：Cardiomegaly 146 图 × 17 条件 = 2482 样本；患者级 5 折（splits/nih_box_patients.csv，
seed 42）；折内训练 / 折外评估。与 raw / prior hard / center_field 同表对比。

用法（gpu09）：
    /root/miniconda3/bin/python repairer.py --stage cache   # 预计算 CAM/mask/q（~40min GPU）
    /root/miniconda3/bin/python repairer.py --stage train   # 训练 + 折外评估
    /root/miniconda3/bin/python repairer.py --stage train --limit 32   # 冒烟

作者：ClawsGO Science Agent
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from cam_benchmark import (BBOX_CSV, FINDING, IMAGE_SIZE, MEAN, OUTDIR, SEED,
                           SHARD_SIZE, STD, THRESHOLDS, batch_cam, build_index,
                           cam_iou_curve, gt_boxes_224, load_model,
                           load_shard_images)
from cam_prior import S1024, detect_field, mask_224, medsam_prep, segment
from patient_split import main as split_main  # 确保划分文件存在
import degradations as dg

ALPHA = 0.5
LAM_ID, LAM_OUT, LAM_SM = 0.5, 0.25, 0.05
GATE_DELTA = 0.05
EPOCHS = 40
BS = 32
LR = 1e-3
NFOLD = 5
TAU45_IDX = int(round((0.45 - 0.05) / 0.05))
SPLIT_CSV = OUTDIR / "splits" / "nih_box_patients.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- q 特征 --
def img_quality(gray: np.ndarray) -> tuple[float, float, float, float]:
    """亮度 / 对比度 / 噪声估计 / Laplacian 锐度（0-1 灰度）。"""
    bright = float(gray.mean())
    contrast = float(gray.std())
    lap = gray - ndi.gaussian_filter(gray, 1.0)
    noise = float(np.abs(lap).mean())
    sharp = float((ndi.laplace(gray) ** 2).mean())
    return bright, contrast, noise, sharp


def mask_feats(mask: np.ndarray) -> tuple[float, int]:
    lab, n = ndi.label(mask.astype(bool))
    return float(mask.mean()), int(n)


def cam_conflict(cam: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """CAM 在 mask 内的质量占比 / 质心距离（px）。"""
    total = cam.sum() + 1e-8
    inside = float((cam * mask).sum() / total)
    ys, xs = np.mgrid[0:cam.shape[0], 0:cam.shape[1]]
    cx, cy = (cam * xs).sum() / total, (cam * ys).sum() / total
    tm = mask.sum() + 1e-8
    mx, my = (mask * xs).sum() / tm, (mask * ys).sum() / tm
    dist = float(np.hypot(cx - mx, cy - my))
    return inside, dist


def q_vector(gray, mask, field_ok) -> np.ndarray:
    b, c, n, s = img_quality(gray)
    frac, ncomp = mask_feats(mask)
    return np.array([field_ok, frac, min(ncomp, 5) / 5.0, b, c,
                     min(n, 0.3), min(np.log10(s + 1e-6) / 4, 1)], np.float32)


def cam_feats(cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inside, dist = cam_conflict(cam, mask)
    return np.array([inside, min(dist, 112) / 112.0], np.float32)


# ---------------------------------------------------------------- 缓存 --
@torch.no_grad()
def stage_cache(args) -> int:
    device = torch.device("cuda")
    dmodel = load_model(device)
    log("加载 MedSAM")
    from transformers import SamModel
    smodel = SamModel.from_pretrained(OUTDIR / "weights" / "medsam").to(device).eval()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    files = sorted(bbox["Image Index"].unique())
    if args.limit:
        files = files[: args.limit]
    lst, loc = build_index()
    by_shard: dict[int, list[int]] = {}
    for f in files:
        si, r = loc[f]
        by_shard.setdefault(si, []).append(r)

    C, H, W = len(dg.CONDITION_ORDER), IMAGE_SIZE, IMAGE_SIZE
    cams = np.zeros((len(files), C, H, W), np.float16)
    masks = np.zeros((len(files), C, H, W), np.uint8)
    probs = np.zeros((len(files), C), np.float32)
    qs = np.zeros((len(files), C, 7), np.float32)
    cs = np.zeros((len(files), C, 2), np.float32)
    fidx = {f: i for i, f in enumerate(files)}
    cond_idx = {c: k for k, c in enumerate(dg.CONDITION_ORDER)}

    for si, rlist in by_shard.items():
        imgs = load_shard_images(si, sorted(rlist))
        row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
        img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                  for r in rlist}
        for r in sorted(rlist):
            i = fidx[row_of[r]]
            # oracle 场框（GT-free 常量，不随条件变）
            for cond in dg.CONDITION_ORDER:
                k = cond_idx[cond]
                deg = dg.degrade(img224[r], cond, dg.rng_for(SEED, r))
                x = tf(deg.convert("RGB"))[None].to(device)
                cam, prob = batch_cam(dmodel, x, device)
                cams[i, k] = cam[0].astype(np.float16)
                probs[i, k] = prob[0]
                fr, ok_r = detect_field(deg)
                mre = mask_224(segment(smodel, medsam_prep(deg), fr, device))
                masks[i, k] = mre.astype(np.uint8)
                g = np.asarray(deg.convert("L"), np.float32) / 255.0
                qs[i, k] = q_vector(g, mre, ok_r)
                cs[i, k] = cam_feats(cam[0], mre)
            log(f"  {row_of[r]} 完成")
    np.savez_compressed(OUTDIR / "repairer" / "cache.npz",
                        cams=cams, masks=masks, probs=probs, qs=qs, cs=cs,
                        files=np.array(files), conds=np.array(dg.CONDITION_ORDER))
    log("cache.npz 完成")
    return 0


# ---------------------------------------------------------------- 模型 --
class RepairHead(nn.Module):
    def __init__(self, nq: int = 9):
        super().__init__()
        ch = 3 + nq             # H_raw, M, H*M + q broadcast
        self.net = nn.Sequential(
            nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, hraw, mask, q):
        B, _, H, W = hraw.shape
        qb = q[:, :, None, None].expand(B, q.shape[1], H, W)
        x = torch.cat([hraw, mask, hraw * mask, qb], dim=1)
        return torch.tanh(self.net(x))


class GateHead(nn.Module):
    def __init__(self, n_in: int = 9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, q):
        return torch.sigmoid(self.net(q)).squeeze(-1)


def iou_at(cam: np.ndarray, boxes, tau_idx: int = TAU45_IDX) -> float:
    return float(cam_iou_curve(cam, boxes)[tau_idx])


def best_iou(cam: np.ndarray, boxes) -> float:
    return float(max(cam_iou_curve(cam, boxes)))


def box_mask(boxes) -> np.ndarray:
    m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
    for x1, y1, x2, y2 in boxes:
        m[int(max(0, y1)):int(min(IMAGE_SIZE, y2)),
          int(max(0, x1)):int(min(IMAGE_SIZE, x2))] = 1.0
    return m


def center_field_mask(mask: np.ndarray) -> np.ndarray:
    """复刻 cam_crop_baselines.center_mask_field：场内居中 35% 面积 w/h=0.75。
    （用 mask 外接框近似场框，避免重复场检测。）"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.ones_like(mask)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    W, H = x2 - x1, y2 - y1
    area = 0.35 * W * H
    w, h = int(np.sqrt(area * 0.75)), int(np.sqrt(area / 0.75))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    m = np.zeros_like(mask)
    m[max(0, cy - h // 2):cy + h // 2, max(0, cx - w // 2):cx + w // 2] = 1
    return m


# ---------------------------------------------------------------- 训练 --
def stage_train(args) -> int:
    device = torch.device("cuda")
    d = np.load(OUTDIR / "repairer" / "cache.npz", allow_pickle=True)
    files = list(d["files"])
    conds = list(d["conds"])
    cams = d["cams"].astype(np.float32)
    masks = d["masks"].astype(np.float32)
    probs = d["probs"]
    qs = np.concatenate([d["qs"], d["cs"]], axis=2)          # (N,C,9)
    N, C = len(files), len(conds)

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    boxes_of = {f: gt_boxes_224(bbox, f) for f in files}
    Y = np.stack([box_mask(boxes_of[f]) for f in files])      # (N,H,W)

    if not SPLIT_CSV.exists():
        split_main()
    sp = pd.read_csv(SPLIT_CSV)
    fold_of = {r.file: r.fold for r in sp[sp.finding == FINDING].itertuples()}
    folds = [[] for _ in range(NFOLD)]
    for i, f in enumerate(files):
        folds[fold_of[f]].append(i)

    ci = {c: k for k, c in enumerate(conds)}
    is_clean = np.array([1.0 if c == "clean" else 0.0 for c in conds])

    rows = []
    rng = np.random.default_rng(SEED)
    for k in range(NFOLD):
        te = folds[k]
        tr = sorted(i for j in range(NFOLD) if j != k for i in folds[j])
        log(f"fold{k}: train {len(tr)} 图 / test {len(te)} 图")

        # ---------------- 阶段一：修复头 ----------------
        head = RepairHead().to(device)
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        tr_pairs = [(i, c) for i in tr for c in range(C)]
        for ep in range(EPOCHS):
            perm = rng.permutation(len(tr_pairs))
            tot = 0.0
            for s0 in range(0, len(perm), BS):
                sel = perm[s0:s0 + BS]
                idx = [tr_pairs[s] for s in sel]
                ii = np.array([a for a, _ in idx])
                cc = np.array([b for _, b in idx])
                h = torch.from_numpy(cams[ii, cc]).unsqueeze(1).to(device)
                m = torch.from_numpy(masks[ii, cc]).unsqueeze(1).to(device)
                y = torch.from_numpy(Y[ii]).unsqueeze(1).to(device)
                q = torch.from_numpy(qs[ii, cc]).to(device)
                delta = head(h, m, q)
                cand = torch.clamp(h + ALPHA * delta, 0, 1)
                l_box = F.binary_cross_entropy(cand, y)
                clean_m = torch.from_numpy(is_clean[cc] == 1).to(device)
                if clean_m.any():
                    l_id = (delta[clean_m] ** 2).mean()
                else:
                    l_id = torch.tensor(0.0, device=device)
                l_out = (cand * (1 - m)).mean()
                l_sm = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean() + \
                       (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
                loss = l_box + LAM_ID * l_id + LAM_OUT * l_out + LAM_SM * l_sm
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(sel)
            if ep % 10 == 9:
                log(f"  ep{ep+1}: loss {tot/len(tr_pairs):.4f}")
        head.eval()

        # ---------------- 阶段二：门控（训练折收益标签） ----------------
        gate = GateHead().to(device)
        gopt = torch.optim.Adam(gate.parameters(), lr=LR)
        with torch.no_grad():
            ys_lab, qs_lab = [], []
            for i in tr:
                for c in range(C):
                    h = cams[i, c]
                    mm = masks[i, c]
                    inp = torch.from_numpy(h[None, None]).to(device)
                    mk = torch.from_numpy(mm[None, None]).to(device)
                    qt = torch.from_numpy(qs[i, c][None]).to(device)
                    cft = torch.from_numpy(d["cs"][i, c][None]).to(device)
                    dl = head(inp, mk, qt)
                    cand = torch.clamp(inp + ALPHA * dl, 0, 1)[0, 0].cpu().numpy()
                    y = 1.0 if iou_at(cand, boxes_of[files[i]]) - iou_at(h, boxes_of[files[i]]) > GATE_DELTA else 0.0
                    ys_lab.append(y)
                    qs_lab.append(qs[i, c])
        ys_lab = np.array(ys_lab, np.float32)
        qs_lab = np.stack(qs_lab)
        pos = ys_lab.mean()
        pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(80):
            perm = rng.permutation(len(ys_lab))
            for s0 in range(0, len(perm), 256):
                sel = perm[s0:s0 + 256]
                q = torch.from_numpy(qs_lab[sel]).to(device)
                y = torch.from_numpy(ys_lab[sel]).to(device)
                out = gate.net(q).squeeze(-1)
                loss = bce(out, y)
                gopt.zero_grad(set_to_none=True)
                loss.backward()
                gopt.step()
        gate.eval()
        log(f"  gate 训练收益标签正例率 {pos:.3f}")

        # ---------------- 折外评估 ----------------
        with torch.no_grad():
            for i in te:
                for c in range(C):
                    h = cams[i, c]
                    mm = masks[i, c]
                    bx = boxes_of[files[i]]
                    cond = conds[c]
                    inp = torch.from_numpy(h[None, None]).to(device)
                    mk = torch.from_numpy(mm[None, None]).to(device)
                    qt = torch.from_numpy(qs[i, c][None]).to(device)
                    cft = torch.from_numpy(d["cs"][i, c][None]).to(device)
                    dl = head(inp, mk, qt)
                    cand = torch.clamp(inp + ALPHA * dl, 0, 1)[0, 0].cpu().numpy()
                    g = float(gate(qt)[0])
                    out = cand if g >= 0.5 else h
                    cf = center_field_mask(mm)
                    rows.append({
                        "fold": k, "nih_file": files[i], "condition": cond,
                        "gate": g, "repaired": int(g >= 0.5),
                        "raw_best": best_iou(h, bx), "raw_45": iou_at(h, bx),
                        "prior_best": best_iou(h * mm, bx), "prior_45": iou_at(h * mm, bx),
                        "center_best": best_iou(h * cf, bx), "center_45": iou_at(h * cf, bx),
                        "cand_best": best_iou(cand, bx), "cand_45": iou_at(cand, bx),
                        "gated_best": best_iou(out, bx), "gated_45": iou_at(out, bx),
                        "prob": float(probs[i, c]),
                    })
        log(f"  fold{k} 评估完成")

    df = pd.DataFrame(rows)
    out = OUTDIR / "repairer" / "repairer_eval.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")

    print("\n== 折外汇总（全部 17 条件）==")
    for m in ["45", "best"]:
        print(f"-- IoU_{m} 均值 --")
        for col in ["raw", "prior", "center", "cand", "gated"]:
            print(f"  {col:7s} {df[col+'_'+m].mean():.3f}")
    print(f"\n门控触发率: {df.repaired.mean():.3f}（clean 上 "
          f"{df[df.condition=='clean'].repaired.mean():.3f}）")
    err = df[df.cand_45 < df.raw_45 - 0.1]
    print(f"修复致伤（cand_45 比 raw_45 掉 >0.1）: {len(err)}/{len(df)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "train"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    (OUTDIR / "repairer").mkdir(exist_ok=True)
    if args.stage == "cache":
        return stage_cache(args)
    return stage_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
