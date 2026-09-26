"""思路A 第3步：解剖先验约束 CAM vs raw CAM，在同一降质阶梯上的定位对比。

先验管线（第1b步已验证）：场检测 v2（候选打分）→ 场内左/右肺分 box 提示 →
MedSAM（真权重）分割 → 并集 + 闭运算 + 填洞 + 去小连通域。

约束变体
--------
* raw        ：无约束（复刻第2步，作对照）。
* prior_hard ：CAM × mask（mask 外置零）。
* prior_soft ：CAM × (0.2 + 0.8·mask)（软加权，保留 20% 场外激活，容忍心影凸出肺野）。
* prior_dil  ：CAM × dilate(mask, 8 轮 ≈ 224 上 16px)（膨胀先验，兼顾心影缘）。

先验来源（两层都要报——先验鲁棒性 = 场检测鲁棒性 × 结构提示鲁棒性）
------------------------------------------------------------------
* oracle    ：mask 从 clean 原图计算（先验完美的上界）。
* realistic ：mask 从**降质后的输入图**计算（部署真实情形，先验自身也承受降质）。

sanity 阶段：先测 146 张 clean 图的 GT 心影框被肺野 mask 的覆盖率
（coverage = GT∩mask / GT 面积）——覆盖不住心影的先验不能硬约束。

用法（GPU07）：
    /root/miniconda3/bin/python cam_prior.py --stage sanity
    /root/miniconda3/bin/python cam_prior.py --stage full
    /root/miniconda3/bin/python cam_prior.py --stage full --limit 8   # 冒烟

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import scipy.ndimage as ndi
from torchvision import transforms
from transformers import SamModel

from cam_benchmark import (BBOX_CSV, CKPT, FINDING, IMAGE_SIZE, MEAN, OUTDIR,
                           PARQUET_DIR, SEED, SHARD_SIZE, STD, T1,
                           TEST_LIST, THRESHOLDS, batch_cam, build_index,
                           cam_iou_curve, gt_boxes_224, load_model,
                           load_shard_images)
import degradations as dg

MEDSAM_DIR = OUTDIR / "weights" / "medsam"
S1024 = 1024
MEAN_SAM = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_SAM = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SOFT_BG = 0.2
DIL_ITERS = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- 先验管线（移植自 seg_check3.py / seg_check2.py） ----------------
def detect_field(gray: Image.Image) -> tuple[tuple[int, int, int, int], bool]:
    """场 bbox（1024 空间坐标）+ 是否触发全图回退。"""
    w0, h0 = gray.size
    s = 256 / max(w0, h0)
    small = np.asarray(gray.resize((max(1, round(w0 * s)), max(1, round(h0 * s))),
                                   Image.BILINEAR), dtype=np.float32)
    thr = 0.5 * (np.quantile(small, 0.25) + np.quantile(small, 0.85))
    m = small > thr
    m = ndi.binary_closing(m, structure=np.ones((5, 5)))
    lab, n = ndi.label(m)
    if n == 0:
        return (0, 0, S1024, S1024), False
    Hs, Ws = m.shape
    best, best_score = None, -1.0
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        area = len(xs)
        if area < 0.05 * Hs * Ws:
            continue
        bx1, bx2 = xs.min(), xs.max()
        by1, by2 = ys.min(), ys.max()
        bw, bh = bx2 - bx1 + 1, by2 - by1 + 1
        fill = area / (bw * bh)
        aspect = bh / bw
        border = ((bx1 <= 1) + (bx2 >= Ws - 2) + (by1 <= 1) + (by2 >= Hs - 2)) / 4
        score = (area / (Hs * Ws)) * fill * (1.0 - 0.5 * border)
        if not (0.7 <= aspect <= 2.2):
            score *= 0.3
        if score > best_score:
            best_score, best = score, (bx1, bx2, by1, by2)
    if best is None:
        return (0, 0, S1024, S1024), False
    bx1, bx2, by1, by2 = best
    x1 = int(bx1 * S1024 / (s * w0))
    x2 = int(bx2 * S1024 / (s * w0)) + 1
    y1 = int(by1 * S1024 / (s * h0))
    y2 = int(by2 * S1024 / (s * h0)) + 1
    mx = int(0.02 * S1024)
    return (max(0, x1 - mx), max(0, y1 - mx),
            min(S1024, x2 + mx), min(S1024, y2 + mx)), True


def medsam_prep(g: Image.Image) -> torch.Tensor:
    a = np.asarray(g.resize((S1024, S1024), Image.BILINEAR), dtype=np.float32)
    lo, hi = a.min(), max(a.max(), a.min() + 1e-8)
    a = (a - lo) / (hi - lo) * 255.0
    t = torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1) / 255.0
    return (t - MEAN_SAM) / STD_SAM


def field_boxes(f: tuple[int, int, int, int]) -> torch.Tensor:
    """左肺 + 右肺 + 心影/纵隔 三结构 box（场内相对坐标）→ (1,3,4)。"""
    x1, y1, x2, y2 = f
    W, H = x2 - x1, y2 - y1
    rel = [(0.02, 0.03, 0.50, 0.85),    # 左肺（图像左半）
           (0.50, 0.03, 0.98, 0.85),    # 右肺
           (0.28, 0.30, 0.75, 0.92)]    # 心影/纵隔
    return torch.tensor([[[x1 + a * W, y1 + b * H, x1 + c * W, y1 + d * H]
                          for a, b, c, d in rel]], dtype=torch.float32)


@torch.no_grad()
def segment(model, pixel: torch.Tensor, f: tuple[int, int, int, int],
            device: torch.device) -> np.ndarray:
    """左肺+右肺 → 并集 → 清理 → 1024 空间二值 mask。"""
    out = model(pixel_values=pixel.to(device),
                input_boxes=field_boxes(f).to(device))
    pm = out.pred_masks[:, :, 0].float().sigmoid().cpu().numpy()
    H_f, W_f = f[3] - f[1], f[2] - f[0]
    mask_f = np.zeros((H_f, W_f), dtype=np.uint8)
    for k in range(pm.shape[1]):
        m = Image.fromarray((pm[0, k] > 0.5).astype(np.uint8) * 255)
        m = np.asarray(m.resize((W_f, H_f), Image.BILINEAR)) > 127
        mask_f |= m.astype(np.uint8)
    mask_f = ndi.binary_closing(mask_f, structure=np.ones((7, 7)))
    mask_f = ndi.binary_fill_holes(mask_f)
    lab, n = ndi.label(mask_f)
    if n:
        sizes = ndi.sum(mask_f, lab, range(1, n + 1))
        mask_f = np.isin(lab, np.where(sizes >= 0.003 * mask_f.size)[0] + 1)
    mask = np.zeros((S1024, S1024), dtype=np.uint8)
    mask[f[1]:f[3], f[0]:f[2]] = mask_f.astype(np.uint8)
    return mask


def mask_224(mask1024: np.ndarray) -> np.ndarray:
    m = Image.fromarray(mask1024 * 255).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return (np.asarray(m) > 127).astype(np.float32)


def coverage(mask: np.ndarray,
             boxes: list[tuple[float, float, float, float]]) -> float:
    """GT 框被 mask 覆盖的比例（多框取均值）。"""
    vals = []
    for x1, y1, x2, y2 in boxes:
        xi1, yi1 = int(max(0, x1)), int(max(0, y1))
        xi2, yi2 = int(min(IMAGE_SIZE, x2)), int(min(IMAGE_SIZE, y2))
        if xi2 <= xi1 or yi2 <= yi1:
            vals.append(0.0)
            continue
        gt = np.zeros_like(mask, dtype=bool)
        gt[yi1:yi2, xi1:xi2] = True
        vals.append(float((mask.astype(bool) & gt).sum() / gt.sum()))
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "full"], default="sanity")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    dmodel = load_model(device)
    log("加载 MedSAM（真权重）")
    smodel = SamModel.from_pretrained(MEDSAM_DIR).to(device).eval()
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])

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

    outdir = OUTDIR / "cam_prior"
    outdir.mkdir(exist_ok=True)
    rows, qcs = [], []
    conds = dg.CONDITION_ORDER if args.stage == "full" else ["clean"]

    for si, rlist in by_shard.items():
        log(f"--- 分卷 {si}：{len(rlist)} 张 ---")
        imgs = load_shard_images(si, sorted(rlist))
        row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
        img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                  for r in rlist}
        boxes_of = {r: gt_boxes_224(bbox, row_of[r]) for r in rlist}

        # ---- oracle mask（clean 原图，每图一次） ----
        oracle = {}
        cov_oracle = {}
        for r in sorted(rlist):
            f, ok = detect_field(img224[r])
            m224 = mask_224(segment(smodel, medsam_prep(img224[r]), f, device))
            oracle[r] = m224
            cov_oracle[r] = coverage(m224, boxes_of[r])
        if args.stage == "sanity":
            c = np.array([cov_oracle[r] for r in sorted(rlist)])
            log(f"  oracle 覆盖率：均值 {c.mean():.3f}，中位 {np.median(c):.3f}，"
                f"<0.5 占比 {(c < 0.5).mean():.3f}，<0.8 占比 {(c < 0.8).mean():.3f}")
            continue

        for cond in conds:
            for s0 in range(0, len(rlist), 16):
                chunk = sorted(rlist)[s0:s0 + 16]
                x = torch.stack([
                    tf(dg.degrade(img224[r], cond, dg.rng_for(SEED, r)).convert("RGB"))
                    for r in chunk])
                cam, prob = batch_cam(dmodel, x, device)
                for j, r in enumerate(chunk):
                    deg = dg.degrade(img224[r], cond, dg.rng_for(SEED, r))
                    # realistic mask（降质输入）
                    fr, ok_r = detect_field(deg)
                    mre = mask_224(segment(smodel, medsam_prep(deg), fr, device))
                    mo = oracle[r]
                    variants = {
                        "raw": np.ones_like(cam[j]),
                        "prior_oracle_hard": mo,
                        "prior_oracle_soft": SOFT_BG + (1 - SOFT_BG) * mo,
                        "prior_oracle_dil": ndi.binary_dilation(
                            mo.astype(bool), iterations=DIL_ITERS).astype(np.float32),
                        "prior_real_hard": mre,
                        "prior_real_soft": SOFT_BG + (1 - SOFT_BG) * mre,
                        "prior_real_dil": ndi.binary_dilation(
                            mre.astype(bool), iterations=DIL_ITERS).astype(np.float32),
                    }
                    for vname, w in variants.items():
                        curve = cam_iou_curve(cam[j] * w, boxes_of[r])
                        rows.append({"nih_file": row_of[r], "condition": cond,
                                     "variant": vname, "prob": float(prob[j]),
                                     "best_iou": max(curve),
                                     "t_best": float(THRESHOLDS[int(np.argmax(curve))]),
                                     "iou_curve": ";".join(f"{v:.4f}" for v in curve),
                                     "cov_oracle": cov_oracle[r],
                                     "cov_real": coverage(mre, boxes_of[r]),
                                     "field_ok_real": int(ok_r)})
                        if cond == "contrast_04" and vname == "prior_real_hard" \
                                and len(qcs) < 24:
                            qcs.append((row_of[r], cond, deg, cam[j],
                                        mre, boxes_of[r]))
            log(f"  {cond} 完成")

    if args.stage == "sanity":
        return 0

    df = pd.DataFrame(rows)
    out = outdir / "cam_prior_nih.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")

    if qcs:
        from cam_benchmark import qc_overlay
        sel = qcs[::max(1, len(qcs) // 6)]
        sheet = Image.new("RGB", (len(sel) * 226, 224), "white")
        from PIL import ImageDraw
        d0 = ImageDraw.Draw(sheet)
        for i, (fname, cond, deg, cam0, mre, bx) in enumerate(sel):
            sheet.paste(qc_overlay(deg, cam0 * mre, bx), (i * 226, 0))
            d0.text((i * 226 + 2, 2), fname[:10], fill=(255, 255, 0))
        sheet.save(outdir / "qc_prior.png")
        log(f"QC -> {outdir / 'qc_prior.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
