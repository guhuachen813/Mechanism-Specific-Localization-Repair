"""第四阶段 1：无学习修复基线——中心框裁剪 / 图场裁剪（无 MedSAM）。

目的：把约束 CAM 的收益分解为
  图像中心效应（center_fixed: 固定中心矩形，零图像自适应信息）
  × 图场裁剪效应（field_crop: 场检测 v2 矩形，无 MedSAM）
  × 解剖形状效应（prior_real_hard，来自 cam_prior_nih.csv，本脚本不复算）
  × CAM 修复效应（第四阶段 bounded repairer，未来）

变体（全部 hard 约束，CAM × mask）：
  raw           无约束（自包含复算，与 cam_prior_nih.csv 的 raw 交叉验证）
  center_fixed  中央 50%×70% 矩形（224 空间，全图中心）
  center_field  场内居中矩形：面积 = 0.35 × 场面积，纵横比 w/h = 0.75
  field_crop    场检测 v2 输出的胶片场矩形（realistic：从降质图算）

用法（gpu09）：
    /root/miniconda3/bin/python cam_crop_baselines.py            # Cardiomegaly 全阶梯
    /root/miniconda3/bin/python cam_crop_baselines.py --limit 8  # 冒烟

作者：ClawsGO Science Agent
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cam_benchmark import (BBOX_CSV, FINDING, IMAGE_SIZE, MEAN, OUTDIR, SEED,
                           SHARD_SIZE, STD, THRESHOLDS, batch_cam, build_index,
                           cam_iou_curve, gt_boxes_224, load_model,
                           load_shard_images)
from cam_prior import S1024, coverage, detect_field, mask_224
import degradations as dg


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def center_mask_fixed() -> np.ndarray:
    m = np.zeros((S1024, S1024), dtype=np.uint8)
    y0, y1 = int(0.15 * S1024), int(0.85 * S1024)     # 高度 70%
    x0, x1 = int(0.25 * S1024), int(0.75 * S1024)     # 宽度 50%
    m[y0:y1, x0:x1] = 1
    return mask_224(m)


def center_mask_field(f: tuple[int, int, int, int]) -> np.ndarray:
    """场内居中：面积 0.35×场，w/h=0.75 → 1024 空间矩形。"""
    x1, y1, x2, y2 = f
    W, H = x2 - x1, y2 - y1
    area = 0.35 * W * H
    w = int(round(np.sqrt(area * 0.75)))
    h = int(round(np.sqrt(area / 0.75)))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    m = np.zeros((S1024, S1024), dtype=np.uint8)
    m[max(0, cy - h // 2):min(S1024, cy + h // 2),
      max(0, cx - w // 2):min(S1024, cx + w // 2)] = 1
    return mask_224(m)


def field_mask(f: tuple[int, int, int, int]) -> np.ndarray:
    m = np.zeros((S1024, S1024), dtype=np.uint8)
    m[f[1]:f[3], f[0]:f[2]] = 1
    return mask_224(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    dmodel = load_model(device)
    tf = dg.tf if hasattr(dg, "tf") else None
    if tf is None:
        from torchvision import transforms
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

    outdir = OUTDIR / "cam_crop"
    outdir.mkdir(exist_ok=True)
    rows = []
    conds = dg.CONDITION_ORDER
    cf_mask = center_mask_fixed()
    log(f"variants: raw / center_fixed / center_field / field_crop; {len(conds)} 条件")

    for si, rlist in by_shard.items():
        log(f"--- 分卷 {si}：{len(rlist)} 张 ---")
        imgs = load_shard_images(si, sorted(rlist))
        row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
        img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                  for r in rlist}
        boxes_of = {r: gt_boxes_224(bbox, row_of[r]) for r in rlist}

        for cond in conds:
            for s0 in range(0, len(rlist), 16):
                chunk = sorted(rlist)[s0:s0 + 16]
                degs = {r: dg.degrade(img224[r], cond, dg.rng_for(SEED, r))
                        for r in chunk}
                x = torch.stack([tf(degs[r].convert("RGB")) for r in chunk])
                cam, prob = batch_cam(dmodel, x, device)
                for j, r in enumerate(chunk):
                    deg = degs[r]
                    f_r, ok_r = detect_field(deg)          # realistic 场检测
                    f_o, ok_o = detect_field(img224[r])    # oracle 场检测
                    variants = {
                        "raw": np.ones_like(cam[j]),
                        "center_fixed": cf_mask,
                        "center_field": center_mask_field(f_r),
                        "field_crop": field_mask(f_r),
                    }
                    for vname, w in variants.items():
                        curve = cam_iou_curve(cam[j] * w, boxes_of[r])
                        rows.append({"nih_file": row_of[r], "condition": cond,
                                     "variant": vname, "prob": float(prob[j]),
                                     "best_iou": max(curve),
                                     "t_best": float(THRESHOLDS[int(np.argmax(curve))]),
                                     "iou_curve": ";".join(f"{v:.4f}" for v in curve),
                                     "cov_oracle": coverage(
                                         w if vname != "raw" else np.ones_like(w),
                                         boxes_of[r]),
                                     "cov_real": coverage(w, boxes_of[r]),
                                     "field_ok_real": int(ok_r)})
            log(f"  {cond} 完成")

    df = pd.DataFrame(rows)
    out = outdir / "cam_crop_nih.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")

    # 快速汇总：各变体 clean 与关键条件的 best-IoU
    piv = df.groupby(["variant", "condition"]).best_iou.mean().unstack()
    cols = ["clean", "contrast_04", "combo_sev", "noise_006", "jpeg_q50"]
    print("\n== best-IoU 快速汇总 ==")
    print(piv[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
