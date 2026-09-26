"""第四阶段 0：官方 1024px NIH 图像复跑主 Cardiomegaly 条件（设计 A）。

目的：验证 300px 镜像上的绝对 IoU 偏差与配对差结论是否在官方分辨率成立。
对照来源：results/cam_bench_summary.csv（300px 第2步）与
cam_crop/cam_crop_nih.csv（300px 第四阶段1）。

图像源：/root/autodl-tmp/nih_cxr14_1024/images（官方 1024 PNG，326 张已提取）
变体：raw / center_fixed / center_field / field_crop（无 MedSAM，快速）

用法（GPU10）：
    /root/miniconda3/bin/python cam_bench_1024.py            # 146 图全阶梯
    /root/miniconda3/bin/python cam_bench_1024.py --limit 8  # 冒烟
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cam_benchmark import (BBOX_CSV, FINDING, IMAGE_SIZE, MEAN, OUTDIR, SEED,
                           STD, THRESHOLDS, batch_cam, cam_iou_curve,
                           gt_boxes_224)
from cam_prior import S1024, coverage, detect_field
import degradations as dg
from cam_crop_baselines import (center_mask_field, center_mask_fixed,
                                field_mask)

IMG_DIR = "/root/autodl-tmp/nih_cxr14_1024/images"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    from cam_benchmark import load_model as _lm
    dmodel = _lm(device)
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    files = sorted(bbox["Image Index"].unique())
    if args.limit:
        files = files[: args.limit]

    outdir = OUTDIR / "cam_1024"
    outdir.mkdir(exist_ok=True)
    rows = []
    cf_mask = center_mask_fixed()
    fidx = {f: i for i, f in enumerate(files)}   # 确定性 RNG 索引
    log(f"{len(files)} 图，官方 1024 源")

    for s0 in range(0, len(files), 16):
        chunk = files[s0:s0 + 16]
        img224, degs = {}, {}
        for f in chunk:
            im = Image.open(f"{IMG_DIR}/{f}").convert("L")
            im224 = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img224[f] = im224
            degs[f] = {c: dg.degrade(im224, c, dg.rng_for(SEED, fidx[f]))
                       for c in dg.CONDITION_ORDER}
        boxes_of = {f: gt_boxes_224(bbox, f) for f in chunk}
        for cond in dg.CONDITION_ORDER:
            x = torch.stack([tf(degs[f][cond].convert("RGB")) for f in chunk])
            cam, prob = batch_cam(dmodel, x, device)
            for j, f in enumerate(chunk):
                deg = degs[f][cond]
                f_r, ok_r = detect_field(deg)
                variants = {
                    "raw": np.ones_like(cam[j]),
                    "center_fixed": cf_mask,
                    "center_field": center_mask_field(f_r),
                    "field_crop": field_mask(f_r),
                }
                for vname, w in variants.items():
                    curve = cam_iou_curve(cam[j] * w, boxes_of[f])
                    rows.append({"nih_file": f, "condition": cond,
                                 "variant": vname, "prob": float(prob[j]),
                                 "best_iou": max(curve),
                                 "t_best": float(THRESHOLDS[int(np.argmax(curve))]),
                                 "iou_curve": ";".join(f"{v:.4f}" for v in curve),
                                 "field_ok_real": int(ok_r)})
        log(f"  {chunk[0]}..{chunk[-1]} 完成")

    df = pd.DataFrame(rows)
    out = outdir / "cam_1024_nih.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")

    piv = df.groupby(["variant", "condition"]).best_iou.mean().unstack()
    cols = ["clean", "contrast_04", "combo_sev", "noise_006", "jpeg_q50"]
    print("\n== 1024 源 best-IoU ==")
    print(piv[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
