"""思路A 第4b步：随机区域约束消融——「解剖」成分是否必要？

对照设计：把三结构解剖 mask 换成**面积匹配的随机矩形**（面积 = 该图先验 mask
面积，纵横比 ~U[0.5,2]，位置均匀），每图每条件 5 个随机种子，其余完全同
cam_prior.py 管线。若随机矩形也能救 contrast_04/combo_sev，则救援主要来自
「切掉图边上的漂移热区」而非解剖知识本身；若解剖显著更好，则先验的解剖
成分必要。

条件：clean、contrast_04、combo_sev、noise_006（第3步的救援轴 + 反例轴）。

用法（GPU07）：
    /root/miniconda3/bin/python cam_random_ablation.py [--limit 8]

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
from torchvision import transforms

from cam_benchmark import (IMAGE_SIZE, MEAN, OUTDIR, SEED, SHARD_SIZE, STD, THRESHOLDS,
                           batch_cam, build_index, cam_iou_curve, gt_boxes_224,
                           load_model, load_shard_images)
from cam_prior import detect_field, mask_224, medsam_prep, segment
import degradations as dg

BBOX_CSV = OUTDIR / "BBox_List_2017.csv"
FINDING = "Cardiomegaly"
CONDS = ["clean", "contrast_04", "combo_sev", "noise_006"]
NRAND = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def random_boxes(mask: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """与 mask 面积匹配的 n 个随机矩形（224 空间，float mask）。"""
    area = float(mask.sum())
    out = []
    for _ in range(n):
        aspect = rng.uniform(0.5, 2.0)
        w = min(IMAGE_SIZE, np.sqrt(area * aspect))
        h = min(IMAGE_SIZE, area / max(w, 1.0))
        if w >= IMAGE_SIZE and h >= IMAGE_SIZE:
            out.append(np.ones_like(mask))
            continue
        cx = rng.uniform(w / 2, IMAGE_SIZE - w / 2)
        cy = rng.uniform(h / 2, IMAGE_SIZE - h / 2)
        box = np.zeros_like(mask)
        box[max(0, int(cy - h / 2)):int(cy + h / 2),
            max(0, int(cx - w / 2)):int(cx + w / 2)] = 1.0
        out.append(box)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    dmodel = load_model(device)
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

    outdir = OUTDIR / "cam_random"
    outdir.mkdir(exist_ok=True)
    rows = []
    for si, rlist in by_shard.items():
        imgs = load_shard_images(si, sorted(rlist))
        row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
        img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                  for r in rlist}
        boxes_of = {r: gt_boxes_224(bbox, row_of[r]) for r in rlist}
        for cond in CONDS:
            for s0 in range(0, len(rlist), 16):
                chunk = sorted(rlist)[s0:s0 + 16]
                x = torch.stack([
                    tf(dg.degrade(img224[r], cond, dg.rng_for(SEED, r)).convert("RGB"))
                    for r in chunk])
                cam, prob = batch_cam(dmodel, x, device)
                for j, r in enumerate(chunk):
                    deg = dg.degrade(img224[r], cond, dg.rng_for(SEED, r))
                    f, _ = detect_field(deg)
                    mre = mask_224(segment(smodel, medsam_prep(deg), f, device))
                    rng = np.random.default_rng(SEED * 7 + r)
                    for vname, w in [("raw", np.ones_like(cam[j])),
                                     ("prior_real_hard", mre),
                                     *[(f"random_{k}", rb) for k, rb in
                                       enumerate(random_boxes(mre, NRAND, rng))]]:
                        curve = cam_iou_curve(cam[j] * w, boxes_of[r])
                        rows.append({"nih_file": row_of[r], "condition": cond,
                                     "variant": vname,
                                     "best_iou": max(curve),
                                     "iou_curve": ";".join(f"{v:.4f}" for v in curve)})
            log(f"  {cond} 完成")
    df = pd.DataFrame(rows)
    out = outdir / "cam_random_nih.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
