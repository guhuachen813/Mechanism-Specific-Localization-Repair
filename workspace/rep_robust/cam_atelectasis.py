"""思路A 第4c步：Atelectasis 多病种扩展——肺类病种的先验设计与救援效果。

设计对照：Cardiomegaly（心影病，三结构 box）vs Atelectasis（肺内病，
**纯双肺 box**——病灶在肺野内，心影 box 反而引入无关区域）。先验证
GT 框被纯肺 mask 的覆盖率（预期显著高于 Cardiomegaly 的 0.53），再跑
17 条件 × raw / prior_oracle_hard / prior_real_hard。

用法（GPU07）：
    /root/miniconda3/bin/python cam_atelectasis.py [--limit 8] [--stage sanity|full]

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
from transformers import SamModel

from cam_benchmark import (BBOX_CSV, IMAGE_SIZE, MEAN, OUTDIR, SEED, SHARD_SIZE, STD,
                           THRESHOLDS, batch_cam, build_index, cam_iou_curve,
                           gt_boxes_224, load_model, load_shard_images)
from cam_prior import S1024, coverage, detect_field, mask_224, medsam_prep
import degradations as dg

FINDING = "Atelectasis"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def field_boxes_lungs(f: tuple[int, int, int, int]) -> torch.Tensor:
    """场内双肺 box（无第三结构）→ (1,2,4)。"""
    x1, y1, x2, y2 = f
    W, H = x2 - x1, y2 - y1
    rel = [(0.02, 0.03, 0.50, 0.85), (0.50, 0.03, 0.98, 0.85)]
    return torch.tensor([[[x1 + a * W, y1 + b * H, x1 + c * W, y1 + d * H]
                          for a, b, c, d in rel]], dtype=torch.float32)


@torch.no_grad()
def segment_lungs(model, pixel: torch.Tensor, f: tuple[int, int, int, int],
                  device: torch.device) -> np.ndarray:
    import scipy.ndimage as ndi
    out = model(pixel_values=pixel.to(device),
                input_boxes=field_boxes_lungs(f).to(device))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "full"], default="sanity")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    dmodel = load_model(device)
    smodel = SamModel.from_pretrained(OUTDIR / "weights" / "medsam").to(device).eval()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    files = sorted(bbox["Image Index"].unique())
    if args.limit:
        files = files[: args.limit]
    lst, loc = build_index()
    log(f"{FINDING} 框图 {len(files)} 张")

    by_shard: dict[int, list[int]] = {}
    for f in files:
        si, r = loc[f]
        by_shard.setdefault(si, []).append(r)

    outdir = OUTDIR / "cam_atelectasis"
    outdir.mkdir(exist_ok=True)
    rows = []
    for si, rlist in by_shard.items():
        imgs = load_shard_images(si, sorted(rlist))
        row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
        img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                  for r in rlist}
        boxes_of = {r: gt_boxes_224(bbox, row_of[r]) for r in rlist}

        oracle, cov_oracle = {}, {}
        for r in sorted(rlist):
            f, _ = detect_field(img224[r])
            m = mask_224(segment_lungs(smodel, medsam_prep(img224[r]), f, device))
            oracle[r] = m
            cov_oracle[r] = coverage(m, boxes_of[r])
        c = np.array([cov_oracle[r] for r in sorted(rlist)])
        log(f"  分卷 {si} 纯肺 mask 覆盖率：均值 {c.mean():.3f}，中位 {np.median(c):.3f}")
        if args.stage == "sanity":
            continue

        for cond in dg.CONDITION_ORDER:
            for s0 in range(0, len(rlist), 16):
                chunk = sorted(rlist)[s0:s0 + 16]
                x = torch.stack([
                    tf(dg.degrade(img224[r], cond, dg.rng_for(SEED, r)).convert("RGB"))
                    for r in chunk])
                cam, prob = batch_cam(dmodel, x, device)
                for j, r in enumerate(chunk):
                    deg = dg.degrade(img224[r], cond, dg.rng_for(SEED, r))
                    f, ok = detect_field(deg)
                    mre = mask_224(segment_lungs(smodel, medsam_prep(deg), f, device))
                    for vname, w in [("raw", np.ones_like(cam[j])),
                                     ("prior_oracle_hard", oracle[r]),
                                     ("prior_real_hard", mre)]:
                        curve = cam_iou_curve(cam[j] * w, boxes_of[r])
                        rows.append({"nih_file": row_of[r], "condition": cond,
                                     "variant": vname, "prob": float(prob[j]),
                                     "best_iou": max(curve),
                                     "cov_oracle": cov_oracle[r],
                                     "cov_real": coverage(mre, boxes_of[r]),
                                     "field_ok_real": int(ok),
                                     "iou_curve": ";".join(f"{v:.4f}" for v in curve)})
            log(f"  {cond} 完成")

    if args.stage == "sanity":
        return 0
    df = pd.DataFrame(rows)
    out = outdir / "cam_atelectasis_nih.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
