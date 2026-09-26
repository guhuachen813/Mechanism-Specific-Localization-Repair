r"""非等价 CAM 对照：Grad-CAM++ / LayerCAM 下的先验结论稳健性。

动机：主基准与修复器全部建立在 Grad-CAM（features×正类权重，relu）上；
需检验"几何先验救回定位"的结论是否 CAM 家族特异。

协议：与 cam_bench_1024 同源——146 张 NIH Cardiomegaly 图，17 条件降质
（degradations 模块按 SEED+索引复现），对每个 (图, 条件) 计算
  - gradcam（复用 cache.npz，作为基准族）
  - gradcam_pp（alpha 加权，pytorch-grad-cam 公式）
  - layercam（relu(grad)·feats 逐像素和）
变体 raw / prior(MedSAM mask) / center_field；指标 best-IoU、τ45。

用法：/root/miniconda3/bin/python cam_multi.py [--limit 8]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

import degradations as dg
from cam_benchmark import (IMAGE_SIZE, MEAN, STD, THRESHOLDS, batch_cam,
                           cam_iou_curve, gt_boxes_224, load_model)
from cam_benchmark import BBOX_CSV, FINDING, SEED
from repairer import center_field_mask
from cam_benchmark import build_index, load_shard_images

OUT = "/root/autodl-tmp/experiments/rep_robust"
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def _norm_up(x: torch.Tensor) -> torch.Tensor:
    x = x / x.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    return torch.nn.functional.interpolate(x, size=IMAGE_SIZE, mode="bilinear",
                                           align_corners=False)


def multi_cam(model, x: torch.Tensor, device) -> dict[str, np.ndarray]:
    """返回 gradcam_pp / layercam 的 (N,224,224)。"""
    feats = model.features(x.to(device))
    logits = model.classifier(feats.mean(dim=(2, 3)))
    score = logits[:, 1].sum()
    grads = torch.autograd.grad(score, feats, retain_graph=False)[0]  # (B,K,h,w)
    gp2, gp3 = grads.pow(2), grads.pow(3)
    sum_f = feats.sum(dim=(2, 3), keepdim=True)
    alpha = gp2 / (2 * gp2 + gp3 * sum_f + 1e-8)
    w = (alpha * grads.relu()).sum(dim=(2, 3), keepdim=True)          # (B,K,1,1)
    cam_pp = _norm_up((w * feats).sum(1, keepdim=True).relu_())[:, 0]
    cam_lc = _norm_up((grads.relu() * feats).sum(1, keepdim=True))[:, 0]
    return {"gradcam_pp": cam_pp.cpu().numpy(), "layercam": cam_lc.cpu().numpy()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")
    model = load_model(device)

    # 与 cache.npz 完全一致的图集
    d = np.load(f"{OUT}/repairer/cache.npz", allow_pickle=True)
    files, conds = list(d["files"]), list(d["conds"])
    masks = d["masks"].astype(np.float32)
    gc_cam = d["cams"].astype(np.float32)
    if args.limit:
        files, keep = files[: args.limit], None
        keep = [list(d["files"]).index(f) for f in files]
        masks = masks[keep]
        gc_cam = gc_cam[keep]
    N, C = len(files), len(conds)

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    boxes_of = {f: gt_boxes_224(bbox, f) for f in files}

    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    fidx = {f: i for i, f in enumerate(sorted(
        bbox["Image Index"].unique()))}

    # 身份索引一次构建；按分卷批量取图
    with contextlib.redirect_stdout(io.StringIO()):
        lst, loc = build_index()
    rows = []
    for s0 in range(0, N, 8):
        chunk = list(range(s0, min(s0 + 8, N)))
        imgs = {}
        by_shard: dict[int, list[int]] = {}
        for i in chunk:
            by_shard.setdefault(loc[files[i]][0], []).append(loc[files[i]][1])
        shard_imgs = {}
        for si, rl in by_shard.items():
            shard_imgs[si] = load_shard_images(si, rl)
        for i in chunk:
            si, ri = loc[files[i]]
            im = shard_imgs[si][ri]
            imgs[i] = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        for c, cond in enumerate(conds):
            x = torch.stack([tf(dg.degrade(imgs[i], cond,
                                           dg.rng_for(SEED, fidx[files[i]]))
                                .convert("RGB")) for i in chunk])
            gc_cams, gc_probs = batch_cam(model, x, device)
            for fam, cams in multi_cam(model, x, device).items():
                for j, i in enumerate(chunk):
                    h, mm, bx = cams[j], masks[i, c], boxes_of[files[i]]
                    cf = center_field_mask(mm)
                    rec = {"nih_file": files[i], "condition": cond, "family": fam,
                           "prob": float(gc_probs[j])}
                    for v, cam in [("raw", h), ("prior", h * mm),
                                   ("center_field", h * cf)]:
                        curve = cam_iou_curve(cam, bx)
                        rec[f"{v}_best"] = float(max(curve))
                        rec[f"{v}_45"] = float(curve[TAU45])
                    rows.append(rec)
            if s0 == 0 and c == 0:
                dgc = np.abs(gc_cams - gc_cam[chunk, c]).max()
                log(f"gradcam 校验 vs cache：max|Δ|={dgc:.2e}")
        log(f"  {s0 + len(chunk)}/{N}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cam_multi/cam_multi_nih.csv", index=False)
    print("\n== 非 Grad-CAM 家族（mean best / τ45） ==")
    piv = df.groupby(["family", "condition"])[["raw_best", "raw_45",
                                               "prior_best", "prior_45",
                                               "center_field_best",
                                               "center_field_45"]].mean()
    key_conds = ["clean", "contrast_04", "combo_sev", "noise_006", "combo_mild"]
    print(piv.loc[(slice(None), key_conds), ["raw_best", "prior_best",
                                             "center_field_best",
                                             "raw_45", "prior_45",
                                             "center_field_45"]]
          .round(3).to_string())
    print("\n== 家族聚合（17 条件均值） ==")
    print(df.groupby("family")[["raw_best", "raw_45", "prior_best", "prior_45",
                                "center_field_best", "center_field_45"]]
          .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
