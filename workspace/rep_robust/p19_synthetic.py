r"""P19（优先级3）：合成退化泛化（valid/synthetic digital+photographic）。

CheXphoto synthetic 是对同一批 CheXpert valid 图的**仿真退化**（滤镜/压缩等，
预期无几何变换）——银标签掩码仅在几何恒等时可用。脚本对每个 tag×variant：
  1) SIFT 配准检查几何位移（corners(T) vs 恒等，中位 px）→ 掩码可用性门槛；
  2) 通过门槛的键上评估 raw45 / pairloc45 / rectnet45（各 tag 的 adapter/head）。

判读：非几何退化下 rectnet 应退化为近恒等（无害性），pairloc 延续修正读出端。
新设备（iPhone/OnePlus 之外）本地无数据，随 test natural 668 外测解决。

输出 p19_synthetic.csv + 控制台汇总。
用法：/root/miniconda3/bin/python p19_synthetic.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, load_model
from p18_new_findings import (Adapter, CORNERS, CornHead, CLEAN_VALID, OUT, _grid,
                              iou45, mask_bbox, register_T)

BASE = "/root/autodl-tmp"
PH_REG, RATIO, RANSAC_THR, MIN_INLIERS = 800, 0.7, 5.0, 20
SEED = 42
TAGS = {"cardi": {"ckpt": "/root/project/outputs/repaired_seed42/densenet121/best.pt",
                  "ev": "cache_photo_adapt"},
        "atel": {"ckpt": "/root/project/outputs/atel_seed42_densenet121/best.pt",
                 "ev": "cache_photo_atel"}}
VARIANTS = {"digital": f"{BASE}/chexphoto/valid/synthetic/digital",
            "photographic": f"{BASE}/chexphoto/valid/synthetic/photographic"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()
    rows = []
    for tag, cfg in TAGS.items():
        cam_benchmark.CKPT = cfg["ckpt"]
        model = load_model(device)
        for p in model.parameters():
            p.requires_grad_(False)
        w1 = model.classifier.weight[1].detach()
        Dv = np.load(f"{OUT}/{cfg['ev']}.npz", allow_pickle=True)
        vkeys = [str(k) for k in Dv["keys"]]
        gt_boxes = [mask_bbox(Dv["gts"][i]) for i in range(len(vkeys))]
        ad = Adapter().to(device)
        ad.load_state_dict(torch.load(f"{OUT}/pairloc_adapter_{tag}.pt",
                                      map_location=device))
        ad.eval()
        head = CornHead().to(device)
        head.load_state_dict(torch.load(f"{OUT}/rectnet_head_{tag}.pt",
                                        map_location=device))
        head.eval()
        for var, PDIR in VARIANTS.items():
            log(f"[{tag}/{var}] {len(vkeys)} 键")
            for i, k in enumerate(vkeys):
                gt = gt_boxes[i]
                if not gt:
                    continue
                parts = k.split("_")
                stem = "_".join(parts[2:])
                try:
                    pim = Image.open(f"{PDIR}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
                except FileNotFoundError:
                    continue
                # 几何位移检查
                T = register_T(k, sift, bf, CLEAN_VALID, PDIR)
                disp = np.nan
                if T is not None:
                    pts = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), T).reshape(4, 2)
                    disp = float(np.linalg.norm(pts - CORNERS, axis=1).mean())
                x = tf(pim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
                with torch.no_grad():
                    f = model.features(x.to(device))
                    cam_r = cam_of(f, w1)[0].cpu().numpy()
                    cam_p = cam_of(ad(f), w1)[0].cpu().numpy()
                    pred = head(f.mean(dim=(2, 3))).cpu().numpy().reshape(4, 2) * 224.0
                raw_i = iou45(cam_r, gt)
                pl_i = iou45(cam_p, gt)
                # rectnet
                ph = pred.astype(np.float32)
                M = cv2.getPerspectiveTransform(ph, CORNERS)
                Mback = cv2.getPerspectiveTransform(CORNERS, ph)
                rect = F.grid_sample(x.to(device), _grid(M, 224, device),
                                     align_corners=False)
                with torch.no_grad():
                    fr = model.features(rect)
                    cam = torch.einsum("k,bkyx->byx", w1, fr).relu_()
                    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                    cam224 = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                           align_corners=False)[:, 0]
                cam_back = F.grid_sample(cam224[:, None], _grid(Mback, 224, device),
                                         align_corners=False)[0, 0].cpu().numpy()
                rect_i = iou45(np.clip(cam_back, 0, 1), gt)
                rows.append({"tag": tag, "variant": var, "key": k,
                             "patient": k.split("_")[0], "disp_px": disp,
                             "raw45": raw_i, "pairloc45": pl_i, "rectnet45": rect_i})
                if (i + 1) % 40 == 0:
                    log(f"  {i + 1}/{len(vkeys)}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p19_synthetic.csv", index=False)
    print("\n== P19 合成退化（掩码可用：disp<6px） ==")
    ok = df.disp_px < 6.0
    print(f"  几何检查: 中位位移 {df.disp_px.median():.2f}px，可用键 {int(ok.sum())}/{len(df)}")
    for (tag, var), g in df[ok].groupby(["tag", "variant"]):
        print(f"  [{tag}/{var}] n={len(g)}  raw {g.raw45.mean():.4f}"
              f"  pairloc Δ {(g.pairloc45 - g.raw45).mean():+.4f}"
              f"  rectnet Δ {(g.rectnet45 - g.raw45).mean():+.4f}")
    log("完成")
    return 0


def cam_of(feat, w1):
    cam = torch.einsum("k,bkyx->byx", w1, feat).relu_()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                         align_corners=False)[:, 0]


if __name__ == "__main__":
    raise SystemExit(main())
