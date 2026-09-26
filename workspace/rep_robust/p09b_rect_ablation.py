r"""P3B 对照：rectnet 超过 rect45（144%）的来源隔离。

两个假设：
  H1 224 空间 warp 管线本身优于 p06 的 native-warp→224 链路（插值/几何复合差异）
  H2 预测角点比 RANSAC GT 单应更平滑（MLP 归纳偏置滤掉配准高频抖动）
对照：valid73 上用 GT 配准角点走与 rectnet 完全相同的 224-warp 管线（rect224_gt），
再报预测 vs GT 角点的逐图 px 误差。若 rect224_gt ≈ rectnet → H1 为主；
若 rectnet > rect224_gt → H2（平滑化带来正则收益）。

用法：/root/miniconda3/bin/python p09b_rect_ablation.py
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
from p09_geo_rect import CORNERS, _grid, iou45, log, mask_bbox, register_T

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
BASE = "/root/autodl-tmp"
SEED = 42


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    cam_benchmark.CKPT = "/root/project/outputs/atel_seed42_densenet121/best.pt"
    model = load_model(device)
    w1 = model.classifier.weight[1].detach()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    CLEAN, PHOTO = f"{BASE}/CheXpert-v1.0-small/valid", f"{BASE}/chexphoto/valid/natural/oneplus"

    Dv = np.load(f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
    vkeys = [str(k) for k in Dv["keys"]]
    gts_v = Dv["gts"]
    df9 = pd.read_csv(f"{OUT}/p09_rectnet_atel.csv").set_index("key")
    gt_boxes = [mask_bbox(gts_v[i]) for i in range(len(vkeys))]

    import cv2 as _cv
    sift, bf = _cv.SIFT_create(), _cv.BFMatcher()
    rect_gt224, px_err = [], []
    for i, k in enumerate(vkeys):
        gt = gt_boxes[i]
        if not gt:
            rect_gt224.append(np.nan)
            px_err.append(np.nan)
            continue
        T = register_T(k, sift, bf, CLEAN, PHOTO)   # clean224→photo224
        if T is None:
            rect_gt224.append(np.nan)
            px_err.append(np.nan)
            continue
        gt_c = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), T).reshape(4, 2)
        M = cv2.getPerspectiveTransform(gt_c.astype(np.float32), CORNERS)
        Mback = cv2.getPerspectiveTransform(CORNERS, gt_c.astype(np.float32))
        parts = k.split("_")
        stem = "_".join(parts[2:])
        p224 = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
                          ).convert("L").resize((224, 224), Image.BILINEAR)
        x = tf(p224.convert("RGB"))[None]
        rect = F.grid_sample(x.to(device), _grid(M, 224, device), align_corners=False)
        with torch.no_grad():
            fr = model.features(rect)
            cam = torch.einsum("k,bkyx->byx", w1, fr).relu_()
            cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            cam224 = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                   align_corners=False)[:, 0]
        cam_back = F.grid_sample(cam224[:, None], _grid(Mback, 224, device),
                                 align_corners=False)[0, 0].cpu().numpy()
        rect_gt224.append(iou45(np.clip(cam_back, 0, 1), gt))
        px_err.append(np.nan)
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(vkeys)}")

    # 角点误差：valid73 的预测角点（重算 head 前向）vs GT
    FE = np.load(f"{OUT}/feat_pair_atel.npz", allow_pickle=True)
    f_val = FE["f_val"]
    import torch.nn as nn
    head = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 8)).to(device)
    head.load_state_dict({k.replace("net.", ""): v for k, v in torch.load(f"{OUT}/rectnet_head_atel.pt", map_location=device).items()})
    head.eval()
    with torch.no_grad():
        fv = torch.from_numpy(np.asarray(f_val, np.float32)).to(device)
        pred8 = head(fv.mean(dim=(2, 3))).cpu().numpy().reshape(-1, 4, 2) * 224.0
    for i, k in enumerate(vkeys):
        T = register_T(k, sift, bf, CLEAN, PHOTO)
        if T is None:
            px_err[i] = np.nan
            continue
        gt_c = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), T).reshape(4, 2)
        px_err[i] = float(np.linalg.norm(pred8[i] - gt_c, axis=1).mean())

    df = pd.DataFrame({"key": vkeys, "patient": [k.split("_")[0] for k in vkeys],
                       "raw45": df9.loc[vkeys, "raw45"].to_numpy(),
                       "rect45_gt_native": df9.loc[vkeys, "rect45_gt"].to_numpy(),
                       "rect224_gt": rect_gt224, "rectnet45": df9.loc[vkeys, "rectnet45"].to_numpy(),
                       "corner_px_err": px_err})
    df.to_csv(f"{OUT}/p09b_rect_ablation.csv", index=False)

    def boot_ci(d, key, n=400, seed=42):
        up = pd.unique(key)
        fmap = pd.Series(key).map({f: i for i, f in enumerate(up)}).to_numpy()
        r = np.random.default_rng(seed)
        vals = [d[np.isin(fmap, r.choice(len(up), len(up)))].mean() for _ in range(n)]
        return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    print("\n== P3B rectnet > rect45 来源隔离（Atel valid 73） ==")
    for c in ["rect45_gt_native", "rect224_gt", "rectnet45"]:
        d = df[c].to_numpy() - df.raw45.to_numpy()
        m, lo, hi = boot_ci(d, df.patient.to_numpy())
        print(f"  {c:18s} Δ {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    err = np.array(px_err)
    print(f"  预测 vs GT 角点误差: 中位 {np.nanmedian(err):.1f}px  p90 {np.nanpercentile(err, 90):.1f}px")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
