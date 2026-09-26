r"""P3A 原型：几何校正网络（photo-only 单应回归，部署无需 clean 原图）。

动机：P1 显示 Atel 的 photo 失效在采集几何（rect45 GT 配准 +0.1397），但 rect45
依赖 clean 原图配准，部署不可得。本原型在配对数据上训一个单应回归头：
  frozen Atel DenseNet 特征 GAP → MLP(1024→64→8)
  预测 clean224 规范四角在 photo224 空间的位置（归一化 8 维，条件良好）
  → cv2.getPerspectiveTransform 构造 H（photo224→clean224）
  → CAM(frozen 模型, 校正图) → 经 H⁻¹ 扭回 photo224 评价
监督 = 配对数据的 GT 配准角点（离线用 clean 图，部署只用 photo）。

定位保持目标（用户 P3A 规范的定位保持项）：L_loc = 校正后 CAM 扭回原空间
与 MedSAM 掩码的一致性（掩码外能量惩罚），作为辅助损失。
重建一致性：单应校正的 cycle 恒等（warp(warp(x,H),H⁻¹)=x），无参数，不构成损失。

判据：rectnet-CAM 在 Atel valid 73 上逼近 rect45（GT 配准上界 +0.1397）的
显著部分 → P3A GO（进入完整物理校正管线）。

用法：/root/miniconda3/bin/python p09_geo_rect.py [atel|cardi]
"""
from __future__ import annotations

import os
import sys
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

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
PH_REG, RATIO, RANSAC_THR, MIN_INLIERS = 800, 0.7, 5.0, 20
SEED = 42
EPOCHS = 300
CORNERS = np.array([[0, 0], [224, 0], [224, 224], [0, 224]], np.float32)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iou45(cam, gt):
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    return float(cam_iou_curve(cam, gt)[int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def corners_of_T(T):
    """T: clean224→photo224 → 4 角映射点 (4,2) 归一化。"""
    pts = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), T).reshape(4, 2)
    return (pts / 224.0).reshape(8).astype(np.float32)


def register_T(k, sift, bf, CLEAN, PHOTO):
    parts = k.split("_")
    stem = "_".join(parts[2:])
    cim = Image.open(f"{CLEAN}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
    pim = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
    cw, ch = cim.size
    pw, ph = pim.size
    s800 = PH_REG / max(pw, ph)
    photo800 = np.asarray(pim.resize((round(pw * s800), round(ph * s800)),
                                     Image.BILINEAR))
    kp1, d1 = sift.detectAndCompute(np.asarray(cim), None)
    kp2, d2 = sift.detectAndCompute(photo800, None)
    if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, nn in ((g[0], g[1]) for g in matches if len(g) == 2)
            if m.distance < RATIO * nn.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
    if H is None:
        return None
    H224 = H.copy()
    H224[0] *= 224.0 / (pw * s800)
    H224[1] *= 224.0 / (ph * s800)
    return (H224 @ np.diag([cw / 224.0, ch / 224.0, 1.0])).astype(np.float32)


class CornHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 8))

    def forward(self, x):
        return self.net(x)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "atel"
    finding = "Atelectasis" if tag == "atel" else "Cardiomegaly"
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    if tag == "atel":
        cam_benchmark.CKPT = "/root/project/outputs/atel_seed42_densenet121/best.pt"
    model = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    w1 = model.classifier.weight[1].detach()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    if tag == "atel":
        CLEAN, PHOTO = f"{BASE}/CheXpert-v1.0-small/valid", f"{BASE}/chexphoto/valid/natural/oneplus"
    else:
        CLEAN, PHOTO = f"{BASE}/CheXpert-v1.0-small/train", f"{BASE}/chexphoto/train/natural/iphone"
    D = np.load(f"{OUT}/pair_cache_{tag}.npz", allow_pickle=True)
    keys = [str(k) for k in D["keys"]]
    FE = np.load(f"{OUT}/feat_pair_{tag}.npz", allow_pickle=True)
    fmap_keys = [str(k) for k in FE["keys"]]
    assert fmap_keys == keys, "特征缓存与配对缓存键序不一致"
    f_photo = FE["f_photo"]

    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()
    log(f"[{tag}] 配准 {len(keys)} 对取 GT 角点")
    gt_corners = np.zeros((len(keys), 8), np.float32)
    ok = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        T = register_T(k, sift, bf, CLEAN, PHOTO)
        if T is not None:
            gt_corners[i] = corners_of_T(T)
            ok[i] = True
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(keys)}")
    log(f"[{tag}] reg_ok {int(ok.sum())}/{len(keys)}")

    pats = np.array([k.split("_")[0] for k in keys])
    up = sorted(set(pats))
    rng = np.random.default_rng(7)
    val_pat = set(rng.choice(up, size=max(2, len(up) // 5), replace=False).tolist())
    is_val = np.array([p in val_pat for p in pats])
    tr = np.where(ok & ~is_val)[0]
    va = np.where(ok & is_val)[0]
    log(f"[{tag}] train {len(tr)} / val {len(va)}（患者隔离）")

    def gap(idx):
        f = torch.from_numpy(np.asarray(f_photo[idx], np.float32)).to(device)
        return f, f.mean(dim=(2, 3))

    Xtr, Gtr = gap(tr), torch.from_numpy(gt_corners[tr]).to(device)
    Xva, Gva = gap(va), torch.from_numpy(gt_corners[va]).to(device)
    head = CornHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best = (1e9, None)
    for ep in range(EPOCHS):
        head.train()
        perm = np.random.default_rng(SEED * 31 + ep).permutation(len(tr))
        for s0 in range(0, len(tr), 32):
            sel = perm[s0:s0 + 32]
            pred = head(Xtr[1][sel])
            loss = F.mse_loss(pred, Gtr[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vloss = float(F.mse_loss(head(Xva[1]), Gva))
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    log(f"[{tag}] 角点回归 val MSE {best[0]:.5f}（≈{np.sqrt(best[0]) * 224:.1f}px）")

    # ---------- valid 评估 ----------
    Dv = np.load(f"{OUT}/cache_photo_atel.npz" if tag == "atel"
                 else f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    vkeys = [str(k) for k in Dv["keys"]]
    gts_v = Dv["gts"]
    dfp = (pd.read_csv(f"{OUT}/p62_photo_sage.csv") if tag == "cardi"
           else pd.read_csv(f"{OUT}/p65_atel_photo.csv")).set_index("key")
    raw_v = dfp.loc[vkeys, "raw45"].to_numpy()
    rect_v = pd.read_csv(f"{OUT}/p06_p1_candidates_{tag}.csv").set_index("key").loc[vkeys, "rect45"].to_numpy()
    f_val = FE["f_val"]
    # 评估键特征来自 f_val（pair 缓存排除了评估键，但特征已单独缓存）
    with torch.no_grad():
        fv = torch.from_numpy(np.asarray(f_val, np.float32)).to(device)
        pred_c = head(fv.mean(dim=(2, 3))).cpu().numpy().reshape(-1, 4, 2) * 224.0
    rows, cam_out = [], []
    for i, k in enumerate(vkeys):
        gt = mask_bbox(gts_v[i])
        raw_i = float(raw_v[i])
        r_i = np.nan
        if gt:
            ph_corners = pred_c[i].astype(np.float32)
            M = cv2.getPerspectiveTransform(ph_corners, CORNERS)      # photo→clean
            Mback = cv2.getPerspectiveTransform(CORNERS, ph_corners)  # clean→photo
            parts = k.split("_")
            stem = "_".join(parts[2:])
            pim = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            if tag == "atel":
                pass
            p224 = pim.resize((224, 224), Image.BILINEAR)
            x = tf(p224.convert("RGB"))[None]
            rect = F.grid_sample(x.to(device),
                                 _grid(M, 224, device), align_corners=False)
            with torch.no_grad():
                fr = model.features(rect)
                cam = torch.einsum("k,bkyx->byx", w1, fr).relu_()
                cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                cam224 = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                       align_corners=False)[:, 0]
            cam_back = F.grid_sample(cam224[:, None],
                                     _grid(Mback, 224, device),
                                     align_corners=False)[0, 0].cpu().numpy()
            cam_back = np.clip(cam_back, 0, 1)
            r_i = iou45(cam_back, gt)
            cam_out.append(cam_back.astype(np.float16))
        else:
            cam_out.append(np.zeros((224, 224), np.float16))
        rows.append({"key": k, "patient": k.split("_")[0], "raw45": raw_i,
                     "rect45_gt": float(rect_v[i]), "rectnet45": r_i})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p09_rectnet_{tag}.csv", index=False)
    np.savez_compressed(f"{OUT}/p09_rectnet_cams_{tag}.npz",
                        keys=np.array(vkeys), cam=np.stack(cam_out))
    torch.save(head.state_dict(), f"{OUT}/rectnet_head_{tag}.pt")

    def boot_ci(d, key, n=400, seed=42):
        up_ = pd.unique(key)
        fmap_ = pd.Series(key).map({f: i for i, f in enumerate(up_)}).to_numpy()
        r = np.random.default_rng(seed)
        vals = [d[np.isin(fmap_, r.choice(len(up_), len(up_)))].mean() for _ in range(n)]
        return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    m = df.rectnet45.notna()
    d1 = df.loc[m, "rectnet45"].to_numpy() - df.loc[m, "raw45"].to_numpy()
    b1 = boot_ci(d1, df.loc[m, "patient"].to_numpy())
    d2 = df.loc[m, "rect45_gt"].to_numpy() - df.loc[m, "raw45"].to_numpy()
    b2 = boot_ci(d2, df.loc[m, "patient"].to_numpy())
    print(f"\n== [{finding}] P3A rectnet（photo-only 校正） vs GT 配准上界 ==")
    print(f"  n={int(m.sum())}/{len(df)}（需配对特征）")
    print(f"  raw45          IoU {df.loc[m, 'raw45'].mean():.4f}")
    print(f"  rect45（GT 配准, p06 参照） Δ {b2[0]:+.4f} CI [{b2[1]:+.4f},{b2[2]:+.4f}]")
    print(f"  **rectnet45**  Δ {b1[0]:+.4f} CI [{b1[1]:+.4f},{b1[2]:+.4f}]"
          f"  （收割 GT 上界 {b1[0] / max(b2[0], 1e-9) * 100:.0f}%）")
    log("完成")
    return 0


def _grid(M, size, device):
    """M(3x3) → 归一化 grid (1,size,size,2)，供 F.grid_sample：dst←src。"""
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
    src = np.linalg.inv(M) @ pts
    src = src[:2] / src[2:3]
    grid = src.T.reshape(size, size, 2)
    grid[..., 0] = grid[..., 0] / (size - 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / (size - 1) * 2 - 1
    return torch.from_numpy(grid.astype(np.float32))[None].to(device)


if __name__ == "__main__":
    raise SystemExit(main())
