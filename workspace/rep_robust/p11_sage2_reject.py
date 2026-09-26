r"""P4b：SAGE-2 拒绝信号原型（Cardi wrong 9.2% → 目标 ≤5%）。

任务：Cardi 恒选 pairloc 的 wrong-repair 9.2% 超过 α=5%。构造无标签拒绝信号：
  agree_p   pairloc CAM 的 7 视图一致性（id/hflip/rot±6/gamma0.75/1.33/bright0.85，
            二值化后两两 IoU 均值——p62 同款视图组）
  shift     结构分歧 = IoU(pairloc CAM 二值, raw CAM 二值)
  mag       pairloc CAM 质量（prob_photo 与 prob(pairloc 特征) 漂移）

评估纪律（沿 P0 三层）：
  Tier-b（可主张）：预注册分位规则——保留一致性 top-q 的图执行修复，τ 不看标签
  Tier-c（上界）：标签知情最优 τ（AUROC + wrong≤5% 下最大覆盖）

用法：/root/miniconda3/bin/python p11_sage2_reject.py [cardi|atel]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
SEED = 42
VIEWS = [("id", None), ("hflip", "hflip"), ("rot+6", ("rot", 6)), ("rot-6", ("rot", -6)),
         ("gamma075", ("gamma", 0.75)), ("gamma133", ("gamma", 1.33)),
         ("bright085", ("bright", 0.85))]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iou45(cam, gt):
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    return float(cam_iou_curve(cam, gt)[int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])


def cam_binary(cam, frac=0.15):
    th = np.quantile(cam, 1 - frac)
    return (cam >= th).astype(np.uint8)


def cam_iou(a, b):
    A, B = cam_binary(a), cam_binary(b)
    inter = (A & B).sum()
    union = (A | B).sum()
    return inter / union if union else 1.0


class Adapter(torch.nn.Module):
    def __init__(self, c=1024, r=16):
        super().__init__()
        self.d1 = torch.nn.Conv2d(c, r, 1)
        self.d2 = torch.nn.Conv2d(r, c, 1, bias=False)

    def forward(self, f):
        return f + self.d2(self.d1(f))


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "cardi"
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    if tag == "atel":
        cam_benchmark.CKPT = "/root/project/outputs/atel_seed42_densenet121/best.pt"
    model = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    w1 = model.classifier.weight[1].detach()
    ad = Adapter().to(device)
    sd = torch.load(f"{OUT}/pairloc_adapter_{tag}.pt", map_location=device)
    ad.load_state_dict(sd)
    ad.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    Dv = np.load(f"{OUT}/cache_photo_adapt.npz" if tag == "cardi"
                 else f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
    vkeys = [str(k) for k in Dv["keys"]]
    gts_v = Dv["gts"]
    PHOTO = (f"{BASE}/chexphoto/valid/natural/oneplus")
    gt_boxes = []
    for i in range(len(vkeys)):
        m = gts_v[i]
        ys, xs = np.where(m > 0)
        gt_boxes.append([(float(xs.min()), float(ys.min()),
                          float(xs.max()) + 1.0, float(ys.max()) + 1.0)]
                        if len(xs) else [])

    def apply_view(p224, v):
        name, par = v
        im = p224
        if par == "hflip":
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif isinstance(par, tuple) and par[0] == "rot":
            im = im.rotate(par[1], resample=Image.BILINEAR)
        elif isinstance(par, tuple) and par[0] == "gamma":
            im = Image.fromarray((np.asarray(im, np.float32) / 255.0) ** par[1] * 255
                                 ).convert(im.mode)
        elif isinstance(par, tuple) and par[0] == "bright":
            im = Image.fromarray(np.clip(np.asarray(im, np.float32) * par[1], 0, 255)
                                 .astype(np.uint8)).convert(im.mode)
        return im

    def pairloc_cam(p224):
        x = tf(p224.convert("RGB"))[None]
        with torch.no_grad():
            f = ad(model.features(x.to(device)))
            cam = torch.einsum("k,bkyx->byx", w1, f).relu_()
            cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            cam = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                align_corners=False)[0, 0].cpu().numpy()
        return cam

    def raw_cam(p224):
        x = tf(p224.convert("RGB"))[None]
        with torch.no_grad():
            f = model.features(x.to(device))
            cam = torch.einsum("k,bkyx->byx", w1, f).relu_()
            cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            cam = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                align_corners=False)[0, 0].cpu().numpy()
        return cam

    rows = []
    for i, k in enumerate(vkeys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        p224 = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
                          ).convert("L").resize((224, 224), Image.BILINEAR)
        cam_pl = pairloc_cam(p224)
        cam_rw = raw_cam(p224)
        views = []
        for v in VIEWS:
            views.append(cam_binary(pairloc_cam(apply_view(p224, v))))
        ag = []
        for a in range(len(views)):
            for b in range(a + 1, len(views)):
                A, B = views[a], views[b]
                u = (A | B).sum()
                ag.append((A & B).sum() / u if u else 1.0)
        gt = gt_boxes[i]
        rows.append({
            "key": k, "patient": k.split("_")[0],
            "pairloc45": iou45(cam_pl, gt) if gt else np.nan,
            "raw45_ref": iou45(cam_rw, gt) if gt else np.nan,
            "agree_p": float(np.mean(ag)),
            "shift": cam_iou(cam_pl, cam_rw)})
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(vkeys)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p11_sage2_reject_{tag}.csv", index=False)
    if tag == "cardi":
        ref = np.load(f"{OUT}/p08_pairloc_valid_iou_cardi.npy")
    else:
        ref = np.load(f"{OUT}/p08_pairloc_valid_iou_atel.npy")
    df["pl45_cached"] = ref[0]
    df["d"] = df.pl45_cached - ref[1]
    df = df.drop(columns=["pairloc45", "raw45_ref"])

    def auc(y, x):
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.argsort(np.argsort(x[o]))
        n1, n0 = y.sum(), (1 - y).sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

    dfv = df.dropna(subset=["d"])
    y = (dfv.d < -0.05).to_numpy()
    print(f"\n== [{tag}] 拒绝信号判别力（有害修复 d<-0.05, {int(y.sum())}/{len(dfv)}） ==")
    for c in ["agree_p", "shift"]:
        sgn = 1 if c == "shift" else -1   # shift 低→危险；agree 低→危险
        print(f"  AUROC({c}) = {auc(y, sgn * dfv[c].to_numpy()):.3f}")

    print("\n-- Tier-b：预注册分位（保留一致性 top-q 执行） --")
    for q in (0.5, 0.7, 0.9):
        for c in ["agree_p", "shift"]:
            th = np.quantile(df[c], q)
            m = df[c] >= th
            d = df.d[m].dropna()
            print(f"  {c}≥q{int(q*100)}  cov {m.mean():.3f}  Δ(执行集) {d.mean():+.4f}"
                  f"  wrong {(d < -0.05).mean():.3f}  系统Δ {df.d[m].sum() / len(df):+.4f}")
    print("-- Tier-c：标签知情（wrong≤5% 下最大覆盖） --")
    best = (0, None)
    for c in ["agree_p", "shift"]:
        for th in np.quantile(df[c], np.linspace(0.3, 0.95, 14)):
            m = df[c] >= th
            d = df.d[m].dropna()
            w = (d < -0.05).mean()
            if w <= 0.05 and m.mean() > best[0]:
                best = (m.mean(), (c, float(th), float(d.mean()), float(w)))
    print(f"  最优: {best[1]}  cov {best[0]:.3f}" if best[1] else "  无可行解")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
