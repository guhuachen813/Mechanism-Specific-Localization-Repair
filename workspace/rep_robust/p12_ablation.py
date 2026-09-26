r"""P5 前置：PAIR-Loc 消融（Cardi 主战场，全量拟合 + early stop on valid65）。

变体（每变体独立重训 adapter，特征级训练 ~2min）：
  full       五损失全开（=p08 复现）
  no_pair    去配对蒸馏（logit/cam/geo 全关，只剩 anat+ident）
  no_camdist 去 CAM 蒸馏（cam/geo 关，保留 logit+anat+ident）
  no_reg     配准目标换成不配准的 clean teacher CAM（同布局直蒸）
  no_anat    去解剖约束
  no_geo     去几何等变一致
  no_ident   去 clean 身份损失（预期 clean 漂移增大）

输出：p12_ablation_cardi.csv + 控制台表。
用法：/root/miniconda3/bin/python p12_ablation.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, load_model

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
SEED = 42
EPOCHS = 40
BS = 32
W_FULL = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.5}
VARIANTS = {
    "full": (W_FULL, "reg"),
    "no_pair": ({"logit": 0.0, "cam": 0.0, "geo": 0.0, "anat": 0.1, "ident": 0.5}, "reg"),
    "no_camdist": ({"logit": 0.5, "cam": 0.0, "geo": 0.0, "anat": 0.1, "ident": 0.5}, "reg"),
    "no_reg": (W_FULL, "noreg"),
    "no_anat": ({"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.0, "ident": 0.5}, "reg"),
    "no_geo": ({"logit": 0.5, "cam": 1.0, "geo": 0.0, "anat": 0.1, "ident": 0.5}, "reg"),
    "no_ident": ({"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.0}, "reg"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


class Adapter(torch.nn.Module):
    def __init__(self, c=1024, r=16):
        super().__init__()
        self.d1 = torch.nn.Conv2d(c, r, 1)
        self.d2 = torch.nn.Conv2d(r, c, 1, bias=False)
        torch.nn.init.zeros_(self.d2.weight)
        torch.nn.init.normal_(self.d1.weight, std=1e-3)

    def forward(self, f):
        return f + self.d2(self.d1(f))


def cam_from_feat(feat, w1):
    cam = torch.einsum("k,bkyx->byx", w1, feat).relu_()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                         align_corners=False)[:, 0]


def iou45(cam, gt):
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    return float(cam_iou_curve(cam, gt)[int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    model = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    w1 = model.classifier.weight[1].detach()

    D = np.load(f"{OUT}/pair_cache_cardi.npz", allow_pickle=True)
    keys = [str(k) for k in D["keys"]]
    cam_t = D["cam_t"].astype(np.float32)
    mask_p = D["mask_p"].astype(np.float32)
    prob_t = D["prob_t"].astype(np.float32)
    reg_ok = D["reg_ok"].astype(bool)
    FE = np.load(f"{OUT}/feat_pair_cardi.npz", allow_pickle=True)
    f_photo, f_clean = FE["f_photo"], FE["f_clean"]
    vkeys = [str(k) for k in FE["vkeys"]]

    Dv = np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    gts_v = Dv["gts"]
    raw_v = np.load(f"{OUT}/p08_pairloc_valid_iou_cardi.npy")[1]
    gt_boxes_v = [mask_bbox(gts_v[i]) for i in range(len(vkeys))]
    f_val = FE["f_val"]

    def valid_iou(ad):
        ad.eval()
        with torch.no_grad():
            f = torch.from_numpy(np.asarray(f_val, np.float32)).to(device)
            cam = cam_from_feat(ad(f), w1).cpu().numpy()
        return np.array([iou45(cam[i], gt_boxes_v[i]) if gt_boxes_v[i] else np.nan
                         for i in range(len(vkeys))])

    def train(w, cam_mode):
        torch.manual_seed(SEED + 99)
        ad = Adapter().to(device)
        opt = torch.optim.Adam(ad.parameters(), lr=1e-3, weight_decay=1e-5)
        ok_idx = np.where(reg_ok)[0]
        best = (-1e9, None)
        for ep in range(EPOCHS):
            ad.train()
            perm = np.random.default_rng(SEED * 1000 + ep).permutation(len(ok_idx))
            for s0 in range(0, len(ok_idx), BS):
                sel = ok_idx[perm[s0:s0 + BS]]
                fb = torch.from_numpy(np.asarray(f_photo[sel], np.float32)).to(device)
                fc = torch.from_numpy(np.asarray(f_clean[sel], np.float32)).to(device)
                mk = torch.from_numpy(mask_p[sel]).to(device)
                pt = torch.from_numpy(prob_t[sel]).to(device)
                tgt = (torch.from_numpy(cam_t[sel]).to(device) if cam_mode == "reg"
                       else cam_from_feat(fc, w1))
                fs = ad(fb)
                cam_s = cam_from_feat(fs, w1)
                cam_s_fl = cam_from_feat(ad(torch.flip(fb, dims=[3])), w1)
                cam_t_fl = torch.flip(tgt, dims=[2])
                cam_c = cam_from_feat(ad(fc), w1)
                cam_c_ref = cam_from_feat(fc, w1)
                prob_s = torch.softmax(model.classifier(fs.mean(dim=(2, 3))), 1)[:, 1]
                l = {
                    "logit": w["logit"] * F.mse_loss(prob_s, pt),
                    "cam": w["cam"] * F.mse_loss(cam_s, tgt),
                    "geo": w["geo"] * F.mse_loss(cam_s_fl, cam_t_fl),
                    "anat": w["anat"] * (cam_s * (1 - F.interpolate(
                        mk[:, None], size=IMAGE_SIZE, mode="bilinear",
                        align_corners=False)[:, 0])).mean(),
                    "ident": w["ident"] * F.mse_loss(cam_c, cam_c_ref),
                }
                loss = sum(l.values())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            iv = valid_iou(ad)
            score = np.nanmean(iv) - np.nanmean(raw_v)
            if score > best[0]:
                best = (float(score), {k: v.detach().clone() for k, v in ad.state_dict().items()})
        ad.load_state_dict(best[1])
        return ad

    def clean_drift(ad):
        ok_idx = np.where(reg_ok)[0][::7]
        with torch.no_grad():
            fc = torch.from_numpy(np.asarray(f_clean[ok_idx], np.float32)).to(device)
            cs = cam_from_feat(ad(fc), w1).cpu().numpy()
            cr = cam_from_feat(fc, w1).cpu().numpy()
        return float(np.abs(cs - cr).mean())

    rows = []
    for name, (w, mode) in VARIANTS.items():
        ad = train(w, mode)
        iv = valid_iou(ad)
        d = iv - raw_v
        rng = np.random.default_rng(42)
        vp = np.array([k.split("_")[0] for k in vkeys])
        upv = pd.unique(vp)
        fmap = pd.Series(vp).map({f: i for i, f in enumerate(upv)}).to_numpy()
        vals = [d[np.isin(fmap, rng.choice(len(upv), len(upv)))].mean() for _ in range(400)]
        rows.append({"variant": name, "photo_d": float(np.nanmean(d)),
                     "lo": float(np.percentile(vals, 2.5)),
                     "hi": float(np.percentile(vals, 97.5)),
                     "wrong": float((d < -0.05).mean()),
                     "clean_drift": clean_drift(ad)})
        print(f"  {name:11s} Δ {rows[-1]['photo_d']:+.4f} "
              f"[{rows[-1]['lo']:+.4f},{rows[-1]['hi']:+.4f}] "
              f"wrong {rows[-1]['wrong']:.3f} clean_drift {rows[-1]['clean_drift']:.4f}",
              flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/p12_ablation_cardi.csv", index=False)
    print("\n== PAIR-Loc 消融（Cardi valid 65，全量拟合，early stop） ==")
    for r in rows:
        print(f"  {r['variant']:11s} Δ {r['photo_d']:+.4f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]"
              f" wrong {r['wrong']:.3f} clean_drift {r['clean_drift']:.4f}")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
