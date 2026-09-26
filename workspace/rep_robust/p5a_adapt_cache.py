r"""5.1a 前置：photo 侧适配缓存。对 65 张 oneplus 配准银标签图，逐图计算：
  photo224 raw CAM（DenseNet）、MedSAM 先验 mask、场框、q 向量（与训练侧逐位一致）、
  GT（配准银标签 224 mask → 紧致框）、clean 侧同款（供适配训练时混合轴用）
存 cache_photo.npz。后续嵌套 CV 全部从缓存读，不重复推理。

协议冻结点（写死，防漂移）：
  - 键集：silver_masks_224.npz 中 Cardiomegaly 通道非空的 65 键（= silver_v2_cardi.csv 同一工作集）
  - 银标签 GT：npz 里 224 空间 mask（配准传播产物），紧致框评估
  - q 向量：repairer.q_vector(g, m, ok) + repairer.cam_feats(cam, m)，9 维
  - patient = key 前 9 字符（patientXXXXX）

用法：/root/miniconda3/bin/python p5a_adapt_cache.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from cam_benchmark import MEAN, STD, batch_cam, load_model
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import cam_feats, q_vector

BASE = "/root/autodl-tmp"
PHOTO = f"{BASE}/chexphoto/valid/natural/oneplus"
CLEAN = f"{BASE}/CheXpert-v1.0-small/valid"
OUT = f"{BASE}/experiments/rep_robust/chexlocalize"
FINDING = "Cardiomegaly"
ALPHA = 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def main() -> int:
    device = torch.device("cuda")
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    log("加载 MedSAM")
    from transformers import SamModel
    smodel = SamModel.from_pretrained(f"{BASE}/experiments/rep_robust/weights/medsam"
                                      ).to(device).eval()

    z = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = list(z["findings"])
    ci = findings.index(FINDING)
    keys = sorted(k for k in z.files if k != "findings" and z[k][ci].sum() > 0)
    log(f"工作集 {len(keys)} 键")

    cams_p, cams_c, masks_p, q9, gts, probs = [], [], [], [], [], []
    keys_ok = []
    for n, k in enumerate(keys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        try:
            photo_im = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            clean_im = Image.open(f"{CLEAN}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            photo224 = photo_im.resize((224, 224), Image.BILINEAR)
            clean224 = clean_im.resize((224, 224), Image.BILINEAR)
            gt = z[k][ci].astype(np.uint8)

            with torch.no_grad():
                cam_p, prob_p = batch_cam(model, tf(photo224.convert("RGB"))[None], device)
                cam_c, _ = batch_cam(model, tf(clean224.convert("RGB"))[None], device)
            cam_p, cam_c = cam_p[0], cam_c[0]

            fr, ok_r = detect_field(photo224)
            mm = mask_224(segment(smodel, medsam_prep(photo224), fr, device)
                          ).astype(np.uint8)
            g = np.asarray(photo224.convert("L"), np.float32) / 255.0
            qv = np.concatenate([q_vector(g, mm, float(ok_r)), cam_feats(cam_p, mm)])

            cams_p.append(cam_p.astype(np.float32))
            cams_c.append(cam_c.astype(np.float32))
            masks_p.append(mm.astype(np.uint8))
            q9.append(qv.astype(np.float32))
            gts.append(gt)
            probs.append(float(prob_p[0]))
            keys_ok.append(k)
        except Exception as e:  # noqa: BLE001
            log(f"  err {k}: {e}")
        if (n + 1) % 10 == 0:
            log(f"  {n + 1}/{len(keys)}")

    np.savez_compressed(
        f"{OUT}/cache_photo_adapt.npz",
        keys=np.array(keys_ok), cams_p=np.stack(cams_p), cams_c=np.stack(cams_c),
        masks_p=np.stack(masks_p), q9=np.stack(q9), gts=np.stack(gts),
        probs=np.array(probs), finding=FINDING)
    log(f"cache_photo_adapt.npz 完成：{len(keys_ok)} 键")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
