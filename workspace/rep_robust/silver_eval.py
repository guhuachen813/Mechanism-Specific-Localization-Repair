r"""翻拍空间银标签 CAM 评估：真实翻拍轴的绝对定位精度（paired clean vs photo）。

工作集：silver_masks_224.npz 中 Cardiomegaly 通道非空的图（65 张，registration_ok）。
每图两个空间、三个变体：
  clean 空间：clean 小图 → 224，CAM raw（GT = clean mask → 224）
  photo 空间：翻拍 → 224，CAM raw / center_field / field_crop（GT = 银标签 mask @224）
指标：best-IoU、固定 τ=0.45、BoxAcc@0.25/0.5（对银 GT 紧致框）、正类概率。
GT-free 变体的 mask 由场检测给出（cam_crop_baselines）。

输出：chexlocalize/silver_eval.csv（gpu10）。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cam_benchmark import (IMAGE_SIZE, MEAN, STD, THRESHOLDS, batch_cam,
                           cam_iou_curve, load_model)
from cam_crop_baselines import center_mask_field, field_mask
from cam_prior import detect_field
from silver_register import PHOTO, SEG_JSON, rle_decode

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
FINDING = "Cardiomegaly"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mask_to_224(m: np.ndarray) -> np.ndarray:
    from PIL import Image as I
    im = I.fromarray(m * 255).resize((IMAGE_SIZE, IMAGE_SIZE), I.BILINEAR)
    return (np.asarray(im, np.float32) / 255.0 > 0.5).astype(np.uint8)


def mask_bbox(m: np.ndarray) -> list[tuple[float, float, float, float]] | list:
    if m.sum() == 0:
        return []
    ys, xs = np.where(m > 0)
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def main() -> int:
    global FINDING
    ap = argparse.ArgumentParser()
    ap.add_argument("--finding", default="Cardiomegaly")
    ap.add_argument("--ckpt", default="/root/project/outputs/repaired_seed42/densenet121/best.pt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    FINDING = args.finding
    tag = args.tag or FINDING.lower()[:5]
    import cam_benchmark
    cam_benchmark.CKPT = __import__("pathlib").Path(args.ckpt)
    device = torch.device("cuda")
    model = load_model(device)
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])

    seg = json.load(open(SEG_JSON))
    z = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = list(z["findings"])
    ci = findings.index(FINDING)
    keys = [k for k in z.files
            if k != "findings" and z[k][ci].sum() > 0]
    log(f"银标签工作集：{len(keys)} 张（{FINDING}）")

    cf_cache, rows = {}, []
    for n, k in enumerate(keys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        path_c = f"/root/autodl-tmp/CheXpert-v1.0-small/valid/{parts[0]}/{parts[1]}/{stem}.jpg"
        path_p = f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
        try:
            # 银标签 GT
            m_photo = z[k][ci].astype(np.uint8)                    # @224 翻拍空间
            rle = seg[k].get(FINDING)
            m_clean_full = rle_decode(rle)
            clean_im = Image.open(path_c).convert("L")
            cw, ch = clean_im.size
            mi = Image.fromarray(m_clean_full * 255).resize((cw, ch), Image.BILINEAR)
            m_clean0 = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            m_clean = mask_to_224(m_clean0)

            # 两个空间的输入
            clean224 = clean_im.resize((224, 224), Image.BILINEAR).convert("RGB")
            photo224 = Image.open(path_p).convert("RGB").resize(
                (224, 224), Image.BILINEAR)

            # 场检测（photo 空间变体）
            f_r, ok_r = detect_field(photo224.convert("L"))
            if ok_r and (f_r is not None):
                variants_p = {
                    "photo_raw": np.ones((224, 224), np.float32),
                    "photo_center_field": center_mask_field(f_r),
                    "photo_field_crop": field_mask(f_r),
                }
            else:
                variants_p = {"photo_raw": np.ones((224, 224), np.float32)}

            with torch.no_grad():
                cam_c, prob_c = batch_cam(
                    model, tf(clean224)[None], device)
                cam_p, prob_p = batch_cam(
                    model, tf(photo224)[None], device)
            cam_c, cam_p = cam_c[0], cam_p[0]

            gt_p = mask_bbox(m_photo)
            gt_c = mask_bbox(m_clean)
            rec = {"key": k, "field_ok_photo": int(ok_r),
                   "prob_clean": float(prob_c[0]), "prob_photo": float(prob_p[0]),
                   "silver_area_photo": float(m_photo.mean()),
                   "silver_area_clean": float(m_clean.mean())}
            curves = {"clean_raw": cam_iou_curve(cam_c, gt_c)}
            for vname, w in variants_p.items():
                curves[vname] = cam_iou_curve(cam_p * w, gt_p)
            for vname, curve in curves.items():
                rec[f"{vname}_best"] = float(max(curve))
                rec[f"{vname}_iou45"] = float(curve[THRESHOLDS.tolist().index(0.45)])
                rec[f"{vname}_boxacc25"] = float(np.max(
                    np.array(curve)[THRESHOLDS <= 0.25]))
                rec[f"{vname}_boxacc50"] = float(np.max(
                    np.array(curve)[THRESHOLDS <= 0.5]))
                rec[f"{vname}_tbest"] = float(THRESHOLDS[int(np.argmax(curve))])
            rows.append(rec)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"key": k, "err": f"{type(e).__name__}: {e}"[:120]})
            if len(rows) > 1:
                break
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(keys)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/silver_eval_{tag}.csv", index=False)
    log(f"写出 silver_eval_{tag}.csv（{len(df)} 行，err {int(df.get('err').notna().sum()) if 'err' in df else 0}）")

    mets = [c for c in df.columns if c.endswith("_best") or c.endswith("_iou45")
            or c.endswith("_boxacc25")]
    print("\n== 银标签绝对定位（mean） ==")
    print(df[mets].mean().round(3).to_string())
    if {"clean_raw_iou45", "photo_raw_iou45"} <= set(df.columns):
        d = df.photo_raw_iou45 - df.clean_raw_iou45
        print(f"\n配对 photo−clean τ45: mean {d.mean():+.3f}（真实翻拍净退化）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
