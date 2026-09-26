r"""CheXlocalize TEST 集（413 frontal）clean 侧银标签定位评估。

数据：AIMI 包 CheXpert/test 原图 + 已有 gt_segmentations_test.json（RLE）。
test 图与 valid 图同为 CheXpert 原始分辨率 jpg；frontal 413 张全部对齐。
评估：DenseNet Grad-CAM 的 raw vs银标签 GT（224 空间 RLE→bbox），报告
τ45/best/BA25/BA50，与 valid 65 张 silver_eval 同口径，bootstrap 400。

用法：/root/miniconda3/bin/python silver_eval_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cam_benchmark import (MEAN, STD, THRESHOLDS, batch_cam, cam_iou_curve,
                           load_model)
from silver_register import rle_string_to_counts  # 复用手写解码（LEB128+zigzag）

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
TEST_DIR = ("/root/autodl-tmp/chexlocalize_aimi/CheXlocalize/chexlocalize"
            "/CheXpert/test")
SEG_TEST = "/root/autodl-tmp/chexlocalize/gt_segmentations_test.json"
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    cnts = (rle["counts"] if isinstance(rle["counts"], list)
            else rle_string_to_counts(rle["counts"]))
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in cnts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val = 1 - val
    assert pos == h * w
    return flat.reshape((w, h)).T


def mask_bbox(m):
    if m.sum() == 0:
        return []
    ys, xs = np.where(m > 0)
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--finding", default="Cardiomegaly")
    args = ap.parse_args()
    FINDING = args.finding
    device = torch.device("cuda")
    model = load_model(device)
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    seg = json.load(open(SEG_TEST))
    keys = [k for k in seg if k.endswith("_frontal") and FINDING in seg[k]
            and seg[k][FINDING].get("counts")]
    log(f"test {FINDING} frontal 工作集 {len(keys)}")

    rows = []
    for n, k in enumerate(keys):
        # seg 键 patientXXXXX_studyY_viewZ_frontal → 路径
        path = f"{TEST_DIR}/{k.replace('_study', '/study').replace('_view', '/view')}.jpg"
        try:
            im = Image.open(path).convert("L")
            im224 = im.resize((224, 224), Image.BILINEAR)
            m_full = rle_decode(seg[k][FINDING])
            w, h = im.size
            mi = Image.fromarray(m_full * 255).resize((w, h), Image.BILINEAR)
            m0 = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            mi2 = Image.fromarray(m0 * 255).resize((224, 224), Image.BILINEAR)
            m224 = (np.asarray(mi2, np.float32) / 255.0 > 0.5).astype(np.uint8)
            gt = mask_bbox(m224)
            with torch.no_grad():
                cam, prob = batch_cam(model, tf(im224.convert("RGB"))[None], device)
            curve = cam_iou_curve(cam[0], gt)
            rec = {"key": k,
                   "raw_45": float(curve[TAU45]), "raw_best": float(max(curve)),
                   "raw_ba25": float(np.array(curve)[THRESHOLDS <= 0.25].max()),
                   "raw_ba50": float(np.array(curve)[THRESHOLDS <= 0.5].max()),
                   "prob": float(prob[0]), "gt_area": float(m224.mean())}
            rows.append(rec)
        except Exception as e:  # noqa: BLE001
            rows.append({"key": k, "err": f"{type(e).__name__}: {e}"[:120]})
        if (n + 1) % 50 == 0:
            log(f"  {n + 1}/{len(keys)}")

    df = pd.DataFrame(rows)
    if "err" in df.columns and df.err.notna().any():
        log(f"err {int(df.err.notna().sum())}")
        df = df[df.err.isna()]
    df.to_csv(f"{OUT}/silver_eval_test_{FINDING.lower()}.csv", index=False)

    print(f"\n== CheXlocalize TEST clean 侧（{FINDING}，n={len(df)}） ==")
    print(f"  raw τ45 {df.raw_45.mean():.3f}   best {df.raw_best.mean():.3f}   "
          f"BA25 {df.raw_ba25.mean():.3f}   BA50 {df.raw_ba50.mean():.3f}")
    rng = np.random.default_rng(42)
    uniq = df.key.unique()
    fmap = df.key.map({f: i for i, f in enumerate(uniq)}).to_numpy()
    vals = [df.raw_45[np.isin(fmap, rng.choice(len(uniq), len(uniq)))].mean()
            for _ in range(400)]
    print(f"  τ45 bootstrap 95% CI [{np.percentile(vals, 2.5):.3f}, "
          f"{np.percentile(vals, 97.5):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
