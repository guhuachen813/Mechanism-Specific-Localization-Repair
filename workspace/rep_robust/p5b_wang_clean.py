r"""5.1b 补充：Wang 2024 推理期复现在 test 154 阳性子集 clean 侧的对照。
域内（clean, 无翻拍）上细化是否有效？对照 photo 侧负结果，分离"方法本身 vs 域偏移"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from cam_benchmark import MEAN, STD, THRESHOLDS, batch_cam, cam_iou_curve, load_model
from p5b_wang_baseline import multi_class_cam, wang_ff, wang_nlm
from silver_eval_test import OUT, SEG_TEST, TEST_DIR, rle_decode

FINDING = "Cardiomegaly"
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


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
    seg = json.load(open(SEG_TEST)) if False else None
    import json
    seg = json.load(open(SEG_TEST))
    keys = [k for k in seg if k.endswith("_frontal") and FINDING in seg[k]
            and seg[k][FINDING].get("counts")]
    rows = []
    for n, k in enumerate(keys):
        path = f"{TEST_DIR}/{k.replace('_study', '/study').replace('_view', '/view')}.jpg"
        try:
            im = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
            m_full = rle_decode(seg[k][FINDING])
            w, h = im.size
            mi = Image.fromarray(m_full * 255).resize((224, 224), Image.BILINEAR)
            m224 = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            gt = mask_bbox(m224)
            if not gt:
                continue
            x = tf(im.convert("RGB"))[None]
            with torch.no_grad():
                cam, prob = batch_cam(model, x, device)
            cam = cam[0]
            mo, _ = multi_class_cam(model, x, device)
            mo = mo[0].detach().cpu().numpy()
            mo = np.array(Image.fromarray((mo / max(mo.max(), 1e-8) * 255).astype(np.uint8)
                                          ).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
            nlm = wang_nlm(cam, cam)   # 单类 CAM 口径（与 photo 侧 nlm_s 一致）
            ff = wang_ff(nlm)
            rec = {"key": k}
            for v, c in [("raw", cam), ("wang_nlm", nlm), ("wang_ff", ff)]:
                curve = np.array(cam_iou_curve(c, gt))
                rec[f"{v}_45"] = float(curve[TAU45])
                rec[f"{v}_best"] = float(curve.max())
            rows.append(rec)
        except Exception as e:  # noqa: BLE001
            print(f"err {k}: {e}", flush=True)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(keys)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p5b_wang_test_clean.csv", index=False)
    print(f"\n== Wang2024 复现（test clean 阳性子集 n={len(df)}） ==")
    for v in ["raw", "wang_nlm", "wang_ff"]:
        print(f"  {v:10s} τ45 {df[f'{v}_45'].mean():.3f}  best {df[f'{v}_best'].mean():.3f}")
    rng = np.random.default_rng(42)
    uniq = df.key.unique()
    fmap = df.key.map({f: i for i, f in enumerate(uniq)}).to_numpy()
    for v in ["wang_nlm", "wang_ff"]:
        d = df[f"{v}_45"] - df.raw_45
        vals = [d[np.isin(fmap, rng.choice(len(uniq), len(uniq)))].mean() for _ in range(400)]
        print(f"  {v:10s} vs raw: {d.mean():+.3f} CI [{np.percentile(vals,2.5):+.3f},{np.percentile(vals,97.5):+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
