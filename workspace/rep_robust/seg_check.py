"""思路A 第一步：真·MedSAM 在胸片上的肺野 mask 质量检查（50 张）。

流程
----
* 抽样：CheXpert valid 正位 30 张 + CheXphoto natural（真实翻拍）10 张 +
  synthetic/photographic 10 张。
* MedSAM 原生口径：squash resize 1024×1024 + 逐图 min-max；肺野 box 提示用
  统一启发式（PA 胸片肺野约占 x:[0.06,0.94], y:[0.10,0.90]）。
* 输出：每张图一行覆盖图（原图 | mask 叠加 | 二值 mask），拼成 contact sheet，
  供肉眼验收；同时记录 mask 面积占比、连通域数、左右对称性等客观统计。

用法（GPU06）：
    /root/miniconda3/bin/python seg_check.py --n-per-group 10 30 10

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize
from transformers import SamModel

OUTDIR = Path("/root/autodl-tmp/experiments/rep_robust")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIZE = 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def medsam_prep(g: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    """返回 (1,3,1024,1024) 张量与原始尺寸 (W,H)。"""
    W, H = g.size
    a = np.asarray(g.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    lo, hi = a.min(), max(a.max(), a.min() + 1e-8)
    a = (a - lo) / (hi - lo) * 255.0
    t = torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1) / 255.0
    return (t - MEAN) / STD, (W, H)


def lung_box_1024() -> np.ndarray:
    """统一肺野 box 提示（1024 空间）。"""
    return np.array([[0.06 * SIZE, 0.10 * SIZE, 0.94 * SIZE, 0.90 * SIZE]],
                    dtype=np.float64)


@torch.no_grad()
def segment(model, pixel: torch.Tensor, device: torch.device) -> np.ndarray:
    """返回 1024×1024 二值 mask。"""
    pixel = pixel.to(device)
    box = torch.tensor(lung_box_1024(), dtype=torch.float32)
    out = model(pixel_values=pixel, input_boxes=box[None].to(pixel.device))
    pm = out.pred_masks[0, 0, 0]                       # (256,256)，MedSAM 单 mask
    pm = pm.float().sigmoid().cpu().numpy()
    mask = (pm > 0.5).astype(np.uint8)
    # 低分辨率 mask 上采样回 1024，并按 MedSAM 惯例裁剪到 box 外为 0
    m = Image.fromarray(mask * 255).resize((SIZE, SIZE), Image.NEAREST)
    m = np.asarray(m) > 127
    keep = np.zeros_like(m)
    x1, y1, x2, y2 = [int(v) for v in box[0].tolist()]
    keep[y1:y2, x1:x2] = m[y1:y2, x1:x2]
    return keep.astype(np.uint8)


def overlay(orig: Image.Image, mask1024: np.ndarray) -> Image.Image:
    """(原图 | 叠加 | mask) 三联图，各 340 宽。"""
    W, H = orig.size
    m = Image.fromarray(mask1024 * 255).resize((W, H), Image.BILINEAR)
    red = Image.new("RGB", (W, H), (255, 40, 40))
    ov = Image.composite(red, orig.convert("RGB"), m.point(lambda v: int(v * 0.45)))
    tri = Image.new("RGB", (3 * 340 + 20, round(340 * H / W) + 4), "white")
    for i, im in enumerate([orig.convert("RGB"), ov, Image.merge("RGB",
                            [Image.fromarray(mask1024 * 255)] * 3)]):
        im = im.resize((340, round(340 * H / W)))
        tri.paste(im, (i * (340 + 10), 2))
    return tri


def mask_stats(mask: np.ndarray) -> dict:
    import scipy.ndimage as ndi
    frac = float(mask.mean())
    lab, n = ndi.label(mask)
    areas = ndi.sum(mask, lab, range(1, n + 1)) if n else np.array([0])
    big = int((areas > 0.002 * mask.size).sum())       # 大于 0.2% 图面积的连通域
    lr = mask[:, :SIZE // 2].mean() / max(mask[:, SIZE // 2:].mean(), 1e-6)
    return {"frac": frac, "components": int(n), "big_components": big,
            "lr_ratio": float(np.clip(lr, 0, 10))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-valid", type=int, default=30)
    ap.add_argument("--n-natural", type=int, default=10)
    ap.add_argument("--n-synphoto", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    log("加载 medsam")
    model = SamModel.from_pretrained(OUTDIR / "weights" / "medsam").to(device).eval()

    man = pd.read_csv(OUTDIR / "eval_list.csv")
    rng = np.random.RandomState(args.seed)
    groups = [("valid", man[man["cond"] == "clean"], args.n_valid),
              ("natural", man[man["cond"] == "natural/oneplus"], args.n_natural),
              ("synphoto", man[man["cond"] == "synthetic/photographic"], args.n_synphoto)]

    outdir = OUTDIR / "seg_check"
    outdir.mkdir(exist_ok=True)
    stats = []
    sheets = []
    for gname, df, n in groups:
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        for k, (_, row) in enumerate(df.iloc[idx].iterrows()):
            with Image.open(row["disk_path"]) as im:
                g = im.convert("L")
            pixel, _ = medsam_prep(g)
            m = segment(model, pixel, device)
            st = mask_stats(m)
            st.update({"group": gname, "key": row["key"]})
            stats.append(st)
            sheets.append(overlay(g, m))
            if len(sheets) == 5:
                sheet = Image.new("RGB", (sheets[0].width, sum(s.height + 6 for s in sheets)),
                                  "white")
                y = 0
                for s in sheets:
                    sheet.paste(s, (0, y)); y += s.height + 6
                sheet.save(outdir / f"sheet_{gname}_{k // 5}.png")
                sheets = []
            log(f"{gname} {k + 1}/{n} frac={st['frac']:.3f} comp={st['components']}")
    pd.DataFrame(stats).to_csv(outdir / "mask_stats.csv", index=False)
    log(f"完成 -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
