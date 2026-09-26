"""思路A 第1b步：图场检测 + 分结构 box 提示 + 形态学清理，修复真实翻拍上的先验失效。

针对 seg_check.py 暴露的三个问题逐一处理：
1. 真实翻拍上固定图像级 box 锁定屏幕而非肺野 → 先做**图场检测**：找亮度最大
   连通域（胶片/屏幕区域），得到 field bbox；后续一切提示与 mask 都限制在场内。
2. 纵隔/心影混入 → 不再用整肺 box，改为**左肺、右肺两个独立 box**（场内相对
   坐标 [0.02,0.50]×[0.03,0.85] 与 [0.50,0.98]×[0.03,0.85]），取并集。
3. 碎片噪点与泄漏 → 形态学闭运算 + 去除小于场面积 0.3% 的连通域 + 填洞。

用法（GPU07）：
    /root/miniconda3/bin/python seg_check2.py                        # 全量 50 张
    /root/miniconda3/bin/python seg_check2.py --n-valid 2 --n-natural 1 --n-synphoto 1

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
import scipy.ndimage as ndi
from transformers import SamModel

OUTDIR = Path("/root/autodl-tmp/experiments/rep_robust")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIZE = 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 图场检测 --
def detect_field(gray: Image.Image) -> tuple[int, int, int, int]:
    """返回场 bbox (x1,y1,x2,y2)，全分辨率坐标。

    胶片/屏幕是图中最大的亮连通域：降采样 → 亮度阈值 → 闭运算 → 最大域 bbox。
    """
    w0, h0 = gray.size
    s = 256 / max(w0, h0)
    small = np.asarray(gray.resize((max(1, round(w0 * s)), max(1, round(h0 * s))),
                                   Image.BILINEAR), dtype=np.float32)
    thr = 0.5 * (np.quantile(small, 0.25) + np.quantile(small, 0.85))
    m = small > thr
    m = ndi.binary_closing(m, structure=np.ones((5, 5)))
    lab, n = ndi.label(m)
    if n == 0:
        return 0, 0, SIZE, SIZE
    sizes = ndi.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == k)
    # small 坐标 → 1024 空间：x_1024 = x_small * SIZE / (s * w0)
    x1 = int(xs.min() * SIZE / (s * w0))
    x2 = int(xs.max() * SIZE / (s * w0)) + 1
    y1 = int(ys.min() * SIZE / (s * h0))
    y2 = int(ys.max() * SIZE / (s * h0)) + 1
    mx, my = int(0.02 * SIZE), int(0.02 * SIZE)
    return max(0, x1 - mx), max(0, y1 - my), min(SIZE, x2 + mx), min(SIZE, y2 + my)


# ---------------------------------------------------------------- MedSAM --
def medsam_prep(g: Image.Image) -> torch.Tensor:
    a = np.asarray(g.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    lo, hi = a.min(), max(a.max(), a.min() + 1e-8)
    a = (a - lo) / (hi - lo) * 255.0
    t = torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1) / 255.0
    return (t - MEAN) / STD


def field_boxes(f: tuple[int, int, int, int]) -> torch.Tensor:
    """场（1024 空间）内相对坐标的两个肺 box → (1,2,4) 张量。"""
    x1, y1, x2, y2 = f
    W, H = x2 - x1, y2 - y1
    rel = [(0.02, 0.03, 0.50, 0.85), (0.50, 0.03, 0.98, 0.85)]
    boxes = [[x1 + a * W, y1 + b * H, x1 + c * W, y1 + d * H]
             for a, b, c, d in rel]
    return torch.tensor([boxes], dtype=torch.float32)


@torch.no_grad()
def segment(model, pixel: torch.Tensor, f: tuple[int, int, int, int],
            device: torch.device) -> np.ndarray:
    """左肺+右肺 → 并集 → 清理 → 1024 空间二值 mask（场外为 0）。"""
    boxes = field_boxes(f)
    out = model(pixel_values=pixel.to(device), input_boxes=boxes.to(device))
    pm = out.pred_masks[:, :, 0]                    # (B, nb, 256,256) 单 mask
    pm = pm.float().sigmoid().cpu().numpy()
    H_f, W_f = f[3] - f[1], f[2] - f[0]
    mask_f = np.zeros((H_f, W_f), dtype=np.uint8)
    for k in range(pm.shape[1]):                    # 两肺并集
        m = Image.fromarray((pm[0, k] > 0.5).astype(np.uint8) * 255)
        m = np.asarray(m.resize((W_f, H_f), Image.BILINEAR)) > 127
        mask_f |= m.astype(np.uint8)
    # ---- 形态学清理 ----
    mask_f = ndi.binary_closing(mask_f, structure=np.ones((7, 7)))
    mask_f = ndi.binary_fill_holes(mask_f)
    lab, n = ndi.label(mask_f)
    if n:
        sizes = ndi.sum(mask_f, lab, range(1, n + 1))
        keep = np.isin(lab, np.where(sizes >= 0.003 * mask_f.size)[0] + 1)
        mask_f = keep.astype(np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[f[1]:f[3], f[0]:f[2]] = mask_f
    return mask


def overlay(orig: Image.Image, mask: np.ndarray, f: tuple[int, int, int, int]) -> Image.Image:
    W, H = orig.size
    m = Image.fromarray(mask * 255).resize((W, H), Image.BILINEAR)
    red = Image.new("RGB", (W, H), (255, 40, 40))
    ov = Image.composite(red, orig.convert("RGB"), m.point(lambda v: int(v * 0.45)))
    d = ov.copy()
    from PIL import ImageDraw
    dr = ImageDraw.Draw(d)
    dr.rectangle([f[0], f[1], f[2], f[3]], outline=(0, 160, 255), width=4)
    tri = Image.new("RGB", (3 * 340 + 20, round(340 * H / W) + 4), "white")
    for i, im in enumerate([orig.convert("RGB"), d,
                            Image.merge("RGB", [Image.fromarray(mask * 255)] * 3)]):
        im = im.resize((340, round(340 * H / W)))
        tri.paste(im, (i * (340 + 10), 2))
    return tri


def mask_stats(mask: np.ndarray) -> dict:
    frac = float(mask.mean())
    lab, n = ndi.label(mask)
    areas = ndi.sum(mask, lab, range(1, n + 1)) if n else np.array([0])
    big = int((areas > 0.002 * mask.size).sum())
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

    outdir = OUTDIR / "seg_check2"
    outdir.mkdir(exist_ok=True)
    stats, sheets = [], []
    for gname, df, n in groups:
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        for k, (_, row) in enumerate(df.iloc[idx].iterrows()):
            with Image.open(row["disk_path"]) as im:
                g = im.convert("L")
            f = detect_field(g)
            pixel = medsam_prep(g)
            m = segment(model, pixel, f, device)
            st = mask_stats(m)
            st.update({"group": gname, "key": row["key"],
                       "field_frac": (f[2] - f[0]) * (f[3] - f[1]) / (SIZE * SIZE)})
            stats.append(st)
            sheets.append(overlay(g, m, f))
            if len(sheets) == 5:
                sheet = Image.new("RGB", (sheets[0].width,
                                          sum(s.height + 6 for s in sheets)), "white")
                y = 0
                for s in sheets:
                    sheet.paste(s, (0, y)); y += s.height + 6
                sheet.save(outdir / f"sheet_{gname}_{k // 5}.png")
                sheets = []
            log(f"{gname} {k + 1}/{n} field={st['field_frac']:.2f} "
                f"frac={st['frac']:.3f} comp={st['components']}")
    pd.DataFrame(stats).to_csv(outdir / "mask_stats.csv", index=False)
    log(f"完成 -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
