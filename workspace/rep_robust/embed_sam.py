"""思路C 前哨实验 第1步：冻结 SAM/MedSAM 图像编码器抽嵌入。

口径
----
* 灰度化与 DenseNet 管线一致：PIL convert("L")（CheXphoto 图先整图转灰度）。
* SAM 原生预处理：ResizeLongestSide(1024) + 右下补零 + SAM pixel_mean/std。
  两个骨干（SAM、MedSAM）使用完全相同的预处理，保证差异只来自表征本身。
* 嵌入 = get_image_features() 输出 (B,256,64,64) 的空间均值 → 256 维。

用法（GPU06）：
    /root/miniconda3/bin/python embed_sam.py \
        --weights-dir /root/autodl-tmp/experiments/rep_robust/weights/sam \
        --tag sam --lists train_list.csv eval_list.csv
    /root/miniconda3/bin/python embed_sam.py \
        --weights-dir /root/autodl-tmp/experiments/rep_robust/weights/medsam \
        --tag medsam --lists train_list.csv eval_list.csv

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
from transformers import SamModel

PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1)
SIZE = 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preprocess_batch(paths: list[str]) -> torch.Tensor:
    """灰度 → 最长边缩放到 1024 → 右下补零 → 归一化。"""
    out = torch.zeros(len(paths), 3, SIZE, SIZE)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            g = im.convert("L")
        w, h = g.size
        s = SIZE / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        g = g.resize((nw, nh), Image.BILINEAR).convert("RGB")
        a = torch.from_numpy(np.asarray(g, dtype=np.float32).transpose(2, 0, 1))
        out[i, :, :nh, :nw] = a
    return (out - PIXEL_MEAN) / PIXEL_STD


@torch.no_grad()
def embed(model, paths: list[str], device: torch.device, bs: int) -> np.ndarray:
    feats = []
    for s in range(0, len(paths), bs):
        x = preprocess_batch(paths[s:s + bs]).to(device)
        out = model.vision_encoder(pixel_values=x)        # (B,256,64,64)
        f = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        feats.append(f.float().mean(dim=(2, 3)).cpu().numpy())
        if (s // bs) % 20 == 0:
            log(f"  {min(s + bs, len(paths))}/{len(paths)}")
    return np.concatenate(feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True, choices=["sam", "medsam"])
    ap.add_argument("--lists", type=Path, nargs="+", required=True)
    ap.add_argument("--outdir", type=Path,
                    default=Path("/root/autodl-tmp/experiments/rep_robust"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda")
    log(f"加载 {args.tag}: {args.weights_dir}")
    model = SamModel.from_pretrained(args.weights_dir).to(device).eval()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for lp in args.lists:
        df = pd.read_csv(args.outdir / lp)
        log(f"{lp}: {len(df)} 张")
        emb = embed(model, df["disk_path"].tolist(), device, args.batch_size)
        np.save(args.outdir / f"emb_{args.tag}_{lp.stem}.npy", emb)
        log(f"  -> emb_{args.tag}_{lp.stem}.npy {emb.shape}")
    log("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
