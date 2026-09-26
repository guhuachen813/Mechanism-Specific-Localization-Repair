"""MedSAM 原生预处理对照（思路C 局限补丁）。

官方 MedSAM 预处理（与 SAM 原生的差别）：
1. 直接 resize 到 1024×1024（纵横比不保留，squash），不做补零；
2. 逐图 min-max 归一化到 [0,255]（不是固定窗宽窗位）；
3. /255 后减 SAM pixel_mean、除 pixel_std。

本脚本用该预处理重抽 train+eval 嵌入并重训探针（tag=medsam_native），
回答：思路C 中 MedSAM 在 real 条件的崩溃，有多少来自预处理失配。

用法（GPU06）：
    /root/miniconda3/bin/python embed_native.py --lists train_list.csv eval_list.csv
    /root/miniconda3/bin/python train_probe.py --tag medsam_native

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

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIZE = 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preprocess_batch(paths: list[str]) -> torch.Tensor:
    out = torch.zeros(len(paths), 3, SIZE, SIZE)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR),
                           dtype=np.float32)
        lo, hi = g.min(), max(g.max(), g.min() + 1e-8)
        g = (g - lo) / (hi - lo) * 255.0
        a = torch.from_numpy(g)[None].repeat(3, 1, 1) / 255.0
        out[i] = a
    return (out - MEAN) / STD


@torch.no_grad()
def embed(model, paths: list[str], device: torch.device, bs: int) -> np.ndarray:
    feats = []
    for s in range(0, len(paths), bs):
        x = preprocess_batch(paths[s:s + bs]).to(device)
        out = model.vision_encoder(pixel_values=x)
        f = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        feats.append(f.float().mean(dim=(2, 3)).cpu().numpy())
        if (s // bs) % 20 == 0:
            log(f"  {min(s + bs, len(paths))}/{len(paths)}")
    return np.concatenate(feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", type=Path,
                    default=Path("/root/autodl-tmp/experiments/rep_robust/weights/medsam"))
    ap.add_argument("--lists", type=Path, nargs="+",
                    default=[Path("train_list.csv"), Path("eval_list.csv")])
    ap.add_argument("--outdir", type=Path,
                    default=Path("/root/autodl-tmp/experiments/rep_robust"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda")
    log("加载 medsam（原生预处理对照）")
    model = SamModel.from_pretrained(args.weights_dir).to(device).eval()
    for lp in args.lists:
        df = pd.read_csv(args.outdir / lp)
        log(f"{lp}: {len(df)} 张")
        emb = embed(model, df["disk_path"].tolist(), device, args.batch_size)
        np.save(args.outdir / f"emb_medsam_native_{lp.stem}.npy", emb)
        log(f"  -> emb_medsam_native_{lp.stem}.npy {emb.shape}")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
