"""在合成降质图像上重跑两模型推理，保存 logits 与降质后的 QC 特征。

为什么保存 logits 而不是概率
----------------------------
温度缩放是 logits 上的单调变换，任何 T 都可以事后施加。
保存原始 logits 后，温度、先验偏移、ECE、AUROC 全部可在离线分析里任意重算，
不需要为每个分析口径重跑 GPU。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, "/root/project/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_baseline import make_model, resolve_image  # noqa: E402
import quality_features as qf  # noqa: E402
from degradations import CONDITIONS, degrade, rng_for  # noqa: E402

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class DegradedDataset(Dataset):
    """在模型输入分辨率上施加降质。

    预加载的已经 resize 到 (size,size) 的灰度 PIL 图像，逐个条件降质。
    """

    def __init__(self, images, condition, seed, model_size, flip=False):
        self.images = images
        self.condition = condition
        self.seed = seed
        self.model_size = model_size
        self.flip = flip
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # 降质在工作分辨率上完成；模型侧再 resize，与 QC 侧共享同一张降质图
        img = degrade(self.images[i], self.condition, rng_for(self.seed, i))
        if img.size != (self.model_size, self.model_size):
            img = img.resize((self.model_size, self.model_size), Image.BILINEAR)
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(img.convert("RGB")), i


@torch.no_grad()
def forward_logits(model, loader, device):
    model.eval()
    out = None
    for images, idx in loader:
        logits = model(images.to(device, non_blocking=True)).float().cpu().numpy()
        if out is None:
            out = np.zeros((len(loader.dataset), logits.shape[1]), dtype=np.float32)
        out[idx.numpy()] = logits
    return out


def preload(frame, data_root, size):
    """解码并 resize 一次，之后所有条件共用，避免 14 次重复解码。"""
    images, keep = [], []
    for i, (_, r) in enumerate(frame.iterrows()):
        try:
            img = Image.open(resolve_image(r, data_root)).convert("RGB")
        except Exception as e:  # 坏图跳过，保持与 pipeline 一致的可审计性
            print(f"  [warn] 跳过 {r.get('Path')}: {e}", flush=True)
            continue
        img = img.resize((size, size), Image.BILINEAR).convert("L")
        images.append(img)
        keep.append(i)
    return images, frame.iloc[keep].reset_index(drop=True)


def qc_row(img: Image.Image, row: pd.Series) -> dict:
    """复用仓库原版 quality_features.features，避免口径漂移。

    通过 BytesIO 传 PNG 让原函数走同一条 Image.open 路径；
    PNG 无损，不引入额外降质。
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return qf.features(buf, row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model1", type=Path, required=True)
    p.add_argument("--model2", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--split-column", default="agent_split")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--working-size", type=int, default=0,
                   help="降质工作分辨率；0 表示与 image-size 相同。QC 需要 >=256")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--conditions", default="all")
    p.add_argument("--tta", action="store_true", help="额外跑水平翻转，用于 TTA 一致性")
    p.add_argument("--qc", action="store_true", help="同时计算降质后的 QC 特征")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.read_csv(args.manifest)
    frame = frame[frame[args.split_column].eq(args.split)].copy()
    frame = frame[frame["Frontal/Lateral"].eq("Frontal")]
    frame["Cardiomegaly"] = pd.to_numeric(frame["Cardiomegaly"], errors="coerce")
    frame = frame[frame["Cardiomegaly"].isin([0, 1, 2])].reset_index(drop=True)

    if args.max_rows and len(frame) > args.max_rows:
        # 按标签分层抽样，保持患病率
        parts = []
        for lab, g in frame.groupby("Cardiomegaly"):
            k = max(1, int(round(args.max_rows * len(g) / len(frame))))
            parts.append(g.sample(n=min(k, len(g)), random_state=args.seed))
        frame = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"split={args.split} 候选 {len(frame)} 行  "
          f"标签分布 {dict(frame['Cardiomegaly'].value_counts().sort_index())}", flush=True)

    conditions = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")

    ck1 = torch.load(args.model1, map_location="cpu")
    ck2 = torch.load(args.model2, map_location="cpu")
    nc1, nc2 = int(ck1.get("num_classes", 3)), int(ck2.get("num_classes", 3))
    m1 = make_model(nc1, ck1.get("arch", "densenet121")).to(device)
    m2 = make_model(nc2, ck2.get("arch", "resnet50")).to(device)
    m1.load_state_dict(ck1["model"])
    m2.load_state_dict(ck2["model"])

    t0 = time.time()
    wsize = args.working_size or args.image_size
    images, frame = preload(frame, args.data_root, wsize)
    print(f"预加载 {len(images)} 张 → {wsize}px（模型输入 {args.image_size}px），"
          f"用时 {time.time()-t0:.1f}s", flush=True)

    base_cols = ["Path", "Patient", "Study", "label", "idx"]
    timing = {}
    for cond in conditions:
        out_csv = args.output_dir / f"{cond}.csv"
        if out_csv.exists():
            print(f"[skip] {cond} 已存在", flush=True)
            continue
        tc = time.time()

        ds = DegradedDataset(images, cond, args.seed, args.image_size, flip=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        l1 = forward_logits(m1, loader, device)
        l2 = forward_logits(m2, loader, device)

        data = {
            "Path": frame["Path"].to_numpy(),
            "Patient": frame["Patient"].to_numpy(),
            "Study": frame["Study"].to_numpy(),
            "label": frame["Cardiomegaly"].astype(int).to_numpy(),
            "idx": np.arange(len(frame)),
        }
        for j in range(nc1):
            data[f"l1_{j}"] = l1[:, j]
        for j in range(nc2):
            data[f"l2_{j}"] = l2[:, j]

        if args.tta:
            dsf = DegradedDataset(images, cond, args.seed, args.image_size, flip=True)
            lf = forward_logits(m1, DataLoader(dsf, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True), device)
            for j in range(nc1):
                data[f"l1f_{j}"] = lf[:, j]

        if args.qc:
            feats = []
            for i, img in enumerate(images):
                img_d = degrade(img, cond, rng_for(args.seed, i))
                feats.append(qc_row(img_d, frame.iloc[i]))
            qdf = pd.DataFrame(feats)
            for c in qdf.columns:
                data[f"qc_{c}"] = qdf[c].to_numpy()

        out = pd.DataFrame(data)
        out.to_csv(out_csv, index=False)
        dt = time.time() - tc
        timing[cond] = dt
        print(f"[done] {cond:12s} n={len(out)} 用时 {dt:.1f}s", flush=True)

    (args.output_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"总用时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
