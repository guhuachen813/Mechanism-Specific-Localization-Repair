"""思路A 第5步：Atelectasis 定位地板归因——CAM 读出的锅还是任务本身？

设计：2×2 归因（同 GT 框 IoU 协议、同 best-IoU 指标）
----------------------------------------------------
                | CAM 读出（分类驱动） | 框监督线性头（1×1 conv + sigmoid）
  --------------+---------------------+----------------------------------
  DenseNet 特征 | 已有（第2/4c步）     | 本脚本
  MedSAM 特征   | —                    | 本脚本

* 框监督线性头：冻结特征 (C,h,w) → 1×1 conv → sigmoid，BCE 对齐框填充
  mask（64×64 或 7×7 网格）。5 折 CV（图像级随机切分，镜像无患者 ID，
  泄漏风险如实记录），报告折间波动。
* 归因逻辑：
  - MedSAM 探针 ≫ DenseNet CAM（同为 Atelectasis）→ 地板在 DenseNet 的
    表征/读出，不在任务；
  - MedSAM 探针 ≈ DenseNet CAM ≈ 0.1 → 任务地板（Atelectasis 外观本身
    难定位 / 框 GT 模糊）；
  - DenseNet 探针 vs DenseNet CAM 的差 = 读出方式（框监督 vs 分类驱动）的
    贡献；MedSAM 探针 vs DenseNet 探针的差 = 特征的贡献。
* Cardiomegaly 作对照（探针方法学的天花板校准）。

用法（GPU07）：
    /root/miniconda3/bin/python probe_localizer.py --stage feats   # 抽特征（两病种两骨干）
    /root/miniconda3/bin/python probe_localizer.py --stage train   # 5 折 CV + 评估

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from cam_benchmark import (BBOX_CSV, IMAGE_SIZE, MEAN, OUTDIR, SEED, SHARD_SIZE,
                           STD, batch_cam, build_index, cam_iou_curve,
                           gt_boxes_224, load_model, load_shard_images)
from cam_prior import medsam_prep
import degradations as dg

S64 = 64
NFOLD = 5
EPOCHS = 60
BS = 16
LR = 1e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def box_mask_grid(boxes: list[tuple[float, float, float, float]],
                  grid: int) -> np.ndarray:
    """GT 框（224 空间）→ grid×grid 填充 mask（float）。"""
    s = grid / IMAGE_SIZE
    m = np.zeros((grid, grid), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        a, b = int(max(0, np.floor(y1 * s))), int(min(grid, np.ceil(y2 * s)))
        c, d = int(max(0, np.floor(x1 * s))), int(min(grid, np.ceil(x2 * s)))
        m[a:max(b, a + 1), c:max(d, c + 1)] = 1.0
    return m


def load_finding_images(bbox: pd.DataFrame, finding: str,
                        limit: int = 0) -> tuple[list[str], dict[int, list[int]]]:
    b = bbox[bbox["Finding Label"] == finding]
    files = sorted(b["Image Index"].unique())
    if limit:
        files = files[:limit]
    lst, loc = build_index()
    by_shard: dict[int, list[int]] = {}
    for f in files:
        si, r = loc[f]
        by_shard.setdefault(si, []).append(r)
    return files, by_shard, lst, {f: loc[f] for f in files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["feats", "train"], required=True)
    ap.add_argument("--findings", nargs="*",
                    default=["Atelectasis", "Cardiomegaly"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    fdir = OUTDIR / "probe_loc"
    fdir.mkdir(exist_ok=True)
    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])

    # ---------------------------------------------------------- 特征抽取 --
    if args.stage == "feats":
        from transformers import SamModel
        smodel = SamModel.from_pretrained(OUTDIR / "weights" / "medsam").to(device).eval()
        dmodel = load_model(device)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        for finding in args.findings:
            files, by_shard, lst, loc = load_finding_images(bbox, finding, args.limit)
            log(f"{finding}: {len(files)} 图")
            feats_m, feats_d, meta = [], [], []
            for si, rlist in by_shard.items():
                imgs = load_shard_images(si, sorted(rlist))
                row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
                img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                          for r in rlist}
                for s0 in range(0, len(rlist), 8):
                    chunk = sorted(rlist)[s0:s0 + 8]
                    with torch.no_grad():
                        xm = torch.stack([medsam_prep(img224[r])[0] for r in chunk]).to(device)
                        fm = smodel.vision_encoder(pixel_values=xm)
                        fm = (fm.last_hidden_state if hasattr(fm, "last_hidden_state")
                              else fm).float().cpu().numpy()
                        xd = torch.stack([tf(img224[r].convert("RGB")) for r in chunk]).to(device)
                        fd = dmodel.features(xd).float().cpu().numpy()
                    for j, r in enumerate(chunk):
                        feats_m.append(fm[j]); feats_d.append(fd[j])
                        meta.append(row_of[r])
                log(f"  分卷 {si} 完成")
            np.save(fdir / f"feats_medsam_{finding}.npy", np.stack(feats_m))
            np.save(fdir / f"feats_densenet_{finding}.npy", np.stack(feats_d))
            pd.DataFrame({"nih_file": meta}).to_csv(
                fdir / f"meta_{finding}.csv", index=False)
            log(f"  -> feats_medsam_{finding}.npy {np.stack(feats_m).shape}")
        return 0

    # ---------------------------------------------------- 5 折 CV 训练 --
    rng_global = np.random.default_rng(SEED)
    all_rows = []
    for finding in args.findings:
        meta = pd.read_csv(fdir / f"meta_{finding}.csv")
        files = meta["nih_file"].tolist()
        f_m = np.load(fdir / f"feats_medsam_{finding}.npy")
        f_d = np.load(fdir / f"feats_densenet_{finding}.npy")
        boxes_of = {f: gt_boxes_224(bbox, f) for f in files}
        idx = rng_global.permutation(len(files))
        folds = np.array_split(idx, NFOLD)
        log(f"{finding}: {len(files)} 图，{NFOLD} 折")

        for backbone, F, grid in [("medsam", f_m, S64), ("densenet", f_d, 7)]:
            # 监督 mask 一次性构建
            sup = {k: box_mask_grid(boxes_of[files[k]], grid) for k in range(len(files))}
            C = F.shape[1]
            probe_preds = {}
            for k in range(NFOLD):
                te = folds[k]
                tr = np.concatenate([folds[j] for j in range(NFOLD) if j != k])
                head = nn.Conv2d(C, 1, 1).to(device)      # 输出 logits
                opt = torch.optim.Adam(head.parameters(), lr=LR)
                # 类不平衡：正样本占比低，pos_weight = 负/正
                pos = np.mean([sup[i].mean() for i in tr])
                pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=device)
                bce = nn.BCEWithLogitsLoss(pos_weight=pw)
                Ftr = torch.from_numpy(F[tr]).to(device)
                Ytr = torch.from_numpy(np.stack([sup[i] for i in tr])).to(device)
                n_tr = len(tr)
                for ep in range(EPOCHS):
                    perm = torch.randperm(n_tr, device=device)
                    tot = 0.0
                    for s0 in range(0, n_tr, BS):
                        sel = perm[s0:s0 + BS]
                        opt.zero_grad(set_to_none=True)
                        out = head(Ftr[sel]).squeeze(1)
                        loss = bce(out, Ytr[sel])
                        loss.backward(); opt.step()
                        tot += float(loss) * len(sel)
                head.eval()
                with torch.no_grad():
                    Fte = torch.from_numpy(F[te]).to(device)
                    pm = torch.sigmoid(head(Fte)).squeeze(1).cpu().numpy()
                for j, i in enumerate(te):
                    m = Image.fromarray((pm[j] * 255).astype(np.uint8))
                    m = np.asarray(m.resize((IMAGE_SIZE, IMAGE_SIZE),
                                            Image.BILINEAR)) / 255.0
                    probe_preds[i] = m
                log(f"  {finding}/{backbone} fold{k} train_loss末值 {tot / n_tr:.4f}")
            for i in range(len(files)):
                curve = cam_iou_curve(probe_preds[i], boxes_of[files[i]])
                all_rows.append({"finding": finding, "backbone": backbone,
                                 "nih_file": files[i],
                                 "best_iou": max(curve),
                                 "iou_curve": ";".join(f"{v:.4f}" for v in curve)})
            pd.DataFrame([r for r in all_rows
                          if r["finding"] == finding and r["backbone"] == backbone]
                         ).to_csv(fdir / f"probe_{finding}_{backbone}.csv", index=False)

    pd.DataFrame(all_rows).to_csv(fdir / "probe_all.csv", index=False)
    log(f"写出 {fdir / 'probe_all.csv'}")

    # ---------------------------------------------------------- 汇总 --
    df = pd.DataFrame(all_rows)
    for finding in args.findings:
        print(f"\n== {finding}（5 折 CV，held-out best-IoU）==")
        for bb in ["medsam", "densenet"]:
            d = df[(df["finding"] == finding) & (df["backbone"] == bb)]
            print(f"  probe_{bb:9s} {d['best_iou'].mean():.3f} (n={len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
