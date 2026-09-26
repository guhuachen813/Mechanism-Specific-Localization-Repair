"""思路A 第2步：NIH Cardiomegaly 框标注上的 CAM 定位鲁棒性基准（raw CAM 半边）。

研究问题：CAM 类定位图在真实拍摄降质下会怎么坏？本脚本先建立**基准的
baseline 半边**——在 NIH 框标注金标准上测 raw CAM（DenseNet-121 类激活图）
在 clean + 合成降质阶梯下的定位退化曲线。第3步再叠加解剖先验约束 CAM 对比。

数据身份恢复（与 perturb_v14/nih_attach_ids.py 同一依据）
--------------------------------------------------------
300px 镜像 parquet 不保留文件名，但已验证按文件名排序：第 g 行 =
`sorted(test_list.txt)[g]`（逐卷 15 类边际计数全部匹配）。本脚本内置
自检：清单长度、每卷行数、目标图全部可解析，任何一项不过即终止。

口径
----
* 设计 A：原图 → 224 → 降质（degradations.py，工作分辨率=模型输入分辨率）→ 模型。
* CAM：DenseNet 分类器是 Linear(1024,3) 作用于 GAP 特征，故
  CAM_1(x,y) = Σ_k w_{1,k} · A_k(x,y)（正类=1 的权重线性组合，ReLU 后归一化），
  与 Grad-CAM 等价且免反传。
* 定位指标：CAM 上采样到 224，阈值 τ ∈ [0.05,0.95] 步长 0.05 二值化 →
  外接框与 GT 框（1024 坐标 × 224/1024）算 IoU；每图每条件存整条 IoU-τ 曲线，
  MaxBoxAcc（max_τ mean IoU）、mean best-IoU、BoxAcc@{0.1,0.25,0.5} 离线聚合。
* 同一图的 (seed, 图像序号) 派生确定性 RNG，跨条件可精确配对。

用法（GPU07）：
    /root/miniconda3/bin/python cam_benchmark.py                # 全量 17 条件
    /root/miniconda3/bin/python cam_benchmark.py --limit 8      # 冒烟

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

import degradations as dg

OUTDIR = Path("/root/autodl-tmp/experiments/rep_robust")
PARQUET_DIR = Path("/root/autodl-tmp/nih_cxr14/parquet")
BBOX_CSV = OUTDIR / "BBox_List_2017.csv"
TEST_LIST = OUTDIR / "test_list.txt"
CKPT = Path("/root/project/outputs/repaired_seed42/densenet121/best.pt")
SHARD_SIZE = 6399
IMAGE_SIZE = 224
T1 = 0.4943421483039856          # 冻结温度：正类概率 = softmax(L/T)[1]
FINDING = "Cardiomegaly"
SEED = 0

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(device: torch.device) -> torch.nn.Module:
    import sys
    sys.path.insert(0, "/root/project/src")
    from train_baseline import make_model  # noqa: E402
    ck = torch.load(CKPT, map_location="cpu")
    model = make_model(int(ck["num_classes"]), ck["arch"])
    model.load_state_dict(ck["model"])
    assert int(ck["num_classes"]) == 3, "CAM 正类权重按 3 类（1=阳性）写定"
    log(f"载入 {ck['arch']}（{ck['num_classes']} 类，epoch {ck.get('epoch')}）")
    return model.to(device).eval()


def build_index() -> tuple[list[str], dict[str, tuple[int, int]]]:
    """返回 (排序清单, 文件名 -> (分卷号, 卷内行号))。"""
    lst = sorted(ln.strip() for ln in TEST_LIST.read_text().splitlines() if ln.strip())
    shards = sorted(PARQUET_DIR.glob("test-*.parquet"))
    assert len(lst) == len(shards) * SHARD_SIZE, f"清单 {len(lst)} != {len(shards)}×{SHARD_SIZE}"
    loc = {}
    for si, sh in enumerate(shards):
        n = pq.read_metadata(sh).num_rows
        assert n == SHARD_SIZE, f"{sh.name} 有 {n} 行，预期 {SHARD_SIZE}"
        for r in range(n):
            loc[lst[si * SHARD_SIZE + r]] = (si, r)
    log(f"身份索引就绪：{len(lst)} 张 / {len(shards)} 卷")
    return lst, loc


def load_shard_images(si: int, rows: list[int]) -> dict[int, Image.Image]:
    tab = pq.read_table(sorted(PARQUET_DIR.glob("test-*.parquet"))[si],
                        columns=["image"])
    images = tab.column("image").to_pylist()
    return {r: Image.open(io.BytesIO(images[r]["bytes"])).convert("L") for r in rows}


def gt_boxes_224(bbox: pd.DataFrame, fname: str) -> list[tuple[float, float, float, float]]:
    s = IMAGE_SIZE / 1024.0
    out = []
    for _, r in bbox[bbox["Image Index"] == fname].iterrows():
        x1, y1 = max(0.0, r["x"] * s), max(0.0, r["y"] * s)
        x2 = min(float(IMAGE_SIZE), (r["x"] + r["w"]) * s)
        y2 = min(float(IMAGE_SIZE), (r["y"] + r["h"]) * s)
        out.append((x1, y1, x2, y2))
    return out


def cam_iou_curve(cam: np.ndarray,
                  boxes: list[tuple[float, float, float, float]]) -> list[float]:
    """cam ∈ [0,1]（224×224），返回各阈值下的 max-over-boxes IoU（无前景点记 0）。"""
    ious = []
    for t in THRESHOLDS:
        m = cam >= t * cam.max()
        if not m.any():
            ious.append(0.0)
            continue
        ys, xs = np.where(m)
        pb = (xs.min(), ys.min(), xs.max() + 1.0, ys.max() + 1.0)
        best = 0.0
        for bx in boxes:
            ix1, iy1 = max(pb[0], bx[0]), max(pb[1], bx[1])
            ix2, iy2 = min(pb[2], bx[2]), min(pb[3], bx[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = ((pb[2] - pb[0]) * (pb[3] - pb[1])
                     + (bx[2] - bx[0]) * (bx[3] - bx[1]) - inter)
            best = max(best, inter / union if union > 0 else 0.0)
        ious.append(best)
    return ious


@torch.no_grad()
def batch_cam(model, x: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """返回 (N,224,224) CAM 与 (N,) 正类概率。"""
    feats = model.features(x.to(device))                  # (B,1024,7,7)
    w = model.classifier.weight[1]                        # 正类（1=Cardiomegaly）
    cam = torch.einsum("k,bkyx->byx", w, feats).relu_()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    cam = torch.nn.functional.interpolate(cam[:, None], size=IMAGE_SIZE,
                                          mode="bilinear", align_corners=False)[:, 0]
    logits = model.classifier(feats.mean(dim=(2, 3)))
    prob = torch.softmax(logits / T1, dim=1)[:, 1]
    return cam.cpu().numpy(), prob.cpu().numpy()


def qc_overlay(img: Image.Image, cam: np.ndarray,
               boxes: list[tuple[float, float, float, float]]) -> Image.Image:
    """降质图 + CAM 热力叠加 + GT 红框，220 宽单图。"""
    base = img.convert("RGB").resize((220, 220), Image.BILINEAR)
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((220, 220))
    pal = np.zeros((256, 3), dtype=np.uint8)
    pal[:, 0] = np.clip(np.linspace(-1, 2, 256) * 255, 0, 255)      # 简化 jet
    pal[:, 1] = np.clip(np.linspace(-2, 1.5, 256) * 255, 0, 255)
    pal[:, 2] = np.clip(np.linspace(1.5, -2, 256) * 255, 0, 255)
    rgb = Image.fromarray(pal[np.asarray(heat)]).convert("RGB")
    ov = Image.blend(base, rgb, 0.45)
    d = ImageDraw.Draw(ov)
    s = 220 / IMAGE_SIZE
    for bx in boxes:
        d.rectangle([bx[0] * s, bx[1] * s, bx[2] * s, bx[3] * s],
                    outline=(255, 30, 30), width=2)
    return ov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试）")
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    conds = args.conditions or dg.CONDITION_ORDER
    device = torch.device("cuda")
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # ---- 框标注 → 目标图 ----
    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    files = sorted(bbox["Image Index"].unique())
    if args.limit:
        files = files[: args.limit]
    log(f"{FINDING} 框图 {bbox['Image Index'].nunique()} 张 / "
        f"{len(bbox)} 框；本次跑 {len(files)} 张 × {len(conds)} 条件")

    lst, loc = build_index()
    missing = [f for f in files if f not in loc]
    if missing:
        log(f"!! {len(missing)} 张不在镜像中，例如 {missing[:3]}，终止")
        return 1

    # ---- 按分卷取图，逐条件批量前向 ----
    by_shard: dict[int, list[int]] = {}
    for f in files:
        si, r = loc[f]
        by_shard.setdefault(si, []).append(r)

    outdir = OUTDIR / "cam_bench"
    outdir.mkdir(exist_ok=True)
    rows, qcs = [], []
    for si, rlist in by_shard.items():
        log(f"--- 分卷 {si}：{len(rlist)} 张 ---")
        imgs = load_shard_images(si, sorted(rlist))
        row_of = {r: lst[si * SHARD_SIZE + r] for r in rlist}
        img224 = {r: imgs[r].resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                  for r in rlist}
        boxes_of = {r: gt_boxes_224(bbox, row_of[r]) for r in rlist}

        for cond in conds:
            tensors, cams, probs = [], {}, {}
            for s in range(0, len(rlist), 64):
                chunk = sorted(rlist)[s:s + 64]
                x = torch.stack([
                    tf(dg.degrade(img224[r], cond, dg.rng_for(SEED, r)).convert("RGB"))
                    for r in chunk])
                cam, pr = batch_cam(model, x, device)
                for j, r in enumerate(chunk):
                    cams[r] = cam[j]
                    probs[r] = float(pr[j])
            for r in sorted(rlist):
                curve = cam_iou_curve(cams[r], boxes_of[r])
                rows.append({"nih_file": row_of[r], "condition": cond,
                             "prob": probs[r],
                             "best_iou": max(curve),
                             "t_best": float(THRESHOLDS[int(np.argmax(curve))]),
                             "iou_curve": ";".join(f"{v:.4f}" for v in curve)})
                if cond in ("clean", "jpeg_q30", "noise_025", "combo_sev") \
                        and sum(1 for q in qcs if q[1] == cond) < 8:
                    qcs.append((row_of[r], cond, img224[r], cams[r], boxes_of[r]))
            log(f"  {cond} 完成（clean 类 best-IoU 均值 "
                f"{np.mean([r['best_iou'] for r in rows if r['condition'] == cond]):.3f}）")

    df = pd.DataFrame(rows)
    out = outdir / "cam_raw_nih.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")

    # ---- QC 拼图：8 图 × 4 条件 ----
    if qcs:
        seen, sel = set(), []
        for item in qcs:
            if (item[0], item[1]) not in seen:
                seen.add((item[0], item[1]))
                sel.append(item)
        sel = sel[:32]
        sheet = Image.new("RGB", (4 * 234 + 6, len(sel) * 224 + 10), "white")
        from PIL import ImageDraw as _D
        d0 = _D.Draw(sheet)
        for i, (fname, cond, img, cam, bx) in enumerate(sel):
            x0, y0 = (i % 4) * 234, i // 4 * 224
            sheet.paste(qc_overlay(img, cam, bx), (x0, y0))
            d0.text((x0 + 2, y0 + 2), f"{cond}", fill=(255, 255, 0))
        sheet.save(outdir / "qc_overlays.png")
        log(f"QC 拼图 -> {outdir / 'qc_overlays.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
