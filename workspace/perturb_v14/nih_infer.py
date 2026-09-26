"""NIH ChestX-ray14 外部验证：在 300px parquet 镜像上重跑与 CheXpert 完全相同的推理管线。

产出 CSV 的列与 `out_seed42_official/clean.csv` 一致，既有分析脚本可直接复用。

口径（与 v1.5 设计 A 对齐）
--------------------------
* 模型输入 224；QC 特征在**同一张模型输入图**（224）上计算——
  这是 v1.5 §3 的核心教训：质量测量分辨率必须等于模型输入分辨率。
* 受此影响，`quality_features.py` 的 `min_size=256` 守卫会让 `hard_fail`
  对每张图都触发（与设计 A 相同）。本实验的结论均不依赖该字段。
* NIH 不含 AP/PA 元数据，故 `projection_unknown` 恒为 1，
  使 `quality_risk` 带一个常量偏移 +1/9。分析以秩统计为主，
  另报告剔除两个投影项的敏感性版本。

本镜像的限制（必须写进报告）
--------------------------
`g-ronimo/NIH-Chest-X-ray-dataset_resized300px` 不保留文件名与患者 ID，
因此 Patient 只能取图像级唯一值 → **只能做图像级 Bootstrap，做不了患者级整群 Bootstrap**。
此外图像已被第三方从 1024 降到 300 px，本管线再降到 224。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/root/project/src")
from train_baseline import make_model  # noqa: E402

import quality_features as qf  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CARDIO_IDX = 2  # 镜像的 class_label 定义：0=No Finding, 1=Atelectasis, 2=Cardiomegaly


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path: Path, arch_hint: str, device: torch.device):
    """按仓库既有 checkpoint 格式加载：{'model','arch','num_classes','epoch','seed',...}。

    直接读 checkpoint 里的 arch / num_classes，避免与训练时的配置漂移。
    """
    ck = torch.load(path, map_location="cpu")
    for k in ("model", "state_dict"):
        if k in ck:
            ck = ck if k == "model" else {"model": ck[k]}
            break
    nc = int(ck.get("num_classes", 3))
    arch = ck.get("arch", arch_hint)
    model = make_model(nc, arch)
    model.load_state_dict(ck["model"])
    log(f"    载入 {arch}（{nc} 类，epoch {ck.get('epoch', '?')}，seed {ck.get('seed', '?')}）")
    return model.to(device).eval()


@torch.no_grad()
def forward_batch(model, tensors: torch.Tensor, device: torch.device) -> np.ndarray:
    return model(tensors.to(device, non_blocking=True)).float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--model1", type=Path, required=True)
    ap.add_argument("--model2", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备 {device}")

    m1 = load_model(args.model1, "densenet121", device)
    m2 = load_model(args.model2, "resnet50", device)

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])

    shards = sorted(args.parquet_dir.glob("*.parquet"))
    log(f"找到 {len(shards)} 个分卷")

    out_rows: list[dict] = []
    processed = 0
    t0 = time.time()

    for si, shard in enumerate(shards):
        tab = pq.read_table(shard)
        images = tab.column("image").to_pylist()
        labels = tab.column("labels").to_pylist()
        n = len(images)
        log(f"--- 分卷 {si + 1}/{len(shards)}：{n} 行 ---")

        for s in range(0, n, args.batch_size):
            if args.limit and processed >= args.limit:
                break
            sl = slice(s, min(s + args.batch_size, n))

            batch_pil: list[Image.Image] = []   # 224 灰度，模型与 QC 共用
            batch_tensors: list[torch.Tensor] = []
            batch_flip: list[torch.Tensor] = []

            for k in range(sl.start, sl.stop):
                blob = images[k]["bytes"]
                img = Image.open(io.BytesIO(blob)).convert("L")
                img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
                batch_pil.append(img)
                rgb = img.convert("RGB")
                batch_tensors.append(tf(rgb))
                batch_flip.append(tf(rgb.transpose(Image.FLIP_LEFT_RIGHT)))

            t1 = torch.stack(batch_tensors)
            t2 = torch.stack(batch_flip)
            l1 = forward_batch(m1, t1, device)
            l2 = forward_batch(m2, t1, device)
            l1f = forward_batch(m1, t2, device)

            # QC 在模型输入分辨率上逐张计算
            row_stub = pd.Series({"Frontal/Lateral": "Frontal", "AP/PA": ""})
            for j, k in enumerate(range(sl.start, sl.stop)):
                buf = io.BytesIO()
                batch_pil[j].save(buf, format="PNG")
                buf.seek(0)
                q = qf.features(buf, row_stub)
                lab_list = [int(v) for v in labels[k]]
                rec = {
                    "Path": f"nih/{shard.stem}/{k:06d}.png",
                    "Patient": f"nih_{shard.stem}_{k:06d}",
                    "Study": f"nih_{shard.stem}_{k:06d}",
                    "label": int(CARDIO_IDX in lab_list),
                    "idx": k,
                }
                for c in range(3):
                    rec[f"l1_{c}"] = float(l1[j, c])
                    rec[f"l2_{c}"] = float(l2[j, c])
                    rec[f"l1f_{c}"] = float(l1f[j, c])
                for key, val in q.items():
                    rec[f"qc_{key}"] = val
                # 附加：原始标签向量与「无投影项」质量风险（敏感性分析用）
                rec["nih_label_ids"] = ";".join(str(v) for v in lab_list)
                risk_terms = [
                    float(q["dynamic_range"] < 80 / 255),
                    float(q["foreground_ratio"] < 0.06 or q["foreground_ratio"] > 0.99),
                    float(q["contrast"] < 0.12),
                    float(q["blur_score"] < 0.0008),
                    float(q["noise_score"] > 0.08),
                    float(q["border_crop_score"] > 0.12),
                    float(q["left_right_symmetry"] > 0.20),
                ]
                rec["qc_quality_risk_noproj"] = float(np.mean(risk_terms))
                out_rows.append(rec)

            processed += len(batch_pil)
            if s % (args.batch_size * 20) == 0:
                el = time.time() - t0
                log(f"    已处理 {processed} 张（{processed / max(el, 1e-9):.1f} 张/秒）")

        if args.limit and processed >= args.limit:
            break

    df = pd.DataFrame(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log(f"完成：{len(df)} 行 → {args.output}，总耗时 {time.time() - t0:.0f}s")
    log(f"Cardiomegaly 阳性率 {df['label'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
