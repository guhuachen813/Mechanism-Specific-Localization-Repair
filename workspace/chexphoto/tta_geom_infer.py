"""方向二（实验 A）：退化感知 TTA —— 几何变换集合下路由信号能否救回来。

问题
----
CheXphoto 真实翻拍（natural/oneplus）条件下，模型的分数分布被整体推到决策阈值
左侧（0/202 判阳，见外部验证报告 §3.1）。与此同时，此前唯一的零标注自洽性信号
——水平翻转 TTA 分歧 |p − p_flip|——对软错误的排序能力从干净域的 0.899
掉到 0.682（同报告 §3.4，表 5）。

一个待检验的假设：**水平翻转对手机翻拍而言不变性太弱**。真实翻拍引入透视畸变、
屏幕莫尔纹与几何形变，这些在左右镜像下并不保持；小幅旋转/缩放/平移更可能
保持语义不变，从而暴露模型真正的不稳定所在。若假设成立，用几何变换集合替代
单次翻转应当把 0.682 推回去。

本脚本只做推理、不做分析：对每张图输出 16 个视图各自的 3 类 logits。
分析脚本可自由组合任意视图子集（视图预算 K 扫描）与任意分歧定义。

视图集合（16）
-------------
v00 原图                        v01 水平翻转
v02 / v03 旋转 ±5°              v04 / v05 旋转 ±10°
v12 / v13 旋转 ±15°
v06 / v07 缩放 0.90 / 1.10      v14 / v15 缩放 0.80 / 1.25
v08 / v09 平移 x ±4%            v10 / v11 平移 y ±4%

口径
----
* 主模型 densenet121（seed42）；冻结温度 T1 = 0.4943421483039856 供分析阶段使用
* 变换施加在 224 灰度图上，与既有设计 A 管线一致（QC 与模型看同一张 224 图）
* 双线性重采样，填充值 0（黑），与放射图像的背景一致
* v00 与既有 preds.csv 的 l1_* 应为同一张图，可用于哨兵校验

用法（gpu05）
------------
    /root/miniconda3/bin/python tta_geom_infer.py \
        --manifest /root/chexphoto/manifest.csv \
        --model /root/project/outputs/repaired_seed42/densenet121/best.pt \
        --output /root/chexphoto/out/tta_geom.csv

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/root/project/src")
from train_baseline import make_model  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# (视图名, 变换参数)。angle 单位度；scale > 1 为放大；tx/ty 单位像素（224 尺度下 4% ≈ 9 px）
VIEWS: list[tuple[str, dict]] = [
    ("v00", {}),
    ("v01", {"flip": True}),
    ("v02", {"angle": 5.0}),
    ("v03", {"angle": -5.0}),
    ("v04", {"angle": 10.0}),
    ("v05", {"angle": -10.0}),
    ("v06", {"scale": 0.90}),
    ("v07", {"scale": 1.10}),
    ("v08", {"tx": 9.0}),
    ("v09", {"tx": -9.0}),
    ("v10", {"ty": 9.0}),
    ("v11", {"ty": -9.0}),
    ("v12", {"angle": 15.0}),
    ("v13", {"angle": -15.0}),
    ("v14", {"scale": 0.80}),
    ("v15", {"scale": 1.25}),
]

# 视图预算 K 的累积前缀，与分析脚本共用（此处仅作说明与自检打印）
K_PREFIX = [1, 2, 4, 6, 8, 12, 16]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def geom_view(img: Image.Image, angle: float = 0.0, scale: float = 1.0,
              tx: float = 0.0, ty: float = 0.0) -> Image.Image:
    """绕图像中心旋转 + 缩放的仿射视图，越界填 0。

    PIL 的 AFFINE 矩阵把**目标**像素坐标映射到**源**坐标：

        x_src = m0 * x + m1 * y + m2
        y_src = m3 * x + m4 * y + m5

    scale > 1 表示放大（目标取源上更小的区域）。
    """
    if angle == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
        return img
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    a = math.radians(angle)
    cos_a, sin_a = math.cos(a), math.sin(a)
    s = 1.0 / scale
    m0, m1 = s * cos_a, s * sin_a
    m3, m4 = -s * sin_a, s * cos_a
    m2 = cx - m0 * cx - m1 * cy - tx
    m5 = cy - m3 * cx - m4 * cy - ty
    return img.transform((w, h), Image.AFFINE, (m0, m1, m2, m3, m4, m5),
                         resample=Image.BILINEAR, fillcolor=0)


def load_model(path: Path, arch_hint: str, device: torch.device):
    ck = torch.load(path, map_location="cpu")
    for k in ("model", "state_dict"):
        if k in ck:
            ck = ck if k == "model" else {"model": ck[k]}
            break
    nc = int(ck.get("num_classes", 3))
    arch = ck.get("arch", arch_hint)
    model = make_model(nc, arch)
    model.load_state_dict(ck["model"])
    log(f"    载入 {arch}（{nc} 类，epoch {ck.get('epoch','?')}，seed {ck.get('seed','?')}）")
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--orient", default="frontal", help="只跑该体位；留空则全跑")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备 {device}")
    model = load_model(args.model, "densenet121", device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    man = pd.read_csv(args.manifest)
    if args.orient:
        man = man[man["orient"] == args.orient]
    if args.limit:
        man = man.head(args.limit)
    man = man.reset_index(drop=True)
    log(f"清单 {len(man)} 行（orient={args.orient or '全部'}），"
        f"{len(VIEWS)} 个视图，共 {len(man) * len(VIEWS)} 次前向")

    # ---- 先把 224 灰度图读进内存（约 40 MB / 808 张），避免每个视图重复解码
    t0 = time.time()
    greys: list[Image.Image] = []
    for i, r in enumerate(man.to_dict("records")):
        with Image.open(r["disk_path"]) as im:
            grey = im.convert("L").resize((args.image_size, args.image_size),
                                          Image.BILINEAR)
        greys.append(grey)
        if (i + 1) % 200 == 0:
            log(f"    解码 {i + 1}/{len(man)}")
    log(f"解码完成 {len(greys)} 张，耗时 {time.time() - t0:.0f}s")

    out = {f"{n}_{c}": np.zeros(len(greys), dtype=np.float32)
           for n, _ in VIEWS for c in range(3)}

    with torch.no_grad():
        for name, spec in VIEWS:
            t1 = time.time()
            for s in range(0, len(greys), args.batch_size):
                chunk = greys[s:s + args.batch_size]
                tens = []
                for g in chunk:
                    if spec.get("flip"):
                        v = g.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        v = geom_view(g, spec.get("angle", 0.0),
                                      spec.get("scale", 1.0),
                                      spec.get("tx", 0.0), spec.get("ty", 0.0))
                    tens.append(tf(v.convert("RGB")))
                T = torch.stack(tens).to(device, non_blocking=True)
                L = model(T).float().cpu().numpy()
                for c in range(3):
                    out[f"{name}_{c}"][s:s + len(chunk)] = L[:, c]
            log(f"  {name} 完成（{time.time() - t1:.1f}s）")

    df = pd.DataFrame({
        "disk_path": man["disk_path"],
        "base_key": man["base_key"],
        "cond": man["cond"],
        "orient": man["orient"],
        "Patient": man["Patient"],
        "Study": man["Study"],
        "View": man["View"],
        "label": man["label"].astype(int),
    })
    for k, v in out.items():
        df[k] = v
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log(f"完成：{len(df)} 行 × {len(VIEWS)} 视图 -> {args.output}，"
        f"总耗时 {time.time() - t0:.0f}s")
    log(f"  视图预算 K 的累积前缀：{K_PREFIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
