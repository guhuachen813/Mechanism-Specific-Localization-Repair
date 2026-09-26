r"""5.1b：Wang et al. 2024 (Sci Rep 14:29181) CAM 细化基线——迁移复现版。

范围声明（写进论文）：原方法 = VMamba 主干 + NLM + FF + FGControl loss 全训练管线，
在 NIH 上 T(0.1) 达 0.45。本复现**迁移到本协议**：DenseNet-121（已冻结分类器）
之上只实现推理期可用的核心——
  (a) NLM（non-linear modulation）：Mo = 多病种 CAM 均值 → 高斯/T 分支调制
      （G: exp(-(x-μ)²/2σ²) 峰在中等激活处；T: 单调递减高值补丁），加权和扩展激活区
  (b) FF（FPM fusion）：M_fg = μ_fg·M + μ_bg·(1-M)，μ_fg=1, μ_bg=-1（原值）后过 sigmoid
  (c) FGControl 是训练期 loss，推理期基线 = (a)+(b) 后处理
  训练期部分（θ1-θ4, R=0.5）不迁移——分类器冻结协议下无法重训 VMamba，
  这是协议差异，论文标明"推理期核心复现"。

三变体（消融归因）：
  wang_nlm   : NLM 加权和（λ1=λ2=λ3=1，同原论文）
  wang_ff    : NLM + FF 融合
  wang_multiclass : Mo 用全部 10 类 CAM 均值（原论文 FPM 定义）vs 单类（附注）
评估：65 张 photo 银标签 τ45/best + 413 test clean 侧同款对照。
用法：/root/miniconda3/bin/python p5b_wang_baseline.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from cam_benchmark import MEAN, STD, THRESHOLDS, batch_cam, load_model
from silver_eval_test import (OUT, SEG_TEST, TEST_DIR, rle_decode)
from silver_register import PHOTO

FINDING = "Cardiomegaly"
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def multi_class_cam(model, x, device):
    """全部类 CAM 均值（原论文 Mo 定义：CAMs of all categories）。
    DenseNet 3 类输出，取全部类。"""
    x = x.to(device)
    feats = model._modules.get("features")(x)
    feats = torch.nn.functional.relu(feats, inplace=False)
    fc = model._modules.get("classifier")
    w_fc = fc.weight  # (num_classes, c)
    cams = torch.einsum("chw,kc->khw", feats[0], w_fc)
    return cams.mean(0)[None], None  # (1,h,w) Mo


def wang_nlm(cam_single: np.ndarray, cam_all_mean: np.ndarray) -> np.ndarray:
    """NLM 推理期近似（乘性重加权口径）。

    原论文：Mo 经 G/T 非线性后各自乘特征图再过独立 head 生成新 FPM；分类器冻结
    协议下无法重训 head，故近似为：G/T 分支输出作为调制权重与 CAM 相乘后归一。
      G(x)=exp(-(x-μ)²/2σ²)：峰在激活均值处 → 过渡区被增强、高/低激活区被抑制
      T(x)=1/(1+exp(3(x-μ)))：单调递减 → 低激活区保持、碎片区被补充
    M = (λ1·CAM + λ2·CAM⊙G + λ3·CAM⊙T)/(λ1+λ2+λ3)，λ 同原论文 =1。"""
    x = cam_single.astype(np.float64)
    mu, sd = x.mean(), x.std() + 1e-8
    G = np.exp(-((x - mu) ** 2) / (2 * sd ** 2))
    T = 1.0 / (1.0 + np.exp(3.0 * (x - mu)))
    base = cam_all_mean.astype(np.float64)
    M = (1.0 * base + 1.0 * base * G + 1.0 * base * T) / 3.0
    M = M / max(M.max(), 1e-8)
    return M.astype(np.float32)


def wang_ff(M: np.ndarray) -> np.ndarray:
    """FF: M_fg = μ_fg·M + μ_bg·M_bg，μ_fg=1, μ_bg=-1，M_bg=1-M → 2M-1 后过 sigmoid 保 [0,1]"""
    fg = 1.0 * M - 1.0 * (1.0 - M)
    # 原论文直接取该值（可为负）作 FPM；本协议评估需 [0,1]：sigmoid
    return 1.0 / (1.0 + np.exp(-6.0 * fg))  # 陡度 6 保持对比放大


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def eval_photo(device, model, tf) -> pd.DataFrame:
    z = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = list(z["findings"])
    ci = findings.index(FINDING)
    keys = sorted(k for k in z.files if k != "findings" and z[k][ci].sum() > 0)
    rows = []
    for n, k in enumerate(keys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        try:
            photo224 = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
                                  ).convert("L").resize((224, 224), Image.BILINEAR)
            gt = mask_bbox(z[k][ci].astype(np.uint8))
            with torch.no_grad():
                cam, prob = batch_cam(model, tf(photo224.convert("RGB"))[None], device)
            cam = cam[0]
            with torch.no_grad():
                mo = multi_class_cam(model, tf(photo224.convert("RGB"))[None], device)[0][0].cpu().numpy()
            mo = np.array(Image.fromarray((mo / max(mo.max(), 1e-8) * 255).astype(np.uint8)
                                          ).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
            nlm = wang_nlm(cam, mo)
            nlm2 = wang_nlm(cam, cam)  # Mo=单类 CAM 变体（更贴近本协议）
            ff = wang_ff(nlm2)
            ff2 = wang_ff(nlm)
            rec = {"key": k, "prob": float(prob[0])}
            for v, c in [("raw", cam), ("wang_nlm", nlm), ("wang_ff", ff),
                         ("wang_nlm_s", nlm2), ("wang_ff_s", ff2)]:
                curve = np.array([0.0] * 19)
                from cam_benchmark import cam_iou_curve
                curve = np.array(cam_iou_curve(c, gt))
                rec[f"{v}_45"] = float(curve[TAU45])
                rec[f"{v}_best"] = float(curve.max())
            rows.append(rec)
        except Exception as e:  # noqa: BLE001
            log(f"err {k}: {e}")
        if (n + 1) % 20 == 0:
            log(f"  photo {n + 1}/{len(keys)}")
    return pd.DataFrame(rows)


def main() -> int:
    device = torch.device("cuda")
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dfp = eval_photo(device, model, tf)
    dfp.to_csv(f"{OUT}/p5b_wang_photo.csv", index=False)
    print("\n== Wang2024 推理期复现 vs raw（photo 银标签 65） ==")
    for v in ["raw", "wang_nlm", "wang_ff", "wang_nlm_s", "wang_ff_s"]:
        print(f"  {v:10s} τ45 {dfp[f'{v}_45'].mean():.3f}  best {dfp[f'{v}_best'].mean():.3f}")
    rng = np.random.default_rng(42)
    uniq = dfp.key.unique()
    fmap = dfp.key.map({f: i for i, f in enumerate(uniq)}).to_numpy()
    for v in ["wang_nlm", "wang_ff", "wang_nlm_s", "wang_ff_s"]:
        d = dfp[f"{v}_45"] - dfp.raw_45
        vals = [d[np.isin(fmap, rng.choice(len(uniq), len(uniq)))].mean() for _ in range(400)]
        print(f"  {v:10s} vs raw: {d.mean():+.3f} CI [{np.percentile(vals,2.5):+.3f},{np.percentile(vals,97.5):+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
