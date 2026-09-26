"""cheXphoto 式合成降质算子。

约定（与 ImageNet-C 一致）
------------------------
降质施加在一个固定的**工作分辨率** W 上：先 resize 到 (W, W)，再降质。
下游两条路径共用同一张降质图：
  · 模型输入：降质图再 resize 到模型尺寸 (S, S)，再 ToTensor/Normalize；
  · QC 特征：直接在降质图上算，W 取 512 以满足 quality_features 的 min_size=256 门。
ImageNet-C 采用同一约定（resize 256 → crop 224 → corruption）。
若把降质施加在原始全分辨率上，高斯噪声会在后续降采样中按面积比被平均掉，
构不成有效的严重度阶梯。

确定性
------
所有算子对同一 (seed, image_index) 产生完全相同的像素，因此：
  1. 「模型输入」与「QC 特征」两条路径看到的是同一批像素，不引入额外噪声；
  2. 整组实验可精确复现。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------------------------------
# 基础算子：输入/输出均为 PIL "L"（灰度）图像
# --------------------------------------------------------------------------

def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("L")


def _blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(float(sigma)))


def _noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a + rng.normal(0.0, float(sigma), size=a.shape).astype(np.float32)
    return Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def _brightness(img: Image.Image, factor: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    return Image.fromarray((np.clip(a * float(factor), 0.0, 1.0) * 255.0).round().astype(np.uint8))


def _contrast(img: Image.Image, factor: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    return Image.fromarray((np.clip((a - 0.5) * float(factor) + 0.5, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def _downsample(img: Image.Image, factor: int) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, w // int(factor)), max(1, h // int(factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


_OPS = {
    "jpeg": lambda img, p, rng: _jpeg(img, p),
    "blur": lambda img, p, rng: _blur(img, p),
    "noise": lambda img, p, rng: _noise(img, p, rng),
    "brightness": lambda img, p, rng: _brightness(img, p),
    "contrast": lambda img, p, rng: _contrast(img, p),
    "downsample": lambda img, p, rng: _downsample(img, p),
}


# --------------------------------------------------------------------------
# 条件表：名字 -> [(算子, 参数), ...]，按顺序施加
# --------------------------------------------------------------------------

CONDITIONS: dict[str, list[tuple[str, float]]] = {
    "clean": [],
    # 采集/存储压缩
    "jpeg_q50": [("jpeg", 50)],
    "jpeg_q30": [("jpeg", 30)],
    "jpeg_q10": [("jpeg", 10)],
    # 运动/失焦模糊
    "blur_1": [("blur", 1.0)],
    "blur_3": [("blur", 3.0)],
    # 量子噪声 / 电子噪声（0.03/0.06 为临床合理档，0.10/0.25 为压力档）
    "noise_003": [("noise", 0.03)],
    "noise_006": [("noise", 0.06)],
    "noise_010": [("noise", 0.10)],
    "noise_025": [("noise", 0.25)],
    # 曝光
    "dark_05": [("brightness", 0.5)],
    "bright_16": [("brightness", 1.6)],
    # 对比度
    "contrast_04": [("contrast", 0.4)],
    # 低分辨率探测器
    "ds_2": [("downsample", 2)],
    "ds_4": [("downsample", 4)],
    # 复合
    "combo_mild": [("jpeg", 50), ("blur", 1.0), ("noise", 0.05)],
    "combo_sev": [("jpeg", 20), ("blur", 3.0), ("noise", 0.20), ("contrast", 0.5), ("downsample", 2)],
}

CONDITION_ORDER = list(CONDITIONS.keys())


def degrade(img: Image.Image, condition: str, rng: np.random.Generator) -> Image.Image:
    """对灰度 PIL 图像施加指定条件的降质，返回新图像。"""
    out = img.convert("L")
    for op_name, param in CONDITIONS[condition]:
        out = _OPS[op_name](out, param, rng)
    return out


def rng_for(seed: int, index: int) -> np.random.Generator:
    """由 (seed, 图像序号) 派生确定性 RNG。"""
    return np.random.default_rng((int(seed) * 1_000_003 + int(index)) % (2**32))
