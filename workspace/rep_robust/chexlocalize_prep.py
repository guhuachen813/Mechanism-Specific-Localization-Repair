r"""CheXlocalize 银标签预处理：RLE 解码 → CheXpert-small 尺寸 → npz。

输入：/root/autodl-tmp/chexlocalize/gt_segmentations_val.json（187 图 × 10 病种 RLE）
输出：/root/autodl-tmp/experiments/rep_robust/chexlocalize/gt_masks_val.npz
      （key = patientXXXXX_studyY_viewZ_frontal/lateral, value = (390,390) float mask,
        按 finding 存多通道）+ 汇总 CSV

RLE 为 COCO 压缩串（LEB128 变体，pycocotools rleFrString 兼容实现），
按列主序展开；mask 高宽 = RLE size（CheXpert 原始分辨率，横向图）。
"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

SEG_JSON = "/root/autodl-tmp/chexlocalize/gt_segmentations_val.json"
OUTDIR = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
SMALL = 390


def rle_string_to_counts(s: str) -> list[int]:
    """pycocotools rleFrString：LEB128 + zigzag delta（i>2 加回 cnts[-2]）。"""
    cnts, i = [], 0
    while i < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_decode(rle: dict) -> np.ndarray:
    """COCO RLE → (h,w) uint8。列主序展开。"""
    h, w = rle["size"]
    cnts = (rle["counts"] if isinstance(rle["counts"], list)
            else rle_string_to_counts(rle["counts"]))
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in cnts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val = 1 - val
    assert pos == h * w, f"RLE 总和 {pos} != h*w {h*w}"
    return flat.reshape((w, h)).T


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    seg = json.load(open(SEG_JSON))
    findings = sorted(next(iter(seg.values())).keys())
    print("findings:", findings)

    keys, rows = [], []
    masks_out = {}
    for k, per in seg.items():
        chans = np.zeros((len(findings), SMALL, SMALL), np.float32)
        stats = {"key": k}
        for ci, f in enumerate(findings):
            rle = per.get(f)
            if not (isinstance(rle, dict) and rle.get("counts")):
                continue
            m = rle_decode(rle)
            if m.sum() == 0:
                continue
            im = Image.fromarray(m * 255).resize((SMALL, SMALL), Image.BILINEAR)
            msmall = (np.asarray(im, np.float32) / 255.0 > 0.5).astype(np.float32)
            chans[ci] = msmall
            stats[f"area_{f}"] = float(msmall.mean())
            ys, xs = np.where(msmall > 0)
            stats[f"bbox_{f}"] = ";".join(map(str, [xs.min(), ys.min(), xs.max(), ys.max()]))
        masks_out[k] = chans
        rows.append(stats)
    df = pd.DataFrame(rows)
    np.savez_compressed(f"{OUTDIR}/gt_masks_val.npz",
                        findings=np.array(findings), **masks_out)
    df.to_csv(f"{OUTDIR}/mask_stats_val.csv", index=False)
    print(f"-> {OUTDIR}/gt_masks_val.npz（{len(masks_out)} 图 × {len(findings)} 病种）")

    # 汇总：各病种非空 mask 数与面积分布
    print("\n各病种非空 mask 统计（390px，面积占比）:")
    for f in findings:
        col = df.get(f"area_{f}")
        if col is not None and col.notna().any():
            v = col.dropna()
            print(f"  {f}: n={len(v)}  area mean {v.mean():.3f} "
                  f"median {v.median():.3f}  [{v.min():.3f},{v.max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
