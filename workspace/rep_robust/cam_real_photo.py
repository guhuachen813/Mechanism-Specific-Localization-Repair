"""思路A 第4步：CheXphoto 真实翻拍上的定位一致性（raw vs 先验约束 CAM）。

背景：CheXphoto 的翻拍对象是 CheXpert valid（无框 GT），定位精度指标用不了；
改用**配对一致性协议**：同一研究的 clean CAM 与降质条件 CAM 在固定阈值
τ=0.45 下的二值 IoU 与质心位移——衡量定位图在真实翻拍下「稳不稳」，
raw vs 先验约束对比；同时记录先验管线两层鲁棒性（场检测成功率、mask 统计）。

变体：raw · prior_oracle_hard（mask 从 clean 原图算）· prior_real_hard
（mask 从各条件图算；clean 条件下两者相同）。

一致性对：natural/oneplus（真实翻拍）、synthetic/photographic、synthetic/digital
三条件各 202 对；clean 作为自身基线（IoU=1）。

用法（GPU07）：
    /root/miniconda3/bin/python cam_real_photo.py            # 全量 202 研究
    /root/miniconda3/bin/python cam_real_photo.py --limit 6  # 冒烟

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scipy.ndimage as ndi
from PIL import Image
from torchvision import transforms

from cam_benchmark import IMAGE_SIZE, MEAN, OUTDIR, STD, T1, batch_cam, load_model
from cam_prior import detect_field, mask_224, medsam_prep, segment

EVAL_LIST = OUTDIR / "eval_list.csv"
CONDS = ["clean", "natural/oneplus", "synthetic/photographic", "synthetic/digital"]
TAU = 0.45
DIL_ITERS = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mask_stats(mask: np.ndarray) -> dict:
    frac = float(mask.mean())
    lab, n = ndi.label(mask)
    return {"frac": round(frac, 4), "components": int(n)}


@torch.no_grad()
def agreement(cam_a: np.ndarray, cam_b: np.ndarray, tau: float = TAU) -> dict:
    """两 CAM 在固定阈值的二值 IoU 与质心位移（px, 224 空间）。

    阈值敏感性：同时给 τ∈{0.30,0.45,0.60} 三档的 agree-IoU。
    """
    out = {}
    for t in (0.30, 0.45, 0.60):
        ma = cam_a >= t * cam_a.max()
        mb = cam_b >= t * cam_b.max()
        inter = float((ma & mb).sum())
        union = float((ma | mb).sum())
        out[f"agree_{int(t * 100)}"] = inter / union if union > 0 else 0.0
    ma, mb = cam_a >= tau * cam_a.max(), cam_b >= tau * cam_b.max()
    iou = out["agree_45"]
    def centroid(m):
        ys, xs = np.where(m)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan)
    ca, cb = centroid(ma), centroid(mb)
    shift = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    return {"agree_iou": iou, "centroid_shift": shift, **out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    dmodel = load_model(device)
    log("加载 MedSAM（真权重）")
    from transformers import SamModel
    smodel = SamModel.from_pretrained(OUTDIR / "weights" / "medsam").to(device).eval()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    man = pd.read_csv(EVAL_LIST)
    man = man[man["orient"] == "frontal"]
    if args.limit:
        keep = man["base_key"].drop_duplicates().head(args.limit)
        man = man[man["base_key"].isin(keep)]
    studies = man["base_key"].drop_duplicates().tolist()
    log(f"{len(studies)} 个研究 × {len(CONDS)} 条件")

    outdir = OUTDIR / "cam_real"
    outdir.mkdir(exist_ok=True)
    rows, qcs = [], []
    for k, base in enumerate(studies):
        sub = man[man["base_key"] == base].set_index("cond")
        imgs, cams, probs = {}, {}, {}
        tensors = []
        for c in CONDS:
            if c not in sub.index:
                continue
            with Image.open(sub.loc[c, "disk_path"]) as im:
                imgs[c] = im.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            tensors.append(tf(imgs[c].convert("RGB")))
        x = torch.stack(tensors)
        cam, pr = batch_cam(dmodel, x, device)
        j = 0
        for c in CONDS:
            if c not in sub.index:
                continue
            cams[c], probs[c] = cam[j], float(pr[j])
            j += 1

        # ---- 先验：oracle（clean 图）+ realistic（各条件图） ----
        moracle = None
        covc = {}
        for c in CONDS:
            if c not in imgs:
                continue
            f, ok = detect_field(imgs[c])
            m = mask_224(segment(smodel, medsam_prep(imgs[c]), f, device))
            covc[c] = {"field_ok": int(ok), **mask_stats(m)}
            if c == "clean":
                moracle = m
        if moracle is None:
            continue

        for c in CONDS:
            if c not in cams:
                continue
            variants = {"raw": np.ones_like(cams[c]),
                        "prior_oracle_hard": moracle,
                        "prior_real_hard": (moracle if c == "clean" else None)}
            if variants["prior_real_hard"] is None:
                variants["prior_real_hard"] = mask_224(
                    segment(smodel, medsam_prep(imgs[c]),
                            detect_field(imgs[c])[0], device))
            for vname, w in variants.items():
                rec = {"base_key": base, "condition": c, "variant": vname,
                       "prob": probs[c], **covc[c]}
                cw = cams[c] * w
                if c == "clean":
                    rec.update({"agree_iou": 1.0, "centroid_shift": 0.0})
                else:
                    rec.update(agreement(cw, cams["clean"] * variants[vname]))
                rows.append(rec)
                if c == "natural/oneplus" and vname == "raw" and len(qcs) < 8:
                    qcs.append((base, imgs[c], cw))
        if (k + 1) % 40 == 0:
            log(f"  {k + 1}/{len(studies)}")

    df = pd.DataFrame(rows)
    out = outdir / "cam_real_photo.csv"
    df.to_csv(out, index=False)
    log(f"写出 {out}（{len(df)} 行）")

    if qcs:
        from cam_benchmark import qc_overlay
        sheet = Image.new("RGB", (len(qcs) * 226, 224), "white")
        for i, (base, img, cw) in enumerate(qcs):
            sheet.paste(qc_overlay(img, cw, []), (i * 226, 0))
        sheet.save(outdir / "qc_real.png")
        log(f"QC -> {outdir / 'qc_real.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
