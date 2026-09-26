r"""CheXlocalize 银标签配准管线：翻拍 ↔ clean 单应配准 + GT 传播。

三方对齐：
  clean = CheXpert-v1.0-small/valid/...（每图宽高不同，如 390x320）
  photo = chexphoto/valid/natural/oneplus/...（大尺寸翻拍显示器照片）
  mask  = CheXlocalize gt_segmentations_val RLE（CheXpert 原始分辨率 h0xw0）

流程（每图）：
  1. 解码 RLE → 原始分辨率 → 按该图 clean 小图实际 (w,h) 重采样（与 CheXpert-small
     的长边 390 保持纵横比口径一致）。
  2. SIFT 特征（clean 长边 390 / photo 长边 800）+ BF 匹配 ratio 0.7 + RANSAC 单应
     （clean→photo，阈值 5px @800）。
  3. registration_ok：内点 ≥25 且内点率 ≥0.20 且中位重投影误差 ≤4px 且 warp 后
     mask 面积占比 ∈[0.005,0.5]。
  4. GT 传播：mask --H--> photo(800) --resize--> 224x224（与评价管线 resize 口径
     一致），存 npz。

输出（gpu10 rep_robust/chexlocalize/）：
  silver_register.csv / silver_masks_224.npz / overlay_*.png（sanity）

用法：/root/miniconda3/bin/python silver_register.py [--limit 6]
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/root/autodl-tmp"
CLEAN = f"{BASE}/CheXpert-v1.0-small/valid"
PHOTO = f"{BASE}/chexphoto/valid/natural/oneplus"
SEG_JSON = f"{BASE}/chexlocalize/gt_segmentations_val.json"
OUT = f"{BASE}/experiments/rep_robust/chexlocalize"

PH_REG = 800        # photo 配准尺度（长边）
PH_EVAL = 224       # 评价空间
RATIO = 0.7
RANSAC_THR = 5.0
MIN_INLIERS, MIN_RATIO, MAX_MEDERR = 25, 0.20, 4.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rle_string_to_counts(s: str) -> list[int]:
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
    assert pos == h * w
    return flat.reshape((w, h)).T


def load_gray(path: str, long_side: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    s = long_side / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    seg = json_load()
    findings = sorted(next(iter(seg.values())).keys())

    # 工作集：有 clean 图、有翻拍、非 lateral 的 val 键
    work = []
    for k, per in seg.items():
        parts = k.split("_")
        if parts[-1] != "frontal":
            continue
        path_c = f"{CLEAN}/{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}.jpg"
        path_p = f"{PHOTO}/{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}.jpg"
        if os.path.exists(path_c) and os.path.exists(path_p):
            work.append((k, path_c, path_p))
    log(f"frontal 交集工作集：{len(work)}")
    if args.limit:
        work = work[: args.limit]

    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher()
    rows, masks224 = [], {}
    for n, (k, path_c, path_p) in enumerate(work):
        parts = k.split("_")
        try:
            clean_im = Image.open(path_c).convert("L")
            cw, ch = clean_im.size
            clean = np.asarray(clean_im)
            photo = load_gray(path_p, PH_REG)
            ph, pw = photo.shape

            # 各病种 mask @clean 尺寸
            mclean = {}
            for f in findings:
                rle = per[f] if (per := seg[k]) and f in seg[k] else None
                if not (isinstance(rle, dict) and rle.get("counts")):
                    continue
                m = rle_decode(rle)
                if m.sum() == 0:
                    continue
                mi = Image.fromarray(m * 255).resize((cw, ch), Image.BILINEAR)
                mclean[f] = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            if not mclean:
                continue

            # SIFT + 单应（clean→photo @PH_REG）
            kp1, d1 = sift.detectAndCompute(clean, None)
            kp2, d2 = sift.detectAndCompute(photo, None)
            if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
                rows.append({"key": k, "ok": 0, "reason": "few_kp"})
                continue
            matches = bf.knnMatch(d1, d2, k=2)
            good = [m for m, nn in ((g[0], g[1]) for g in matches if len(g) == 2)
                    if m.distance < RATIO * nn.distance]
            if len(good) < MIN_INLIERS:
                rows.append({"key": k, "ok": 0, "reason": "few_match",
                             "n_match": len(good)})
                continue
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
            if H is None:
                rows.append({"key": k, "ok": 0, "reason": "no_H"})
                continue
            inl = inl.ravel().astype(bool)
            n_in, ratio = int(inl.sum()), float(inl.mean())
            proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
            err = np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)
            med = float(np.median(err))

            # GT 传播：H(390-space clean→photo800) 与 resize(photo800→224) 复合
            H224 = H.copy()
            H224[0] *= PH_EVAL / pw
            H224[1] *= PH_EVAL / ph
            ok_area = True
            w224 = {f: cv2.warpPerspective(m, H224, (PH_EVAL, PH_EVAL))
                    for f, m in mclean.items()}
            # 面积合理性按病种记录（细线条病种如 Support Devices 不参与图像级判定）
            bad_area = [f for f, m in w224.items()
                        if float(m.mean()) < 0.002 or float(m.mean()) > 0.6]
            ok = int(n_in >= MIN_INLIERS and ratio >= MIN_RATIO
                     and med <= MAX_MEDERR)

            rows.append({"key": k, "ok": ok, "reason": "" if ok else "thr",
                         "cw": cw, "ch": ch, "n_kp_clean": len(kp1),
                         "n_kp_photo": len(kp2), "n_match": len(good),
                         "n_inlier": n_in, "inlier_ratio": round(ratio, 3),
                         "med_reproj": round(med, 2),
                         "bad_area": ";".join(bad_area),
                         "mask_area_224": ";".join(
                             f"{f}:{w224[f].mean():.3f}" for f in sorted(w224))})
            if ok:
                arr = np.zeros((len(findings), PH_EVAL, PH_EVAL), np.uint8)
                for i, f in enumerate(findings):
                    if f in w224 and f not in bad_area:
                        arr[i] = w224[f]
                masks224[k] = arr

            # sanity 叠加（前 3 张 ok 的）
            if ok and sum(1 for r in rows if r.get("ok")) <= 3:
                vis = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
                Hw = H.copy()
                Hw[0] *= vis.shape[1] / pw
                Hw[1] *= vis.shape[0] / ph
                for f, m in mclean.items():
                    mw = cv2.warpPerspective(m * 255, Hw, (vis.shape[1], vis.shape[0]))
                    sel = mw > 127
                    vis[sel, 2] = np.minimum(255, vis[sel, 2].astype(int) + 120)
                    vis[sel, 0] = (vis[sel, 0] * 0.4).astype(np.uint8)
                Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(
                    f"{OUT}/overlay_{k}.png")
        except Exception as e:  # noqa: BLE001 —— 单图失败不中断批
            rows.append({"key": k, "ok": 0, "reason": f"err:{type(e).__name__}:{e}"[:120]})
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(work)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/silver_register.csv", index=False)
    np.savez_compressed(f"{OUT}/silver_masks_224.npz",
                        findings=np.array(findings), **masks224)
    ok_n = int(df["ok"].sum()) if "ok" in df else 0
    log(f"完成：{len(df)} 张，registration_ok {ok_n}（{ok_n / max(1, len(df)):.1%}）")
    print(df["reason"].value_counts().to_string() if "reason" in df else "")
    return 0


def json_load():
    import json
    return json.load(open(SEG_JSON))


if __name__ == "__main__":
    raise SystemExit(main())
