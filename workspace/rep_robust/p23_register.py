r"""P23 自主翻拍注册跑道（silver_register.py 的 P23 变体，协议冻结 v1.0 §6）。

仅改动：
  - 工作集来自 results/p6/p23_shoot_manifest.csv（shot_id → clean_relpath，1:1 患者）
  - PHOTO 指向 /root/autodl-tmp/p23_capture/ 的 pass1 文件（S###_1.jpg）
  - 输出目录 = p23_register/；sanity overlay 改为 reg-ok 中随机抽 6 张（协议 §6）
SIFT/BF/RANSAC 全部参数逐字不动（clean 长边 390 / photo 长边 800，ratio 0.7，thr 5px）。
reg-ok 口径：内点 ≥25 且内点率 ≥0.20 且中位重投影误差 ≤4px；mask warp 后面积占比 ∈[0.002,0.6]。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/root/autodl-tmp"
CLEAN = f"{BASE}/CheXpert-v1.0-small/valid"
PHOTO = f"{BASE}/p23_capture"          # pass1: S###_1.jpg
MANIFEST = f"{BASE}/p23_capture/p23_shoot_manifest.csv"
SEG_JSON = f"{BASE}/chexlocalize/gt_segmentations_val.json"
OUT = f"{BASE}/experiments/rep_robust/p23_register"

PH_REG = 800        # photo 配准尺度（长边）
PH_EVAL = 224       # 评价空间
RATIO = 0.7
RANSAC_THR = 5.0
MIN_INLIERS, MIN_RATIO, MAX_MEDERR = 25, 0.20, 4.0
AREA_LO, AREA_HI = 0.002, 0.6


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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    random.seed(args.seed)

    with open(SEG_JSON) as f:
        seg = json.load(f)
    findings = sorted(next(iter(seg.values())).keys())

    man = pd.read_csv(MANIFEST)
    work = []
    for _, r in man.iterrows():
        path_c = f"{CLEAN}/{r['clean_relpath'].replace('CheXpert-v1.0-small/valid/', '', 1)}"
        path_p = f"{PHOTO}/{r['file_pass1']}"
        if not os.path.exists(path_c):
            work.append((r["shot_id"], r["key"], path_c, path_p, "clean_missing"))
            continue
        if not os.path.exists(path_p):
            work.append((r["shot_id"], r["key"], path_c, path_p, "photo_missing"))
            continue
        if r["key"] not in seg:
            work.append((r["shot_id"], r["key"], path_c, path_p, "no_seg"))
            continue
        work.append((r["shot_id"], r["key"], path_c, path_p, ""))
    log(f"P23 工作集（manifest pass1）：{len(work)}")
    if args.limit:
        work = work[: args.limit]

    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher()
    rows, masks224, overlays = [], {}, {}
    for n, (sid, k, path_c, path_p, pre_fail) in enumerate(work):
        parts = k.split("_")
        if pre_fail:
            rows.append({"shot_id": sid, "key": k, "ok": 0, "reason": pre_fail})
            continue
        per = seg[k]
        try:
            clean_im = Image.open(path_c).convert("L")
            cw, ch = clean_im.size
            clean = np.asarray(clean_im)
            photo = load_gray(path_p, PH_REG)
            ph, pw = photo.shape

            mclean = {}
            for f in findings:
                rle = per.get(f)
                if not (isinstance(rle, dict) and rle.get("counts")):
                    continue
                m = rle_decode(rle)
                if m.sum() == 0:
                    continue
                mi = Image.fromarray(m * 255).resize((cw, ch), Image.BILINEAR)
                mclean[f] = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            if not mclean:
                rows.append({"shot_id": sid, "key": k, "ok": 0, "reason": "no_mask"})
                continue

            kp1, d1 = sift.detectAndCompute(clean, None)
            kp2, d2 = sift.detectAndCompute(photo, None)
            if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
                rows.append({"shot_id": sid, "key": k, "ok": 0, "reason": "few_kp"})
                continue
            matches = bf.knnMatch(d1, d2, k=2)
            good = [m for m, nn in ((g[0], g[1]) for g in matches if len(g) == 2)
                    if m.distance < RATIO * nn.distance]
            if len(good) < MIN_INLIERS:
                rows.append({"shot_id": sid, "key": k, "ok": 0, "reason": "few_match",
                             "n_match": len(good)})
                continue
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
            if H is None:
                rows.append({"shot_id": sid, "key": k, "ok": 0, "reason": "no_H"})
                continue
            inl = inl.ravel().astype(bool)
            n_in, ratio = int(inl.sum()), float(inl.mean())
            proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
            err = np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)
            med = float(np.median(err))

            H224 = H.copy()
            H224[0] *= PH_EVAL / pw
            H224[1] *= PH_EVAL / ph
            w224 = {f: cv2.warpPerspective(m, H224, (PH_EVAL, PH_EVAL))
                    for f, m in mclean.items()}
            bad_area = [f for f, m in w224.items()
                        if float(m.mean()) < AREA_LO or float(m.mean()) > AREA_HI]
            ok = int(n_in >= MIN_INLIERS and ratio >= MIN_RATIO
                     and med <= MAX_MEDERR)

            rows.append({"shot_id": sid, "key": k, "ok": ok,
                         "reason": "" if ok else "thr",
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
                overlays[(sid, k)] = (photo, H, mclean, pw, ph)
        except Exception as e:  # noqa: BLE001
            rows.append({"shot_id": sid, "key": k, "ok": 0, "reason": f"err:{type(e).__name__}"})
        if (n + 1) % 20 == 0:
            log(f"{n + 1}/{len(work)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p23_register.csv", index=False)
    np.savez_compressed(f"{OUT}/p23_masks_224.npz",
                        findings=np.array(findings),
                        **{k: v for k, v in masks224.items()})
    n_ok = int(df["ok"].sum())
    log(f"reg-ok：{n_ok}/{len(df)} = {n_ok / len(df) * 100:.1f}%")
    log("失败列表：" + json.dumps(
        df[df["ok"] == 0][["shot_id", "reason"]].to_dict("records"), ensure_ascii=False))

    # sanity：reg-ok 中随机 6 张 overlay（协议 §6）
    ok_ids = [x for x in overlays if df.loc[df["key"] == x[1], "ok"].iloc[0]]
    for sid, k in random.sample(ok_ids, min(6, len(ok_ids))):
        photo, H, mclean, pw, ph = overlays[(sid, k)]
        vis = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
        Hw = H.copy()
        Hw[0] *= vis.shape[1] / pw
        Hw[1] *= vis.shape[0] / ph
        for f, m in mclean.items():
            mw = cv2.warpPerspective(m, Hw, (vis.shape[1], vis.shape[0]))
            cnts, _ = cv2.findContours(mw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
        cv2.imwrite(f"{OUT}/overlay_{sid}.png", vis)
    log("sanity overlay 已写出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
