r"""P1a：扩充候选动作 + 新 oracle（免训练部分）。

新增候选（用户 P1 清单的 ④⑤，⑥ feature adapter / ⑦ PAIR-Loc 后续）：
  rect45   ④ 单应校正后 CAM：photo 原生分辨率经 H⁻¹ 扭到 clean 几何 → 224 →
           DenseNet CAM → 经 H 复合矩阵扭回 photo-224 空间评价
  teach45  ⑤ clean teacher CAM 传播：clean 侧 CAM（cache 的 cams_c，Cardi 已有，
           Atel 现算）经 clean224→photo224 复合单应传播
既有候选：raw45 / prior45 / cand_zs45。
新 oracle = max(五动作)；银标签 = photo-224 空间 silver mask（评价空间不变）。

配准：SIFT + BF ratio 0.7 + RANSAC（与 silver_register.py 同参数），按 key 重跑
并保存 H224（clean-native → photo224）供 P2 复用。

判据（预注册）：新 oracle 相对 raw ≥ +0.02、患者级 CI 下界>0、新候选不是只在
少数异常样本上有效（报 argmax 来源分布）。

用法：/root/miniconda3/bin/python p06_p1_candidates.py [cardi|atel|all]
"""
from __future__ import annotations

import csv
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import MEAN, STD, THRESHOLDS, batch_cam, cam_iou_curve, load_model
from repairer import cam_feats, q_vector  # noqa: F401 (保持 import 路径一致)

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
CLEAN = f"{BASE}/CheXpert-v1.0-small/valid"
PHOTO = f"{BASE}/chexphoto/valid/natural/oneplus"
PH_REG = 800
RATIO, RANSAC_THR, MIN_INLIERS = 0.7, 5.0, 20
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iou45(cam, gt):
    return float(cam_iou_curve(cam, gt)[TAU45])


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def boot_ci(d, key, n=400, seed=42):
    up = pd.unique(key)
    fmap = pd.Series(key).map({f: i for i, f in enumerate(up)}).to_numpy()
    rng = np.random.default_rng(seed)
    vals = [d[np.isin(fmap, rng.choice(len(up), len(up)))].mean() for _ in range(n)]
    return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def register_H(k, sift, bf):
    """返回 (H224: clean-native→photo224, cw, ch, ok, reason)。"""
    parts = k.split("_")
    stem = "_".join(parts[2:])
    path_c = f"{CLEAN}/{parts[0]}/{parts[1]}/{stem}.jpg"
    path_p = f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
    try:
        cim = Image.open(path_c).convert("L")
        cw, ch = cim.size
        pim = Image.open(path_p).convert("L")
        pw, ph = pim.size
        s800 = PH_REG / max(pw, ph)
        photo = np.asarray(pim.resize((round(pw * s800), round(ph * s800)),
                                      Image.BILINEAR))
        kp1, d1 = sift.detectAndCompute(np.asarray(cim), None)
        kp2, d2 = sift.detectAndCompute(photo, None)
        if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
            return None, cw, ch, 0, "few_kp"
        matches = bf.knnMatch(d1, d2, k=2)
        good = [m for m, nn in ((g[0], g[1]) for g in matches if len(g) == 2)
                if m.distance < RATIO * nn.distance]
        if len(good) < MIN_INLIERS:
            return None, cw, ch, 0, "few_match"
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
        if H is None:
            return None, cw, ch, 0, "no_H"
        H224 = H.copy()
        H224[0] *= 224.0 / pw800(pw, s800)
        H224[1] *= 224.0 / ph800(ph, s800)
        return H224, cw, ch, 1, ""
    except Exception as e:  # noqa: BLE001
        return None, 0, 0, 0, f"err:{e}"


def pw800(pw, s800):
    return pw * s800


def ph800(ph, s800):
    return ph * s800


def run_domain(finding: str, ckpt_path: str) -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    cam_benchmark.CKPT = ckpt_path
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    tag = {"Cardiomegaly": "cardi", "Atelectasis": "atel"}[finding]
    cache_name = "cache_photo_adapt.npz" if finding == "Cardiomegaly" else f"cache_photo_{tag}.npz"
    D = np.load(f"{OUT}/{cache_name}", allow_pickle=True)
    keys = [str(x) for x in D["keys"]]
    gts = D["gts"]
    cams_c = D["cams_c"] if "cams_c" in D.files else None
    p62f = (f"{OUT}/p62_photo_sage.csv" if finding == "Cardiomegaly"
            else f"{OUT}/p65_atel_photo.csv")
    dfp = pd.read_csv(p62f).set_index("key").loc[keys].reset_index()

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    log(f"[{finding}] 配准 {len(keys)} 键")
    Hs, sizes, oks = {}, {}, {}
    for k in keys:
        H, cw, ch, ok, reason = register_H(k, sift, bf)
        Hs[k], sizes[k], oks[k] = H, (cw, ch), (ok, reason)
    n_ok = sum(v[0] for v in oks.values())
    log(f"[{finding}] 配准 ok {n_ok}/{len(keys)}")

    # clean 侧 CAM（Cardi 用 cache；Atel 现算）
    if cams_c is None:
        log(f"[{finding}] 现算 clean 侧 CAM（Atel ckpt）")
        cams_c = np.zeros((len(keys), 224, 224), np.float32)
        for i, k in enumerate(keys):
            parts = k.split("_")
            stem = "_".join(parts[2:])
            cim = Image.open(f"{CLEAN}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            x = tf(cim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            with torch.no_grad():
                cam, _ = batch_cam(model, x, device)
            cams_c[i] = cam[0]

    rows = []
    cams_out, teach_out, rect_out = [], [], []
    for i, k in enumerate(keys):
        ok, reason = oks[k]
        raw_i = float(dfp.raw45.iloc[i])
        prior_i = float(dfp.prior45.iloc[i])
        cand_i = float(dfp.cand_zs45.iloc[i])
        teach_i, rect_i = np.nan, np.nan
        gt = mask_bbox(gts[i])
        if ok and gt:
            H224, (cw, ch) = Hs[k], sizes[k]
            # ---- ⑤ teacher 传播：clean224 → photo224 = H224 @ diag(cw/224, ch/224)
            T = H224 @ np.diag([cw / 224.0, ch / 224.0, 1.0])
            cam_t = cv2.warpPerspective(cams_c[i].astype(np.float32), T, (224, 224))
            teach_i = iou45(np.clip(cam_t, 0, 1), gt)
            # ---- ④ 单应校正：photo native → clean native → 224 → CAM → 扭回
            parts = k.split("_")
            stem = "_".join(parts[2:])
            pim = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg")
            pw, ph = pim.size
            s800 = PH_REG / max(pw, ph)
            # H 是 clean-native→photo800；photo native→photo800 缩放
            S = np.diag([pw * s800 / pw, ph * s800 / ph, 1.0])  # =diag(s800,s800,1)
            M = np.linalg.inv(H) @ np.diag([s800, s800, 1.0])
            pnat = np.asarray(pim.convert("L"))
            rect = cv2.warpPerspective(pnat, M, (cw, ch))
            x = tf(Image.fromarray(rect).convert("RGB").resize(
                (224, 224), Image.BILINEAR))[None]
            with torch.no_grad():
                cam_r, _ = batch_cam(model, x, device)
            cam_r224 = cam_r[0]
            cam_back = cv2.warpPerspective(cam_r224.astype(np.float32),
                                           T, (224, 224))
            rect_i = iou45(np.clip(cam_back, 0, 1), gt)
            cams_out.append(cam_r224.astype(np.float16))
            teach_out.append(cam_t.astype(np.float16))
            rect_out.append(cam_back.astype(np.float16))
        else:
            cams_out.append(np.zeros((224, 224), np.float16))
            teach_out.append(np.zeros((224, 224), np.float16))
            rect_out.append(np.zeros((224, 224), np.float16))
        rows.append({"key": k, "patient": k.split("_")[0], "reg_ok": ok,
                     "reason": reason, "raw45": raw_i, "prior45": prior_i,
                     "cand_zs45": cand_i, "teach45": teach_i, "rect45": rect_i})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p06_p1_candidates_{tag}.csv", index=False)
    np.savez_compressed(f"{OUT}/p06_p1_cams_{tag}.npz",
                        keys=np.array(keys),
                        cam_rect=np.stack(cams_out), cam_teach=np.stack(teach_out),
                        cam_back=np.stack(rect_out),
                        H224=np.stack([Hs[k] if Hs[k] is not None
                                       else np.eye(3) for k in keys]).astype(np.float32),
                        reg_ok=np.array([oks[k][0] for k in keys]))

    # ---- 新 oracle ----
    vals = {"raw45": df.raw45, "prior45": df.prior45, "cand_zs45": df.cand_zs45,
            "teach45": df.teach45, "rect45": df.rect45}
    V = np.stack([v.to_numpy() for v in vals.values()], 1)
    Vn = np.where(np.isnan(V), -1.0, V)
    oracle_old = Vn[:, :3].max(1)
    oracle_new = Vn.max(1)
    names = list(vals)
    print(f"\n== [{finding}] P1a 候选与 oracle（τ45，photo 银标签，n={len(df)}） ==")
    for j, nm in enumerate(names):
        d = Vn[:, j] - Vn[:, 0]
        m, lo, hi = boot_ci(d, df.patient.to_numpy())
        print(f"  {nm:10s} Δ {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]  IoU {np.nanmean(V[:, j]):.4f}")
    d = oracle_old - Vn[:, 0]
    m, lo, hi = boot_ci(d, df.patient.to_numpy())
    print(f"  oracle_old {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    d = oracle_new - Vn[:, 0]
    m, lo, hi = boot_ci(d, df.patient.to_numpy())
    print(f"  **oracle_new** {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]  (n_reg_ok {n_ok})")
    src = np.argmax(Vn, 1)
    print("  argmax 来源分布:", {names[j]: int((src == j).sum()) for j in range(len(names))})
    # 新候选只在少数异常样本上有效吗：rect/teach 作为唯一 argmax 且 > old oracle 的图数
    for j in (3, 4):
        imp = (Vn[:, j] > oracle_old + 1e-6).sum()
        print(f"  {names[j]} 超过旧 oracle 的图数: {imp}/{len(df)}")
    log(f"[{finding}] 完成")
    return 0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ckpt_c = "/root/project/outputs/repaired_seed42/densenet121/best.pt"
    if not os.path.exists(ckpt_c):
        import glob as g
        cand = g.glob("/root/project/outputs/*densenet121*/best.pt")
        ckpt_c = cand[0] if cand else ckpt_c
    ckpt_a = "/root/project/outputs/atel_seed42_densenet121/best.pt"
    rc = 0
    if which in ("cardi", "all"):
        rc = run_domain("Cardiomegaly", ckpt_c)
    if which in ("atel", "all") and rc == 0:
        rc = run_domain("Atelectasis", ckpt_a)
    raise SystemExit(rc)
