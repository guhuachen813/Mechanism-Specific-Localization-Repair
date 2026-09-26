r"""P2 阶段 1：PAIR-Loc 配对数据缓存（Cardi ckpt 为主，可指定病种）。

数据：CheXphoto train natural/iphone（847 张可用，iPhone 8 手拍）
      × clean 配对 = CheXpert-v1.0-small/train 同 patient/study/view 文件名。
配准：SIFT+BF+RANSAC（silver_register 同参数）→ H224（clean-native→photo224）。
teacher：clean 图 → 冻结 DenseNet CAM/prob → 经 T=H224@diag(cw/224,ch/224,1)
         传播到 photo224 空间 = 蒸馏目标。
anatomy：photo 图 MedSAM 掩码（L4 约束用）。

输出 pair_cache_{tag}.npz：
  keys(847) / photo(847,224,224) f32 / cam_t(847,224,224) f16 / prob_t(847)
  / mask_p(847,224,224) u8 / reg_ok(847) / finding
用法：/root/miniconda3/bin/python p07_pair_data.py [cardi|atel]
"""
from __future__ import annotations

import glob
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import MEAN, STD, batch_cam, load_model
from cam_prior import detect_field, mask_224, medsam_prep, segment

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
PH_REG = 800
RATIO, RANSAC_THR, MIN_INLIERS = 0.7, 5.0, 20
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    finding = {"cardi": "Cardiomegaly", "atel": "Atelectasis"}[
        sys.argv[1] if len(sys.argv) > 1 else "cardi"]
    tag = "cardi" if finding == "Cardiomegaly" else "atel"
    if finding == "Cardiomegaly":
        CLEAN = f"{BASE}/CheXpert-v1.0-small/train"
        PHOTO = f"{BASE}/chexphoto/train/natural/iphone"
    else:
        CLEAN = f"{BASE}/CheXpert-v1.0-small/valid"
        PHOTO = f"{BASE}/chexphoto/valid/natural/oneplus"
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    if finding == "Atelectasis":
        cam_benchmark.CKPT = "/root/project/outputs/atel_seed42_densenet121/best.pt"
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    from transformers import SamModel
    smodel = SamModel.from_pretrained(f"{ROOT}/weights/medsam").to(device).eval()

    import cv2 as _cv2
    sift = _cv2.SIFT_create()
    bf = _cv2.BFMatcher()

    files = sorted(f for f in glob.glob(f"{PHOTO}/patient*/study*/view*_frontal.jpg")
                   if not os.path.basename(f).startswith("._"))
    if finding == "Atelectasis":
        # 排除 73 张 Atel 评估键（同 key 不入训练对）
        Da = np.load(f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
        excl = {str(k) for k in Da["keys"]}
        files = [f for f in files
                 if f.split("natural/oneplus/")[1][:-4].replace("/", "_") not in excl]
    log(f"[{finding}] 配对 {len(files)} 张")

    photos, cam_ts, prob_ts, masks, keys, regoks = [], [], [], [], [], []
    t0 = time.time()
    for n, fp in enumerate(files):
        try:
            rel = fp.split("natural/")[1]                 # device/patient/study/view_frontal.jpg
            rel = rel.split("/", 1)[1]                    # patient/study/view_frontal.jpg
            path_c = f"{CLEAN}/{rel}"
            key = rel.replace("/", "_")[:-4]
            pim = Image.open(fp).convert("L")
            cim = Image.open(path_c).convert("L")
            cw, ch = cim.size
            pw, ph = pim.size
            # ---- 配准：clean-native → photo800 → photo224
            s800 = PH_REG / max(pw, ph)
            photo800 = np.asarray(pim.resize((round(pw * s800), round(ph * s800)),
                                             Image.BILINEAR))
            kp1, d1 = sift.detectAndCompute(np.asarray(cim), None)
            kp2, d2 = sift.detectAndCompute(photo800, None)
            ok = 0
            T = np.eye(3, dtype=np.float32)
            photo224 = np.asarray(pim.resize((224, 224), Image.BILINEAR))
            if d1 is not None and d2 is not None and len(kp1) >= 20 and len(kp2) >= 20:
                matches = bf.knnMatch(d1, d2, k=2)
                good = [m for m, nn in ((g[0], g[1]) for g in matches if len(g) == 2)
                        if m.distance < RATIO * nn.distance]
                if len(good) >= MIN_INLIERS:
                    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, _ = _cv2.findHomography(src, dst, _cv2.RANSAC, RANSAC_THR)
                    if H is not None:
                        H224 = H.copy()
                        H224[0] *= 224.0 / (pw * s800)
                        H224[1] *= 224.0 / (ph * s800)
                        T = (H224 @ np.diag([cw / 224.0, ch / 224.0, 1.0])).astype(np.float32)
                        ok = 1
            # ---- teacher：clean CAM → 传播到 photo224
            x_c = tf(cim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            with torch.no_grad():
                cam_c, prob_c = batch_cam(model, x_c, device)
            cam_t = _cv2.warpPerspective(cam_c[0].astype(np.float32), T, (224, 224))
            cam_t = np.clip(cam_t, 0, 1)
            # ---- anatomy：photo MedSAM 掩码
            fr, ok_r = detect_field(pim.resize((224, 224), Image.BILINEAR))
            mm = mask_224(segment(smodel, medsam_prep(pim), fr, device)).astype(np.uint8)
            photos.append(photo224.astype(np.float32) / 255.0)
            cam_ts.append(cam_t.astype(np.float16))
            prob_ts.append(float(prob_c[0]))
            masks.append(mm)
            keys.append(key)
            regoks.append(ok)
        except Exception as e:  # noqa: BLE001
            log(f"err {fp}: {e}")
        if (n + 1) % 100 == 0:
            el = time.time() - t0
            log(f"  {n + 1}/{len(files)} {el:.0f}s eta {el / (n + 1) * (len(files) - n - 1):.0f}s"
                f"  reg_ok {sum(regoks)}/{n + 1}")
    np.savez_compressed(f"{OUT}/pair_cache_{tag}.npz",
                        keys=np.array(keys), photo=np.stack(photos).astype(np.float16),
                        cam_t=np.stack(cam_ts), prob_t=np.array(prob_ts),
                        mask_p=np.stack(masks), reg_ok=np.array(regoks),
                        finding=np.array(finding))
    log(f"[{finding}] 完成：{len(keys)} 对，reg_ok {sum(regoks)} → pair_cache_{tag}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
