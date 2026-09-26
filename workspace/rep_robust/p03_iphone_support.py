r"""P0.3：支持度 s(x) 在未参与训练的新采集语境（iPhone 8，847 张）上的外测。

背景：s(x)=logistic(q9) 只在 NIH 合成退化（正类）与 OnePlus 6 翻拍（负类）上
训练过。用户审计要求：用完全未参与支持度训练的新采集语境测试，否则只能
称"对已知 photo 域的来源检测器"。

iPhone 8 数据无 q9 缓存（p5_lite 只存了派生统计），本脚本重算：
  qv = q_vector(gray, mask, field_ok) ⊕ cam_feats(cam, mask)   （与 p62b 同一 9 维）
然后用 p62b 训练的全量 clf（synth vs photo 逻辑回归）输出 s(x)。
零样本外测：iPhone 不参与 s 拟合。

对照读数：photo OnePlush 65 s 中位 0.548（拦截域）、synth 外测 0.998（放行域）。
判读：
  - s(iPhone) 中位 << 1 → s(x) 跨采集设备可迁移，支持度语义成立
  - s(iPhone) 中位 ≈ 1 → s(x) 是 OnePlus 特异检测器，语义坍塌为"来源检测"

输出：p03_iphone_support.csv + 控制台汇总
用法：/root/miniconda3/bin/python p03_iphone_support.py
"""
from __future__ import annotations

import glob
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import transforms

import cam_benchmark
from cam_benchmark import MEAN, STD, batch_cam, load_model
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import cam_feats, q_vector

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
IPHONE = "/root/autodl-tmp/chexphoto/train/natural/iphone"
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    from transformers import SamModel
    smodel = SamModel.from_pretrained(f"{ROOT}/weights/medsam").to(device).eval()

    # ---------- s(x) 训练：与 p62b 全量版一致 ----------
    log("重建 s(x) 训练集（NIH synth + OnePlus photo q9）")
    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
    q9_n_flat = np.concatenate([C["qs"], C["cs"]], 2).reshape(-1, 9)
    D = np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    q9_p = D["q9"]                     # (65,9) OnePlus
    Xs = np.concatenate([q9_n_flat, q9_p], 0)
    ys = np.array([1] * len(q9_n_flat) + [0] * len(q9_p))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xs, ys)
    s_photo_ref = clf.predict_proba(q9_p)[:, 1]
    log(f"参照：OnePlus 65 s 中位 {np.median(s_photo_ref):.3f}（拦截域）")

    # ---------- iPhone 8：重算 q9 ----------
    files = sorted(glob.glob(f"{IPHONE}/patient*/study*/view*_frontal.jpg"))
    log(f"iPhone 8 待处理 {len(files)} 张")
    rows = []
    t0 = time.time()
    for n, fp in enumerate(files):
        try:
            im = Image.open(fp).convert("L")
            im224 = im.resize((224, 224), Image.BILINEAR)
            x = tf(im224.convert("RGB"))[None]
            with torch.no_grad():
                cam, prob = batch_cam(model, x, device)
            cam = cam[0]
            fr, ok_r = detect_field(im224)
            mm = mask_224(segment(smodel, medsam_prep(im), fr, device)).astype(np.uint8)
            g_arr = np.asarray(im224.convert("L"), np.float32) / 255.0
            qv = np.concatenate([q_vector(g_arr, mm, float(ok_r)),
                                 cam_feats(cam, mm)])
            rows.append({"key": fp.split("natural/iphone/")[1],
                         "field_ok": int(ok_r), "frac": float(qv[1]),
                         "prob": float(prob[0]), **{f"q{i}": float(qv[i]) for i in range(9)}})
        except Exception as e:  # noqa: BLE001
            log(f"err {fp}: {e}")
        if (n + 1) % 100 == 0:
            el = time.time() - t0
            log(f"  {n + 1}/{len(files)}  {el:.0f}s  eta {el / (n + 1) * (len(files) - n - 1):.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p03_iphone_q9.csv", index=False)

    # ---------- s(x) 零样本应用 ----------
    q9_ip = df[[f"q{i}" for i in range(9)]].to_numpy(np.float32)
    s_ip = clf.predict_proba(q9_ip)[:, 1]
    df["s_support"] = s_ip
    df.to_csv(f"{OUT}/p03_iphone_support.csv", index=False)

    # ---------- 汇总 ----------
    q9_ext = None
    try:
        E = np.load(f"{OUT}/p61_test_cams.npz", allow_pickle=True)
        q9_ext = E["qv"]
    except Exception:  # noqa: BLE001
        pass
    print("\n== P0.3 s(x) 新采集语境外测（iPhone 8，零样本） ==")
    print(f"  n = {len(df)}")
    print(f"  s(iPhone) 分位 p10/p25/p50/p75/p90 = "
          f"{np.percentile(s_ip, [10, 25, 50, 75, 90]).round(3)}")
    print(f"  s(iPhone) < κ(q25 源域) 比例 = "
          f"{(s_ip < np.quantile(clf.predict_proba(q9_n_flat)[:, 1], 0.25)).mean():.3f}")
    if q9_ext is not None:
        s_ext = clf.predict_proba(q9_ext.astype(np.float32))[:, 1]
        print(f"  参照 synth 外测 s 中位 {np.median(s_ext):.3f} | OnePlus 65 中位 "
              f"{np.median(s_photo_ref):.3f}")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
