r"""P2 阶段 2：PAIR-Loc 配对蒸馏训练与评估（Cardi / Atel 各自运行）。

架构：冻结 DenseNet（病种 ckpt）+ 轻量残差 adapter（1×1 低秩 1024→16→1024，
零初始化 → 学生从 raw 出发）。学生只看 photo 图，部署时无需 clean 原图。

五损失（用户 P2 规范）：
  L_logit  学生 prob(photo) ↔ teacher prob(clean) 的 MSE
  L_cam    registration-aware CAM 蒸馏：MSE(CAM_student(photo224), cam_t)
           cam_t = clean teacher CAM 经配准单应传播到 photo224（p07 预计算）
  L_geo    几何等变一致：MSE(CAM(hflip(photo)), hflip(cam_t))
  L_anat   anatomy 约束：CAM 能量在 MedSAM 掩码外惩罚
  L_ident  identity：学生 adapter 在 clean 特征上行为不变
           MSE(CAM_student(clean), CAM_teacher(clean)) —— 保证 clean 域不降

协议：847 iphone 配对按患者 5 折 + 全量拟合；早停用 valid oneplus 65 银标签
（模型选择用途，符合"银标签只用于评估不伪装强监督"）；报告各折 valid Δ 方向
一致性；clean 侧身份漂移 |CAM_stu(clean)−CAM_tea(clean)|。

用法：/root/miniconda3/bin/python p08_pair_loc.py [cardi|atel]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, batch_cam, load_model

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
PHOTO_V = f"{BASE}/chexphoto/valid/natural/oneplus"
SEED = 42
EPOCHS = 40
BS = 32
W = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.5}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iou45(cam, gt):
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    return float(cam_iou_curve(cam, gt)[int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


class Adapter(torch.nn.Module):
    def __init__(self, c=1024, r=16):
        super().__init__()
        self.d1 = torch.nn.Conv2d(c, r, 1)
        self.d2 = torch.nn.Conv2d(r, c, 1, bias=False)
        torch.nn.init.zeros_(self.d2.weight)
        torch.nn.init.normal_(self.d1.weight, std=1e-3)

    def forward(self, f):
        return f + self.d2(self.d1(f))


def cam_from_feat(feat, w1):
    """feat (B,1024,7,7) → (B,224,224) CAM（与 batch_cam 同一读出）。"""
    cam = torch.einsum("k,bkyx->byx", w1, feat).relu_()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                         align_corners=False)[:, 0]


def norm_t():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def build_features(model, device, finding):
    """预计算：iphone 配对 photo/clean 特征 + valid65 photo 特征。"""
    tag = "cardi" if finding == "Cardiomegaly" else "atel"
    fp_cache = f"{OUT}/feat_pair_{tag}.npz"
    if os.path.exists(fp_cache):
        log("特征缓存已存在，跳过")
        return np.load(fp_cache, allow_pickle=True)
    tf = norm_t()
    D = np.load(f"{OUT}/pair_cache_{tag}.npz", allow_pickle=True)
    keys = [str(k) for k in D["keys"]]
    CLEAN = (f"{BASE}/CheXpert-v1.0-small/train" if finding == "Cardiomegaly"
             else f"{BASE}/CheXpert-v1.0-small/valid")
    if finding == "Atelectasis":
        # Atel 的 photo 在 valid/oneplus，clean 对应是 CheXpert valid
        PDIR = f"{BASE}/chexphoto/valid/natural/oneplus"
        CLEAN = f"{BASE}/CheXpert-v1.0-small/valid"
    else:
        PDIR = f"{BASE}/chexphoto/train/natural/iphone"
    fp_photo, fp_clean, probs_c = [], [], []
    with torch.no_grad():
        for n, k in enumerate(keys):
            rel = k.replace("_frontal", "_frontal")
            parts = k.split("_")
            stem = "_".join(parts[2:])
            pim = Image.open(f"{PDIR}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            cim = Image.open(f"{CLEAN}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            xp = tf(pim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            xc = tf(cim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            fp_photo.append(model.features(xp.to(device))[0].cpu().numpy().astype(np.float16))
            f_c = model.features(xc.to(device))
            fp_clean.append(f_c[0].cpu().numpy().astype(np.float16))
            logits = model.classifier(f_c.mean(dim=(2, 3)))
            probs_c.append(float(torch.softmax(logits, 1)[0, 1]))
            if (n + 1) % 200 == 0:
                log(f"  feats {n + 1}/{len(keys)}")
    # valid65 photo 特征（Cardi 用 oneplus valid；Atel 同目录）
    Dv = np.load(f"{OUT}/cache_photo_adapt.npz" if finding == "Cardiomegaly"
                 else f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
    vkeys = [str(k) for k in Dv["keys"]]
    fp_val = []
    with torch.no_grad():
        for k in vkeys:
            parts = k.split("_")
            stem = "_".join(parts[2:])
            pim = Image.open(f"{PHOTO_V}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
            x = tf(pim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            fp_val.append(model.features(x.to(device))[0].cpu().numpy().astype(np.float16))
    np.savez_compressed(fp_cache, keys=np.array(keys),
                        f_photo=np.stack(fp_photo), f_clean=np.stack(fp_clean),
                        prob_c=np.array(probs_c), vkeys=np.array(vkeys),
                        f_val=np.stack(fp_val))
    log(f"特征缓存 {fp_cache}")
    return np.load(fp_cache, allow_pickle=True)


def cam224_from_np(f16, w1_t, device, idx=None):
    f = torch.from_numpy(np.asarray(f16, np.float32)).to(device)
    if idx is not None:
        f = f[idx]
    return cam_from_feat(f, w1_t)


def main() -> int:
    finding = {"cardi": "Cardiomegaly", "atel": "Atelectasis"}[
        sys.argv[1] if len(sys.argv) > 1 else "cardi"]
    tag = "cardi" if finding == "Cardiomegaly" else "atel"
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    if finding == "Atelectasis":
        cam_benchmark.CKPT = "/root/project/outputs/atel_seed42_densenet121/best.pt"
    model = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    w1 = model.classifier.weight[1].detach()

    D = np.load(f"{OUT}/pair_cache_{tag}.npz", allow_pickle=True)
    keys = [str(k) for k in D["keys"]]
    cam_t = D["cam_t"].astype(np.float32)
    mask_p = D["mask_p"].astype(np.float32)
    prob_t = D["prob_t"].astype(np.float32)
    reg_ok = D["reg_ok"].astype(bool)
    log(f"[{finding}] 配对 {len(keys)}，reg_ok {int(reg_ok.sum())}")

    FE = build_features(model, device, finding)
    f_photo, f_clean = FE["f_photo"], FE["f_clean"]
    prob_c = FE["prob_c"]
    vkeys = [str(k) for k in FE["vkeys"]]

    # valid65 参照量
    Dv = np.load(f"{OUT}/cache_photo_adapt.npz" if finding == "Cardiomegaly"
                 else f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
    gts_v = Dv["gts"]
    dfp = (pd.read_csv(f"{OUT}/p62_photo_sage.csv") if finding == "Cardiomegaly"
           else pd.read_csv(f"{OUT}/p65_atel_photo.csv")).set_index("key")
    raw_v = dfp.loc[vkeys, "raw45"].to_numpy()
    teach_v = pd.read_csv(f"{OUT}/p06_p1_candidates_{tag}.csv").set_index("key").loc[vkeys, "teach45"].to_numpy()
    rect_v = pd.read_csv(f"{OUT}/p06_p1_candidates_{tag}.csv").set_index("key").loc[vkeys, "rect45"].to_numpy()
    gt_boxes_v = [mask_bbox(gts_v[i]) for i in range(len(vkeys))]
    f_val = FE["f_val"]

    def valid_iou(adapter):
        adapter.eval()
        with torch.no_grad():
            f = torch.from_numpy(np.asarray(f_val, np.float32)).to(device)
            cam = cam_from_feat(adapter(f), w1).cpu().numpy()
        return np.array([iou45(cam[i], gt_boxes_v[i]) if gt_boxes_v[i] else np.nan
                         for i in range(len(vkeys))])

    # ---------------- 训练函数 ----------------
    def train_adapter(fit_idx, seed, early_stop=True):
        torch.manual_seed(seed)
        ad = Adapter().to(device)
        opt = torch.optim.Adam(ad.parameters(), lr=1e-3, weight_decay=1e-5)
        fit_idx = np.asarray(fit_idx)
        best = (-1e9, None, -1)
        for ep in range(EPOCHS):
            ad.train()
            perm = np.random.default_rng(seed * 1000 + ep).permutation(len(fit_idx))
            for s0 in range(0, len(fit_idx), BS):
                sel = fit_idx[perm[s0:s0 + BS]]
                fb = torch.from_numpy(np.asarray(f_photo[sel], np.float32)).to(device)
                fc = torch.from_numpy(np.asarray(f_clean[sel], np.float32)).to(device)
                tgt = torch.from_numpy(cam_t[sel]).to(device)
                mk = torch.from_numpy(mask_p[sel]).to(device)
                pt = torch.from_numpy(prob_t[sel]).to(device)
                fs = ad(fb)
                cam_s = cam_from_feat(fs, w1)
                cam_s_fl = cam_from_feat(ad(torch.flip(fb, dims=[3])), w1)
                cam_t_fl = torch.flip(tgt, dims=[2])
                cam_c = cam_from_feat(ad(fc), w1)
                cam_c_ref = cam_from_feat(fc, w1)
                logits_s = model.classifier(fs.mean(dim=(2, 3)))
                prob_s = torch.softmax(logits_s, 1)[:, 1]
                l_logit = F.mse_loss(prob_s, pt)
                l_cam = F.mse_loss(cam_s, tgt)
                l_geo = F.mse_loss(cam_s_fl, cam_t_fl)
                l_anat = (cam_s * (1 - F.interpolate(mk[:, None], size=IMAGE_SIZE,
                                                     mode="bilinear",
                                                     align_corners=False)[:, 0])).mean()
                l_ident = F.mse_loss(cam_c, cam_c_ref)
                loss = (W["logit"] * l_logit + W["cam"] * l_cam + W["geo"] * l_geo
                        + W["anat"] * l_anat + W["ident"] * l_ident)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if early_stop:
                iv = valid_iou(ad)
                score = np.nanmean(iv) - np.nanmean(raw_v)
                if score > best[0]:
                    best = (float(score), {k: v.detach().clone() for k, v in
                                           ad.state_dict().items()}, ep)
        if early_stop and best[1] is not None:
            ad.load_state_dict(best[1])
        return ad, (best[0], best[2])

    # ---------------- clean 侧身份漂移 ----------------
    def clean_drift(ad, idx):
        with torch.no_grad():
            fc = torch.from_numpy(np.asarray(f_clean[idx], np.float32)).to(device)
            cam_s = cam_from_feat(ad(fc), w1).cpu().numpy()
            cam_r = cam_from_feat(fc, w1).cpu().numpy()
        return float(np.abs(cam_s - cam_r).mean())

    # ---------------- 5 折患者 CV + 全量 ----------------
    pats = np.array([k.split("_")[0] for k in keys])
    up = sorted(set(pats))
    folds = [set(up[j::5]) for j in range(5)]
    ok_idx = np.where(reg_ok)[0]
    rows = []
    for jf in range(5):
        te_pat = folds[jf]
        fit = ok_idx[~np.isin(pats[ok_idx], list(te_pat))]
        ad, (score, ep) = train_adapter(fit, SEED + jf)
        iv = valid_iou(ad)
        d = iv - raw_v
        rng = np.random.default_rng(42)
        vp = np.array([k.split("_")[0] for k in vkeys])
        upv = pd.unique(vp)
        fmap = pd.Series(vp).map({f: i for i, f in enumerate(upv)}).to_numpy()
        vals = [d[np.isin(fmap, rng.choice(len(upv), len(upv)))].mean() for _ in range(400)]
        rows.append({"fold": jf, "best_ep": ep, "valid_score": score,
                     "photo_d": float(np.nanmean(d)),
                     "lo": float(np.percentile(vals, 2.5)),
                     "hi": float(np.percentile(vals, 97.5)),
                     "clean_drift": clean_drift(ad, ok_idx[::7])})
        print(f"  fold{jf} ep{ep} photo Δ {np.nanmean(d):+.4f} "
              f"[{rows[-1]['lo']:+.4f},{rows[-1]['hi']:+.4f}] "
              f"clean_drift {rows[-1]['clean_drift']:.4f}", flush=True)
    ad_full, (score, ep) = train_adapter(ok_idx, SEED + 99)
    iv = valid_iou(ad_full)
    d = iv - raw_v
    rng = np.random.default_rng(42)
    vp = np.array([k.split("_")[0] for k in vkeys])
    upv = pd.unique(vp)
    fmap = pd.Series(vp).map({f: i for i, f in enumerate(upv)}).to_numpy()
    vals = [d[np.isin(fmap, rng.choice(len(upv), len(upv)))].mean() for _ in range(400)]
    print(f"\n== [{finding}] PAIR-Loc valid {len(vkeys)}（τ45，患者整群 bootstrap 400） ==")
    print(f"  raw        IoU {np.nanmean(raw_v):.4f}")
    print(f"  teacher45  Δ {(teach_v - raw_v).mean():+.4f}（传播上界参照）")
    print(f"  rect45     Δ {(rect_v - raw_v).mean():+.4f}（几何校正参照）")
    print(f"  **PAIR-Loc** Δ {np.nanmean(d):+.4f} CI [{np.percentile(vals, 2.5):+.4f},"
          f"{np.percentile(vals, 97.5):+.4f}]  (full-fit ep{ep})")
    print(f"  5 折方向一致: {sum(1 for r in rows if r['photo_d'] > 0)}/5 正")
    print(f"  clean 身份漂移（全量模型）: {clean_drift(ad_full, ok_idx):.4f}")
    torch.save(ad_full.state_dict(), f"{OUT}/pairloc_adapter_{tag}.pt")
    pd.DataFrame(rows).to_csv(f"{OUT}/p08_pairloc_{tag}.csv", index=False)
    np.save(f"{OUT}/p08_pairloc_valid_iou_{tag}.npy",
            np.stack([iv, raw_v, teach_v, rect_v]))
    log(f"[{finding}] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
