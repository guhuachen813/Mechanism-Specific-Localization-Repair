r"""P18（优先级3）：新病种泛化 + rectnet@Cardi 无害性，统一评估脚本。

三个 tag：
  cardi  无害性确认：rectnet 头在 Cardi 847 iphone 配对上训练，valid 65 评估
         （预期几何非主导 → 近 raw；为"Cardi 选 pairloc 不选 rectnet"提供依据）
  eff    Pleural Effusion（银标签阳性 50）：新 teacher（p17）+ 新 rectnet 头
         + 新 pairloc adapter（cardi 的 847 iphone 配对，跨设备训练→oneplus 评估）
  ede    Edema（银标签阳性 41）：同上

银标签 = silver_masks_224.npz（10 病种 × 168 张 oneplus valid 的 224 翻拍空间
掩码，CheXlocalize GT 经配准传播；eff/ede 取 mask.sum()>50 的键）。

pairloc 训练纪律：cardi 直接加载 p08 已训 adapter（与已发布数字一致）；
eff/ede 固定 40 epochs 全量拟合、无 early stop（避免在评估集上做模型选择）。

输出 p18_new_findings_{tag}.csv：key, patient, raw45, teach45, rectnet45,
pairloc45, oracle_dep45 + 控制台 bootstrap 表。
角点缓存 pair_corners_cardi.npz（847 对的 SIFT GT 角点，各 tag 共享）。

用法：/root/miniconda3/bin/python p18_new_findings.py [cardi|eff|ede]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, load_model

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
PH_REG, RATIO, RANSAC_THR, MIN_INLIERS = 800, 0.7, 5.0, 20
SEED = 42
EPOCHS_PL = 40
BS = 32
CORNERS = np.array([[0, 0], [224, 0], [224, 224], [0, 224]], np.float32)
W_FULL = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.5}
CLEAN_TRAIN = f"{BASE}/CheXpert-v1.0-small/train"
CLEAN_VALID = f"{BASE}/CheXpert-v1.0-small/valid"
PHOTO_IPHONE = f"{BASE}/chexphoto/train/natural/iphone"
PHOTO_ONEPLUS = f"{BASE}/chexphoto/valid/natural/oneplus"
CFG = {
    "cardi": {"ckpt": "/root/project/outputs/repaired_seed42/densenet121/best.pt",
              "pair": "pair_cache_cardi", "pair_photo": PHOTO_IPHONE,
              "pair_clean": CLEAN_TRAIN, "eval": "adapt", "finding": None,
              "adapter": "pairloc_adapter_cardi.pt"},
    "atel": {"ckpt": "/root/project/outputs/atel_seed42_densenet121/best.pt",
             "pair": "pair_cache_atel", "pair_photo": PHOTO_ONEPLUS,
             "pair_clean": CLEAN_VALID, "eval": "atel", "finding": None,
             "adapter": "pairloc_adapter_atel.pt"},
    "eff": {"ckpt": "/root/project/outputs/eff_seed42_densenet121/best.pt",
            "pair": "pair_cache_cardi", "pair_photo": PHOTO_IPHONE,
            "pair_clean": CLEAN_TRAIN, "eval": "silver",
            "finding": "Pleural Effusion", "adapter": None},
    "ede": {"ckpt": "/root/project/outputs/ede_seed42_densenet121/best.pt",
            "pair": "pair_cache_cardi", "pair_photo": PHOTO_IPHONE,
            "pair_clean": CLEAN_TRAIN, "eval": "silver", "finding": "Edema",
            "adapter": None},
}


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


def cam_from_feat(feat, w1):
    cam = torch.einsum("k,bkyx->byx", w1, feat).relu_()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                         align_corners=False)[:, 0]


def _grid(M, size, device):
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
    src = np.linalg.inv(M) @ pts
    src = src[:2] / src[2:3]
    grid = src.T.reshape(size, size, 2)
    grid[..., 0] = grid[..., 0] / (size - 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / (size - 1) * 2 - 1
    return torch.from_numpy(grid.astype(np.float32))[None].to(device)


def register_T(k, sift, bf, CLEAN, PHOTO):
    """clean224→photo224 的 T（与 p09 同款）。"""
    parts = k.split("_")
    stem = "_".join(parts[2:])
    cim = Image.open(f"{CLEAN}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
    pim = Image.open(f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg").convert("L")
    cw, ch = cim.size
    pw, ph = pim.size
    s800 = PH_REG / max(pw, ph)
    photo800 = np.asarray(pim.resize((round(pw * s800), round(ph * s800)), Image.BILINEAR))
    kp1, d1 = sift.detectAndCompute(np.asarray(cim), None)
    kp2, d2 = sift.detectAndCompute(photo800, None)
    if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, nn in ((g[0], g[1]) for g in matches if len(g) == 2)
            if m.distance < RATIO * nn.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
    if H is None:
        return None
    H224 = H.copy()
    H224[0] *= 224.0 / (pw * s800)
    H224[1] *= 224.0 / (ph * s800)
    return (H224 @ np.diag([cw / 224.0, ch / 224.0, 1.0])).astype(np.float32)


class CornHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 8))

    def forward(self, x):
        return self.net(x)


VIEWS18 = [("id", None), ("hflip", "hflip"), ("rot+6", ("rot", 6)), ("rot-6", ("rot", -6)),
           ("gamma075", ("gamma", 0.75)), ("gamma133", ("gamma", 1.33)),
           ("bright085", ("bright", 0.85))]


def _apply_view(p224, v):
    name, par = v
    im = p224
    if par == "hflip":
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif isinstance(par, tuple) and par[0] == "rot":
        im = im.rotate(par[1], resample=Image.BILINEAR)
    elif isinstance(par, tuple) and par[0] == "gamma":
        im = Image.fromarray((np.asarray(im, np.float32) / 255.0) ** par[1] * 255
                             ).convert(im.mode)
    elif isinstance(par, tuple) and par[0] == "bright":
        im = Image.fromarray(np.clip(np.asarray(im, np.float32) * par[1], 0, 255)
                             .astype(np.uint8)).convert(im.mode)
    return im


def _cam_binary(cam, frac=0.15):
    th = np.quantile(cam, 1 - frac)
    return (cam >= th).astype(np.uint8)


def _cam_iou(a, b):
    A, B = _cam_binary(a), _cam_binary(b)
    inter, union = (A & B).sum(), (A | B).sum()
    return inter / union if union else 1.0


class Adapter(torch.nn.Module):
    def __init__(self, c=1024, r=16):
        super().__init__()
        self.d1 = torch.nn.Conv2d(c, r, 1)
        self.d2 = torch.nn.Conv2d(r, c, 1, bias=False)
        torch.nn.init.zeros_(self.d2.weight)
        torch.nn.init.normal_(self.d1.weight, std=1e-3)

    def forward(self, f):
        return f + self.d2(self.d1(f))


def train_cornhead(GAP, corners, ok, device):
    pats = GAP["pats"]
    up = sorted(set(pats))
    rng = np.random.default_rng(7)
    val_pat = set(rng.choice(up, size=max(2, len(up) // 5), replace=False).tolist())
    is_val = np.array([p in val_pat for p in pats])
    tr = np.where(ok & ~is_val)[0]
    va = np.where(ok & is_val)[0]
    Xtr = torch.from_numpy(np.asarray(GAP["feat"][tr], np.float32)).to(device)
    Gtr = torch.from_numpy(corners[tr]).to(device)
    Xva = torch.from_numpy(np.asarray(GAP["feat"][va], np.float32)).to(device)
    Gva = torch.from_numpy(corners[va]).to(device)
    head = CornHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best = (1e9, None)
    for ep in range(300):
        head.train()
        perm = np.random.default_rng(SEED * 31 + ep).permutation(len(tr))
        for s0 in range(0, len(tr), 32):
            sel = perm[s0:s0 + 32]
            loss = F.mse_loss(head(Xtr[sel]), Gtr[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vloss = float(F.mse_loss(head(Xva), Gva))
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    log(f"  rectnet 头 val MSE {best[0]:.5f}（≈{np.sqrt(best[0]) * 224:.1f}px）")
    return head


def train_pairloc(w, model, w1, device, reg_ok, f_photo, f_clean, mask_p, prob_t, cam_t):
    torch.manual_seed(SEED + 99)
    ad = Adapter().to(device)
    opt = torch.optim.Adam(ad.parameters(), lr=1e-3, weight_decay=1e-5)
    ok_idx = np.where(reg_ok)[0]
    for ep in range(EPOCHS_PL):
        ad.train()
        perm = np.random.default_rng(SEED * 1000 + ep).permutation(len(ok_idx))
        for s0 in range(0, len(ok_idx), BS):
            sel = ok_idx[perm[s0:s0 + BS]]
            fb = torch.from_numpy(np.asarray(f_photo[sel], np.float32)).to(device)
            fc = torch.from_numpy(np.asarray(f_clean[sel], np.float32)).to(device)
            mk = torch.from_numpy(mask_p[sel]).to(device)
            pt = torch.from_numpy(prob_t[sel]).to(device)
            tgt = torch.from_numpy(cam_t[sel]).to(device)
            fs = ad(fb)
            cam_s = cam_from_feat(fs, w1)
            cam_s_fl = cam_from_feat(ad(torch.flip(fb, dims=[3])), w1)
            cam_t_fl = torch.flip(tgt, dims=[2])
            cam_c = cam_from_feat(ad(fc), w1)
            cam_c_ref = cam_from_feat(fc, w1)
            prob_s = torch.softmax(model.classifier(fs.mean(dim=(2, 3))), 1)[:, 1]
            loss = (w["logit"] * F.mse_loss(prob_s, pt)
                    + w["cam"] * F.mse_loss(cam_s, tgt)
                    + w["geo"] * F.mse_loss(cam_s_fl, cam_t_fl)
                    + w["anat"] * (cam_s * (1 - F.interpolate(
                        mk[:, None], size=IMAGE_SIZE, mode="bilinear",
                        align_corners=False)[:, 0])).mean()
                    + w["ident"] * F.mse_loss(cam_c, cam_c_ref))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    ad.eval()
    return ad


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "cardi"
    cfg = CFG[tag]
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    cam_benchmark.CKPT = cfg["ckpt"]
    model = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    w1 = model.classifier.weight[1].detach()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()

    # ---------- 1) 配对键角点缓存（SIFT 一次，各 tag 共享） ----------
    D = np.load(f"{OUT}/{cfg['pair']}.npz", allow_pickle=True)
    keys = [str(k) for k in D["keys"]]
    mask_p = D["mask_p"].astype(np.float32)
    reg_ok_pair = D["reg_ok"].astype(bool)
    cpath = f"{OUT}/pair_corners_{cfg['pair']}.npz"
    if os.path.exists(cpath):
        C = np.load(cpath, allow_pickle=True)
        assert [str(k) for k in C["keys"]] == keys
        Tpair, ok_pair = C["T"], C["ok"].astype(bool)
        log(f"角点缓存命中（{int(ok_pair.sum())}/{len(keys)}）")
    else:
        Tpair = np.zeros((len(keys), 3, 3), np.float32)
        ok_pair = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            T = register_T(k, sift, bf, cfg["pair_clean"], cfg["pair_photo"])
            if T is not None:
                Tpair[i] = T
                ok_pair[i] = True
            if (i + 1) % 100 == 0:
                log(f"  SIFT {i + 1}/{len(keys)}")
        np.savez_compressed(cpath, keys=np.array(keys), T=Tpair, ok=ok_pair)
        log(f"角点缓存落盘（{int(ok_pair.sum())}/{len(keys)}）")

    # ---------- 2) 评估键与银标签 ----------
    SM = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = [str(x) for x in SM["findings"]]
    if cfg["eval"] in ("adapt", "atel"):
        Dv = np.load(f"{OUT}/cache_photo_{'adapt' if tag == 'cardi' else 'atel'}.npz",
                     allow_pickle=True)
        vkeys = [str(k) for k in Dv["keys"]]
        vgts = {k: Dv["gts"][i] for i, k in enumerate(vkeys)}
    else:
        fi = findings.index(cfg["finding"])
        vgts = {k: SM[k][fi] for k in SM.keys() if k != "findings"
                and SM[k][fi].sum() > 50}
        vkeys = sorted(vgts)

    # ---------- 3) 配对键特征（当前 tag backbone）+ cam_t ----------
    fpath = f"{OUT}/p18_feat_{tag}.npz"
    if os.path.exists(fpath):
        FE = np.load(fpath, allow_pickle=True)
        f_photo, f_clean, prob_t = FE["f_photo"], FE["f_clean"], FE["prob_t"].astype(np.float32)
        f_val = FE["f_val"]
        assert [str(k) for k in FE["vkeys"]] == vkeys, "特征缓存评估键序不一致"
        log(f"特征缓存命中 {fpath}")
    else:
        fp, fc_, pt_ = [], [], []
        for n, k in enumerate(keys):
            parts = k.split("_")
            stem = "_".join(parts[2:])
            pim = Image.open(f"{cfg['pair_photo']}/{parts[0]}/{parts[1]}/{stem}.jpg"
                             ).convert("L")
            cim = Image.open(f"{cfg['pair_clean']}/{parts[0]}/{parts[1]}/{stem}.jpg"
                             ).convert("L")
            xp = tf(pim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            xc = tf(cim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            with torch.no_grad():
                fp.append(model.features(xp.to(device))[0].cpu().numpy().astype(np.float16))
                fcc = model.features(xc.to(device))
                fc_.append(fcc[0].cpu().numpy().astype(np.float16))
                pt_.append(float(torch.softmax(model.classifier(fcc.mean(dim=(2, 3))), 1)[0, 1]))
            if (n + 1) % 200 == 0:
                log(f"  feats pair {n + 1}/{len(keys)}")
        fv = []
        with torch.no_grad():
            for n, k in enumerate(vkeys):
                parts = k.split("_")
                stem = "_".join(parts[2:])
                pim = Image.open(f"{PHOTO_ONEPLUS}/{parts[0]}/{parts[1]}/{stem}.jpg"
                                 ).convert("L")
                x = tf(pim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
                fv.append(model.features(x.to(device))[0].cpu().numpy().astype(np.float16))
                if (n + 1) % 50 == 0:
                    log(f"  feats val {n + 1}/{len(vkeys)}")
        f_photo, f_clean = np.stack(fp), np.stack(fc_)
        prob_t = np.array(pt_, np.float32)
        f_val = np.stack(fv)
        np.savez_compressed(fpath, keys=np.array(keys), f_photo=f_photo,
                            f_clean=f_clean, prob_t=prob_t,
                            vkeys=np.array(vkeys), f_val=f_val)
        log(f"特征缓存落盘 {fpath}")

    # ---------- 3) cam_t（teacher CAM 经配准 T 传播） ----------
    cam_t = np.zeros((len(keys), 224, 224), np.float32)
    with torch.no_grad():
        for i in np.where(ok_pair & reg_ok_pair)[0]:
            fc_i = torch.from_numpy(np.asarray(f_clean[i], np.float32)).to(device)[None]
            cam_c = cam_from_feat(fc_i, w1)[0].cpu().numpy()
            cam_t[i] = cv2.warpPerspective(cam_c, Tpair[i], (224, 224))
    log(f"cam_t 传播 {int((ok_pair & reg_ok_pair).sum())} 对")

    # ---------- 4) rectnet 头（配对 GAP → 角点） ----------
    GAP = {"feat": f_photo.astype(np.float32).mean(axis=(2, 3)),
           "pats": np.array([k.split("_")[0] for k in keys])}
    corners = np.zeros((len(keys), 8), np.float32)
    for i in np.where(ok_pair)[0]:
        pts = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), Tpair[i]).reshape(4, 2)
        corners[i] = (pts / 224.0).reshape(8)
    head = train_cornhead(GAP, corners, ok_pair, device)
    torch.save(head.state_dict(), f"{OUT}/rectnet_head_{tag}.pt")

    # ---------- 5) pairloc adapter ----------
    if cfg["adapter"]:
        ad = Adapter().to(device)
        ad.load_state_dict(torch.load(f"{OUT}/{cfg['adapter']}", map_location=device))
        ad.eval()
        log(f"adapter 加载 {cfg['adapter']}")
    else:
        log("训练 pairloc（full 权重，40 epochs 固定，无 early stop）")
        ad = train_pairloc(W_FULL, model, w1, device, reg_ok_pair, f_photo, f_clean,
                           mask_p, prob_t, cam_t)
        torch.save(ad.state_dict(), f"{OUT}/pairloc_adapter_{tag}.pt")

    # ---------- 6) 评估 ----------
    gts_v = [mask_bbox(vgts[k]) if k in vgts else [] for k in vkeys]
    Hmap = {}
    for t2 in ("cardi", "atel"):
        try:
            Z = np.load(f"{OUT}/p06_p1_cams_{t2}.npz", allow_pickle=True)
            for j, k in enumerate(Z["keys"]):
                Hmap[str(k)] = (Z["H224"][j], bool(Z["reg_ok"][j]))
        except FileNotFoundError:
            pass
    rows = []
    with torch.no_grad():
        for i, k in enumerate(vkeys):
            gt = gts_v[i]
            fv_i = torch.from_numpy(np.asarray(f_val[i], np.float32)).to(device)[None]
            raw_i = iou45(cam_from_feat(fv_i, w1)[0].cpu().numpy(), gt) if gt else np.nan
            pl_i = iou45(cam_from_feat(ad(fv_i), w1)[0].cpu().numpy(), gt) if gt else np.nan
            teach_i = np.nan
            rect_i = np.nan
            if gt:
                # teach：H224 缓存 / SIFT 兜底
                if k in Hmap and Hmap[k][1]:
                    H224 = Hmap[k][0]
                    parts = k.split("_")
                    stem = "_".join(parts[2:])
                    cw, ch = Image.open(
                        f"{CLEAN_VALID}/{parts[0]}/{parts[1]}/{stem}.jpg").size
                    T = H224 @ np.diag([cw / 224.0, ch / 224.0, 1.0])
                else:
                    T = register_T(k, sift, bf, CLEAN_VALID, PHOTO_ONEPLUS)
                if T is not None:
                    cim = Image.open(f"{CLEAN_VALID}/{k.split('_')[0]}/{k.split('_')[1]}/"
                                     f"{'_'.join(k.split('_')[2:])}.jpg").convert("L")
                    xc = tf(cim.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
                    cam_c = cam_from_feat(model.features(xc.to(device)), w1)[0].cpu().numpy()
                    cam_p = cv2.warpPerspective(cam_c, T, (224, 224))
                    teach_i = iou45(np.clip(cam_p, 0, 1), gt)
                # rectnet：预测角点 → 校正 → CAM → 扭回
                pred = head(fv_i.mean(dim=(2, 3))).cpu().numpy().reshape(4, 2) * 224.0
                ph_corners = pred.astype(np.float32)
                M = cv2.getPerspectiveTransform(ph_corners, CORNERS)
                Mback = cv2.getPerspectiveTransform(CORNERS, ph_corners)
                pim = Image.open(f"{PHOTO_ONEPLUS}/{k.split('_')[0]}/{k.split('_')[1]}/"
                                 f"{'_'.join(k.split('_')[2:])}.jpg").convert("L")
                x = tf(pim.resize((224, 224), Image.BILINEAR).convert("RGB"))[None]
                rect = F.grid_sample(x.to(device), _grid(M, 224, device),
                                     align_corners=False)
                fr = model.features(rect)
                cam = torch.einsum("k,bkyx->byx", w1, fr).relu_()
                cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                cam224 = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                       align_corners=False)[:, 0]
                cam_back = F.grid_sample(cam224[:, None], _grid(Mback, 224, device),
                                         align_corners=False)[0, 0].cpu().numpy()
                rect_i = iou45(np.clip(cam_back, 0, 1), gt)
                # 7 视图一致性（p11 同款视图组，视空间比较）
                p224 = pim.resize((224, 224), Image.BILINEAR)
                views_pl, views_rc = [], []
                for v in VIEWS18:
                    im_v = _apply_view(p224, v)
                    xv = tf(im_v.convert("RGB"))[None]
                    fvv = model.features(xv.to(device))
                    views_pl.append(_cam_binary(
                        cam_from_feat(ad(fvv), w1)[0].cpu().numpy()))
                    pred_v = head(fvv.mean(dim=(2, 3))).cpu().numpy().reshape(4, 2) * 224.0
                    pv = pred_v.astype(np.float32)
                    Mv = cv2.getPerspectiveTransform(pv, CORNERS)
                    Mbv = cv2.getPerspectiveTransform(CORNERS, pv)
                    rv = F.grid_sample(xv.to(device), _grid(Mv, 224, device),
                                       align_corners=False)
                    camv = torch.einsum("k,bkyx->byx", w1, model.features(rv)).relu_()
                    camv = camv / camv.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                    camv224 = F.interpolate(camv[:, None], size=IMAGE_SIZE,
                                            mode="bilinear", align_corners=False)[:, 0]
                    cbv = F.grid_sample(camv224[:, None], _grid(Mbv, 224, device),
                                        align_corners=False)[0, 0].cpu().numpy()
                    views_rc.append(_cam_binary(np.clip(cbv, 0, 1)))
                agree_pl = float(np.mean([_cam_iou(views_pl[a], views_pl[b])
                                          for a in range(7) for b in range(a + 1, 7)]))
                agree_rect = float(np.mean([_cam_iou(views_rc[a], views_rc[b])
                                            for a in range(7) for b in range(a + 1, 7)]))
            else:
                agree_pl = agree_rect = np.nan
            rows.append({"key": k, "patient": k.split("_")[0], "raw45": raw_i,
                         "teach45": teach_i, "rectnet45": rect_i, "pairloc45": pl_i,
                         "agree_p_pl": agree_pl, "agree_p_rect": agree_rect})
    df = pd.DataFrame(rows)
    dep = df[["raw45", "pairloc45", "rectnet45"]].to_numpy()
    oracle_dep = np.nanmax(np.where(np.isnan(dep), -np.inf, dep), axis=1)
    oracle_dep[~np.isfinite(oracle_dep)] = np.nan
    df["oracle_dep45"] = oracle_dep
    df["argmax"] = [ ("raw","pairloc","rectnet")[int(np.nanargmax(r))]
                     if np.isfinite(r).any() else "na"
                     for r in np.where(np.isnan(dep), -np.inf, dep) ]
    df.to_csv(f"{OUT}/p18_new_findings_{tag}.csv", index=False)

    def boot_ci(d, key, n=400, seed=42):
        up_ = pd.unique(key)
        fmap_ = pd.Series(key).map({f: i for i, f in enumerate(up_)}).to_numpy()
        r = np.random.default_rng(seed)
        vals = [d[np.isin(fmap_, r.choice(len(up_), len(up_)))].mean() for _ in range(n)]
        return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    print(f"\n== [{tag}] 泛化评估（n={len(df)}） ==")
    for c in ["teach45", "rectnet45", "pairloc45", "oracle_dep45"]:
        m = df[c].notna() & df.raw45.notna()
        d = df.loc[m, c].to_numpy() - df.loc[m, "raw45"].to_numpy()
        b = boot_ci(d, df.loc[m, "patient"].to_numpy())
        wr = float((d < -0.05).mean())
        print(f"  {c:12s} Δ {b[0]:+.4f} CI [{b[1]:+.4f},{b[2]:+.4f}]"
              f"  wrong {wr:.3f}")
    print(f"  argmax 分布: {df.argmax.value_counts().to_dict()}")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
