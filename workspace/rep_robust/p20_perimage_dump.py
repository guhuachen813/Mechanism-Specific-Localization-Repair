r"""P20：主结果逐图 IoU 落盘（统计证据包 P15 的输入）。

p12 消融只输出了聚合值；P15 需要 no_ident/full 的**逐图** IoU45 才能算
患者级配对 CI vs raw 与 vs gated_v2。本脚本重训 cardi/atel × full/no_ident
（与 p12/p14 完全同配置同 seed），把 65/73 张 eval 键的逐图 IoU45 落盘。

输出 p20_perimage.csv：tag, key, patient, raw45, full45, noident45
用法：/root/miniconda3/bin/python p20_perimage_dump.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, load_model

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
PHOTO_V = f"{BASE}/chexphoto/valid/natural/oneplus"
SEED = 42
EPOCHS = 40
BS = 32
W_FULL = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.5}
W_NOID = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.0}
TAGS = {"cardi": {"ckpt": "/root/project/outputs/repaired_seed42/densenet121/best.pt",
                  "pair": "pair_cache_cardi", "feat": "feat_pair_cardi",
                  "ev": "cache_photo_adapt", "ref": "p08_pairloc_valid_iou_cardi.npy"},
        "atel": {"ckpt": "/root/project/outputs/atel_seed42_densenet121/best.pt",
                 "pair": "pair_cache_atel", "feat": "feat_pair_atel",
                 "ev": "cache_photo_atel", "ref": "p08_pairloc_valid_iou_atel.npy"}}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    cam = torch.einsum("k,bkyx->byx", w1, feat).relu_()
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                         align_corners=False)[:, 0]


def iou45(cam, gt):
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    return float(cam_iou_curve(cam, gt)[int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])


def train_variant(w, cam_mode, model, w1, reg_ok, f_photo, f_clean, mask_p, prob_t,
                  cam_t, f_val, gt_boxes_v, raw_v, device):
    torch.manual_seed(SEED + 99)
    ad = Adapter().to(device)
    opt = torch.optim.Adam(ad.parameters(), lr=1e-3, weight_decay=1e-5)
    ok_idx = np.where(reg_ok)[0]
    best = (-1e9, None)

    def valid_iou(ad):
        ad.eval()
        with torch.no_grad():
            f = torch.from_numpy(np.asarray(f_val, np.float32)).to(device)
            cam = cam_from_feat(ad(f), w1).cpu().numpy()
        return np.array([iou45(cam[i], gt_boxes_v[i]) if gt_boxes_v[i] else np.nan
                         for i in range(len(f_val))])

    for ep in range(EPOCHS):
        ad.train()
        perm = np.random.default_rng(SEED * 1000 + ep).permutation(len(ok_idx))
        for s0 in range(0, len(ok_idx), BS):
            sel = ok_idx[perm[s0:s0 + BS]]
            fb = torch.from_numpy(np.asarray(f_photo[sel], np.float32)).to(device)
            fc = torch.from_numpy(np.asarray(f_clean[sel], np.float32)).to(device)
            mk = torch.from_numpy(mask_p[sel]).to(device)
            pt = torch.from_numpy(prob_t[sel]).to(device)
            tgt = (torch.from_numpy(cam_t[sel]).to(device) if cam_mode == "reg"
                   else cam_from_feat(fc, w1))
            fs = ad(fb)
            cam_s = cam_from_feat(fs, w1)
            cam_s_fl = cam_from_feat(ad(torch.flip(fb, dims=[3])), w1)
            cam_t_fl = torch.flip(tgt, dims=[2])
            cam_c = cam_from_feat(ad(fc), w1)
            cam_c_ref = cam_from_feat(fc, w1)
            prob_s = torch.softmax(model.classifier(fs.mean(dim=(2, 3))), 1)[:, 1]
            l = {
                "logit": w["logit"] * F.mse_loss(prob_s, pt),
                "cam": w["cam"] * F.mse_loss(cam_s, tgt),
                "geo": w["geo"] * F.mse_loss(cam_s_fl, cam_t_fl),
                "anat": w["anat"] * (cam_s * (1 - F.interpolate(
                    mk[:, None], size=IMAGE_SIZE, mode="bilinear",
                    align_corners=False)[:, 0])).mean(),
                "ident": w["ident"] * F.mse_loss(cam_c, cam_c_ref),
            }
            loss = sum(l.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        iv = valid_iou(ad)
        score = np.nanmean(iv) - np.nanmean(raw_v)
        if score > best[0]:
            best = (float(score), {k: v.detach().clone() for k, v in ad.state_dict().items()})
    ad.load_state_dict(best[1])
    return valid_iou(ad), ad


VIEWS = [("id", None), ("hflip", "hflip"), ("rot+6", ("rot", 6)), ("rot-6", ("rot", -6)),
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


def view_agreement(model, w1, ad, vkeys, device):
    """no_ident adapter 的 7 视图一致性 agree_p 与 shift（vs raw）。"""
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def cam_of_img(p224, use_ad):
        x = tf(p224.convert("RGB"))[None]
        with torch.no_grad():
            f = model.features(x.to(device))
            f = ad(f) if use_ad else f
            cam = torch.einsum("k,bkyx->byx", w1, f).relu_()
            cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            cam = F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                                align_corners=False)[0, 0].cpu().numpy()
        return cam

    agree_p, shift_p = [], []
    for n, k in enumerate(vkeys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        p224 = Image.open(f"{PHOTO_V}/{parts[0]}/{parts[1]}/{stem}.jpg"
                          ).convert("L").resize((224, 224), Image.BILINEAR)
        views = [_cam_binary(cam_of_img(_apply_view(p224, v), True)) for v in VIEWS]
        ag = [_cam_iou(views[a], views[b])
              for a in range(len(views)) for b in range(a + 1, len(views))]
        agree_p.append(float(np.mean(ag)))
        shift_p.append(_cam_iou(cam_of_img(p224, True), cam_of_img(p224, False)))
        if (n + 1) % 40 == 0:
            print(f"    view agree {n + 1}/{len(vkeys)}", flush=True)
    return np.array(agree_p), np.array(shift_p)


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = []
    for tag, cfg in TAGS.items():
        torch.manual_seed(SEED)
        cam_benchmark.CKPT = cfg["ckpt"]
        model = load_model(device)
        for p in model.parameters():
            p.requires_grad_(False)
        w1 = model.classifier.weight[1].detach()
        D = np.load(f"{OUT}/{cfg['pair']}.npz", allow_pickle=True)
        cam_t = D["cam_t"].astype(np.float32)
        mask_p = D["mask_p"].astype(np.float32)
        prob_t = D["prob_t"].astype(np.float32)
        reg_ok = D["reg_ok"].astype(bool)
        FE = np.load(f"{OUT}/{cfg['feat']}.npz", allow_pickle=True)
        f_photo, f_clean, f_val = FE["f_photo"], FE["f_clean"], FE["f_val"]
        Dv = np.load(f"{OUT}/{cfg['ev']}.npz", allow_pickle=True)
        vkeys = [str(k) for k in Dv["keys"]]
        gts_v = Dv["gts"]
        raw_v = np.load(f"{OUT}/{cfg['ref']}")[1]
        gt_boxes_v = [mask_bbox(gts_v[i]) for i in range(len(vkeys))]
        log(f"[{tag}] 训练 full / no_ident")
        iv_full, ad_full = train_variant(W_FULL, "reg", model, w1, reg_ok, f_photo,
                                         f_clean, mask_p, prob_t, cam_t, f_val,
                                         gt_boxes_v, raw_v, device)
        iv_noid, ad_noid = train_variant(W_NOID, "reg", model, w1, reg_ok, f_photo,
                                         f_clean, mask_p, prob_t, cam_t, f_val,
                                         gt_boxes_v, raw_v, device)
        # 7 视图一致性（no_ident adapter，图像级；p11 同款视图组）
        agree_p, shift_p = view_agreement(model, w1, ad_noid, vkeys, device)
        for i, k in enumerate(vkeys):
            rows.append({"tag": tag, "key": k, "patient": k.split("_")[0],
                         "raw45": float(raw_v[i]), "full45": float(iv_full[i]),
                         "noident45": float(iv_noid[i]),
                         "agree_p_noid": float(agree_p[i]), "shift_noid": float(shift_p[i])})
        d = iv_noid - raw_v
        print(f"  [{tag}] no_ident Δ {np.nanmean(d):+.4f}  full Δ "
              f"{np.nanmean(iv_full - raw_v):+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p20_perimage.csv", index=False)
    log(f"完成 → p20_perimage.csv（{len(df)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
