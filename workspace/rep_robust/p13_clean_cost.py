r"""P5：no_ident 变体的 clean 域代价验证（NIH 146 图 clean 侧）。

消融发现去掉 identity 损失 photo Δ +0.129→+0.185，但 clean CAM 漂移翻倍
（0.054→0.080）。漂移≠变差：本脚本在 NIH 146 图 clean 上直接量 IoU45：
  full adapter / no_ident adapter / raw 三方对比（Cardi GT 框）。
判读：若 no_ident 的 clean IoU 不降（甚至升），identity 可放宽（权衡表素材）；
若明显降，则 identity 必要，权衡表如实记录。

用法：/root/miniconda3/bin/python p13_clean_cost.py
"""
from __future__ import annotations

import csv
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import MEAN, STD, load_model

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
NIH = "/root/autodl-tmp/nih_cxr14/images"
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iou45(cam, gt):
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    return float(cam_iou_curve(cam, gt)[int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])


class Adapter(torch.nn.Module):
    def __init__(self, c=1024, r=16):
        super().__init__()
        self.d1 = torch.nn.Conv2d(c, r, 1)
        self.d2 = torch.nn.Conv2d(r, c, 1, bias=False)

    def forward(self, f):
        return f + self.d2(self.d1(f))


def load_ad(device):
    ad = Adapter().to(device)
    ad.load_state_dict(torch.load(f"{OUT}/pairloc_adapter_cardi.pt", map_location=device))
    ad.eval()
    return ad


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    model = load_model(device)
    w1 = model.classifier.weight[1].detach()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # 保存的 adapter 只有 full（p08）；no_ident 需重训——直接重训一次（同 p12 no_ident 配置）
    D = np.load(f"{OUT}/pair_cache_cardi.npz", allow_pickle=True)
    reg_ok = D["reg_ok"].astype(bool)
    FE = np.load(f"{OUT}/feat_pair_cardi.npz", allow_pickle=True)
    f_photo, f_clean = FE["f_photo"], FE["f_clean"]
    cam_t = D["cam_t"].astype(np.float32)
    mask_p = D["mask_p"].astype(np.float32)
    prob_t = D["prob_t"].astype(np.float32)
    IMAGE_SIZE = 224
    BS, EPOCHS = 32, 40
    W = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.0}

    def cam_from_feat(feat):
        cam = torch.einsum("k,bkyx->byx", w1, feat).relu_()
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return F.interpolate(cam[:, None], size=IMAGE_SIZE, mode="bilinear",
                             align_corners=False)[:, 0]

    def train(ident_w):
        torch.manual_seed(SEED + 99)
        ad = Adapter().to(device)
        opt = torch.optim.Adam(ad.parameters(), lr=1e-3, weight_decay=1e-5)
        ok_idx = np.where(reg_ok)[0]
        for ep in range(EPOCHS):
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
                cam_s = cam_from_feat(fs)
                cam_s_fl = cam_from_feat(ad(torch.flip(fb, dims=[3])))
                cam_t_fl = torch.flip(tgt, dims=[2])
                cam_c = cam_from_feat(ad(fc))
                cam_c_ref = cam_from_feat(fc)
                prob_s = torch.softmax(model.classifier(fs.mean(dim=(2, 3))), 1)[:, 1]
                loss = (ident_w * F.mse_loss(prob_s, pt)
                        + 1.0 * F.mse_loss(cam_s, tgt)
                        + 0.5 * F.mse_loss(cam_s_fl, cam_t_fl)
                        + 0.1 * (cam_s * (1 - F.interpolate(mk[:, None], size=IMAGE_SIZE,
                                                            mode="bilinear",
                                                            align_corners=False)[:, 0])).mean()
                        + 0.5 * ident_w * F.mse_loss(cam_c, cam_c_ref))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        ad.eval()
        return ad

    log("重训 full（ident=0.5，无早停与 p12 对齐）")
    ad_full = train(1.0)
    log("重训 no_ident（ident=0）")
    ad_noid = train(0.0)

    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == "Cardiomegaly":
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = (x * s, y * s, (x + w) * s, (y + h) * s)

    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
    files = [str(x) for x in C["files"]]
    keys = [k for k in files if k in gt_box]
    log(f"NIH clean 评估 {len(keys)} 图")
    rows = []
    with torch.no_grad():
        for n, k in enumerate(keys):
            im = Image.open(f"{NIH}/{k}").convert("L").resize((224, 224), Image.BILINEAR)
            x = tf(im.convert("RGB"))[None]
            f = model.features(x.to(device))
            cam_r = cam_from_feat(f)[0].cpu().numpy()
            cam_f = cam_from_feat(ad_full(f))[0].cpu().numpy()
            cam_n = cam_from_feat(ad_noid(f))[0].cpu().numpy()
            gt = [gt_box[k]]
            rows.append({"key": k, "patient": k.split("_")[0],
                         "raw": iou45(cam_r, gt), "full": iou45(cam_f, gt),
                         "no_ident": iou45(cam_n, gt)})
            if (n + 1) % 40 == 0:
                log(f"  {n + 1}/{len(keys)}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p13_clean_cost.csv", index=False)

    def boot(d, key, n=400, seed=42):
        up = pd.unique(key)
        fmap = pd.Series(key).map({f: i for i, f in enumerate(up)}).to_numpy()
        r = np.random.default_rng(seed)
        vals = [d[np.isin(fmap, r.choice(len(up), len(up)))].mean() for _ in range(n)]
        return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    print("\n== NIH clean 146 图 IoU45（Cardi） ==")
    print(f"  raw        {df.raw.mean():.4f}")
    for c in ["full", "no_ident"]:
        d = df[c] - df.raw
        m, lo, hi = boot(d.to_numpy(), df.patient.to_numpy())
        print(f"  {c:9s} {df[c].mean():.4f}  Δ {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    d = df.no_ident - df.full
    m, lo, hi = boot(d.to_numpy(), df.patient.to_numpy())
    print(f"  no_ident − full 配对差 {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
