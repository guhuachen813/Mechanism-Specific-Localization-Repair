r"""P5-lite：iPhone1k（iPhone 8，第二设备）无定位 GT 跨设备验证。

数据：CheXphoto train natural/iphone（iPhone 8 手拍，与被扣留的 test natural
同设备同协议），frontal 图；clean 原图取本地 CheXpert-v1.0-small/train。
对照：OnePlus 6 翻拍（silver_v2_cardi.csv 的 65 张，本脚本补算其 frac 分布）。

无定位 GT，因此不报 IoU，只报：
  1. 采集语境守卫行为：field_ok / medsam_ok / frac 分布 / 硬守卫触发率
     （P3 口径：frac>0.6 或 frac<0.1 或 field_ok<0.5 → no-op）
  2. 门控行为：q 特征分布、三路决策（choice/picked）分布、margin
  3. 分类概率漂移 photo−clean（Cardiomegaly 标签=1 的子集，与 OnePlus 对照）

用法：/root/miniconda3/bin/python p5_lite.py [--limit 20]
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cam_benchmark import MEAN, STD, batch_cam, load_model
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import ALPHA, RepairHead, cam_feats, mask_feats, q_vector

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
ART = "/root/autodl-tmp/experiments/rep_robust/repairer_v2_artifacts"
IPHONE = "/root/autodl-tmp/chexphoto/train/natural/iphone"
CLEAN = "/root/autodl-tmp/CheXpert-v1.0-small/train"
CKPT = "/root/project/outputs/repaired_seed42/densenet121/best.pt"
ALPHAA = ALPHA


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GateV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                       torch.nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def eval_photo(photo224, clean224, model, smodel, tf, device,
               heads, gates, d_med) -> dict:
    from torchvision import transforms
    with torch.no_grad():
        cam_p, prob_p = batch_cam(model, tf(photo224.convert("RGB"))[None], device)
        cam_c, prob_c = batch_cam(model, tf(clean224.convert("RGB"))[None], device)
    cam_p, cam_c = cam_p[0], cam_c[0]
    fr, ok_r = detect_field(photo224)
    mm = mask_224(segment(smodel, medsam_prep(photo224), fr, device)).astype(np.uint8)
    frac, ncomp = mask_feats(mm)
    g_arr = np.asarray(photo224.convert("L"), np.float32) / 255.0
    qv = np.concatenate([q_vector(g_arr, mm, float(ok_r)), cam_feats(cam_p, mm)])
    inp = torch.from_numpy(cam_p[None, None]).to(device)
    mk = torch.from_numpy(mm[None, None].astype(np.float32)).to(device)
    qt = torch.from_numpy(qv[None]).to(device)
    with torch.no_grad():
        cands = [torch.clamp(inp + ALPHAA * h(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                 for h in heads]
        cand = np.median(np.stack(cands), 0)
        probs_g = torch.softmax(torch.stack([g(qt) for g in gates]), dim=2).cpu().numpy()
    pg = np.median(probs_g[:, 0], 0)
    choice = int(pg.argmax())
    margin = float(pg[choice] - pg[0])
    pick = choice if (choice == 0 or margin >= d_med) else 0
    return {
        "field_ok": int(ok_r), "medsam_ok": int(mm.sum() > 0),
        "frac": float(frac), "ncomp": int(ncomp),
        "q_field_ok": float(qv[0]), "q_b": float(qv[3]), "q_c": float(qv[4]),
        "cam_inside": float(qv[7]),
        "choice": choice, "picked": pick, "margin": margin,
        "g_prior": float(pg[1]), "g_cand": float(pg[2]),
        "prob_photo": float(prob_p[0]), "prob_clean": float(prob_c[0]),
        "cand_mean": float(cand.mean()),
    }


def load_system(device):
    import cam_benchmark
    import pathlib
    cam_benchmark.CKPT = pathlib.Path(CKPT)
    model = load_model(device)
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    heads, gates, deltas = [], [], []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads.append(h)
        g = GateV2().to(device)
        g.load_state_dict(torch.load(f"{ART}/fold{kf}_gate.pt", map_location=device))
        g.eval()
        gates.append(g)
        deltas.append(float(np.load(f"{ART}/fold{kf}_delta.npy")[0]))
    d_med = float(np.median(deltas))
    from transformers import SamModel
    smodel = SamModel.from_pretrained(
        "/root/autodl-tmp/experiments/rep_robust/weights/medsam").to(device).eval()
    return model, smodel, tf, heads, gates, d_med


def summarize(df: pd.DataFrame, tag: str, oneplus: pd.DataFrame | None) -> None:
    print(f"\n== [{tag}] n={len(df)} ==")
    print(f"  field_ok {(df.field_ok == 1).mean():.3f}  medsam_ok {(df.medsam_ok == 1).mean():.3f}")
    f = df.frac
    print(f"  frac median {f.median():.3f}  IQR [{f.quantile(.25):.3f},{f.quantile(.75):.3f}]"
          f"  范围 [{f.min():.3f},{f.max():.3f}]")
    guard = ((f > 0.6) | (f < 0.1) | (df.field_ok < 0.5))
    print(f"  硬守卫触发率 {guard.mean():.3f}（frac>0.6: {(f > 0.6).mean():.3f},"
          f" frac<0.1: {(f < 0.1).mean():.3f}, field_ok<0.5: {(df.field_ok < 0.5).mean():.3f}）")
    print(f"  choice 分布 raw/prior/cand: "
          f"{df.choice.value_counts(normalize=True).reindex([0,1,2]).fillna(0).round(3).to_dict()}")
    print(f"  picked 分布 raw/prior/cand: "
          f"{df.picked.value_counts(normalize=True).reindex([0,1,2]).fillna(0).round(3).to_dict()}")
    print(f"  margin median {df.margin.median():.3f}  IQR [{df.margin.quantile(.25):.3f},{df.margin.quantile(.75):.3f}]")
    d = df.prob_photo - df.prob_clean
    print(f"  prob 漂移 photo−clean mean {d.mean():+.3f} median {d.median():+.3f}")
    if oneplus is not None:
        f1 = oneplus.frac
        g1 = ((f1 > 0.6) | (f1 < 0.1) | (oneplus.field_ok_photo < 0.5))
        d1 = oneplus.prob_photo - oneplus.prob_clean
        print(f"  -- 对照 OnePlus 6 (n={len(oneplus)}): field_ok {(oneplus.field_ok_photo == 1).mean():.3f},"
              f" frac median {f1.median():.3f}, 守卫 {g1.mean():.3f},"
              f" prob 漂移 mean {d1.mean():+.3f} median {d1.median():+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")

    # ---- OnePlus 65 张补算 frac（复用 silver_v2_cardi.csv 的图集） ----
    oneplus = None
    try:
        op = pd.read_csv(f"{OUT}/silver_v2_cardi.csv")
        log(f"OnePlus 对照 {len(op)} 张，补算 frac…")
        model, smodel, tf, heads, gates, d_med = load_system(device)
        fr_vals, fok = [], []
        for n, k in enumerate(op.key):
            parts = k.split("_")
            stem = "_".join(parts[2:])
            path_p = f"/root/autodl-tmp/chexphoto/valid/natural/oneplus/{parts[0]}/{parts[1]}/{stem}.jpg"
            p224 = Image.open(path_p).convert("L").resize((224, 224), Image.BILINEAR)
            fr, ok_r = detect_field(p224)
            mm = mask_224(segment(smodel, medsam_prep(p224), fr, device)).astype(np.uint8)
            fv, _ = mask_feats(mm)
            fr_vals.append(float(fv)); fok.append(int(ok_r))
            if (n + 1) % 20 == 0:
                log(f"  oneplus {n + 1}/{len(op)}")
        op["frac"] = fr_vals
        op["field_ok_photo"] = fok
        oneplus = op
    except Exception as e:  # noqa: BLE001
        log(f"OnePlus 对照失败（跳过）：{e}")

    # ---- iPhone1k ----
    files = sorted(glob.glob(f"{IPHONE}/patient*/study*/view*_frontal.jpg"))
    log(f"iPhone frontal {len(files)}")
    labels = pd.read_csv("/root/autodl-tmp/CheXpert-v1.0-small/train.csv")
    lab = {}
    for r in labels.itertuples():
        p = r.Path.split("train/")[1]
        lab[p] = float(r.Cardiomegaly) if r.Cardiomegaly in (0.0, 1.0) else np.nan
    if args.limit:
        files = files[: args.limit]

    rows = []
    for n, fp in enumerate(files):
        rel = fp.split("natural/iphone/")[1]          # patient/study/view_frontal.jpg
        cp = f"{CLEAN}/{rel}"
        try:
            photo224 = Image.open(fp).convert("L").resize((224, 224), Image.BILINEAR)
            clean224 = Image.open(cp).convert("L").resize((224, 224), Image.BILINEAR)
            rec = eval_photo(photo224, clean224, model, smodel, tf, device,
                             heads, gates, d_med)
            rec["key"] = rel
            rec["cardi_label"] = lab.get(rel, np.nan)
            rows.append(rec)
        except Exception as e:  # noqa: BLE001
            rows.append({"key": rel, "err": f"{type(e).__name__}: {e}"[:120]})
        if (n + 1) % 50 == 0:
            log(f"  iphone {n + 1}/{len(files)}")
    df = pd.DataFrame(rows)
    if "err" in df.columns and df.err.notna().any():
        log(f"err {int(df.err.notna().sum())}")
        df = df[df.err.isna()]
    df.to_csv(f"{OUT}/p5_lite_iphone.csv", index=False)

    summarize(df, "iPhone 8 (iPhone1k frontal)", oneplus)
    cardi = df[df.cardi_label == 1]
    if len(cardi):
        summarize(cardi, "iPhone 8 ∩ Cardiomegaly=1", oneplus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
