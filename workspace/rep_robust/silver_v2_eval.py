r"""优先级 1：CheXphoto 自然翻拍银标签上零样本评估 v2 全系统。

零样本协议：65 张 Cardiomegaly 银标签翻拍从不参与合成轴训练/校准。
载入 repairer_v2_artifacts 的每折头/门控/δ，对翻拍图做 5 头中位集成 + 5 门控
中位概率（不引入翻拍信息）。

系统：raw / prior hard（翻拍图 MedSAM mask）/ cand / gated_v2 / center_field。
指标：τ45 / best / BoxAcc@0.25 / 0.5 / prob / 配对 rescue / no-op 比例。
输出：chexlocalize/silver_v2_{tag}.csv + 控制台主表。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cam_benchmark import (IMAGE_SIZE, MEAN, STD, THRESHOLDS, batch_cam,
                           cam_iou_curve, load_model)
from cam_crop_baselines import center_mask_field
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import cam_feats, q_vector  # 与合成轴逐位一致
from silver_register import PHOTO, SEG_JSON

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
ART = "/root/autodl-tmp/experiments/rep_robust/repairer_v2_artifacts"
ALPHA = 0.5
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


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


def mask_bbox(m):
    if m.sum() == 0:
        return []
    ys, xs = np.where(m > 0)
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finding", default="Cardiomegaly")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="cardi")
    ap.add_argument("--artdir", default=ART)
    args = ap.parse_args()
    FINDING = args.finding
    device = torch.device("cuda")
    import cam_benchmark
    import pathlib
    cam_benchmark.CKPT = pathlib.Path(args.ckpt)
    model = load_model(device)
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    heads, gates, deltas = [], [], []
    from repairer import RepairHead
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{args.artdir}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads.append(h)
        g = GateV2().to(device)
        g.load_state_dict(torch.load(f"{args.artdir}/fold{kf}_gate.pt", map_location=device))
        g.eval()
        gates.append(g)
        deltas.append(float(np.load(f"{args.artdir}/fold{kf}_delta.npy")[0]))
    log(f"载入 5 折头/门控/δ（δ={deltas}）")

    log("加载 MedSAM")
    from transformers import SamModel
    smodel = SamModel.from_pretrained(
        "/root/autodl-tmp/experiments/rep_robust/weights/medsam").to(device).eval()

    seg = json.load(open(SEG_JSON))
    z = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = list(z["findings"])
    ci_f = findings.index(FINDING)
    keys = [k for k in z.files if k != "findings" and z[k][ci_f].sum() > 0]
    log(f"{FINDING} 银标签工作集 {len(keys)}（零样本）")

    reg = pd.read_csv(f"{OUT}/silver_register.csv").set_index("key")
    rows = []
    for n, k in enumerate(keys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        path_c = f"/root/autodl-tmp/CheXpert-v1.0-small/valid/{parts[0]}/{parts[1]}/{stem}.jpg"
        path_p = f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
        try:
            clean_im = Image.open(path_c).convert("L")
            photo_im = Image.open(path_p).convert("L")
            clean224 = clean_im.resize((224, 224), Image.BILINEAR)
            photo224 = photo_im.resize((224, 224), Image.BILINEAR)

            m_photo = z[k][ci_f].astype(np.uint8)
            rle = seg[k].get(FINDING)
            m_clean_full = rle_decode(rle)
            cw, ch = clean_im.size
            mi = Image.fromarray(m_clean_full * 255).resize((cw, ch), Image.BILINEAR)
            m_clean0 = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            mi2 = Image.fromarray(m_clean0 * 255).resize((224, 224), Image.BILINEAR)
            m_clean = (np.asarray(mi2, np.float32) / 255.0 > 0.5).astype(np.uint8)

            with torch.no_grad():
                cam_c, prob_c = batch_cam(model, tf(clean224.convert("RGB"))[None], device)
                cam_p, prob_p = batch_cam(model, tf(photo224.convert("RGB"))[None], device)
            cam_c, cam_p = cam_c[0], cam_p[0]

            fr, ok_r = detect_field(photo224)
            mm = mask_224(segment(smodel, medsam_prep(photo224), fr, device)
                          ).astype(np.uint8)
            g_arr = np.asarray(photo224.convert("L"), np.float32) / 255.0
            qv = np.concatenate([q_vector(g_arr, mm, float(ok_r)),
                                 cam_feats(cam_p, mm)])
            cf = center_mask_field(fr) if ok_r else np.ones((224, 224), np.float32)

            # 5 头集成（中位）+ 5 门控中位概率
            inp = torch.from_numpy(cam_p[None, None]).to(device)
            mk = torch.from_numpy(mm[None, None].astype(np.float32)).to(device)
            qt = torch.from_numpy(qv[None]).to(device)
            with torch.no_grad():
                cands = [torch.clamp(inp + ALPHA * h(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                         for h in heads]
                cand = np.median(np.stack(cands), 0)
                probs_g = torch.softmax(torch.stack(
                    [g(qt) for g in gates]), dim=2).cpu().numpy()
            pg = np.median(probs_g[:, 0], 0)          # (3,)
            choice = int(pg.argmax())
            margin = float(pg[choice] - pg[0])
            d_med = float(np.median(deltas))
            pick = choice if (choice == 0 or margin >= d_med) else 0

            variants = {
                "raw": cam_p, "prior": cam_p * mm, "cand": cand,
                "center_field": cam_p * cf,
            }
            variants["gated_v2"] = {"raw": cam_p, "prior": cam_p * mm,
                                    "cand": cand}[["raw", "prior", "cand"][pick]]

            gt_p = mask_bbox(m_photo)
            gt_c = mask_bbox(m_clean)
            rec = {"key": k, "field_ok_photo": int(ok_r),
                   "medsam_ok": int(mm.sum() > 0),
                   "reg_ok": int(reg.loc[k, "ok"]) if k in reg.index else -1,
                   "reg_mederr": float(reg.loc[k, "med_reproj"]) if k in reg.index else np.nan,
                   "prob_clean": float(prob_c[0]), "prob_photo": float(prob_p[0]),
                   "choice": choice, "picked": pick, "margin": margin,
                   "g_prior": float(pg[1]), "g_cand": float(pg[2])}
            for v, cam in variants.items():
                curve = cam_iou_curve(cam, gt_p)
                rec[f"{v}_45"] = float(curve[TAU45])
                rec[f"{v}_best"] = float(max(curve))
                acc = np.array(curve)
                rec[f"{v}_ba25"] = float(acc[THRESHOLDS <= 0.25].max())
                rec[f"{v}_ba50"] = float(acc[THRESHOLDS <= 0.5].max())
            rec["clean_raw_45"] = float(cam_iou_curve(cam_c, gt_c)[TAU45])
            rec["clean_raw_best"] = float(max(cam_iou_curve(cam_c, gt_c)))
            rows.append(rec)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"key": k, "err": f"{type(e).__name__}: {e}"[:120]})
            if len(rows) > 2:
                break
        if (n + 1) % 10 == 0:
            log(f"  {n + 1}/{len(keys)}")

    df = pd.DataFrame(rows)
    if "err" in df.columns and df.err.notna().any():
        log(f"err {int(df.err.notna().sum())}")
        df = df[df.err.isna()]
    df.to_csv(f"{OUT}/silver_v2_{args.tag}.csv", index=False)

    print(f"\n== 零样本主表（{FINDING}，n={len(df)}，τ45/best/BA25/BA50） ==")
    sys_cols = ["raw", "prior", "cand", "gated_v2", "center_field"]
    for v in sys_cols:
        print(f"  {v:12s} {df[f'{v}_45'].mean():.3f} / {df[f'{v}_best'].mean():.3f}"
              f" / {df[f'{v}_ba25'].mean():.3f} / {df[f'{v}_ba50'].mean():.3f}")
    print(f"  clean_raw    {df.clean_raw_45.mean():.3f} / {df.clean_raw_best.mean():.3f}")
    rng = np.random.default_rng(42)
    uniq = df.key.unique()
    fmap = df.key.map({f: i for i, f in enumerate(uniq)}).to_numpy()
    for v in sys_cols:
        d = df[f"{v}_45"] - df.raw_45
        vals = [d[np.isin(fmap, rng.choice(len(uniq), len(uniq)))].mean()
                for _ in range(400)]
        print(f"  {v:12s} vs raw: {d.mean():+.3f} CI95 [{np.percentile(vals,2.5):+.3f},"
              f"{np.percentile(vals,97.5):+.3f}]")
    print(f"\nno-op（picked=raw）比例: {(df.picked==0).mean():.3f}")
    print(f"choice 分布: {df.choice.value_counts(normalize=True).reindex([0,1,2]).round(3).to_dict()}")
    print(f"prob 漂移 photo−clean: mean {(df.prob_photo-df.prob_clean).mean():+.3f} "
          f"median {(df.prob_photo-df.prob_clean).median():+.3f}")
    print("\n== registration_ok 分层（τ45） ==")
    print(df.groupby("reg_ok")[[f"{v}_45" for v in sys_cols] + ["raw_45"]]
          .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
