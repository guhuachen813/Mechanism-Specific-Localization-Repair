r"""P1 探索实验：翻拍域重校准（photo-domain recalibration）。

与零样本主表严格分开报告：65 张 Cardiomegaly 银标签图内部做患者级 5 折 CV——
折内用银标签 GT 收益标签（argmax{0, prior−raw, cand−raw}）训练 GateV2（输入为
翻拍图 q 特征）+ calib δ；折外评估重校准门控。

目的：回答"零样本失败是门控概念失败，还是合成→翻拍分布偏移"。
若域内重校准后 gated 显著超过 max(raw,prior,cand)，说明门控机制本身有效、
缺的是采集语境特征/域适配；若仍不涨，说明翻拍域内"何时用哪个"不可从 q 学出。

输出：chexlocalize/recal_photo.csv + 控制台汇总。
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

import cam_benchmark
from cam_benchmark import (IMAGE_SIZE, MEAN, STD, THRESHOLDS, batch_cam,
                           cam_iou_curve)
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import RepairHead, cam_feats, q_vector
from silver_register import PHOTO, SEG_JSON

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
ART = "/root/autodl-tmp/experiments/rep_robust/repairer_v2_artifacts"
ALPHA = 0.5
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])
SEED = 42


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


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    s = rle["counts"] if isinstance(rle["counts"], str) else None
    if s is None:
        raise ValueError("unexpected counts")
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
    device = torch.device("cuda")
    model = cam_benchmark.load_model(device)
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    heads = []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads.append(h)

    from transformers import SamModel
    smodel = SamModel.from_pretrained(
        "/root/autodl-tmp/experiments/rep_robust/weights/medsam").to(device).eval()

    seg = json.load(open(SEG_JSON))
    z = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = list(z["findings"])
    ci_f = findings.index("Cardiomegaly")
    keys = [k for k in z.files if k != "findings" and z[k][ci_f].sum() > 0]
    log(f"工作集 {len(keys)}")

    # ---- 逐图计算 raw/prior/cand/q 与银标签 IoU ----
    recs = []
    for n, k in enumerate(keys):
        parts = k.split("_")
        stem = "_".join(parts[2:])
        path_p = f"{PHOTO}/{parts[0]}/{parts[1]}/{stem}.jpg"
        photo224 = Image.open(path_p).convert("L").resize((224, 224), Image.BILINEAR)
        m_photo = z[k][ci_f].astype(np.uint8)
        with torch.no_grad():
            cam_p, prob_p = batch_cam(model, tf(photo224.convert("RGB"))[None], device)
        cam_p = cam_p[0]
        fr, ok_r = detect_field(photo224)
        mm = mask_224(segment(smodel, medsam_prep(photo224), fr, device)).astype(np.uint8)
        g_arr = np.asarray(photo224.convert("L"), np.float32) / 255.0
        qv = np.concatenate([q_vector(g_arr, mm, float(ok_r)), cam_feats(cam_p, mm)])
        inp = torch.from_numpy(cam_p[None, None]).to(device)
        mk = torch.from_numpy(mm[None, None].astype(np.float32)).to(device)
        qt = torch.from_numpy(qv[None]).to(device)
        with torch.no_grad():
            cands = [torch.clamp(inp + ALPHA * h(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                     for h in heads]
        cand = np.median(np.stack(cands), 0)
        gt = mask_bbox(m_photo)
        rec = {"key": k, "patient": parts[0],
               "raw45": float(cam_iou_curve(cam_p, gt)[TAU45]),
               "prior45": float(cam_iou_curve(cam_p * mm, gt)[TAU45]),
               "cand45": float(cam_iou_curve(cand, gt)[TAU45])}
        recs.append((rec, qv))
        if (n + 1) % 10 == 0:
            log(f"  {n + 1}/{len(keys)}")

    df = pd.DataFrame([r for r, _ in recs])
    Q = np.stack([q for _, q in recs])
    df.to_csv(f"{OUT}/recal_photo_base.csv", index=False)

    # ---- 患者级 5 折 CV：域内重训三选一门控 ----
    rng = np.random.default_rng(SEED)
    patients = sorted(df.patient.unique())
    perm = rng.permutation(len(patients))
    folds = np.array_split(perm, 5)
    pat_fold = {patients[i]: k for k, idx in enumerate(folds) for i in idx}
    fold_id = df.patient.map(pat_fold).to_numpy()
    y_all = np.stack([np.zeros(len(df)),
                      df.prior45 - df.raw45, df.cand45 - df.raw45]).argmax(0)

    g2 = np.full(len(df), np.nan)
    pick_all = np.full(len(df), -1)
    for k in range(5):
        te = fold_id == k
        tr = ~te
        rng_c = np.random.default_rng(SEED + 100 + k)
        tr_pat = df.patient.to_numpy()[tr]
        pts = np.array(sorted(set(tr_pat)))
        calib_p = set(pts[rng_c.permutation(len(pts))[: max(1, len(pts) // 4)]])
        is_cal = np.array([p in calib_p for p in tr_pat])
        g = GateV2().to(device)
        gopt = torch.optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        Xt = torch.tensor(Q[tr][~is_cal], dtype=torch.float32, device=device)
        yt = torch.tensor(y_all[tr][~is_cal], dtype=torch.long, device=device)
        for ep in range(400):
            gopt.zero_grad(set_to_none=True)
            loss = lossf(g(Xt), yt)
            loss.backward()
            gopt.step()
        g.eval()
        with torch.no_grad():
            pcal = torch.softmax(g(torch.tensor(
                Q[tr][is_cal], dtype=torch.float32, device=device)), 1).cpu().numpy()
        out_cal = {"raw": df.raw45.to_numpy()[tr][is_cal],
                   "prior": df.prior45.to_numpy()[tr][is_cal],
                   "cand": df.cand45.to_numpy()[tr][is_cal]}
        best = (-1e9, 0.0)
        for dl in np.arange(0.0, 0.55, 0.05):
            ch = pcal.argmax(1)
            mg = pcal[np.arange(len(pcal)), ch] - pcal[:, 0]
            pk = ch.copy()
            pk[(ch > 0) & (mg < dl)] = 0
            iou = out_cal["raw"].copy()
            iou[pk == 1] = out_cal["prior"][pk == 1]
            iou[pk == 2] = out_cal["cand"][pk == 2]
            if iou.mean() > best[0]:
                best = (iou.mean(), float(dl))
        d_star = best[1]
        with torch.no_grad():
            pte = torch.softmax(g(torch.tensor(
                Q[te], dtype=torch.float32, device=device)), 1).cpu().numpy()
        ch = pte.argmax(1)
        mg = pte[np.arange(len(pte)), ch] - pte[:, 0]
        pk = ch.copy()
        pk[(ch > 0) & (mg < d_star)] = 0
        out_te = {"raw": df.raw45.to_numpy()[te],
                  "prior": df.prior45.to_numpy()[te],
                  "cand": df.cand45.to_numpy()[te]}
        iou = out_te["raw"].copy()
        iou[pk == 1] = out_te["prior"][pk == 1]
        iou[pk == 2] = out_te["cand"][pk == 2]
        g2[te] = iou
        pick_all[te] = pk
        log(f"fold{k}: δ*={d_star:.2f} gated {iou.mean():.3f} "
            f"(raw {out_te['raw'].mean():.3f})")
    df["recal_45"] = g2
    df["picked"] = pick_all
    df.to_csv(f"{OUT}/recal_photo.csv", index=False)

    print("\n== 域内重校准（患者级 5 折 CV，探索性） ==")
    print(f"  raw     {df.raw45.mean():.3f}")
    print(f"  prior   {df.prior45.mean():.3f}")
    print(f"  cand    {df.cand45.mean():.3f}")
    print(f"  oracle  {df[['raw45','prior45','cand45']].max(1).mean():.3f}")
    print(f"  recal   {df.recal_45.mean():.3f}")
    print(f"  picked 分布: {df.picked.value_counts(normalize=True).reindex([0,1,2]).round(3).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
