r"""第六阶段 6.5：Atelectasis 第二病种复制 —— SAGE-Loc 管线全迁移。

步骤（三个子命令）：
  cache ：photo Atel 73 图缓存（cams_p/masks_p/q9/gts/probs，同 p5a 协议）
  test  ：test 外测特征（Atel ckpt，GT=Atelectasis 面积>0 frontal，p61 同款循环）
  sage  ：GM(Atel) 训练+校准 → 外测 sage/g2 → s(x) 双域 → κ 扫描

用法：/root/miniconda3/bin/python p65_atel.py cache|test|sage
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKCONFIG", ":4096:8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import (MEAN, STD, THRESHOLDS, batch_cam, cam_iou_curve,
                           load_model)
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import cam_feats, q_vector
from silver_eval_test import OUT, SEG_TEST, TEST_DIR, rle_decode

ROOT = "/root/autodl-tmp/experiments/rep_robust"
ROOTA = "/root/autodl-tmp/experiments/rep_robust_atel"
ARTA = f"{ROOTA}/repairer_v2_artifacts_atel"
BASE = "/root/autodl-tmp"
PHOTO_DIR = f"{BASE}/chexphoto/valid/natural/oneplus"
FINDING = "Atelectasis"
ALPHA = 0.5
SEED = 42
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


class RepairHead(torch.nn.Module):
    def __init__(self, nq: int = 9):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3 + nq, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, hraw, mask, q):
        B, _, H, W = hraw.shape
        qb = q[:, :, None, None].expand(B, q.shape[1], H, W)
        x = torch.cat([hraw, mask, hraw * mask, qb], dim=1)
        return torch.tanh(self.net(x))


class GateV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                       torch.nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def iou45(cam, gt):
    return float(cam_iou_curve(cam, gt)[TAU45])


def base_feat(qv9, prob):
    frac, fok, conf = float(qv9[1]), float(qv9[0]), float(qv9[8])
    return np.concatenate([qv9, [frac, fok, conf, float(prob)]]).astype(np.float32)


def gain_nll_train(X, y, seed, epochs=200, bs=64):
    X = np.asarray(X, np.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 32), torch.nn.ReLU())
    mu_h = torch.nn.Linear(32, 1)
    lv_h = torch.nn.Linear(32, 1)
    params = list(net.parameters()) + list(mu_h.parameters()) + list(lv_h.parameters())
    opt = torch.optim.Adam(params, lr=2e-3, weight_decay=1e-4)
    n = len(y)
    for ep in range(epochs):
        perm = np.random.default_rng(seed * 1000 + ep).permutation(n)
        for s0 in range(0, n, bs):
            sel = perm[s0:s0 + bs]
            xb = torch.from_numpy(X[sel])
            yb = torch.from_numpy(y[sel].astype(np.float32))[:, None]
            h = net(xb)
            mu = mu_h(h)
            logvar = lv_h(h).clamp(-4.0, 2.0)
            loss = 0.5 * (logvar + (yb - mu) ** 2 / torch.exp(logvar)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return (net, mu_h, lv_h)


def gain_predict(models, X):
    net, mu_h, lv_h = models
    with torch.no_grad():
        h = net(torch.from_numpy(np.asarray(X, np.float32)))
        mu = mu_h(h)[:, 0].numpy()
        sig = np.exp(0.5 * lv_h(h)[:, 0].clamp(-4.0, 2.0).numpy())
    return mu, sig


def sage_row(qv9, prob, GM, lam, eps, mrg):
    b = base_feat(qv9, prob)
    Xa = np.stack([np.concatenate([b, np.eye(3, dtype=np.float32)[a]])
                   for a in range(3)])
    mus, sgs = [], []
    for a in range(3):
        mu, sig = gain_predict(GM[a], Xa[a][None])
        mus.append(float(mu[0]))
        sgs.append(float(sig[0]))
    L = np.array(mus) - lam * np.array(sgs)
    a_star = int(np.argmax(L))
    pick = 0
    if a_star != 0 and L[a_star] > eps and mus[a_star] - mus[0] >= mrg:
        pick = a_star
    return pick


def load_atel_model(device):
    cam_benchmark.CKPT = Path(f"/root/project/outputs/atel_seed42_densenet121/best.pt")
    return load_model(device)


# ---------------- 子命令：cache ----------------
def cmd_cache() -> int:
    device = torch.device("cuda")
    model = load_atel_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    log("加载 MedSAM")
    from transformers import SamModel
    smodel = SamModel.from_pretrained(f"{ROOT}/weights/medsam").to(device).eval()
    z = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = list(z["findings"])
    ci = findings.index(FINDING)
    keys = sorted(k for k in z.files if k != "findings" and z[k][ci].sum() > 0)
    log(f"工作集 {len(keys)} 键（{FINDING} 非空）")
    cams_p, masks_p, q9s, gts, probs, keys_ok = [], [], [], [], [], []
    for n, k in enumerate(keys):
        try:
            parts = k.split("_")
            stem = "_".join(parts[2:])
            photo_im = Image.open(f"{PHOTO_DIR}/{parts[0]}/{parts[1]}/{stem}.jpg"
                                  ).convert("L")
            m224 = z[k][ci].astype(np.uint8)
            x = tf(photo_im.convert("RGB").resize((224, 224), Image.BILINEAR))[None]
            with torch.no_grad():
                cam, prob = batch_cam(model, x, device)
            cam = cam[0]
            fr, ok_r = detect_field(photo_im.resize((224, 224), Image.BILINEAR))
            mm = mask_224(segment(smodel, medsam_prep(photo_im), fr, device)
                          ).astype(np.uint8)
            g_arr = np.asarray(photo_im.resize((224, 224), Image.BILINEAR).convert("L"),
                               np.float32) / 255.0
            qv = np.concatenate([q_vector(g_arr, mm, float(ok_r)), cam_feats(cam, mm)])
            cams_p.append(cam.astype(np.float32))
            masks_p.append(mm)
            q9s.append(qv)
            gts.append(m224)
            probs.append(float(prob[0]))
            keys_ok.append(k)
        except Exception as e:  # noqa: BLE001
            log(f"err {k}: {e}")
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(keys)}")
    np.savez_compressed(f"{OUT}/cache_photo_atel.npz", keys=np.array(keys_ok),
                        cams_p=np.stack(cams_p), masks_p=np.stack(masks_p),
                        q9=np.stack(q9s), gts=np.stack(gts), probs=np.array(probs))
    log(f"cache_photo_atel.npz 完成：{len(keys_ok)} 键")
    return 0


# ---------------- 子命令：test ----------------
def cmd_test() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    model = load_atel_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    from transformers import SamModel
    smodel = SamModel.from_pretrained(f"{ROOT}/weights/medsam").to(device).eval()
    heads_v2 = []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ARTA}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)
    gates_v2 = []
    for kf in range(5):
        g = GateV2().to(device)
        g.load_state_dict(torch.load(f"{ARTA}/fold{kf}_gate.pt", map_location=device))
        g.eval()
        gates_v2.append(g)
    d_med = float(np.median([float(np.load(f"{ARTA}/fold{kf}_delta.npy")[0])
                             for kf in range(5)]))

    seg = json.load(open(SEG_TEST))
    keys = []
    for k in seg:
        if not k.endswith("_frontal") or FINDING not in seg[k]:
            continue
        rle = seg[k][FINDING]
        if not rle.get("counts"):
            continue
        m_full = rle_decode(rle)
        if m_full.sum() == 0:
            continue
        keys.append(k)
    keys = sorted(keys)
    log(f"Atel 外测工作集 {len(keys)} 图 / {len(set(k.split('_')[0] for k in keys))} 患者")

    rows, cams_out, masks_out, qv_out = [], [], [], []
    for n, k in enumerate(keys):
        path = f"{TEST_DIR}/{k.replace('_study', '/study').replace('_view', '/view')}.jpg"
        try:
            im224 = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
            m_full = rle_decode(seg[k][FINDING])
            w, h = im224.size
            mi = Image.fromarray(m_full * 255).resize((w, h), Image.BILINEAR)
            m0 = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
            mi2 = Image.fromarray(m0 * 255).resize((224, 224), Image.BILINEAR)
            m224 = (np.asarray(mi2, np.float32) / 255.0 > 0.5).astype(np.uint8)
            gt = mask_bbox(m224)
            conds_t = [("clean", im224)] + [
                (c, __import__("degradations", fromlist=["degrade"]).degrade(
                    im224, c,
                    __import__("degradations", fromlist=["rng_for"]).rng_for(
                        SEED, zlib.crc32((k + c).encode()))))
                for c in __import__("degradations", fromlist=["CONDITION_ORDER"]
                                    ).CONDITION_ORDER]
            for cond, deg in conds_t:
                x = tf(deg.convert("RGB"))[None]
                with torch.no_grad():
                    cam, prob = batch_cam(model, x, device)
                cam = cam[0]
                fr, ok_r = detect_field(deg)
                mm = mask_224(segment(smodel, medsam_prep(deg), fr, device)
                              ).astype(np.uint8)
                g_arr = np.asarray(deg.convert("L"), np.float32) / 255.0
                qv = np.concatenate([q_vector(g_arr, mm, float(ok_r)),
                                     cam_feats(cam, mm)])
                inp = torch.from_numpy(cam[None, None]).to(device)
                mk = torch.from_numpy(mm[None, None].astype(np.float32)).to(device)
                qt = torch.from_numpy(qv[None]).to(device)
                with torch.no_grad():
                    cs_zs = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                             for hh in heads_v2]
                    cand_zs = np.median(np.stack(cs_zs), 0)
                    pg = np.median(torch.softmax(torch.stack(
                        [g(qt) for g in gates_v2]), dim=2).cpu().numpy()[:, 0], 0)
                prior = cam * mm
                r_i, p_i, c_i = iou45(cam, gt), iou45(prior, gt), iou45(cand_zs, gt)
                ch = int(pg.argmax())
                pick2 = ch if (ch == 0 or pg[ch] - pg[0] >= d_med) else 0
                g2_i = {0: r_i, 1: p_i, 2: c_i}[pick2]
                rows.append({
                    "key": k, "patient": k.split("_")[0], "cond": cond,
                    "prob": float(prob[0]), "field_ok": int(ok_r),
                    "raw45": r_i, "prior45": p_i, "cand_zs45": c_i,
                    "gated_v2_45": g2_i, "g2_pick": pick2,
                    "oracle45": max(r_i, p_i, c_i),
                    "frac": float(qv[1]), "conflict": float(qv[8]),
                    "gt_area": float(m224.mean())})
                cams_out.append(cam.astype(np.float16))
                masks_out.append(mm.astype(np.uint8))
                qv_out.append(qv.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            log(f"err {k}: {e}")
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(keys)}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p65_atel_test.csv", index=False)
    np.savez_compressed(f"{OUT}/p65_atel_cams.npz",
                        cam=np.stack(cams_out), mask=np.stack(masks_out),
                        qv=np.stack(qv_out),
                        rowkey=np.array([f"{r.key}|{r.cond}" for r in df.itertuples()]))
    print("\n== Atel 外测（τ45） ==")
    print(df.groupby("cond")[["raw45", "prior45", "cand_zs45", "gated_v2_45",
                              "oracle45"]].mean().round(3).to_string())
    rng = np.random.default_rng(42)
    up = df.patient.unique()
    fmap = df.patient.map({f: i for i, f in enumerate(up)}).to_numpy()
    print("\n== vs raw（患者整群 bootstrap 400） ==")
    for v in ["prior45", "cand_zs45", "gated_v2_45", "oracle45"]:
        d = df[v] - df.raw45
        vals = [d[np.isin(fmap, rng.choice(len(up), len(up)))].mean() for _ in range(400)]
        print(f"  {v:12s} {d.mean():+.4f} CI [{np.percentile(vals, 2.5):+.4f},"
              f"{np.percentile(vals, 97.5):+.4f}]")
    return 0


# ---------------- 子命令：sage ----------------
def cmd_sage() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    C = np.load(f"{ROOTA}/repairer/cache.npz", allow_pickle=True)
    cams_n = C["cams"].astype(np.float32)
    masks_n = C["masks"]
    q9_n3 = np.concatenate([C["qs"], C["cs"]], 2)
    probs_n = C["probs"]
    files = list(C["files"])
    conds = list(C["conds"])
    NI, NC = cams_n.shape[:2]

    heads_v2 = []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ARTA}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)

    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == FINDING:
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = [(x * s, y * s, (x + w) * s, (y + h) * s)]
    log(f"Atel GT 框 {len(gt_box)} 图")

    X_rows, Y_rows, KV = [], [], []
    with torch.no_grad():
        for i in range(NI):
            if files[i] not in gt_box:
                continue
            gt = gt_box[files[i]]
            for c in range(NC):
                inp = torch.from_numpy(cams_n[i, c][None, None]).to(device)
                mk = torch.from_numpy(masks_n[i, c][None, None].astype(np.float32)).to(device)
                q9 = q9_n3[i, c].astype(np.float32)
                qt = torch.from_numpy(q9[None]).to(device)
                cs_zs = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                         for hh in heads_v2]
                cand = np.median(np.stack(cs_zs), 0)
                r_i = iou45(cams_n[i, c], gt)
                p_i = iou45(cams_n[i, c] * masks_n[i, c], gt)
                c_i = iou45(cand, gt)
                b = base_feat(q9, float(probs_n[i, c]))
                oneh = np.eye(3, dtype=np.float32)
                for a in range(3):
                    X_rows.append(np.concatenate([b, oneh[a]]))
                Y_rows.append([0.0, p_i - r_i, c_i - r_i])
                KV.append((i, c))
    X_n = np.stack(X_rows)
    Y_n = np.array(Y_rows, np.float32)
    AID_n = np.tile(np.arange(3), len(Y_n))
    GM = [gain_nll_train(X_n[AID_n == a], Y_n[:, a], SEED + 99 + a) for a in range(3)]
    log(f"Atel GM 训练完成（样本 {len(Y_n)}）")

    # 校准（与 p61 同协议：患者 20% seed7）
    kv_pat = np.array([files[int(i)].split("_")[0] for (i, c) in KV])
    upats = sorted(set(kv_pat))
    rng_c = np.random.default_rng(7)
    cal_pat = set(rng_c.choice(upats, size=max(4, len(upats) // 5), replace=False).tolist())
    is_cal = np.isin(kv_pat, list(cal_pat))
    N_n = len(Y_n)
    MU3 = np.zeros((N_n, 3), np.float32)
    SG3 = np.zeros((N_n, 3), np.float32)
    for a in range(3):
        mu, sig = gain_predict(GM[a], X_n[AID_n == a])
        MU3[:, a] = mu
        SG3[:, a] = sig
    IoU3n = np.zeros((N_n, 3), np.float32)
    IoU3n[:, 1] = Y_n[:, 1]
    IoU3n[:, 2] = Y_n[:, 2]
    best = (-1e9, None)
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for eps in [-0.02, 0.0, 0.02, 0.05]:
            for mrg in [0.0, 0.02, 0.05]:
                idx = np.where(is_cal)[0]
                L = MU3[idx] - lam * SG3[idx]
                picks = np.zeros(len(idx), np.int64)
                for j in range(len(idx)):
                    a = int(np.argmax(L[j]))
                    if a != 0 and L[j, a] > eps and MU3[idx[j], a] - MU3[idx[j], 0] >= mrg:
                        picks[j] = a
                g = IoU3n[idx][np.arange(len(idx)), picks] - IoU3n[idx, 0]
                score = g.mean() - 0.05 * (g < -0.05).mean()
                if score > best[0]:
                    best = (score, (lam, eps, mrg))
    lam, eps, mrg = best[1]
    log(f"Atel 校准 λ={lam} ε={eps} m={mrg}")

    # 外测应用
    dft = pd.read_csv(f"{OUT}/p65_atel_test.csv")
    E = np.load(f"{OUT}/p65_atel_cams.npz", allow_pickle=True)
    qv_ext = E["qv"]
    picks_t = np.zeros(len(dft), np.int64)
    mus_t = np.zeros((len(dft), 3))
    for j in range(len(dft)):
        pick = sage_row(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]),
                        GM, lam, eps, mrg)
        picks_t[j] = pick
        b = base_feat(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]))
        Xa = np.stack([np.concatenate([b, np.eye(3, dtype=np.float32)[a]])
                       for a in range(3)])
        for a in range(3):
            mu, _ = gain_predict(GM[a], Xa[a][None])
            mus_t[j, a] = mu[0]
    outs_t = {0: dft.raw45.to_numpy(), 1: dft.prior45.to_numpy(),
              2: dft.cand_zs45.to_numpy()}
    sage_t = outs_t[0].copy()
    for a in (1, 2):
        sage_t[picks_t == a] = outs_t[a][picks_t == a]
    d_t = sage_t - dft.raw45.to_numpy()
    rng = np.random.default_rng(42)
    up = dft.patient.unique()
    fmap = dft.patient.map({f: i for i, f in enumerate(up)}).to_numpy()
    vals = [d_t[np.isin(fmap, rng.choice(len(up), len(up)))].mean() for _ in range(400)]
    print("\n== Atel 外测：sage（零样本，NIH 校准） ==")
    print(f"  sage Δ {d_t.mean():+.4f} CI [{np.percentile(vals, 2.5):+.4f},"
          f"{np.percentile(vals, 97.5):+.4f}] cov {(picks_t > 0).mean():.3f} "
          f"wrong {(d_t < -0.05).mean():.4f}")
    print(f"  [对照] cand_zs {(dft.cand_zs45 - dft.raw45).mean():+.4f} | "
          f"g2 {(dft.gated_v2_45 - dft.raw45).mean():+.4f}")

    # s(x)：synth-atel vs photo-atel
    from sklearn.linear_model import LogisticRegression
    Dp = np.load(f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
    q9_p = Dp["q9"]
    keys_p = [str(k) for k in Dp["keys"]]
    pat_p = np.array([k.split("_")[0] for k in keys_p])
    q9_n_flat = q9_n3.reshape(-1, 9)
    Xs = np.concatenate([q9_n_flat, q9_p], 0)
    ys = np.array([1] * len(q9_n_flat) + [0] * len(q9_p))
    upf = sorted(set(pat_p))
    folds = [set(upf[j::5]) for j in range(5)]
    s_photo = np.zeros(len(dfp := dft)) if False else np.zeros(len(keys_p))
    picks_f = np.zeros(len(keys_p), np.int64)
    # photo 侧 IoU
    gts_p, cams_pp, masks_pp, probs_p = Dp["gts"], Dp["cams_p"], Dp["masks_p"], Dp["probs"]
    raw_p = np.array([iou45(cams_pp[i], mask_bbox(gts_p[i])) for i in range(len(keys_p))])
    prior_p = np.array([iou45(cams_pp[i] * masks_pp[i], mask_bbox(gts_p[i]))
                        for i in range(len(keys_p))])
    cand_p = np.zeros(len(keys_p))
    with torch.no_grad():
        for i in range(len(keys_p)):
            inp = torch.from_numpy(cams_pp[i][None, None]).to(device)
            mk = torch.from_numpy(masks_pp[i][None, None].astype(np.float32)).to(device)
            qt = torch.from_numpy(q9_p[i][None]).to(device)
            cs_zs = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                     for hh in heads_v2]
            cand_p[i] = iou45(np.median(np.stack(cs_zs), 0), mask_bbox(gts_p[i]))
    outs_p = {0: raw_p, 1: prior_p, 2: cand_p}
    for jf in range(5):
        hold = folds[jf]
        te = np.array([p in hold for p in pat_p])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        tr_photo_idx = np.where(~te)[0]
        Xtr = np.concatenate([q9_n_flat, q9_p[tr_photo_idx]], 0)
        ytr = np.array([1] * len(q9_n_flat) + [0] * len(tr_photo_idx))
        clf.fit(Xtr, ytr)
        s_h = clf.predict_proba(q9_p[te])[:, 1]
        s_photo[te] = s_h
        s_syn = clf.predict_proba(q9_n_flat)[:, 1]
        kappa = float(np.quantile(s_syn, 0.25))
        for j, i in enumerate(np.where(te)[0]):
            pick = sage_row(q9_p[i], float(probs_p[i]), GM, lam, eps, mrg)
            if pick != 0 and s_h[j] < kappa:
                pick = 0
            picks_f[i] = pick
    ss_p = outs_p[0].copy()
    for a in (1, 2):
        ss_p[picks_f == a] = outs_p[a][picks_f == a]
    d_p = ss_p - raw_p
    print("\n== Atel photo（5 折患者 CV，s 折外，κ=q25） ==")
    print(f"  sage+support Δ {d_p.mean():+.4f} cov {(picks_f > 0).mean():.3f} "
          f"wrong {(d_p < -0.05).mean():.4f}")
    print(f"  [对照] cand_zs {(cand_p - raw_p).mean():+.4f} | "
          f"photo s 中位 {np.median(s_photo):.3f}")

    # synth 外测 s(x) + κ 扫描
    clf_full = LogisticRegression(max_iter=1000, C=1.0)
    clf_full.fit(Xs, ys)
    s_syn_full = clf_full.predict_proba(q9_n_flat)[:, 1]
    s_ext = clf_full.predict_proba(qv_ext)[:, 1]
    print(f"  外测 s 中位 {np.median(s_ext):.3f}")
    print("\n== Atel κ 扫描 ==")
    for kq in [0.0, 0.25, 0.5]:
        kap = float(np.quantile(s_syn_full, kq))
        pk_t = np.zeros(len(dft), np.int64)
        for j in range(len(dft)):
            pick = sage_row(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]),
                            GM, lam, eps, mrg)
            if pick != 0 and s_ext[j] < kap:
                pick = 0
            pk_t[j] = pick
        st = outs_t[0].copy()
        for a in (1, 2):
            st[pk_t == a] = outs_t[a][pk_t == a]
        dt2 = st - dft.raw45.to_numpy()
        print(f"  κ(q{int(kq * 100):02d})={kap:.3f}  synth Δ {dt2.mean():+.4f} "
              f"cov {(pk_t > 0).mean():.3f} wrong {(dt2 < -0.05).mean():.4f}")
    dft["sage45"] = sage_t
    dft["sage_pick"] = picks_t
    dft["ss_pick"] = pk_t
    dft["ss45"] = st
    dft.to_csv(f"{OUT}/p65_atel_test.csv", index=False)
    pd.DataFrame({"key": keys_p, "raw45": raw_p, "prior45": prior_p,
                  "cand_zs45": cand_p, "s_support": s_photo,
                  "ss_pick": picks_f, "ss45": ss_p}).to_csv(
        f"{OUT}/p65_atel_photo.csv", index=False)
    log("完成")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"cache": cmd_cache, "test": cmd_test, "sage": cmd_sage}[cmd]())
