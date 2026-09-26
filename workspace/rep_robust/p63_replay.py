r"""第六阶段 6.3：R_t 目标域适配 + replay/distillation 抗遗忘（source retention）。

动机（已量化痛点）：photo 适配把合成外测收益从 +0.047 抹到 +0.008（灾难性遗忘）。

变体（3 seed × 4 变体，early-stop 协议与 5.1a/p6 一致）：
  v0 plain   ：仅 photo 52 fit 患者适配（=p6 cand_ad，复现 +0.008 基线）
  v1 dist1.0 ：+ NIH replay（1:1 批），蒸馏损失 MSE(cand_Rt, cand_Rs)（R_s=v2 5 头中位）
  v2 dist0.3 ：蒸馏权重 0.3
  v3 replayGT：+ NIH replay 用 GT 框监督（经典联合重放，无蒸馏）

评估：
  - photo 65（目标域安全）：Δ vs raw、harm rate
  - 合成外测 test 154×18（源域技能保留）：Δ vs raw、retention =
    (Δ_rt − Δ_v0) / (Δ_cand_zs − Δ_v0)

用法：/root/miniconda3/bin/python p63_replay.py
"""
from __future__ import annotations

import csv
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from cam_benchmark import THRESHOLDS, cam_iou_curve
from silver_eval_test import OUT, SEG_TEST, rle_decode

ROOT = "/root/autodl-tmp/experiments/rep_robust"
ART = f"{ROOT}/repairer_v2_artifacts"
ALPHA = 0.5
SEED = 42
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])
NREPLAY = 8          # 每 photo 批后的 replay 批大小
BS = 16


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


def iou45(cam, gt):
    return float(cam_iou_curve(cam, gt)[TAU45])


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)

    # ---------- 数据 ----------
    log("载入 photo 65 缓存与 NIH 源域缓存")
    D = np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    keys_p = [str(k) for k in D["keys"]]
    cams_p, masks_p, q9_p, gts_p = (D["cams_p"], D["masks_p"].astype(np.float32),
                                    D["q9"], D["gts"])
    pat_p = np.array([k.split("_")[0] for k in keys_p])
    up_p = sorted(set(pat_p))

    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
    cams_n = C["cams"].astype(np.float32)      # (146,17,224,224)
    masks_n = C["masks"].astype(np.float32)
    qs_n = C["qs"]
    cs_n = C["cs"]
    NI, NC = cams_n.shape[:2]

    heads_v2 = []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)

    # 蒸馏目标：R_s（v2 5 头中位）在全部 NIH 样本上的输出
    log("预计算 R_s 蒸馏目标（NIH 全样本）")
    cand_rs = np.zeros_like(cams_n)
    with torch.no_grad():
        for i in range(NI):
            for c in range(NC):
                inp = torch.from_numpy(cams_n[i, c][None, None]).to(device)
                mk = torch.from_numpy(masks_n[i, c][None, None]).to(device)
                q9 = np.concatenate([qs_n[i, c], cs_n[i, c]]).astype(np.float32)
                qt = torch.from_numpy(q9[None]).to(device)
                cs_zs = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                         for hh in heads_v2]
                cand_rs[i, c] = np.median(np.stack(cs_zs), 0)
    rep_flat_cams = cams_n.reshape(-1, 224, 224)
    rep_flat_masks = masks_n.reshape(-1, 224, 224)
    rep_flat_rs = cand_rs.reshape(-1, 224, 224)
    rep_flat_q9 = np.concatenate([qs_n, cs_n], 2).reshape(-1, 9)
    NR = len(rep_flat_cams)

    # NIH GT 框（replayGT 变体用）
    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == "Cardiomegaly":
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = (x * s, y * s, (x + w) * s, (y + h) * s)
    rep_file_idx = np.repeat(np.arange(NI), NC)
    rep_cond_idx = np.tile(np.arange(NC), NI)
    rep_has_gt = np.array([files_i in gt_box for files_i in
                           [C["files"][j] for j in rep_file_idx]])

    # ---------- 外测 GT 框（源域技能评估） ----------
    log("解析外测 GT（154×18）")
    E = np.load(f"{OUT}/p61_test_cams.npz", allow_pickle=True)
    rowkey = [str(k) for k in E["rowkey"]]
    ext_cams = E["cam"].astype(np.float32)
    ext_masks = E["mask"].astype(np.float32)
    ext_qv = E["qv"]
    seg = json.load(open(SEG_TEST))
    gt224 = {}

    def gt_of(key):
        if key not in gt224:
            m_full = rle_decode(seg[key]["Cardiomegaly"])
            mi = Image.fromarray(m_full * 255).resize((224, 224), Image.BILINEAR)
            gt224[key] = (np.asarray(mi, np.float32) / 255.0 > 0.5).astype(np.uint8)
        return gt224[key]

    ext_gt = [gt_of(rk.split("|")[0]) for rk in rowkey]
    log(f"外测 {len(rowkey)} 行")

    # ---------- 训练函数 ----------
    def train_rt(fit_idx, cal_idx, mode, seed, beta=1.0, epochs_max=60, eval_every=5):
        """mode: plain | dist | replaygt"""
        head = RepairHead().to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        Yp = np.zeros((len(fit_idx), 1, 224, 224), np.float32)
        for j, i in enumerate(fit_idx):
            b = mask_bbox(gts_p[i])[0]
            Yp[j, 0, int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1.0
        rng = np.random.default_rng(seed)
        rng_rep = np.random.default_rng(seed + 555)
        best = (-1e9, None, 0)

        def cal_iou():
            head.eval()
            with torch.no_grad():
                sc = []
                for i in cal_idx:
                    h = torch.from_numpy(cams_p[i][None, None]).to(device)
                    m = torch.from_numpy(masks_p[i][None, None]).to(device)
                    q = torch.from_numpy(q9_p[i][None]).to(device)
                    c = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[0, 0].cpu().numpy()
                    sc.append(iou45(c, mask_bbox(gts_p[i])))
            head.train()
            return float(np.mean(sc))

        for ep in range(epochs_max):
            perm = rng.permutation(len(fit_idx))
            for s0 in range(0, len(perm), BS):
                sel = perm[s0:s0 + BS]
                ii = np.array([fit_idx[s] for s in sel])
                h = torch.from_numpy(cams_p[ii]).unsqueeze(1).to(device)
                m = torch.from_numpy(masks_p[ii]).unsqueeze(1).to(device)
                y = torch.from_numpy(Yp[sel]).to(device)
                q = torch.from_numpy(q9_p[ii]).to(device)
                delta = head(h, m, q)
                cand = torch.clamp(h + ALPHA * delta, 0, 1)
                loss = F.binary_cross_entropy(cand, y) \
                    + 0.25 * (cand * (1 - m)).mean() \
                    + 0.05 * ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                              + (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                # replay 批
                if mode in ("dist", "replaygt"):
                    sel_r = rng_rep.choice(NR, NREPLAY, replace=False)
                    if mode == "replaygt":
                        ok = rep_has_gt[sel_r]
                        if ok.sum() == 0:
                            continue
                        sel_r = sel_r[ok]
                        hr = torch.from_numpy(rep_flat_cams[sel_r]).unsqueeze(1).to(device)
                        mr = torch.from_numpy(rep_flat_masks[sel_r]).unsqueeze(1).to(device)
                        qr = torch.from_numpy(rep_flat_q9[sel_r]).to(device)
                        yr = np.zeros((len(sel_r), 1, 224, 224), np.float32)
                        for j, rr in enumerate(sel_r):
                            fi = C["files"][rep_file_idx[rr]]
                            bx = gt_box[fi]
                            yr[j, 0, int(bx[1]):int(bx[3]), int(bx[0]):int(bx[2])] = 1.0
                        yr = torch.from_numpy(yr).to(device)
                        dr = head(hr, mr, qr)
                        cr = torch.clamp(hr + ALPHA * dr, 0, 1)
                        loss_r = F.binary_cross_entropy(cr, yr)
                    else:
                        hr = torch.from_numpy(rep_flat_cams[sel_r]).unsqueeze(1).to(device)
                        mr = torch.from_numpy(rep_flat_masks[sel_r]).unsqueeze(1).to(device)
                        qr = torch.from_numpy(rep_flat_q9[sel_r]).to(device)
                        tr = torch.from_numpy(rep_flat_rs[sel_r]).unsqueeze(1).to(device)
                        dr = head(hr, mr, qr)
                        cr = torch.clamp(hr + ALPHA * dr, 0, 1)
                        loss_r = beta * F.mse_loss(cr, tr)
                    opt.zero_grad(set_to_none=True)
                    loss_r.backward()
                    opt.step()
            if (ep + 1) % eval_every == 0 or ep == epochs_max - 1:
                sc = cal_iou()
                if sc > best[0]:
                    best = (sc, {k: v.clone() for k, v in head.state_dict().items()}, ep + 1)
        head.load_state_dict(best[1])
        head.eval()
        return head, best[2]

    # ---------- 4 变体 × 3 seed ----------
    variants = [("plain", {}), ("dist1.0", {"mode": "dist", "beta": 1.0}),
                ("dist0.3", {"mode": "dist", "beta": 0.3}),
                ("replaygt", {"mode": "replaygt"})]
    rows = []
    for vname, kw in variants:
        for sd in (42, 7, 2024):
            rng = np.random.default_rng(sd)
            perm = rng.permutation(len(up_p))
            cal_p = set(up_p[j] for j in perm[: max(4, len(up_p) // 5)])
            fit_idx = [i for i in range(len(pat_p)) if pat_p[i] not in cal_p]
            cal_idx = [i for i in range(len(pat_p)) if pat_p[i] in cal_p]
            mode = kw.get("mode", "plain")
            head, bep = train_rt(fit_idx, cal_idx, mode, sd,
                                 beta=kw.get("beta", 1.0))
            # photo 65 评估（全池，与 p6 口径一致）
            ph_out = []
            with torch.no_grad():
                for i in range(len(pat_p)):
                    h = torch.from_numpy(cams_p[i][None, None]).to(device)
                    m = torch.from_numpy(masks_p[i][None, None]).to(device)
                    q = torch.from_numpy(q9_p[i][None]).to(device)
                    c = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[0, 0].cpu().numpy()
                    ph_out.append(iou45(c, mask_bbox(gts_p[i])))
            # 外测源域技能
            ex_out = []
            with torch.no_grad():
                for s0 in range(0, len(rowkey), 256):
                    sl = slice(s0, min(s0 + 256, len(rowkey)))
                    h = torch.from_numpy(ext_cams[sl]).unsqueeze(1).to(device)
                    m = torch.from_numpy(ext_masks[sl]).unsqueeze(1).to(device)
                    q = torch.from_numpy(ext_qv[sl]).to(device)
                    c = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[:, 0].cpu().numpy()
                    for j in range(c.shape[0]):
                        ex_out.append(iou45(c[j], mask_bbox(ext_gt[s0 + j])))
            d_photo = np.mean(ph_out) - np.mean([iou45(cams_p[i], mask_bbox(gts_p[i]))
                                                 for i in range(len(pat_p))])
            d_ext = float(np.mean(ex_out))
            rows.append({"variant": vname, "seed": sd, "best_ep": bep,
                         "d_photo": float(d_photo),
                         "photo_harm": float(np.mean(np.array(ph_out) -
                                              [iou45(cams_p[i], mask_bbox(gts_p[i]))
                                               for i in range(len(pat_p))]) < -0.05),
                         "d_ext": d_ext})
            log(f"{vname} seed{sd} best_ep {bep}  photo Δ {d_photo:+.4f}  ext {d_ext:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p63_replay.csv", index=False)
    print("\n== 汇总（3 seed 均值） ==")
    print(df.groupby("variant")[["d_photo", "photo_harm", "d_ext"]].mean().round(4).to_string())
    # retention 相对 plain
    base = df[df.variant == "plain"].d_ext.mean()
    zs = 0.0469  # cand_zs 外测（p61 逐位复现 p6）
    for v in df.variant.unique():
        m = df[df.variant == v].d_ext.mean()
        rec = (m - base) / (zs - base) if zs > base else float("nan")
        print(f"  {v:9s} retention {rec:.2%}（相对 plain→cand_zs 差距的恢复）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
