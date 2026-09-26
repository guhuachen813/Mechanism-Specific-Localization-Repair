r"""第六阶段 6.1：SAGE-Loc 收益预测器 + 保守下界选择（go/no-go 检查点）。

中心问题：对每图判断"当前采集语境是否支持某修复动作、该动作收益下界是否为正"。

设计：
- 训练数据：NIH 146 图 × 17 条件，候选动作 {raw, prior, cand_zs(5 头中位)} 的
  真实配对收益 g_a = IoU45(H_a) − IoU45(raw)（GT 框 IoU，BBox_List_2017）。
- 收益预测器：逐动作异方差 MLP，输入 x_a = q9(9) ⊕ [frac,field_ok,conflict,prob](4)
  ⊕ 动作 onehot(3) = 16 维，输出 (μ_a, σ_a)，高斯 NLL。
- 患者级 5 折交叉拟合（折=第四阶段患者划分，与修复头折一致 → cand 收益标签本就折外）。
- 选择规则：a* = argmax_a L_a，L_a = μ_a − λσ_a；执行条件 L_a* > ε 且 μ_a* − μ_raw ≥ m；
  λ/ε/m 在校准患者池（20%）网格选择。
- 锁定外测：CheXlocalize test 154 阳性 × 17 条件 + clean（p6 协议逐位一致），
  系统 raw / gated_v2(复算) / sage / oracle。

判据（预注册）：sage vs raw Δ45 CI>0 且点估计 > gated_v2 的 +0.053 → go。

用法：/root/miniconda3/bin/python p61_gain_selector.py [--limit 6] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zlib

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import degradations as dg
from cam_benchmark import (MEAN, STD, THRESHOLDS, batch_cam, cam_iou_curve,
                           load_model)
from cam_prior import detect_field, mask_224, medsam_prep, segment
from repairer import cam_feats, q_vector
from silver_eval_test import OUT, SEG_TEST, TEST_DIR, rle_decode

ART = "/root/autodl-tmp/experiments/rep_robust/repairer_v2_artifacts"
ROOT = "/root/autodl-tmp/experiments/rep_robust"
ALPHA = 0.5
SEED = 42
TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])
FINDING = "Cardiomegaly"


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


def gain_nll_train(X, y, seed, epochs=200, bs=64):
    """异方差收益头：(16 维) → (μ, σ)。高斯 NLL。"""
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


def iou45(cam, gt):
    return float(cam_iou_curve(cam, gt)[TAU45])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)

    # ================= NIH 训练侧 =================
    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
    cams = C["cams"].astype(np.float32)          # (146,17,224,224)
    masks = C["masks"]                            # uint8
    qs = C["qs"]                                  # (146,17,7)
    cs = C["cs"]                                  # (146,17,2)
    probs_n = C["probs"]                          # (146,17)
    files = list(C["files"])
    conds = list(C["conds"])
    NI, NC = len(files), len(conds)

    sp = pd.read_csv(f"{ROOT}/splits/nih_box_patients.csv")
    fold_of = {r.file: r.fold for r in sp[sp.finding == FINDING].itertuples()}
    pat_of = {f: f.split("_")[0] for f in files}
    log(f"NIH cache {NI} 图 × {NC} 条件")

    heads_v2, gates_v2, deltas = [], [], []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)
        g = GateV2().to(device)
        g.load_state_dict(torch.load(f"{ART}/fold{kf}_gate.pt", map_location=device))
        g.eval()
        gates_v2.append(g)
        deltas.append(float(np.load(f"{ART}/fold{kf}_delta.npy")[0]))
    d_med = float(np.median(deltas))
    log(f"v2 工件载入，δ 中位 {d_med:.2f}")

    with torch.no_grad():
        cand_all = np.zeros_like(cams)
        pg_all = np.zeros((NI, NC, 3), np.float32)
        for i in range(NI):
            for c in range(NC):
                inp = torch.from_numpy(cams[i, c][None, None]).to(device)
                mk = torch.from_numpy(masks[i, c][None, None].astype(np.float32)).to(device)
                q9 = np.concatenate([qs[i, c], cs[i, c]]).astype(np.float32)
                qt = torch.from_numpy(q9[None]).to(device)
                cs_zs = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                         for hh in heads_v2]
                cand_all[i, c] = np.median(np.stack(cs_zs), 0)
                pg_all[i, c] = np.median(torch.softmax(torch.stack(
                    [g(qt) for g in gates_v2]), dim=2).cpu().numpy()[:, 0], 0)
            if (i + 1) % 40 == 0:
                log(f"  cand {i + 1}/{NI}")

    # GT 框（224 坐标）
    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == FINDING:
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                sx = sy = 224.0 / 1024.0
                gt_box[row[0]] = [(x * sx, y * sy, (x + w) * sx, (y + h) * sy)]
    log(f"GT 框 {len(gt_box)} 图")

    raw_iou = np.full((NI, NC), np.nan, np.float32)
    prior_iou = np.full((NI, NC), np.nan, np.float32)
    cand_iou = np.full((NI, NC), np.nan, np.float32)
    for i, f in enumerate(files):
        gt = gt_box.get(f)
        if gt is None:
            continue
        for c in range(NC):
            raw_iou[i, c] = iou45(cams[i, c], gt)
            prior_iou[i, c] = iou45(cams[i, c] * masks[i, c], gt)
            cand_iou[i, c] = iou45(cand_all[i, c], gt)
    okm = ~np.isnan(raw_iou)
    log(f"有效 (图,条件) 对 {int(okm.sum())}")

    # ---- 样本表：N=有效对；X (N*3,16)；AID (N*3)；Y (N,3)；KV (N,2) ----
    X_rows, Y_rows, KV = [], [], []
    for i in range(NI):
        for c in range(NC):
            if not okm[i, c]:
                continue
            q9 = np.concatenate([qs[i, c], cs[i, c]]).astype(np.float32)
            frac, fok, conf = float(q9[1]), float(q9[0]), float(q9[8])
            base = np.concatenate([q9, [frac, fok, conf, float(probs_n[i, c])]])
            oneh = np.eye(3, dtype=np.float32)
            for a in range(3):
                X_rows.append(np.concatenate([base, oneh[a]]))
            Y_rows.append([0.0, prior_iou[i, c] - raw_iou[i, c],
                           cand_iou[i, c] - raw_iou[i, c]])
            KV.append((i, c))
    X = np.stack(X_rows)
    Y = np.array(Y_rows, np.float32)
    AID = np.tile(np.arange(3), len(Y))
    KV = np.array(KV)
    N = len(Y)
    KV_pat = np.array([pat_of[files[int(i)]] for i in KV[:, 0]])
    KV_cond = np.array([conds[int(c)] for c in KV[:, 1]])
    log(f"训练样本 {N}（×3 动作 = {X.shape[0]} 行，{X.shape[1]} 维）")

    # ---- 患者级 5 折交叉拟合（折外 μ,σ） ----
    upats = sorted(set(KV_pat))
    pat_folds = [set(upats[j::5]) for j in range(5)]
    MU_A = np.zeros(N * 3, np.float32)
    SG_A = np.zeros(N * 3, np.float32)
    for jf in range(5):
        hold = pat_folds[jf]
        tr_s = ~np.isin(KV_pat, list(hold))
        tr_row = np.repeat(tr_s, 3)              # X 行级训练掩码（样本主序）
        for a in range(3):
            m = tr_row & (AID == a)
            mdl = gain_nll_train(X[m], Y[tr_s, a], SEED + jf)
            mu, sig = gain_predict(mdl, X[AID == a])
            MU_A[AID == a] = mu
            SG_A[AID == a] = sig
        log(f"  cf fold{jf} done（训练患者 {len(upats) - len(hold)} / 折外 {len(hold)}）")
    MU3 = MU_A.reshape(N, 3)
    SG3 = SG_A.reshape(N, 3)

    # 注：折外收益标签的口径——cand 收益来自 v2 折外头（样本所在折的 head 不含该患者），
    # 收益预测器的交叉拟合亦按同一患者折进行 → (μ,σ) 对训练池患者无自泄漏。

    # ---- 校准池：患者 20%（seed 7），其余为评估池 ----
    rng_c = np.random.default_rng(7)
    cal_pat = set(rng_c.choice(upats, size=max(4, len(upats) // 5), replace=False).tolist())
    is_cal = np.isin(KV_pat, list(cal_pat))
    is_ev = ~is_cal
    log(f"校准患者 {len(cal_pat)} / 评估患者 {len(upats) - len(cal_pat)}")

    IoU3 = np.stack([raw_iou[KV[:, 0], KV[:, 1]],
                     prior_iou[KV[:, 0], KV[:, 1]],
                     cand_iou[KV[:, 0], KV[:, 1]]], 1)

    def sage_pick(mask, lam, eps, mrg):
        idx = np.where(mask)[0]
        picks = np.zeros(len(idx), np.int64)
        if len(idx) == 0:
            return picks, idx
        L = MU3[idx] - lam * SG3[idx]
        MU = MU3[idx]
        for j in range(len(idx)):
            a = int(np.argmax(L[j]))
            if a != 0 and L[j, a] > eps and MU[j, a] - MU[j, 0] >= mrg:
                picks[j] = a
        return picks, idx

    best = (-1e9, None)
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for eps in [-0.02, 0.0, 0.02, 0.05]:
            for mrg in [0.0, 0.02, 0.05]:
                picks, _ = sage_pick(is_cal, lam, eps, mrg)
                outs = IoU3[is_cal][np.arange(len(picks)), picks]
                gain = outs - IoU3[is_cal, 0]
                score = gain.mean() - 0.05 * (gain < -0.05).mean()
                if score > best[0]:
                    best = (score, (lam, eps, mrg))
    lam, eps, mrg = best[1]
    log(f"校准 λ={lam} ε={eps} m={mrg}（校准池效用 {best[0]:+.4f}）")

    # ---- NIH 折外评估：sage vs gated_v2 ----
    picks_all, _ = sage_pick(np.ones(N, bool), lam, eps, mrg)
    outs_all = IoU3[np.arange(N), picks_all]
    ev_gain = outs_all[is_ev] - IoU3[is_ev, 0]
    g2_pick = []
    for (i, c) in KV:
        pg = pg_all[int(i), int(c)]
        ch = int(pg.argmax())
        g2_pick.append(ch if (ch == 0 or pg[ch] - pg[0] >= d_med) else 0)
    g2_pick = np.array(g2_pick)
    g2_out = IoU3[np.arange(N), g2_pick]
    ev2 = g2_out[is_ev] - IoU3[is_ev, 0]
    log(f"NIH 评估池：sage {ev_gain.mean():+.4f} cov {(picks_all[is_ev] > 0).mean():.3f} "
        f"wrong {(ev_gain < -0.05).mean():.4f}")
    log(f"NIH 评估池：gated_v2 {ev2.mean():+.4f} cov {(g2_pick[is_ev] > 0).mean():.3f} "
        f"wrong {(ev2 < -0.05).mean():.4f}")
    print("\n== NIH 评估池 per-condition（Δ vs raw） ==")
    for cd in conds:
        m = (KV_cond == cd) & is_ev
        if m.sum() == 0:
            continue
        print(f"  {cd:12s} sage {(outs_all[m] - IoU3[m, 0]).mean():+.3f}"
              f"  g2 {(g2_out[m] - IoU3[m, 0]).mean():+.3f}  n={int(m.sum())}")

    if args.smoke:
        log("smoke：跳过锁定外测")
        return 0

    # ---- 最终模型（全量训练，用于外测；外测患者与 NIH 无重叠，无泄漏） ----
    GM = [gain_nll_train(X[AID == a], Y[:, a], SEED + 99 + a) for a in range(3)]

    # ================= 锁定外测（test 154 × 17+clean） =================
    log("加载分类器与 MedSAM（外测）")
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    from transformers import SamModel
    smodel = SamModel.from_pretrained(
        f"{ROOT}/weights/medsam").to(device).eval()

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
    if args.limit:
        keys = keys[: args.limit]
    log(f"锁定外测工作集 {len(keys)} 图")

    rows = []
    cams_out, masks_out, qv_out = [], [], []
    for n, k in enumerate(keys):
        parts = k.split("_")
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
            patient = parts[0]
            conds_t = [("clean", im224)] + [
                (c, dg.degrade(im224, c, dg.rng_for(SEED, zlib.crc32((k + c).encode()))))
                for c in dg.CONDITION_ORDER]
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
                # sage 选择
                frac, fok, conf = float(qv[1]), float(qv[0]), float(qv[8])
                base = np.concatenate([qv, [frac, fok, conf, float(prob[0])]])
                Xa = np.stack([np.concatenate([base, np.eye(3, dtype=np.float32)[a]])
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
                sage_out = {0: r_i, 1: p_i, 2: c_i}[pick]
                ch2 = int(pg.argmax())
                pick2 = ch2 if (ch2 == 0 or pg[ch2] - pg[0] >= d_med) else 0
                g2_i = {0: r_i, 1: p_i, 2: c_i}[pick2]
                gated2 = [cam, prior, cand_zs][pick2]
                rows.append({
                    "key": k, "patient": patient, "cond": cond,
                    "prob": float(prob[0]), "field_ok": int(ok_r),
                    "raw45": r_i, "prior45": p_i, "cand_zs45": c_i,
                    "sage45": sage_out, "sage_pick": pick,
                    "sage_mu": mus[pick] if pick else 0.0,
                    "sage_sigma": sgs[pick] if pick else 0.0,
                    "sage_L": float(L[pick]),
                    "gated_v2_45": g2_i, "g2_pick": pick2,
                    "oracle45": max(r_i, p_i, c_i),
                    "frac": frac, "conflict": conf, "gt_area": float(m224.mean())})
                cams_out.append(cam.astype(np.float16))
                masks_out.append(mm.astype(np.uint8))
                qv_out.append(qv.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            log(f"err {k}: {e}")
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(keys)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p61_synth_test.csv", index=False)
    np.savez_compressed(f"{OUT}/p61_test_cams.npz",
                        cam=np.stack(cams_out), mask=np.stack(masks_out),
                        qv=np.stack(qv_out), rowkey=np.array(
                            [f"{r.key}|{r.cond}" for r in df.itertuples()]))
    print("\n== 锁定外测（test 154 × 17+clean，τ45） ==")
    print(df.groupby("cond")[["raw45", "prior45", "cand_zs45", "sage45",
                              "gated_v2_45", "oracle45"]].mean().round(3).to_string())
    rng = np.random.default_rng(42)
    up = df.patient.unique()
    fmap = df.patient.map({f: i for i, f in enumerate(up)}).to_numpy()
    print("\n== vs raw（患者整群 bootstrap 400） ==")
    res = {}
    for v in ["prior45", "cand_zs45", "sage45", "gated_v2_45", "oracle45"]:
        d = df[v] - df.raw45
        vals = [d[np.isin(fmap, rng.choice(len(up), len(up)))].mean() for _ in range(400)]
        lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
        res[v] = (d.mean(), lo, hi)
        print(f"  {v:12s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    cov = (df.sage_pick > 0).mean()
    wr = ((df.sage45 - df.raw45 < -0.05) & (df.sage_pick > 0)).mean()
    log(f"sage 覆盖率 {cov:.3f}  错误修复率 {wr:.4f}  "
        f"(gated_v2 覆盖 {(df.g2_pick > 0).mean():.3f})")
    go = res["sage45"][0] > res["gated_v2_45"][0] and res["sage45"][1] > 0
    print(f"\n判据（sage CI>0 且 > gated_v2 {res['gated_v2_45'][0]:+.4f}）："
          f"{'GO' if go else 'NO-GO（回设计）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
