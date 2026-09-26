r"""第六阶段 6.2：SAGE-Loc 在真实翻拍域（photo 65）——安全检验 + σ/OOD 信号 + CRC 校准。

问题：NIH 合成轴校准的选择策略零样本落到真实翻拍域会怎样？
（photo oracle +0.013，几乎无可收割收益——正确行为是识别"不可修复"并回退 raw。）

内容：
A. 复现 6.1 的收益预测器 GM（同 seed 同协议，全量 NIH）与校准 (λ,ε,m)。
B. photo 65 零样本评估：actions {raw, prior, cand_zs}，sage(NIH 校准) vs
   gated_v2 / v4 已知结果（+0.001, cov 0.06-0.12, err≈0）vs 零样本 cand(−0.079)。
C. 不确定性=支持度代理：photo 上 σ、|μ|、L 对"有害修复"(g<−0.05) 的 AUROC；
   σ 分布 photo vs NIH（配对 t / 中位数对比）——OOD 时不确定性是否膨胀。
D. 多视图分歧（几何+光度 7 视图 CAM 一致性）作为支持度特征的增益：
   对有害修复的 AUROC 增量；photo vs NIH 分歧分布对比。
E. 色度泄漏检验：RGB 统计域分类器（预期 AUROC≈1.0，已知捷径）vs
   q9 灰度特征域分类器（管线本身无色度，预期显著更低）。
F. photo 域 CRC 校准（患者级 5 折嵌套：折内校准 (λ,ε,m) 控制 wrong-repair≤α=0.05，
   折外评估）——"photo 校准的 sage"折外 wrong-repair / cov / Δ。

用法：/root/miniconda3/bin/python p62_sage_photo.py
"""
from __future__ import annotations

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

from cam_benchmark import (MEAN, STD, THRESHOLDS, batch_cam, cam_iou_curve,
                           load_model)
from repairer import cam_feats, q_vector

ROOT = "/root/autodl-tmp/experiments/rep_robust"
ART = f"{ROOT}/repairer_v2_artifacts"
OUT = f"{ROOT}/chexlocalize"
BASE = "/root/autodl-tmp"
PHOTO_DIR = f"{BASE}/chexphoto/valid/natural/oneplus"
NIH_DIR = f"{BASE}/nih_cxr14_1024/images"  # 仅取存在者；不存在则跳过色度对照
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


def iou45(cam, gt):
    return float(cam_iou_curve(cam, gt)[TAU45])


def base_feat(qv9, prob):
    frac, fok, conf = float(qv9[1]), float(qv9[0]), float(qv9[8])
    return np.concatenate([qv9, [frac, fok, conf, float(prob)]]).astype(np.float32)


def sage_pick_row(qv9, prob, GM, lam, eps, mrg):
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
    return pick, np.array(mus), np.array(sgs), L


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)

    # ---------- A. 复现 GM 与 NIH 校准 ----------
    log("A. 重建 NIH 收益预测器与校准")
    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
    cams_n = C["cams"].astype(np.float32)
    masks_n = C["masks"]
    qs_n = C["qs"]
    cs_n = C["cs"]
    probs_n = C["probs"]
    files = list(C["files"])
    conds = list(C["conds"])
    NI, NC = len(files), len(conds)

    heads_v2 = []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)
    gates_v2 = []
    for kf in range(5):
        g = GateV2().to(device)
        g.load_state_dict(torch.load(f"{ART}/fold{kf}_gate.pt", map_location=device))
        g.eval()
        gates_v2.append(g)
    d_med = float(np.median([float(np.load(f"{ART}/fold{kf}_delta.npy")[0])
                             for kf in range(5)]))

    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == "Cardiomegaly":
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = [(x * s, y * s, (x + w) * s, (y + h) * s)]

    X_rows, Y_rows = [], []
    with torch.no_grad():
        for i in range(NI):
            gt = gt_box.get(files[i])
            if gt is None:
                continue
            for c in range(NC):
                inp = torch.from_numpy(cams_n[i, c][None, None]).to(device)
                mk = torch.from_numpy(masks_n[i, c][None, None].astype(np.float32)).to(device)
                q9 = np.concatenate([qs_n[i, c], cs_n[i, c]]).astype(np.float32)
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
    X_n = np.stack(X_rows)
    Y_n = np.array(Y_rows, np.float32)
    AID_n = np.tile(np.arange(3), len(Y_n))
    GM = [gain_nll_train(X_n[AID_n == a], Y_n[:, a], SEED + 99 + a) for a in range(3)]
    # NIH 校准（与 p61 相同协议：患者 20% seed7）
    N_n = len(Y_n)
    KV_pat_n = np.array([f.split("_")[0] for f in files
                         for _ in conds if gt_box.get(f) is not None])
    # 上面推导易错，直接按样本重建：
    kv_pat = []
    for i in range(NI):
        if gt_box.get(files[i]) is None:
            continue
        kv_pat += [files[i].split("_")[0]] * NC
    KV_pat_n = np.array(kv_pat)
    upats = sorted(set(KV_pat_n))
    rng_c = np.random.default_rng(7)
    cal_pat = set(rng_c.choice(upats, size=max(4, len(upats) // 5), replace=False).tolist())
    is_cal_n = np.isin(KV_pat_n, list(cal_pat))
    MU3 = np.zeros((N_n, 3), np.float32)
    SG3 = np.zeros((N_n, 3), np.float32)
    for a in range(3):
        mu, sig = gain_predict(GM[a], X_n[AID_n == a])
        MU3[:, a] = mu
        SG3[:, a] = sig
    IoU3n = np.zeros((N_n, 3), np.float32)
    IoU3n[:, 1] = Y_n[:, 1]
    IoU3n[:, 2] = Y_n[:, 2]  # 相对 raw 的收益；raw 列 IoU 差异无用于校准选择（用收益即可）

    def pick_gain(mask, lam, eps, mrg):
        idx = np.where(mask)[0]
        picks = np.zeros(len(idx), np.int64)
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
                picks, idx = pick_gain(is_cal_n, lam, eps, mrg)
                g = np.where(picks == 0, 0.0, np.where(
                    picks == 1, Y_n[idx, 1], Y_n[idx, 2]))
                score = g.mean() - 0.05 * (g < -0.05).mean()
                if score > best[0]:
                    best = (score, (lam, eps, mrg))
    lam, eps, mrg = best[1]
    log(f"NIH 校准 λ={lam} ε={eps} m={mrg}")

    # ---------- B. photo 65 零样本评估 ----------
    log("B. photo 65 零样本评估")
    D = np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    keys = [str(k) for k in D["keys"]]
    cams_p, masks_p, q9s, gts = D["cams_p"], D["masks_p"], D["q9"], D["gts"]
    probs_p = D["probs"]
    NP = len(keys)
    pats = np.array([k.split("_")[0] for k in keys])

    rows = []
    mus_all, sgs_all, L_all = np.zeros((NP, 3)), np.zeros((NP, 3)), np.zeros((NP, 3))
    with torch.no_grad():
        for i in range(NP):
            gt = mask_bbox(gts[i])
            cam = cams_p[i]
            mm = masks_p[i]
            qv = q9s[i]
            inp = torch.from_numpy(cam[None, None]).to(device)
            mk = torch.from_numpy(mm[None, None].astype(np.float32)).to(device)
            qt = torch.from_numpy(qv[None]).to(device)
            cs_zs = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                     for hh in heads_v2]
            cand = np.median(np.stack(cs_zs), 0)
            prior = cam * mm
            r_i, p_i, c_i = iou45(cam, gt), iou45(prior, gt), iou45(cand, gt)
            pick, mus, sgs, L = sage_pick_row(qv, probs_p[i], GM, lam, eps, mrg)
            mus_all[i], sgs_all[i], L_all[i] = mus, sgs, L
            pg = np.median(torch.softmax(torch.stack(
                [g(qt) for g in gates_v2]), dim=2).cpu().numpy()[:, 0], 0)
            ch = int(pg.argmax())
            pick2 = ch if (ch == 0 or pg[ch] - pg[0] >= d_med) else 0
            outs = {0: r_i, 1: p_i, 2: c_i}
            rows.append({
                "key": keys[i], "patient": pats[i],
                "raw45": r_i, "prior45": p_i, "cand_zs45": c_i,
                "sage45": outs[pick], "sage_pick": pick,
                "gated_v2_45": outs[pick2], "g2_pick": pick2,
                "oracle45": max(r_i, p_i, c_i),
                "gprior": p_i - r_i, "gcand": c_i - r_i,
                "frac": float(qv[1]), "conflict": float(qv[8])})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p62_photo_sage.csv", index=False)
    print("\n== photo 65（τ45） ==")
    for v in ["raw45", "prior45", "cand_zs45", "sage45", "gated_v2_45", "oracle45"]:
        print(f"  {v:12s} {df[v].mean():.4f}")
    d_sage = df.sage45 - df.raw45
    d_g2 = df.gated_v2_45 - df.raw45
    print(f"  sage Δ {d_sage.mean():+.4f} cov {(df.sage_pick > 0).mean():.3f} "
          f"wrong {(d_sage < -0.05).mean():.4f}")
    print(f"  g2   Δ {d_g2.mean():+.4f} cov {(df.g2_pick > 0).mean():.3f} "
          f"wrong {(d_g2 < -0.05).mean():.4f}")

    # ---------- C. 不确定性=支持度代理 ----------
    log("C. σ / L 对有害修复的判别（photo）")
    # 候选修复动作行（prior 或 cand）上的"实际收益<−0.05"为有害
    harmful = []
    scores = {"mu_cand": [], "sig_cand": [], "L_cand": [], "sig_prior": [], "L_prior": []}
    lab = []
    for i in range(NP):
        # cand 动作
        harmful.append(df.gcand[i] < -0.05)
        scores["mu_cand"].append(mus_all[i, 2])
        scores["sig_cand"].append(sgs_all[i, 2])
        scores["L_cand"].append(L_all[i, 2])
        scores["sig_prior"].append(sgs_all[i, 1])
        scores["L_prior"].append(L_all[i, 1])
        lab.append(1.0)
    harmful = np.array(harmful)
    from sklearn.metrics import roc_auc_score
    for k, v in scores.items():
        v = np.array(v)
        if harmful.sum() in (0, len(harmful)):
            continue
        auc = roc_auc_score(harmful, -v if k.startswith(("L", "mu")) else v)
        print(f"  AUROC(有害|{k}) = {auc:.3f}（有害率 {harmful.mean():.2f}）")
    # σ 膨胀：photo vs NIH（cand 动作）
    sig_n_cand = SG3[AID_n == 2] if False else SG3[:, 2]
    print(f"  σ(cand) 中位：NIH {np.median(sig_n_cand):.4f}  photo {np.median(sgs_all[:, 2]):.4f}"
          f"  |μ| 中位：NIH {np.median(np.abs(MU3[:, 2])):.4f}  photo {np.median(np.abs(mus_all[:, 2])):.4f}")

    # ---------- D. 多视图分歧 ----------
    log("D. 多视图分歧（7 视图 CAM 一致性）")
    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def views(im):
        v = [("id", im)]
        v.append(("hflip", im.transpose(Image.FLIP_LEFT_RIGHT)))
        for ang in (-6, 6):
            v.append((f"rot{ang}", im.rotate(ang, resample=Image.BILINEAR, fillcolor=0)))
        for gm in (0.75, 1.33):
            v.append((f"gamma{gm}",
                      Image.fromarray(np.clip(
                          (np.asarray(im, np.float32) / 255.0) ** gm * 255, 0, 255
                      ).astype(np.uint8))))
        v.append(("bright", Image.fromarray(np.clip(
            np.asarray(im, np.float32) * 0.85, 0, 255).astype(np.uint8))))
        return v

    dis_pair = np.zeros(NP)
    dis_l1 = np.zeros(NP)
    with torch.no_grad():
        for i in range(NP):
            parts = keys[i].split("_")
            stem = "_".join(parts[2:])
            path = f"{PHOTO_DIR}/{parts[0]}/{parts[1]}/{stem}.jpg"
            try:
                im = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
            except Exception as e:  # noqa: BLE001
                log(f"  img miss {keys[i]}: {e}")
                continue
            cams_v = []
            for name, vv in views(im):
                x = tf(vv.convert("RGB"))[None]
                cam, _ = batch_cam(model, x, device)
                cams_v.append(cam[0])
            c0 = cams_v[0]
            b0 = c0 > THRESHOLDS[TAU45]
            ious = []
            l1s = []
            for cv in cams_v[1:]:
                bv = cv > THRESHOLDS[TAU45]
                inter = (b0 & bv).sum()
                union = (b0 | bv).sum()
                ious.append(inter / union if union else 1.0)
                l1s.append(np.abs(cv - c0).mean())
            dis_pair[i] = np.mean(ious)
            dis_l1[i] = np.mean(l1s)
    df["agree"] = dis_pair
    df["dis_l1"] = dis_l1
    df.to_csv(f"{OUT}/p62_photo_sage.csv", index=False)
    ok = dis_pair > 0
    if ok.sum() and harmful.sum() not in (0, len(harmful)):
        auc_a = roc_auc_score(harmful[ok], dis_pair[ok])
        auc_l = roc_auc_score(harmful[ok], dis_l1[ok])
        print(f"  AUROC(有害|agree) = {auc_a:.3f}  AUROC(有害|dis_l1) = {auc_l:.3f}")
    print(f"  photo 视图 agree 中位 {np.median(dis_pair[ok]):.3f}")

    # ---------- E. 色度泄漏检验 ----------
    log("E. 色度泄漏检验（RGB 统计 vs q9 灰度特征的域分类 AUROC）")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        rgb_feat_p = []
        for i in range(NP):
            parts = keys[i].split("_")
            stem = "_".join(parts[2:])
            path = f"{PHOTO_DIR}/{parts[0]}/{parts[1]}/{stem}.jpg"
            im = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), np.float32) / 255.0
            rg = im[..., 0] - im[..., 1]
            yb = 0.5 * im[..., 1] - 0.25 * (im[..., 0] + im[..., 2])
            rgb_feat_p.append([rg.mean(), rg.std(), yb.mean(), yb.std(),
                               im[..., 0].std(), im[..., 1].std(), im[..., 2].std()])
        rgb_feat_n = []
        import glob
        nih_files = sorted(glob.glob(f"{NIH_DIR}/*.png"))[:400]
        for pth in nih_files:
            im = np.asarray(Image.open(pth).convert("RGB").resize((64, 64)), np.float32) / 255.0
            rg = im[..., 0] - im[..., 1]
            yb = 0.5 * im[..., 1] - 0.25 * (im[..., 0] + im[..., 2])
            rgb_feat_n.append([rg.mean(), rg.std(), yb.mean(), yb.std(),
                               im[..., 0].std(), im[..., 1].std(), im[..., 2].std()])
        X_rgb = np.stack(rgb_feat_p + rgb_feat_n)
        y_rgb = np.array([1] * len(rgb_feat_p) + [0] * len(rgb_feat_n))
        auc_rgb = cross_val_score(LogisticRegression(max_iter=1000), X_rgb, y_rgb,
                                  cv=5, scoring="roc_auc").mean()
        # q9 灰度特征：photo vs NIH（cache 行）
        X_q9 = np.concatenate([q9s, np.concatenate([qs_n[:, 0], cs_n[:, 0]], 1)], 0)
        y_q9 = np.array([1] * len(q9s) + [0] * (qs_n.shape[0]))
        auc_q9 = cross_val_score(LogisticRegression(max_iter=1000), X_q9, y_q9,
                                 cv=5, scoring="roc_auc").mean()
        print(f"  域分类 AUROC：RGB 统计 {auc_rgb:.3f}（已知捷径） | q9 灰度特征 {auc_q9:.3f}")
    except Exception as e:  # noqa: BLE001
        log(f"  色度检验跳过：{e}")

    # ---------- F. photo 域 CRC 校准（患者级 5 折嵌套） ----------
    log("F. photo 域 CRC 校准（5 折嵌套，α=0.05）")
    up = sorted(set(pats))
    folds = [set(up[j::5]) for j in range(5)]
    alpha_target = 0.05
    all_outs, all_picks, all_raws, all_pat = [], [], [], []
    for jf in range(5):
        hold = folds[jf]
        te = np.array([p in hold for p in pats])
        tr = ~te
        # 折内患者 20% 校准
        tp = sorted(set(pats[tr]))
        rng_f = np.random.default_rng(100 + jf)
        cp = set(rng_f.choice(tp, size=max(2, len(tp) // 5), replace=False).tolist())
        cal = np.array([p in cp for p in pats])
        bcal = (-1e9, None)
        for lam2 in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for eps2 in [-0.02, 0.0, 0.02, 0.05, 0.1]:
                for mrg2 in [0.0, 0.02, 0.05]:
                    picks_f = np.zeros(NP, np.int64)
                    for i in range(NP):
                        Lf = mus_all[i] - lam2 * sgs_all[i]
                        a = int(np.argmax(Lf))
                        if a != 0 and Lf[a] > eps2 and mus_all[i, a] - mus_all[i, 0] >= mrg2:
                            picks_f[i] = a
                    g = np.array([df.gcand[i] if picks_f[i] == 2 else
                                  (df.gprior[i] if picks_f[i] == 1 else 0.0)
                                  for i in range(NP)])
                    gcal = g[cal]
                    wr = (gcal < -0.05).mean()
                    if wr <= alpha_target and gcal.mean() > bcal[0]:
                        bcal = (gcal.mean(), (lam2, eps2, mrg2))
        lam2, eps2, mrg2 = bcal[1] if bcal[1] else (0.0, 1e9, 0.0)  # 无可行→全 raw
        for i in range(NP):
            if not te[i]:
                continue
            Lf = mus_all[i] - lam2 * sgs_all[i]
            a = int(np.argmax(Lf))
            pk = 0
            if a != 0 and Lf[a] > eps2 and mus_all[i, a] - mus_all[i, 0] >= mrg2:
                pk = a
            outs = {0: df.raw45[i], 1: df.prior45[i], 2: df.cand_zs45[i]}
            all_outs.append(outs[pk])
            all_picks.append(pk)
            all_raws.append(df.raw45[i])
            all_pat.append(pats[i])
    all_outs, all_picks, all_raws = map(np.array, (all_outs, all_picks, all_raws))
    g_crc = all_outs - all_raws
    rng = np.random.default_rng(42)
    upat = np.array(sorted(set(all_pat)))
    fmap = {f: i for i, f in enumerate(upat)}
    fidx = np.array([fmap[p] for p in all_pat])
    vals = [g_crc[np.isin(fidx, rng.choice(len(upat), len(upat)))].mean()
            for _ in range(400)]
    print(f"\n== photo CRC-SAGE（5 折嵌套，α=0.05） ==")
    print(f"  Δ vs raw {g_crc.mean():+.4f} CI [{np.percentile(vals, 2.5):+.4f},"
          f"{np.percentile(vals, 97.5):+.4f}]  cov {(all_picks > 0).mean():.3f}  "
          f"wrong {(g_crc < -0.05).mean():.4f}")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
