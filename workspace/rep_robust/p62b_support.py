r"""第六阶段 6.2b：显式支持度检测器 s(x)（标签无关）+ 双域 Pareto 表。

动机（p62 发现）：收益预测器的 μ 阈值在 OOD 无保护力（photo Δ −0.040），
但 q9 特征域可分性 0.996 → 把"域支持"从隐式阈值改为显式检测器：
  s(x) = P(synth | q9 特征)，logistic，训练标签 = 域来源（synth=1 / photo=0），
  **不需要任何目标域收益标签**（label-free OOD detection）。

策略：sage-full = ( s(x) ≥ κ ) AND ( L_{a*} > ε AND margin ≥ m ) 才修复。
κ 取 synth 校准池 s 分布的分位数（κ=0.5 分位 → 合成侧放行 ≥95%）。

评估（两条链，患者级）：
  - photo 65（5 折患者 CV：s 用折外 photo 训练——留出折的 photo 不参与 s 拟合）：
    期望 Δ≈0、err≈0（正确拒绝）；对照 sage-zs(−0.040)/g2(−0.044)/v4(+0.001)。
  - synth 外测 test 154×18（s 用全量训练后零样本应用；合成行预期 s≈1）：
    期望保持 +0.053 收割。
  - κ 扫描的 risk-coverage（photo err vs synth gain 的 Pareto）。

用法：/root/miniconda3/bin/python p62b_support.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/chexlocalize"
ART = f"{ROOT}/repairer_v2_artifacts"
LAM, EPS, MRG = 0.0, -0.02, 0.02   # p61/p62 的 NIH 校准值


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mask_bbox(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    return [(float(xs.min()), float(ys.min()),
             float(xs.max()) + 1.0, float(ys.max()) + 1.0)]


def main() -> int:
    # ---------- 特征与已知 IoU ----------
    log("载入特征与已有评估")
    D = np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    q9_p = D["q9"]                       # (65,9)
    keys_p = [str(k) for k in D["keys"]]
    pat_p = np.array([k.split("_")[0] for k in keys_p])

    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
    qs_n = C["qs"]
    cs_n = C["cs"]
    q9_n3 = np.concatenate([qs_n, cs_n], 2)                # (146,17,9)
    NI, NC = q9_n3.shape[:2]
    q9_n_flat = q9_n3.reshape(-1, 9)

    dfp = pd.read_csv(f"{OUT}/p62_photo_sage.csv")          # photo IoUs + sage
    dft = pd.read_csv(f"{OUT}/p61_synth_test.csv")          # 外测 IoUs
    E = np.load(f"{OUT}/p61_test_cams.npz", allow_pickle=True)
    qv_ext = E["qv"]                                        # (2772,9)
    rowkey = [str(k) for k in E["rowkey"]]
    assert len(qv_ext) == len(dft), (len(qv_ext), len(dft))

    # sage 的 (μ,σ)：外测行需要重算 GM——p62 已存 photo 的 mus/sgs? 未存 → 重训 GM（确定性）
    import torch
    from p62_sage_photo import gain_nll_train, gain_predict, base_feat
    X_rows, Y_rows = [], []
    import csv
    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == "Cardiomegaly":
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = (x * s, y * s, (x + w) * s, (y + h) * s)
    files = list(C["files"])
    conds = list(C["conds"])
    cams_n = C["cams"].astype(np.float32)
    masks_n = C["masks"]
    probs_n = C["probs"]
    device = torch.device("cuda")
    heads_v2 = []
    from torch import nn
    import torch as _t

    class RepairHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(12, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1))

        def forward(self, hraw, mask, q):
            B, _, H, W = hraw.shape
            qb = q[:, :, None, None].expand(B, q.shape[1], H, W)
            xx = torch.cat([hraw, mask, hraw * mask, qb], dim=1)
            return torch.tanh(self.net(xx))

    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)
    ALPHA = 0.5
    TH = None
    from cam_benchmark import THRESHOLDS as _TH, cam_iou_curve
    TAU45 = int(np.where(np.isclose(_TH, 0.45))[0][0])

    def iou45(cam, gt):
        return float(cam_iou_curve(cam, gt)[TAU45])

    log("重算 NIH 收益标签与 GM（确定性复现）")
    with torch.no_grad():
        for i in range(NI):
            if files[i] not in gt_box:
                continue
            gt = [gt_box[files[i]]]
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
    X_n = np.stack(X_rows)
    Y_n = np.array(Y_rows, np.float32)
    AID_n = np.tile(np.arange(3), len(Y_n))
    GM = [gain_nll_train(X_n[AID_n == a], Y_n[:, a], 42 + 99 + a) for a in range(3)]

    def sage_row(qv9, prob):
        b = base_feat(qv9, prob)
        Xa = np.stack([np.concatenate([b, np.eye(3, dtype=np.float32)[a]])
                       for a in range(3)])
        mus, sgs = [], []
        for a in range(3):
            mu, sig = gain_predict(GM[a], Xa[a][None])
            mus.append(float(mu[0]))
            sgs.append(float(sig[0]))
        L = np.array(mus) - LAM * np.array(sgs)
        a_star = int(np.argmax(L))
        pick = 0
        if a_star != 0 and L[a_star] > EPS and mus[a_star] - mus[0] >= MRG:
            pick = a_star
        return pick

    log("s(x) 支持度检测器：logistic(q9)，synth=1 / photo=0")
    Xs = np.concatenate([q9_n_flat, q9_p], 0)
    ys = np.array([1] * len(q9_n_flat) + [0] * len(q9_p))

    # ---------- photo 侧：5 折患者 CV（s 折外） ----------
    log("photo 5 折患者 CV（s 折外训练）")
    up = sorted(set(pat_p))
    folds = [set(up[j::5]) for j in range(5)]
    picks_f = np.zeros(len(dfp), np.int64)
    s_photo = np.zeros(len(dfp))
    for jf in range(5):
        hold = folds[jf]
        te = np.array([p in hold for p in pat_p])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        # 折外：photo 训练部分 = 除 hold 外的 photo
        tr_photo_idx = np.where(~te)[0]
        Xtr = np.concatenate([q9_n_flat, q9_p[tr_photo_idx]], 0)
        ytr = np.array([1] * len(q9_n_flat) + [0] * len(tr_photo_idx))
        clf.fit(Xtr, ytr)
        s_h = clf.predict_proba(q9_p[te])[:, 1]
        s_photo[te] = s_h
        # κ = synth 校准池 s 分布 50 分位（用同一折的 clf 在 NIH 上取）
        s_syn = clf.predict_proba(q9_n_flat)[:, 1]
        kappa = float(np.quantile(s_syn, 0.5))
        for j, i in enumerate(np.where(te)[0]):
            pick = sage_row(q9_p[i], float(D["probs"][i]))
            if pick != 0 and s_h[j] < kappa:
                pick = 0
            picks_f[i] = pick
    outs_map = {0: dfp.raw45.to_numpy(), 1: dfp.prior45.to_numpy(),
                2: dfp.cand_zs45.to_numpy()}
    sage_support_photo = outs_map[0].copy()
    for a in (1, 2):
        sage_support_photo[picks_f == a] = outs_map[a][picks_f == a]
    d_ss = sage_support_photo - dfp.raw45.to_numpy()
    print("\n== photo（5 折患者 CV，s 折外） ==")
    print(f"  sage+support Δ {d_ss.mean():+.4f} cov {(picks_f > 0).mean():.3f} "
          f"wrong {(d_ss < -0.05).mean():.4f}")
    print(f"  [对照] sage-zs {(dfp.sage45 - dfp.raw45).mean():+.4f} | "
          f"g2 {(dfp.gated_v2_45 - dfp.raw45).mean():+.4f} | v4 ≈ +0.001")
    print(f"  photo s(x) 中位 {np.median(s_photo):.3f}（预期低 → 拒绝）")

    # ---------- synth 外测：s 全量训练后零样本应用 ----------
    log("synth 外测（s 全量训练）")
    clf_full = LogisticRegression(max_iter=1000, C=1.0)
    clf_full.fit(Xs, ys)
    s_syn_full = clf_full.predict_proba(q9_n_flat)[:, 1]
    kappa_f = float(np.quantile(s_syn_full, 0.5))
    s_ext = clf_full.predict_proba(qv_ext)[:, 1]
    picks_t = np.zeros(len(dft), np.int64)
    for j in range(len(dft)):
        pick = sage_row(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]))
        if pick != 0 and s_ext[j] < kappa_f:
            pick = 0
        picks_t[j] = pick
    outs_t = {0: dft.raw45.to_numpy(), 1: dft.prior45.to_numpy(),
              2: dft.cand_zs45.to_numpy()}
    sage_support_t = outs_t[0].copy()
    for a in (1, 2):
        sage_support_t[picks_t == a] = outs_t[a][picks_t == a]
    d_t = sage_support_t - dft.raw45.to_numpy()
    import numpy.random as npr
    rng = np.random.default_rng(42)
    up_t = dft.patient.unique()
    fmap = dft.patient.map({f: i for i, f in enumerate(up_t)}).to_numpy()
    vals = [d_t[np.isin(fmap, rng.choice(len(up_t), len(up_t)))].mean()
            for _ in range(400)]
    print("\n== synth 外测（test 154×18） ==")
    print(f"  sage+support Δ {d_t.mean():+.4f} CI [{np.percentile(vals, 2.5):+.4f},"
          f"{np.percentile(vals, 97.5):+.4f}]  cov {(picks_t > 0).mean():.3f}  "
          f"wrong {(d_t < -0.05).mean():.4f}")
    print(f"  [对照] sage-zs {(dft.sage45 - dft.raw45).mean():+.4f} | "
          f"g2 {(dft.gated_v2_45 - dft.raw45).mean():+.4f}")
    print(f"  外测 s(x) 中位 {np.median(s_ext):.3f}（预期高 → 放行）")

    # ---------- κ 扫描 Pareto ----------
    print("\n== κ 扫描（photo wrong vs synth gain） ==")
    for kq in [0.0, 0.25, 0.5, 0.75, 0.95]:
        kap = float(np.quantile(s_syn_full, kq))
        # photo
        pk_p = np.zeros(len(dfp), np.int64)
        for j, i in enumerate(range(len(dfp))):
            pick = sage_row(q9_p[i], float(D["probs"][i]))
            if pick != 0 and s_photo[i] < kap:
                pick = 0
            pk_p[i] = pick
        sp = outs_map[0].copy()
        for a in (1, 2):
            sp[pk_p == a] = outs_map[a][pk_p == a]
        dp = sp - dfp.raw45.to_numpy()
        # synth
        pk_t = np.zeros(len(dft), np.int64)
        for j in range(len(dft)):
            pick = sage_row(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]))
            if pick != 0 and s_ext[j] < kap:
                pick = 0
            pk_t[j] = pick
        st = outs_t[0].copy()
        for a in (1, 2):
            st[pk_t == a] = outs_t[a][pk_t == a]
        dt = st - dft.raw45.to_numpy()
        print(f"  κ(q{int(kq * 100):02d})={kap:.3f}  photo Δ {dp.mean():+.4f} "
          f"wrong {(dp < -0.05).mean():.4f} cov {(pk_p > 0).mean():.3f} | "
          f"synth Δ {dt.mean():+.4f} cov {(pk_t > 0).mean():.3f}")

    # 保存
    dfp["s_support"] = s_photo
    dfp["ss_pick"] = picks_f
    dfp["ss45"] = sage_support_photo
    dfp.to_csv(f"{OUT}/p62_photo_sage.csv", index=False)
    dft["s_support"] = s_ext
    dft["ss_pick"] = picks_t
    dft["ss45"] = sage_support_t
    dft.to_csv(f"{OUT}/p61_synth_test.csv", index=False)
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
