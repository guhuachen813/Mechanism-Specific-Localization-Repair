r"""P0.2b：Atel 域三层阈值审计补算。

补齐 p65 缺失的两组量（本地 Tier-b 分析需要）：
  1. p65_atel_test.csv 缺 s_support（p65 只存了门控后的 ss_pick/ss45）
  2. p65_atel_photo.csv 缺未门控 sage_pick（只有 ss_pick=q25 源域门控版）

流程（与 p65 cmd_sage 完全一致的确定性重训）：
  - Atel GM 重训（seed 42+99+a）
  - Atel s(x) 分类器重训（NIH Atel q9 + OnePlus photo Atel q9，synth=1/photo=0）
  - photo 73：未门控 sage picks（NIH 校准 (λ,ε,m) 需重现——直接沿用 p65 打印的
    校准结果不可靠，这里重新网格搜索，protocol 同 p61：20% 患者校准池）
  - 输出 Tier-a（源域 κ）/ Tier-b（target 分位 κ，无 GT）/ Tier-c（GT 下界）双域表

用法：/root/miniconda3/bin/python p05_atel_s_backfill.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = "/root/autodl-tmp/experiments/rep_robust"
ROOTA = "/root/autodl-tmp/experiments/rep_robust_atel"
OUT = f"{ROOT}/chexlocalize"
ARTA = f"{ROOTA}/repairer_v2_artifacts_atel"
FINDING = "Atelectasis"
ALPHA = 0.5
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    if a_star != 0 and L[a_star] > eps and mus[a_star] - mus[0] >= mrg:
        return a_star
    return 0


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    from cam_benchmark import THRESHOLDS, cam_iou_curve
    TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])

    def iou45(cam, gt):
        return float(cam_iou_curve(cam, gt)[TAU45])

    C = np.load(f"{ROOTA}/repairer/cache.npz", allow_pickle=True)
    q9_n3 = np.concatenate([C["qs"], C["cs"]], 2)
    probs_n = C["probs"]
    files = list(C["files"])
    conds = list(C["conds"])
    cams_n = C["cams"].astype(np.float32)
    masks_n = C["masks"]
    NI, NC = cams_n.shape[:2]

    heads_v2 = []
    for kf in range(5):
        h = RepairHead().to(device)
        h.load_state_dict(torch.load(f"{ARTA}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)

    import csv
    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == FINDING:
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = [(x * s, y * s, (x + w) * s, (y + h) * s)]

    log("Atel GM 重训（确定性）")
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

    # 校准（p61 协议：患者 20% seed7）
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
    best = (-1e9, (0.0, -0.02, 0.02))
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
    log(f"校准 λ={lam} ε={eps} m={mrg}")

    # ---------- photo 73：未门控 sage picks ----------
    log("photo 73 未门控 sage picks")
    Dp = np.load(f"{OUT}/cache_photo_atel.npz", allow_pickle=True)
    q9_p = Dp["q9"]
    keys_p = [str(k) for k in Dp["keys"]]
    probs_p = Dp["probs"]
    dfp = pd.read_csv(f"{OUT}/p65_atel_photo.csv")
    assert len(dfp) == len(keys_p)
    pick_p = np.array([sage_pick_row(q9_p[i].astype(np.float32), float(probs_p[i]),
                                     GM, lam, eps, mrg) for i in range(len(keys_p))])
    dfp["sage_pick"] = pick_p
    outs_p = {0: dfp.raw45.to_numpy(), 1: dfp.prior45.to_numpy(),
              2: dfp.cand_zs45.to_numpy()}

    # ---------- synth 外测：s_support 回填 ----------
    log("synth 外测 s_support 回填 + 未门控 picks 复核")
    dft = pd.read_csv(f"{OUT}/p65_atel_test.csv")
    E = np.load(f"{OUT}/p65_atel_cams.npz", allow_pickle=True)
    qv_ext = E["qv"]
    q9_n_flat = q9_n3.reshape(-1, 9)
    Xs = np.concatenate([q9_n_flat, q9_p], 0)
    ys = np.array([1] * len(q9_n_flat) + [0] * len(q9_p))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xs, ys)
    s_ext = clf.predict_proba(qv_ext.astype(np.float32))[:, 1]
    s_photo = clf.predict_proba(q9_p.astype(np.float32))[:, 1]
    dfp["s_support"] = s_photo
    dft["s_support"] = s_ext
    dft.to_csv(f"{OUT}/p65_atel_test.csv", index=False)
    dfp.to_csv(f"{OUT}/p65_atel_photo.csv", index=False)
    log(f"已回填：photo s 中位 {np.median(s_photo):.3f} | 外测 s 中位 {np.median(s_ext):.3f}")

    # ---------- 三层阈值表 ----------
    def boot_ci(d, key, n=400, seed=42):
        up = pd.unique(key)
        fmap = pd.Series(key).map({f: i for i, f in enumerate(up)}).to_numpy()
        rng = np.random.default_rng(seed)
        vals = [d[np.isin(fmap, rng.choice(len(up), len(up)))].mean() for _ in range(n)]
        return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    def apply_rule(s, pk, outs, kappa):
        out = outs[0].copy()
        for a in (1, 2):
            m = (pk == a) & (s >= kappa)
            out[m] = outs[a][m]
        return out

    print("\n== Atel 三层阈值（p05 审计版） ==")
    s_syn_full = clf.predict_proba(q9_n_flat)[:, 1]
    print("-- Tier-a 源域规则：κ=synth s 分布分位（不看目标域任何数据） --")
    for q in (0.25, 0.5):
        kap = float(np.quantile(s_syn_full, q))
        # photo
        outp = apply_rule(s_photo, pick_p, outs_p, kap)
        dp = outp - outs_p[0]
        # synth
        pk_t = np.array([sage_pick_row(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]),
                                       GM, lam, eps, mrg) for j in range(len(dft))])
        outs_t = {0: dft.raw45.to_numpy(), 1: dft.prior45.to_numpy(),
                  2: dft.cand_zs45.to_numpy()}
        outt = apply_rule(s_ext, pk_t, outs_t, kap)
        dt = outt - outs_t[0]
        mp, lp, hp_ = boot_ci(dp, np.array([k.split("_")[0] for k in keys_p]))
        mt, lt, ht = boot_ci(dt, dft.patient.to_numpy())
        print(f"  κ(q{int(q*100):02d})={kap:.3f}  photo Δ {mp:+.4f} CI [{lp:+.4f},{hp_:+.4f}]"
              f" cov {(pick_p[s_photo >= kap] > 0).mean():.3f} | "
              f"synth Δ {mt:+.4f} CI [{lt:+.4f},{ht:+.4f}] cov {(pk_t[s_ext >= kap] > 0).mean():.3f}")
    print("-- Tier-b 无标签 transductive：κ=目标域自身 s 分位（无 GT） --")
    for q in (0.25, 0.5):
        outp = apply_rule(s_photo, pick_p, outs_p, np.quantile(s_photo, q))
        dp = outp - outs_p[0]
        pk_t = np.array([sage_pick_row(qv_ext[j].astype(np.float32), float(dft.prob.iloc[j]),
                                       GM, lam, eps, mrg) for j in range(len(dft))])
        outs_t = {0: dft.raw45.to_numpy(), 1: dft.prior45.to_numpy(),
                  2: dft.cand_zs45.to_numpy()}
        outt = apply_rule(s_ext, pk_t, outs_t, np.quantile(s_ext, q))
        dt = outt - outs_t[0]
        mp, lp, hp_ = boot_ci(dp, np.array([k.split("_")[0] for k in keys_p]))
        mt, lt, ht = boot_ci(dt, dft.patient.to_numpy())
        print(f"  q{int(q*100):02d}  photo Δ {mp:+.4f} CI [{lp:+.4f},{hp_:+.4f}] | "
              f"synth Δ {mt:+.4f} CI [{lt:+.4f},{ht:+.4f}]")
    print("-- Tier-c GT 下界（κ 网格选最优 photo wrong≤0.05 下的最大 synth gain） --")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
