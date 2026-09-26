r"""第六阶段 6.4a：收益预测器留一退化族外测（泛化 vs 记忆）。

问题：GM 在 NIH 上的收益预测是学到了"退化→收益"的机制，还是逐条件记忆？
协议：退化 17 条件按物理族分组 5 族：
  jpeg{q50,q30,q10} / blur{1,3} / noise{003,006,010,025} / 光照{dark_05,bright_16,contrast_04} / 几何{ds_2,ds_4}+复合{combo_mild,combo_sev}+clean
留一族：GM 用其余族训练（患者级交叉拟合取消——直接全池训练，族外评估），
在该族条件上评估 μ 与真实收益的秩相关（患者级 Spearman）与选择收益。

对照：随机同大小条件留出（5 次重复）——若族外显著差于随机条件外，
说明存在"族级特异记忆"；若相当，说明预测器泛化的是机制。

用法：/root/miniconda3/bin/python p64_family_holdout.py
"""
from __future__ import annotations

import csv
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = "/root/autodl-tmp/experiments/rep_robust"
ART = f"{ROOT}/repairer_v2_artifacts"
ALPHA = 0.5
SEED = 42
TAU45 = int(np.where(np.isclose(__import__("cam_benchmark").THRESHOLDS, 0.45))[0][0])

FAMILIES = {
    "jpeg": ["jpeg_q50", "jpeg_q30", "jpeg_q10"],
    "blur": ["blur_1", "blur_3"],
    "noise": ["noise_003", "noise_006", "noise_010", "noise_025"],
    "light": ["dark_05", "bright_16", "contrast_04"],
    "geom": ["ds_2", "ds_4"],
    "combo": ["combo_mild", "combo_sev"],
    "clean": ["clean"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RepairHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(12, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, hraw, mask, q):
        B, _, H, W = hraw.shape
        qb = q[:, :, None, None].expand(B, q.shape[1], H, W)
        xx = torch.cat([hraw, mask, hraw * mask, qb], dim=1)
        return torch.tanh(self.net(xx))


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


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)

    C = np.load(f"{ROOT}/repairer/cache.npz", allow_pickle=True)
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
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)

    gt_box = {}
    with open(f"{ROOT}/BBox_List_2017.csv") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) >= 6 and row[1] == "Cardiomegaly":
                x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                s = 224.0 / 1024.0
                gt_box[row[0]] = (x * s, y * s, (x + w) * s, (y + h) * s)

    def base_feat(qv9, prob):
        frac, fok, conf = float(qv9[1]), float(qv9[0]), float(qv9[8])
        return np.concatenate([qv9, [frac, fok, conf, float(prob)]]).astype(np.float32)

    log("构建全部 (图,条件) 收益标签")
    Xa_rows, gains, cond_id, pat_id = [], [], [], []
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
                from cam_benchmark import cam_iou_curve
                r_i = float(cam_iou_curve(cams_n[i, c], gt)[TAU45])
                p_i = float(cam_iou_curve(cams_n[i, c] * masks_n[i, c], gt)[TAU45])
                c_i = float(cam_iou_curve(cand, gt)[TAU45])
                b = base_feat(q9, float(probs_n[i, c]))
                oneh = np.eye(3, dtype=np.float32)
                for a in range(3):
                    Xa_rows.append(np.concatenate([b, oneh[a]]))
                gains.append([0.0, p_i - r_i, c_i - r_i])
                cond_id.append(conds[c])
                pat_id.append(files[i].split("_")[0])
    X = np.stack(Xa_rows)
    G = np.array(gains, np.float32)
    CID = np.array(cond_id)
    PID = np.array(pat_id)
    AID = np.tile(np.arange(3), len(G))
    N = len(G)
    log(f"样本 {N} × 3 动作")

    fam_of = {}
    for fam, lst in FAMILIES.items():
        for cd in lst:
            fam_of[cd] = fam
    FID = np.array([fam_of[c] for c in CID])
    fams = [f for f in FAMILIES if f != "clean"]

    def run_holdout(hold_mask, tag):
        """hold_mask: (N,) 布尔，族外样本"""
        tr_s = ~hold_mask
        tr = np.repeat(tr_s, 3)
        MU = np.zeros((N, 3), np.float32)
        SG = np.zeros((N, 3), np.float32)
        for a in range(3):
            mdl = gain_nll_train(X[tr & (AID == a)], G[tr_s, a], SEED + 7)
            mu, sig = gain_predict(mdl, X[AID == a])
            MU[:, a] = mu
            SG[:, a] = sig
        # 评估：族外样本上 cand 动作的秩相关（按患者聚合取均值先）
        res = {}
        for a, an in [(1, "prior"), (2, "cand")]:
            m = hold_mask
            rho_p = []
            for p in sorted(set(PID[m])):
                mp = m & (PID == p)
                if mp.sum() >= 5:
                    rho_p.append(spearmanr(MU[mp, a], G[mp, a]).statistic)
            res[f"rho_{an}"] = float(np.nanmean(rho_p))
        # 选择收益（族外，ε=0 简单规则）
        L = MU - SG
        pick = np.argmax(L, 1)
        outs = np.where(pick == 0, 0.0, np.where(pick == 1, G[:, 1], G[:, 2]))
        res["gain"] = float(outs[hold_mask].mean())
        res["cov"] = float((pick[hold_mask] > 0).mean())
        res["wrong"] = float((outs[hold_mask] < -0.05).mean())
        print(f"  {tag:10s} rho_prior {res['rho_prior']:+.3f} rho_cand {res['rho_cand']:+.3f}"
              f"  gain {res['gain']:+.4f} cov {res['cov']:.3f} wrong {res['wrong']:.3f}")
        return res

    print("\n== 留一退化族外测 ==")
    rows = []
    for fam in fams:
        hold = FID == fam
        r = run_holdout(hold, f"hold:{fam}")
        r["hold"] = fam
        rows.append(r)
    print("\n== 对照：随机同大小条件留出（5 次） ==")
    rng = np.random.default_rng(0)
    all_conds = sorted(set(CID.tolist()))
    for rep in range(5):
        k = rng.choice(len(all_conds), 3, replace=False)
        hold = np.isin(CID, [all_conds[j] for j in k])
        r = run_holdout(hold, f"rand{rep}")
        r["hold"] = f"rand{rep}"
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{ROOT}/chexlocalize/p64_family_holdout.csv", index=False)
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
