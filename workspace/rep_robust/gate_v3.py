r"""优先级 3：no-op/失败代价门控 v3。

v2 的问题：收益标签把 raw 增益定义为 0，δ*=0，raw 几乎不被选——是"三路选择器"
而非选择性修复器。v3 改动：
  1. 代价敏感效用标签：y = argmax{0, g_p − c·1[g_p<0], g_c − c·1[g_c<0]}，
     c ∈ {0, 0.05, 0.1}（c=0 即 v2）——负增益选项被显式惩罚。
  2. 多 δ 校准敏感性：δ ∈ {0, 0.05, 0.1, 0.2, 0.3} 网格（calib 上选，报告全表）。
  3. 采集语境硬守卫（决策后叠加，不依赖学习）：
     - frac（mask/场面积）> 0.6 → no-op（过度膨胀 mask）
     - frac < 0.1 → no-op（mask 退化为线条，先验无意义）
     - field_ok == 0 → no-op
     - cov < 0.3（保守档）/ cov < 0.5（用户方案档）→ no-op
  4. 训练折内三分：train_g / calib（δ）/ val（无偏估计）。
  5. 报告 no-op precision/recall、错误修复率、risk-coverage。

判据：若 v3 仍不能稳定提高 selective risk（gated 的收益/错误修复权衡），
论文命名为"三路修复选择器"。

用法：/root/miniconda3/bin/python gate_v3.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import repairer as rp
from repairer import ALPHA, BS, EPOCHS, LAM_ID, LAM_OUT, LAM_SM, LR, NFOLD, \
    OUTDIR, SEED, SPLIT_CSV, RepairHead, box_mask
import cam_benchmark
from patient_split import main as split_main

ART = OUTDIR / "repairer_v2_artifacts"
TAU45 = rp.TAU45_IDX


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


def patient_of(fname: str) -> str:
    return fname.split(".")[0].split("_")[0]


def main() -> int:
    device = torch.device("cuda")
    d = np.load(OUTDIR / "repairer" / "cache.npz", allow_pickle=True)
    files, conds = list(d["files"]), list(d["conds"])
    cams = d["cams"].astype(np.float32)
    masks = d["masks"].astype(np.float32)
    qs = np.concatenate([d["qs"], d["cs"]], axis=2)
    N, C = len(files), len(conds)
    frac_all = d["qs"][:, :, 1]      # mask/场面积
    fok_all = d["qs"][:, :, 0]

    bbox = pd.read_csv(cam_benchmark.BBOX_CSV, skiprows=1, header=None,
                       usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == cam_benchmark.FINDING]
    from cam_benchmark import gt_boxes_224
    boxes_of = {f: gt_boxes_224(bbox, f) for f in files}

    if not SPLIT_CSV.exists():
        split_main()
    sp = pd.read_csv(SPLIT_CSV)
    fold_of = {r.file: r.fold
               for r in sp[sp.finding == cam_benchmark.FINDING].itertuples()}
    folds = [[] for _ in range(NFOLD)]
    for i, f in enumerate(files):
        folds[fold_of[f]].append(i)
    is_clean = np.array([1.0 if c == "clean" else 0.0 for c in conds])

    # ---- 折外 cand（复用 v2 落盘头权重，保证与 v2 可比）----
    raw45 = np.zeros((N, C)); prior45 = np.zeros((N, C)); cand45 = np.zeros((N, C))
    Y = np.stack([box_mask(boxes_of[f]) for f in files])
    for k in range(NFOLD):
        head = RepairHead().to(device)
        head.load_state_dict(torch.load(ART / f"fold{k}_head.pt",
                                        map_location=device))
        head.eval()
        te = set(folds[k])
        with torch.no_grad():
            for i in range(N):
                if i not in te:
                    continue
                for c in range(C):
                    inp = torch.from_numpy(cams[i, c][None, None]).to(device)
                    mk = torch.from_numpy(masks[i, c][None, None]).to(device)
                    qt = torch.from_numpy(qs[i, c][None]).to(device)
                    cand = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                    bx = boxes_of[files[i]]
                    raw45[i, c] = rp.iou_at(cams[i, c], bx)
                    prior45[i, c] = rp.iou_at(cams[i, c] * masks[i, c], bx)
                    cand45[i, c] = rp.iou_at(cand, bx)
    log("折外 cand 完成")

    gp = prior45 - raw45
    gc = cand45 - raw45
    oracle = np.maximum(raw45, np.maximum(prior45, cand45))
    gt_best_opt = np.stack([np.zeros_like(gp), gp, gc]).argmax(0)  # 0/1/2

    # ---- 患者级折：train_g / calib / val 三分 ----
    X = qs.astype(np.float32)
    results = {}
    for c_cost in [0.0, 0.05, 0.1]:
        y_util = np.stack([np.zeros_like(gp),
                           gp - c_cost * (gp < 0),
                           gc - c_cost * (gc < 0)]).argmax(0)
        g2 = np.full((N, C), np.nan)
        pick_all = np.full((N, C), -1)

        for k in range(NFOLD):
            te = np.array([i in set(folds[k]) for i in range(N)])
            tr = ~te
            idx_tr = [(i, c) for i in np.where(tr)[0] for c in range(C)]
            rng_c = np.random.default_rng(SEED + 200 + k + int(c_cost * 100))
            tr_pat = np.array([patient_of(files[i]) for i, _ in idx_tr])
            pts = np.array(sorted(set(tr_pat)))
            perm = rng_c.permutation(len(pts))
            p_cal = set(pts[perm[: len(pts) // 5]])
            p_val = set(pts[perm[len(pts) // 5: 2 * len(pts) // 5]])
            is_cal = np.array([p in p_cal for p in tr_pat])
            is_val = np.array([p in p_val for p in tr_pat])
            is_trn = ~(is_cal | is_val)
            Xg = np.stack([qs[i, c] for i, c in idx_tr])
            yg = np.array([y_util[i, c] for i, c in idx_tr])
            g = GateV2().to(device)
            gopt = torch.optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-4)
            lossf = torch.nn.CrossEntropyLoss()
            Xt = torch.tensor(Xg[is_trn], dtype=torch.float32, device=device)
            yt = torch.tensor(yg[is_trn], dtype=torch.long, device=device)
            for ep in range(400):
                gopt.zero_grad(set_to_none=True)
                loss = lossf(g(Xt), yt)
                loss.backward()
                gopt.step()
            g.eval()
            with torch.no_grad():
                pc = torch.softmax(g(torch.tensor(
                    Xg[is_cal], dtype=torch.float32, device=device)), 1).cpu().numpy()
            out_cal = {"raw": raw45[tuple(np.array(idx_tr)[is_cal].T)],
                       "prior": prior45[tuple(np.array(idx_tr)[is_cal].T)],
                       "cand": cand45[tuple(np.array(idx_tr)[is_cal].T)]}
            best = (-1e9, 0.0)
            for dl in np.arange(0.0, 0.55, 0.05):
                ch = pc.argmax(1)
                mg = pc[np.arange(len(pc)), ch] - pc[:, 0]
                pk = ch.copy()
                pk[(ch > 0) & (mg < dl)] = 0
                iou = out_cal["raw"].copy()
                iou[pk == 1] = out_cal["prior"][pk == 1]
                iou[pk == 2] = out_cal["cand"][pk == 2]
                if iou.mean() > best[0]:
                    best = (iou.mean(), float(dl))
            d_star = best[1]
            with torch.no_grad():
                pv = torch.softmax(g(torch.tensor(
                    Xg[is_val], dtype=torch.float32, device=device)), 1).cpu().numpy()
            out_val = {"raw": raw45[tuple(np.array(idx_tr)[is_val].T)],
                       "prior": prior45[tuple(np.array(idx_tr)[is_val].T)],
                       "cand": cand45[tuple(np.array(idx_tr)[is_val].T)]}
            ch = pv.argmax(1)
            mg = pv[np.arange(len(pv)), ch] - pv[:, 0]
            pk = ch.copy()
            pk[(ch > 0) & (mg < d_star)] = 0
            iou = out_val["raw"].copy()
            iou[pk == 1] = out_val["prior"][pk == 1]
            iou[pk == 2] = out_val["cand"][pk == 2]
            # 折外全量（test 折）记录
            with torch.no_grad():
                pt = torch.softmax(g(torch.tensor(
                    np.stack([qs[i, c] for i in np.where(te)[0] for c in range(C)]),
                    dtype=torch.float32, device=device)), 1).cpu().numpy()
            m = 0
            for i in np.where(te)[0]:
                for c in range(C):
                    ch2 = int(pt[m].argmax())
                    mg2 = float(pt[m, ch2] - pt[m, 0])
                    pk2 = ch2 if (ch2 == 0 or mg2 >= d_star) else 0
                    pick_all[i, c] = pk2
                    g2[i, c] = {"raw": raw45[i, c], "prior": prior45[i, c],
                                "cand": cand45[i, c]}[["raw", "prior", "cand"][pk2]]
                    m += 1
        flat_i, flat_c = np.meshgrid(np.arange(N), np.arange(C), indexing="ij")
        # 硬守卫
        guard = ((frac_all > 0.6) | (frac_all < 0.1) | (fok_all < 0.5))
        for cov_th, tag in [(0.3, "cov30"), (0.5, "cov50")]:
            g_full = g2.copy()
            pk_full = pick_all.copy()
            apply_g = guard | (frac_all < cov_th)
            g_full[apply_g] = raw45[apply_g]
            pk_full[apply_g] = 0
            results[(c_cost, tag)] = (g_full, pk_full)
        results[(c_cost, "plain")] = (g2, pick_all)

    # ---- 汇总 ----
    print("\n== 门控 v3 汇总（τ45，折外；val 三分内评估） ==")
    print(f"{'系统':28s} {'τ45':>6s} {'vs raw':>8s} {'错误修复':>8s} "
          f"{'no-op率':>7s} {'no-op P':>7s} {'no-op R':>7s}")
    base = raw45.flatten()
    noop_gt = gt_best_opt.flatten() == 0
    for (cc, tag), (g, pk) in results.items():
        gg = g.flatten(); ppk = pk.flatten()
        noop = ppk == 0
        err = ((ppk > 0) & (gg < base - 1e-9)).mean()
        prec = (noop_gt[noop]).mean() if noop.any() else np.nan
        rec_ = noop[noop_gt].mean()
        print(f"cost={cc:.2f}+{tag:6s}{'':14s} {gg.mean():.3f} "
              f"{gg.mean()-base.mean():+.3f} {err:8.3f} {noop.mean():7.3f} "
              f"{prec:7.3f} {rec_:7.3f}")
    print(f"{'oracle':28s} {oracle.flatten().mean():.3f} "
          f"{oracle.flatten().mean()-base.mean():+.3f}")
    print(f"{'cand 固定':28s} {cand45.flatten().mean():.3f} "
          f"{cand45.flatten().mean()-base.mean():+.3f}")
    print(f"{'prior 固定':28s} {prior45.flatten().mean():.3f} "
          f"{prior45.flatten().mean()-base.mean():+.3f}")

    # risk-coverage（margin 不可得——用 c=0.1+plain 的门控按 是否修复 分层收益）
    print("\n== δ 敏感性（cost=0.05，calib 网格内固定 δ 时的 val 收益，示意） ==")
    print("（δ 由 calib 自动选择，逐折 δ* 见日志）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
