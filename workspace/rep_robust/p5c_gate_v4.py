r"""5.2：收益门控 v4（photo 域适配版）——门控真正学"何时修复"。

协议（计划 5.2）：
  - 效用标签：y = argmax{0, g_prior - c·1[g_prior<0], g_cand - c·1[g_cand<0]}，
    c=0.05（负收益显式惩罚）；raw（no-op）允许胜出
  - 交叉拟合：门控训练用的 cand 收益来自"不含该患者的折内适配头"
    （嵌套 CV 内实现 leave-patient-out 交叉拟合）
  - 门控输入：q9 + frac + field_ok + cand_conflict（12 维）
  - 外层折评估：raw / cand_zs / adapt_es / gated_v4 / oracle；δ 在内层校准患者网格选
  - 报告：修复覆盖率、不修复率、错误修复率、risk-coverage

用法：/root/miniconda3/bin/python p5c_gate_v4.py
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
ART = "/root/autodl-tmp/experiments/rep_robust/repairer_v2_artifacts"
ALPHA = 0.5
N_OUTER = 5
C_NEG = 0.05
TAU = 0.45


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


class GateV4(torch.nn.Module):
    def __init__(self, n_in: int = 12):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                       torch.nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def iou_curve(cam, box):
    taus = np.arange(0.05, 1.0, 0.05)
    out = np.zeros(len(taus))
    x1, y1, x2, y2 = box
    for i, t in enumerate(taus):
        m = cam >= t
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ix = max(0, min(x2, bx2) - max(x1, bx1))
        iy = max(0, min(y2, by2) - max(y1, by1))
        inter = ix * iy
        union = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter
        out[i] = inter / max(union, 1)
    return out


def iou_at(cam, box, tau=TAU):
    return float(iou_curve(cam, box)[int(round((tau - 0.05) / 0.05))])


def box_of(m):
    ys, xs = np.where(m > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def train_head_es(idxs, cal_idx, D, device, epochs_max=60, eval_every=5):
    head = RepairHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    boxes = [box_of(D["gts"][i]) for i in idxs]
    Y = np.zeros((len(idxs), 1, 224, 224), np.float32)
    for j, (i, b) in enumerate(zip(idxs, boxes)):
        Y[j, 0, int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1.0
    rng = np.random.default_rng(0)
    best = (-1e9, None, 0)

    def cand_iou(i, hmod):
        h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
        m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
        q = torch.from_numpy(D["q9"][i][None]).to(device)
        with torch.no_grad():
            c = torch.clamp(h + ALPHA * hmod(h, m, q), 0, 1)[0, 0].cpu().numpy()
        return iou_at(c, box_of(D["gts"][i]))

    for ep in range(epochs_max):
        perm = rng.permutation(len(idxs))
        for s0 in range(0, len(perm), 16):
            sel = perm[s0:s0 + 16]
            ii = np.array([idxs[s] for s in sel])
            h = torch.from_numpy(D["cams_p"][ii]).unsqueeze(1).to(device)
            m = torch.from_numpy(D["masks_p"][ii].astype(np.float32)).unsqueeze(1).to(device)
            y = torch.from_numpy(Y[sel]).to(device)
            q = torch.from_numpy(D["q9"][ii]).to(device)
            delta = head(h, m, q)
            cand = torch.clamp(h + ALPHA * delta, 0, 1)
            loss = F.binary_cross_entropy(cand, y) \
                + 0.25 * (cand * (1 - m)).mean() \
                + 0.05 * ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                          + (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % eval_every == 0 or ep == epochs_max - 1:
            head.eval()
            sc = float(np.mean([cand_iou(i, head) for i in cal_idx]))
            if sc > best[0]:
                best = (sc, {k: v.clone() for k, v in head.state_dict().items()}, ep + 1)
            head.train()
    head.load_state_dict(best[1])
    head.eval()
    return head, best[2]


def cand_of(head, idxs, D, device):
    out = {}
    with torch.no_grad():
        for i in idxs:
            h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
            m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
            q = torch.from_numpy(D["q9"][i][None]).to(device)
            out[i] = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[0, 0].cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,2024")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda")

    D = dict(np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True))
    keys = list(D["keys"])
    patients = np.array([k.split("_")[0] for k in keys])
    uniq_p = sorted(set(patients))
    N = len(keys)
    log(f"缓存 {N} 图 / {len(uniq_p)} 患者")

    # 零样本 cand（合成轴 v2 5 头中位）
    from repairer import RepairHead as V2Head
    heads_v2 = []
    for kf in range(5):
        h = V2Head().to(device)
        h.load_state_dict(torch.load(f"{ART}/fold{kf}_head.pt", map_location=device))
        h.eval()
        heads_v2.append(h)

    def cand_zs_one(i):
        h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
        m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
        q = torch.from_numpy(D["q9"][i][None]).to(device)
        with torch.no_grad():
            cs = [torch.clamp(h + ALPHA * hh(h, m, q), 0, 1)[0, 0].cpu().numpy()
                  for hh in heads_v2]
        return np.median(np.stack(cs), 0)

    boxes = [box_of(D["gts"][i]) for i in range(N)]
    raw45 = np.array([iou_at(D["cams_p"][i], boxes[i]) for i in range(N)])
    prior45 = np.array([iou_at(D["cams_p"][i] * D["masks_p"][i], boxes[i]) for i in range(N)])
    cand_zs = {i: cand_zs_one(i) for i in range(N)}
    zs45 = np.array([iou_at(cand_zs[i], boxes[i]) for i in range(N)])
    log(f"全池 raw {raw45.mean():.3f} prior {prior45.mean():.3f} zs {zs45.mean():.3f}")

    rows = []
    rc_rows = []
    for seed in seeds:
        rng0 = np.random.default_rng(seed)
        perm = rng0.permutation(len(uniq_p))
        folds = [sorted(uniq_p[j] for j in perm[k::N_OUTER]) for k in range(N_OUTER)]
        for k in range(N_OUTER):
            te_p = set(folds[k])
            tr_p = [p for j, f in enumerate(folds) if j != k for p in f]
            te_idx = [i for i in range(N) if patients[i] in te_p]
            rng1 = np.random.default_rng(seed * 10 + k)
            tr_shuf = list(rng1.permutation(len(tr_p)))
            n_cal = max(4, len(tr_p) // 5)
            cal_p = set(tr_p[j] for j in tr_shuf[:n_cal])
            fit_p = [p for p in tr_p if p not in cal_p]
            cal_idx = [i for i in range(N) if patients[i] in cal_p]
            fit_idx = [i for i in range(N) if patients[i] in set(fit_p)]

            # 适配头（全 fit 池 52 内）
            head, bep = train_head_es(fit_idx, cal_idx, D, device)
            cand_fit = cand_of(head, fit_idx, D, device)

            # 交叉拟合收益标签：fit 池内每个患者的 cand 由"不含该患者"的头产生
            # （嵌套 4 折，患者级）
            rng_cf = np.random.default_rng(seed + 999 + k)
            fp = sorted(set(fit_p))
            cf_folds = [set(fp[j::4]) for j in range(4)]
            X_lab, y_lab, idx_lab = [], [], []
            for jf in range(4):
                hold_p = cf_folds[jf]
                t_p = [p for p in fp if p not in hold_p]
                t_idx = [i for i in fit_idx if patients[i] in set(t_p)]
                cal_sub = [i for i in cal_idx if patients[i] in set(t_p[: max(1, len(t_p) // 5)])]
                if not cal_sub:
                    cal_sub = t_idx[: max(2, len(t_idx) // 5)]
                hj, _ = train_head_es(t_idx, cal_sub, D, device)
                cands = cand_of(hj, [i for i in fit_idx if patients[i] in hold_p], D, device)
                for i, c in cands.items():
                    g_p = prior45[i] - raw45[i]
                    g_c = iou_at(c, boxes[i]) - raw45[i]
                    u = [0.0, g_p - C_NEG * (g_p < 0), g_c - C_NEG * (g_c < 0)]
                    yv = int(np.argmax(u))
                    # 12 维：q9 + frac + field_ok + cand_conflict
                    frac = float(D["q9"][i][1])
                    fok = float(D["q9"][i][0])
                    conflict = float(D["q9"][i][8])
                    X_lab.append(np.concatenate([D["q9"][i], [frac, fok, conflict]]))
                    y_lab.append(yv)
                    idx_lab.append(i)
            X_lab = np.stack(X_lab).astype(np.float32)
            y_lab = np.array(y_lab)

            gate = GateV4().to(device)
            gopt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
            lossf = torch.nn.CrossEntropyLoss()
            for ep in range(300):
                perm2 = np.random.default_rng(seed + k).permutation(len(y_lab))
                for s0 in range(0, len(perm2), 64):
                    sel = perm2[s0:s0 + 64]
                    q = torch.from_numpy(X_lab[sel]).to(device)
                    y = torch.from_numpy(y_lab[sel]).to(device)
                    loss = lossf(gate(q), y)
                    gopt.zero_grad(set_to_none=True)
                    loss.backward()
                    gopt.step()
            gate.eval()

            # δ 校准（内层 calib 患者上）
            cand_te_src = cand_of(head, te_idx, D, device)
            X_eval = []
            for i in te_idx:
                X_eval.append(np.concatenate([D["q9"][i], [D["q9"][i][1], D["q9"][i][0], D["q9"][i][8]]]))
            X_eval = np.stack(X_eval).astype(np.float32)
            with torch.no_grad():
                p_eval = torch.softmax(gate(torch.tensor(X_eval, dtype=torch.float32).to(device)), 1).cpu().numpy()
            # δ 网格在 calib 上选
            X_cal_eval = []
            for i in cal_idx:
                X_cal_eval.append(np.concatenate([D["q9"][i], [D["q9"][i][1], D["q9"][i][0], D["q9"][i][8]]]))
            X_cal_eval = np.stack(X_cal_eval).astype(np.float32)
            with torch.no_grad():
                p_cal = torch.softmax(gate(torch.tensor(X_cal_eval, dtype=torch.float32).to(device)), 1).cpu().numpy()
            best = (-1e9, 0.0)
            for dl in np.arange(0.0, 0.55, 0.05):
                outs = []
                for m_ in range(len(cal_idx)):
                    i = cal_idx[m_]
                    ch = int(p_cal[m_].argmax())
                    pick = ch if (ch == 0 or p_cal[m_, ch] - p_cal[m_, 0] >= dl) else 0
                    outs.append({0: raw45[i], 1: prior45[i],
                                 2: iou_at(cand_fit.get(i, cand_zs[i]), boxes[i])}[pick])
                if np.mean(outs) > best[0]:
                    best = (np.mean(outs), float(dl))
            d_star = best[1]

            # 外层折评估
            for m_, i in enumerate(te_idx):
                ch = int(p_eval[m_].argmax())
                margin = float(p_eval[m_, ch] - p_eval[m_, 0])
                pick = ch if (ch == 0 or margin >= d_star) else 0
                iou_pick = {0: raw45[i], 1: prior45[i], 2: iou_at(cand_te_src[i], boxes[i])}[pick]
                # oracle 三选一（诊断上界）
                iou_or = max(raw45[i], prior45[i], iou_at(cand_te_src[i], boxes[i]))
                rows.append({
                    "seed": seed, "fold": k, "i": i,
                    "raw45": raw45[i], "prior45": prior45[i],
                    "zs45": zs45[i], "adapt45": iou_at(cand_te_src[i], boxes[i]),
                    "gated45": iou_pick, "oracle45": iou_or,
                    "pick": pick, "choice": ch, "margin": margin, "delta": d_star,
                    "best_ep": bep,
                })
                rc_rows.append({"seed": seed, "fold": k, "i": i, "margin": margin,
                                "gain": iou_pick - raw45[i]})
            log(f"seed{seed} fold{k} δ*={d_star:.2f} bep={bep}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p5c_gate_v4.csv", index=False)
    df["err_flag"] = ((df.gated45 - df.raw45) < -0.05) & (df.pick > 0)
    df["repair_flag"] = df.pick > 0
    agg = df.groupby("seed").agg(
        raw=("raw45", "mean"), prior=("prior45", "mean"), zs=("zs45", "mean"),
        adapt=("adapt45", "mean"), gated=("gated45", "mean"), oracle=("oracle45", "mean"),
        repair_cov=("repair_flag", "mean"),
        err_repair=("err_flag", "mean"),
    )
    print("\n== 5.2 门控 v4（photo 域，外层折） ==")
    print(agg.round(3).to_string())
    d = df.gated45 - df.raw45
    print(f"\ngated vs raw: {d.mean():+.3f}（患者整群 bootstrap 需按 seed 分层）")
    da = df.adapt45 - df.raw45
    print(f"adapt vs raw: {da.mean():+.3f}")
    print(f"oracle vs raw: {(df.oracle45-df.raw45).mean():+.3f}（上界）")
    # risk-coverage（margin 降序累积；gain = gated45 - raw45）
    df["gain"] = df.gated45 - df.raw45
    rc = df.sort_values("margin", ascending=False).reset_index(drop=True)
    print("\nrisk-coverage（margin 前缀）：")
    for cv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        j = int(cv * len(rc)) - 1
        print(f"  cov {cv:.1f}: gain {rc['gain'][:j+1].mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
