r"""第四阶段 4：四组指标补全（定位准确性 / 采集偏移鲁棒性 / 安全性 / 分类保持性）。

复现 stage_train（同 seed 同超参 → 同折外头），在折外评估时记录：
  - 完整 IoU 曲线 → BoxAcc@0.25/0.5（raw/prior/cand/gated_v2）
  - agree-IoU：同一变体在 clean 与退化图上的 τ45 预测 mask 重叠
  - 质心位移：τ45 预测 mask 质心到 GT 框中心距离
  - relative rescue：(var−raw)/(clean_raw−raw)
  - 安全性：修复失败/错误修复/门控触发/risk-coverage（gated_v2 margin）
  - 分类保持性：prob 漂移（修复器只动 CAM，不动分类头，附注 AUROC 需负样本）

sanity：raw_45/prior_45/cand_45 与 repairer_eval.csv 逐值核对（|Δ|<1e-4）。
输出：repairer/metrics_eval.csv + 控制台汇总。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cam_benchmark import BBOX_CSV, FINDING, THRESHOLDS, cam_iou_curve, gt_boxes_224
from repairer import (ALPHA, BS, EPOCHS, GATE_DELTA, LAM_ID, LAM_OUT, LAM_SM,
                      LR, NFOLD, OUTDIR, SEED, SPLIT_CSV, RepairHead,
                      box_mask, center_field_mask, split_main)
from silver_eval import mask_to_224  # noqa: F401  (未用，仅保持导入面一致)

TAU45 = int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    device = torch.device("cuda")
    d = np.load(OUTDIR / "repairer" / "cache.npz", allow_pickle=True)
    files, conds = list(d["files"]), list(d["conds"])
    cams = d["cams"].astype(np.float32)
    masks = d["masks"].astype(np.float32)
    probs = d["probs"]
    qs = np.concatenate([d["qs"], d["cs"]], axis=2)
    N, C = len(files), len(conds)

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    boxes_of = {f: gt_boxes_224(bbox, f) for f in files}
    Y = np.stack([box_mask(boxes_of[f]) for f in files])
    gt_center = {f: ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
                 for f in files for bx in boxes_of[f][:1]} or {}

    if not SPLIT_CSV.exists():
        split_main()
    sp = pd.read_csv(SPLIT_CSV)
    fold_of = {r.file: r.fold for r in sp[sp.finding == FINDING].itertuples()}
    folds = [[] for _ in range(NFOLD)]
    for i, f in enumerate(files):
        folds[fold_of[f]].append(i)

    ci = {c: k for k, c in enumerate(conds)}
    is_clean = np.array([1.0 if c == "clean" else 0.0 for c in conds])
    ref = pd.read_csv(OUTDIR / "repairer" / "repairer_eval.csv")
    refkey = {(r.nih_file, r.condition): r for r in ref.itertuples()}
    gv2 = pd.read_csv(OUTDIR / "repairer" / "gate_v2_eval.csv")
    gv2key = {(r.nih_file, r.condition): r for r in gv2.itertuples()}

    def centroid_err(m: np.ndarray, f: str) -> float:
        if m.sum() == 0 or f not in gt_center:
            return np.nan
        ys, xs = np.where(m)
        gx, gy = gt_center[f]
        return float(np.hypot(xs.mean() - gx, ys.mean() - gy))

    def agree(a: np.ndarray, b: np.ndarray) -> float:
        ma, mb = a >= THRESHOLDS[TAU45] * max(a.max(), 1e-8), \
                 b >= THRESHOLDS[TAU45] * max(b.max(), 1e-8)
        u = (ma | mb).sum()
        return float((ma & mb).sum() / u) if u else 1.0

    rows = []
    rng = np.random.default_rng(SEED)
    for k in range(NFOLD):
        te = folds[k]
        tr = sorted(i for j in range(NFOLD) if j != k for i in folds[j])
        head = RepairHead().to(device)
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        tr_pairs = [(i, c) for i in tr for c in range(C)]
        for ep in range(EPOCHS):
            perm = rng.permutation(len(tr_pairs))
            for s0 in range(0, len(perm), BS):
                sel = perm[s0:s0 + BS]
                idx = [tr_pairs[s] for s in sel]
                ii = np.array([a for a, _ in idx])
                cc = np.array([b for _, b in idx])
                h = torch.from_numpy(cams[ii, cc]).unsqueeze(1).to(device)
                m = torch.from_numpy(masks[ii, cc]).unsqueeze(1).to(device)
                y = torch.from_numpy(Y[ii]).unsqueeze(1).to(device)
                q = torch.from_numpy(qs[ii, cc]).to(device)
                delta = head(h, m, q)
                cand = torch.clamp(h + ALPHA * delta, 0, 1)
                l_box = F.binary_cross_entropy(cand, y)
                clean_m = torch.from_numpy(is_clean[cc] == 1).to(device)
                l_id = (delta[clean_m] ** 2).mean() if clean_m.any() \
                    else torch.tensor(0.0, device=device)
                l_out = (cand * (1 - m)).mean()
                l_sm = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean() + \
                       (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
                loss = l_box + LAM_ID * l_id + LAM_OUT * l_out + LAM_SM * l_sm
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        head.eval()
        # clean 折外输出（agree-IoU 参照）
        clean_out = {}
        with torch.no_grad():
            for i in te:
                inp = torch.from_numpy(cams[i, ci["clean"]][None, None]).to(device)
                mk = torch.from_numpy(masks[i, ci["clean"]][None, None]).to(device)
                qt = torch.from_numpy(qs[i, ci["clean"]][None]).to(device)
                cand_c = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                clean_out[i] = {"raw": cams[i, ci["clean"]],
                                "prior": cams[i, ci["clean"]] * masks[i, ci["clean"]],
                                "cand": cand_c}
            for i in te:
                for c in range(C):
                    f, cond = files[i], conds[c]
                    inp = torch.from_numpy(cams[i, c][None, None]).to(device)
                    mk = torch.from_numpy(masks[i, c][None, None]).to(device)
                    qt = torch.from_numpy(qs[i, c][None]).to(device)
                    cand = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                    h, mm, bx = cams[i, c], masks[i, c], boxes_of[f]
                    prior = h * mm
                    r = gv2key[(f, cond)]
                    variants = {"raw": h, "prior": prior, "cand": cand}
                    sel = ["raw", "prior", "cand"][int(r.gated2_choice)]
                    g2_out = variants[sel]
                    rec = {"fold": k, "nih_file": f, "condition": cond,
                           "gated2_choice": int(r.gated2_choice),
                           "gated2_margin": float(r.gated2_margin),
                           "prob": float(probs[i, c]),
                           "prob_clean": float(probs[i, ci["clean"]])}
                    for v, cam in variants.items():
                        curve = cam_iou_curve(cam, bx)
                        rec[f"{v}_best"] = float(max(curve))
                        rec[f"{v}_45"] = float(curve[TAU45])
                        acc = np.array(curve)
                        rec[f"{v}_ba25"] = float(acc[THRESHOLDS <= 0.25].max())
                        rec[f"{v}_ba50"] = float(acc[THRESHOLDS <= 0.5].max())
                        pm = cam >= THRESHOLDS[TAU45] * max(cam.max(), 1e-8)
                        rec[f"{v}_cent"] = centroid_err(pm.astype(np.uint8), f)
                        if c != ci["clean"]:
                            rec[f"{v}_agree"] = agree(cam, clean_out[i][v])
                    rec["gated2_45"] = float(cam_iou_curve(g2_out, bx)[TAU45])
                    rec["gated2_ba25"] = float(
                        np.array(cam_iou_curve(g2_out, bx))[THRESHOLDS <= 0.25].max())
                    if c != ci["clean"]:
                        rec["gated2_agree"] = agree(g2_out, clean_out[i][sel])
                    rec["gated2_cent"] = centroid_err(
                        (g2_out >= THRESHOLDS[TAU45] * max(g2_out.max(), 1e-8))
                        .astype(np.uint8), f)
                    rows.append(rec)
        log(f"fold{k} 完成")

    df = pd.DataFrame(rows)
    # sanity：与 v1 eval 逐值核对
    diff = []
    for r in df.itertuples():
        o = refkey[(r.nih_file, r.condition)]
        diff += [abs(r.raw_45 - o.raw_45), abs(r.prior_45 - o.prior_45),
                 abs(r.cand_45 - o.cand_45)]
    log(f"sanity vs repairer_eval.csv：max|Δ|={max(diff):.2e}（应 <1e-4）")
    df.to_csv(OUTDIR / "repairer" / "metrics_eval.csv", index=False)

    # ---- 汇总 ----
    deg = df[df.condition != "clean"]
    cl = df[df.condition == "clean"]
    print("\n== 定位准确性（均值） ==")
    print(df[["raw_best", "raw_45", "raw_ba25", "raw_ba50",
              "prior_45", "prior_ba25", "prior_ba50",
              "cand_45", "cand_ba25", "cand_ba50",
              "gated2_45", "gated2_ba25"]].mean().round(3).to_string())
    print("\n== 采集偏移鲁棒性（退化条件） ==")
    for v in ["raw", "prior", "cand", "gated2"]:
        d45 = deg[f"{v}_45"] - deg["raw_45"]
        print(f"  {v:7s} Δτ45 {d45.mean():+.3f}  agree-IoU "
              f"{deg[f'{v}_agree'].mean():.3f}  质心误差 {deg[f'{v}_cent'].mean():.2f}px")
    clean_raw_of = cl.set_index("nih_file").raw_45
    gap = deg.nih_file.map(clean_raw_of) - deg.raw_45      # 该图 clean raw − 退化 raw
    m_ok = gap > 0.01
    resc = ((deg.gated2_45 - deg.raw_45)[m_ok] / gap[m_ok])
    resc_p = ((deg.prior_45 - deg.raw_45)[m_ok] / gap[m_ok])
    resc_c = ((deg.cand_45 - deg.raw_45)[m_ok] / gap[m_ok])
    print(f"  relative rescue（gap>0.01 子集，n={m_ok.sum()}）: "
          f"prior {resc_p.mean():+.3f}  cand {resc_c.mean():+.3f}  "
          f"gated2 {resc.mean():+.3f}")
    print("\n== 安全性 ==")
    hurt = df[df.cand_45 < df.raw_45 - 0.1]
    hurt_applied = df[(df.gated2_choice == 2) & (df.cand_45 < df.raw_45 - 0.1)]
    print(f"  修复致伤（cand 掉>0.1）: {len(hurt)}/{len(df)} ({len(hurt)/len(df):.1%})")
    print(f"  错误修复（gated2 选了 cand 且致伤）: {len(hurt_applied)}/{len(df)}")
    print(f"  clean 损失 gated2−raw: {(cl.gated2_45-cl.raw_45).mean():+.3f}")
    # risk-coverage：按 margin 从高到低累计应用 cand/prior 的收益
    dd = deg.sort_values("gated2_margin", ascending=False)
    gains, covs = [], []
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        m = int(len(dd) * frac)
        sel = dd.iloc[:m]
        alt = np.where(sel.gated2_choice == 2, sel.cand_45,
                       np.where(sel.gated2_choice == 1, sel.prior_45, sel.raw_45))
        gains.append((alt - sel.raw_45).mean())
        covs.append(frac)
    print("  risk-coverage（margin top-k 的平均收益）:")
    for c_, g_ in zip(covs, gains):
        print(f"    cov {c_:.0%}: {g_:+.3f}")
    print("\n== 分类保持性 ==")
    dp = deg.prob - deg.prob_clean
    print(f"  prob 漂移（退化−clean）: mean {dp.mean():+.3f}, median {dp.median():+.3f}")
    print("  注：修复器/门控仅修改 CAM 读出，不触碰分类头；AUROC 需要阴性样本，"
          "在 146 张阳性内不可定义（列入外部验证任务）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
