r"""优先级 6（先做，供后续全部实验复用）+ 优先级 1 的训练侧产物。

repairer_v2.py：在 repairer.py 基础上的可复现版 v2 训练，一次产出：
  - 每折 RepairHead 权重 fold{k}_head.pt
  - 每折 GateV2 权重 fold{k}_gate.pt + 校准阈值 fold{k}_delta.npy
  - gate_v2_eval.csv（折外逐样本 choice/margin/各系统 τ45）
  - torch.use_deterministic_algorithms + 固定 seed；训练后保存 A_cand 不做
    （体积大），以 eval csv + 权重为准。

与 gate_v2.py 的差别：
  1. 头/门控/δ 全部落盘（零样本评估需要）。
  2. 确定性算法开启（torch 2.8 + CUBLAS_WORKSPACE_CONFIG）。
  3. 门控与 gate_v2.py 完全同构（三选一 + calib δ 网格），保证连续性。

用法：/root/miniconda3/bin/python repairer_v2.py           # 全量
      /root/miniconda3/bin/python repairer_v2.py --limit 4 # 冒烟
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import repairer as rp
from repairer import (ALPHA, BS, EPOCHS, LAM_ID, LAM_OUT, LAM_SM, LR, NFOLD,
                      OUTDIR, SEED, SPLIT_CSV, RepairHead, box_mask)
from patient_split import main as split_main  # 确保划分文件存在

torch.use_deterministic_algorithms(True, warn_only=True)

ARTDIR = OUTDIR / "repairer_v2_artifacts"
TAU45 = rp.TAU45_IDX


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GateV2(torch.nn.Module):
    def __init__(self, n_in: int = 9):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                       torch.nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def patient_of(fname: str) -> str:
    return fname.split(".")[0].split("_")[0]


def apply_gate(g: GateV2, X: np.ndarray, out: dict[str, np.ndarray],
               deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dev = next(g.parameters()).device
    with torch.no_grad():
        p = torch.softmax(g(torch.tensor(X, dtype=torch.float32, device=dev)),
                          dim=1).cpu().numpy()
    choice = p.argmax(1)
    margin = p[np.arange(len(p)), choice] - p[:, 0]
    pick = choice.copy()
    pick[(choice > 0) & (margin < deltas)] = 0
    iou = out["raw"].copy()
    iou[pick == 1] = out["prior"][pick == 1]
    iou[pick == 2] = out["cand"][pick == 2]
    return iou, pick


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--finding", default="Cardiomegaly")
    ap.add_argument("--ckpt", default="/root/project/outputs/repaired_seed42/densenet121/best.pt")
    ap.add_argument("--cache", default="")
    ap.add_argument("--artdir", default="")
    args = ap.parse_args()
    device = torch.device("cuda")

    # 病种参数化（repairer/cam_benchmark 的模块级常量）
    import cam_benchmark
    import pathlib
    cam_benchmark.CKPT = pathlib.Path(args.ckpt)
    cam_benchmark.FINDING = args.finding
    rp.FINDING = args.finding
    if args.cache:
        cache_path = pathlib.Path(args.cache)
    else:
        cache_path = OUTDIR / "repairer" / "cache.npz"
    if args.artdir:
        artdir = pathlib.Path(args.artdir)
    else:
        artdir = ARTDIR
    artdir.mkdir(exist_ok=True)

    d = np.load(cache_path, allow_pickle=True)
    files, conds = list(d["files"]), list(d["conds"])
    cams = d["cams"].astype(np.float32)
    masks = d["masks"].astype(np.float32)
    probs = d["probs"]
    qs = np.concatenate([d["qs"], d["cs"]], axis=2)
    N, C = len(files), len(conds)
    if args.limit:
        files, cams, masks, probs, qs = (files[: args.limit], cams[: args.limit],
                                         masks[: args.limit], probs[: args.limit],
                                         qs[: args.limit])
        N = len(files)

    from cam_benchmark import BBOX_CSV, FINDING, gt_boxes_224
    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == FINDING]
    boxes_of = {f: gt_boxes_224(bbox, f) for f in files}
    Y = np.stack([box_mask(boxes_of[f]) for f in files])

    if not SPLIT_CSV.exists():
        split_main()
    sp = pd.read_csv(SPLIT_CSV)
    fold_of = {r.file: r.fold for r in sp[sp.finding == FINDING].itertuples()}
    missing = [f for f in files if f not in fold_of]
    if missing:
        raise RuntimeError(f"划分文件缺 {len(missing)} 图，例: {missing[:3]}")
    folds = [[] for _ in range(NFOLD)]
    for i, f in enumerate(files):
        folds[fold_of[f]].append(i)
    if args.limit and min(len(x) for x in folds) == 0:
        # 冒烟模式：cache 只有前几张图，折可能为空——改为均衡抽折
        folds = [list(range(i, N, NFOLD)) for i in range(NFOLD)]
        log("冒烟模式：改用均衡折")
    is_clean = np.array([1.0 if c == "clean" else 0.0 for c in conds])
    ci = {c: k for k, c in enumerate(conds)}

    Xall = qs.astype(np.float32)
    raw45 = np.zeros((N, C))
    prior45 = np.zeros((N, C))
    cand45 = np.zeros((N, C))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(NFOLD):
        te = folds[k]
        tr = sorted(i for j in range(NFOLD) if j != k for i in folds[j])
        log(f"fold{k}: train {len(tr)} / test {len(te)}")
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
        torch.save(head.state_dict(), artdir / f"fold{k}_head.pt")

        # 折外 cand/45 与门控训练集
        cand_of = {}
        y_all, X_tr = [], []
        with torch.no_grad():
            for i in tr:
                for c in range(C):
                    inp = torch.from_numpy(cams[i, c][None, None]).to(device)
                    mk = torch.from_numpy(masks[i, c][None, None]).to(device)
                    qt = torch.from_numpy(qs[i, c][None]).to(device)
                    cand = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                    bx = boxes_of[files[i]]
                    raw45[i, c] = rp.iou_at(cams[i, c], bx)
                    prior45[i, c] = rp.iou_at(cams[i, c] * masks[i, c], bx)
                    cand45[i, c] = rp.iou_at(cand, bx)
                    cand_of[(i, c)] = cand
                    y_all.append([0.0, prior45[i, c] - raw45[i, c],
                                  cand45[i, c] - raw45[i, c]])
                    X_tr.append(qs[i, c])
            for i in te:
                for c in range(C):
                    inp = torch.from_numpy(cams[i, c][None, None]).to(device)
                    mk = torch.from_numpy(masks[i, c][None, None]).to(device)
                    qt = torch.from_numpy(qs[i, c][None]).to(device)
                    cand = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                    bx = boxes_of[files[i]]
                    raw45[i, c] = rp.iou_at(cams[i, c], bx)
                    prior45[i, c] = rp.iou_at(cams[i, c] * masks[i, c], bx)
                    cand45[i, c] = rp.iou_at(cand, bx)
                    cand_of[(i, c)] = cand
        y_all = np.array(y_all).argmax(1)
        X_tr = np.stack(X_tr)

        gate = GateV2().to(device)
        gopt = torch.optim.Adam(gate.parameters(), lr=LR, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        for ep in range(300):
            perm = rng.permutation(len(y_all))
            for s0 in range(0, len(perm), 256):
                sel = perm[s0:s0 + 256]
                q = torch.from_numpy(X_tr[sel]).to(device)
                y = torch.from_numpy(y_all[sel]).to(device)
                loss = lossf(gate(q), y)
                gopt.zero_grad(set_to_none=True)
                loss.backward()
                gopt.step()
        gate.eval()

        # 训练折内患者级 80/20 calib 校准 δ
        rng_c = np.random.default_rng(SEED + 100 + k)
        tr_pat = np.array([patient_of(files[i]) for i in tr
                           for _ in range(C)])
        pts = np.array(sorted(set(tr_pat)))
        calib_p = set(pts[rng_c.permutation(len(pts))[: max(1, len(pts) // 5)]])
        is_cal = np.array([p in calib_p for p in tr_pat])
        X_cal, out_cal = X_tr[is_cal], {
            "raw": raw45[np.repeat(np.array(tr), C)[is_cal]],
            "prior": prior45[np.repeat(np.array(tr), C)[is_cal]],
            "cand": cand45[np.repeat(np.array(tr), C)[is_cal]]}
        best = (-1e9, 0.0)
        for dl in np.arange(0.0, 0.55, 0.05):
            iou_c, _ = apply_gate(gate, X_cal, out_cal,
                                  np.full(is_cal.sum(), dl))
            if iou_c.mean() > best[0]:
                best = (iou_c.mean(), float(dl))
        d_star = best[1]
        np.save(artdir / f"fold{k}_delta.npy", np.array([d_star]))
        torch.save(gate.state_dict(), artdir / f"fold{k}_gate.pt")
        log(f"  fold{k} δ*={d_star:.2f}（calib {int(is_cal.sum())}）")

        # 折外记录
        if te:
            with torch.no_grad():
                p = torch.softmax(gate(torch.tensor(
                    np.stack([qs[i, c] for i in te for c in range(C)]),
                    dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        pick_all, m = 0, 0
        for i in te:
            for c in range(C):
                xq = qs[i, c]
                choice = int(p[m].argmax())
                margin = float(p[m, choice] - p[m, 0])
                pick = choice if (choice == 0 or margin >= d_star) else 0
                variants = {"raw": cams[i, c],
                            "prior": cams[i, c] * masks[i, c],
                            "cand": cand_of[(i, c)]}
                g2_out = ["raw", "prior", "cand"][pick]
                bx = boxes_of[files[i]]
                rows.append({
                    "fold": k, "nih_file": files[i], "condition": conds[c],
                    "choice": choice, "picked": pick, "margin": margin,
                    "prob": float(probs[i, c]),
                    "raw_45": raw45[i, c], "prior_45": prior45[i, c],
                    "cand_45": cand45[i, c],
                    "raw_best": rp.best_iou(cams[i, c], bx),
                    "prior_best": rp.best_iou(cams[i, c] * masks[i, c], bx),
                    "cand_best": rp.best_iou(cand_of[(i, c)], bx),
                    "gated2_45": rp.iou_at(variants[g2_out], bx),
                    "gated2_best": rp.best_iou(variants[g2_out], bx),
                    "center_45": rp.iou_at(cams[i, c] * rp.center_field_mask(masks[i, c]), bx),
                })
                m += 1

    df = pd.DataFrame(rows)
    df.to_csv(artdir / "gate_v2_eval.csv", index=False)
    print("\n== 患者级 5 折折外（τ45） ==")
    print(df[["raw_45", "prior_45", "cand_45", "gated2_45"]].mean().round(4).to_string())
    print("choice 分布 raw/prior/cand:",
          df.choice.value_counts(normalize=True).reindex([0, 1, 2]).round(3).to_dict())
    print("picked 分布 raw/prior/cand:",
          df.picked.value_counts(normalize=True).reindex([0, 1, 2]).round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
