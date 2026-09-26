r"""P4：跨 CAM 家族修复头重训（GradCAM++ / LayerCAM）。

动机：主链（cache.npz → repairer_v2）全部绑定 Grad-CAM；检验"bounded
residual 修复 + 三选一门控"的结论是否 CAM 家族特异。

协议：
  A. 用 cam_multi.multi_cam 对 cache.npz 的同一批 (图, 条件) 计算
     gradcam_pp / layercam（逐位复现 cam_multi_nih.csv 的公式），落盘
     cam_family_cams.npz（fp16）。
  B. 对每个非 Grad-CAM 家族：q 特征的 cam_feats 两维换成该家族 CAM 重算，
     其余 7 维 q_vector 不变；按与 repairer_v2.py 完全相同的患者级 5 折、
     超参、seed、确定性设置重训 RepairHead + GateV2 + calib δ，折外评估。
  Grad-CAM 参照直接复用 repairer_v2_artifacts/gate_v2_eval.csv（同折同协议）。

比较（按用户 P4 规格）：配对增益方向（P(cand>raw)、meanΔ 按条件）、
risk-coverage（门控 margin 排序的选择性收益曲线）、错误修复率；
不要求零样本迁移、不做跨族绝对 IoU 排名。

用法：/root/miniconda3/bin/python p4_cam_family.py [--limit 8]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import degradations as dg
from cam_benchmark import (BBOX_CSV, IMAGE_SIZE, MEAN, SEED, STD,
                           build_index, gt_boxes_224, load_model,
                           load_shard_images)
import repairer as rp
from repairer import (ALPHA, BS, EPOCHS, LAM_ID, LAM_OUT, LAM_SM, LR, NFOLD,
                      SPLIT_CSV, RepairHead, box_mask, cam_feats)
from cam_multi import multi_cam

torch.use_deterministic_algorithms(True, warn_only=True)

OUT = rp.OUTDIR                      # /root/autodl-tmp/experiments/rep_robust
ART = OUT / "p4_cam_family"
CAMSF = OUT / "repairer" / "cam_family_cams.npz"
TAU45 = rp.TAU45_IDX
FAMILIES = ["gradcam_pp", "layercam"]


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


# ---------------- A. 家族 CAM 缓存 ----------------

def compute_family_cams(device) -> None:
    model = load_model(device)
    d = np.load(OUT / "repairer" / "cache.npz", allow_pickle=True)
    files, conds = list(d["files"]), list(d["conds"])
    N, C = len(files), len(conds)
    pp = np.zeros((N, C, IMAGE_SIZE, IMAGE_SIZE), np.float16)
    lc = np.zeros_like(pp)

    import pandas as pd
    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == "Cardiomegaly"]
    fidx = {f: i for i, f in enumerate(sorted(bbox["Image Index"].unique()))}

    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    with contextlib.redirect_stdout(io.StringIO()):
        _, loc = build_index()

    for s0 in range(0, N, 8):
        chunk = list(range(s0, min(s0 + 8, N)))
        by_shard: dict[int, list[int]] = {}
        for i in chunk:
            by_shard.setdefault(loc[files[i]][0], []).append(loc[files[i]][1])
        shard_imgs = {si: load_shard_images(si, rl) for si, rl in by_shard.items()}
        imgs = {i: shard_imgs[loc[files[i]][0]][loc[files[i]][1]].resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR) for i in chunk}
        for c, cond in enumerate(conds):
            x = torch.stack([tf(dg.degrade(imgs[i], cond,
                                           dg.rng_for(SEED, fidx[files[i]]))
                                .convert("RGB")) for i in chunk])
            # 注意：multi_cam 内部需要 autograd 求梯度，不能包 no_grad
            cams_fam = multi_cam(model, x, device)
            for fam, cams in cams_fam.items():
                arr = pp if fam == "gradcam_pp" else lc
                for j, i in enumerate(chunk):
                    # layercam 逐像素和可为大负值（gradcam_pp 有 relu_，layercam 无）：
                    # 与 cam_iou_curve 语义一致，先截到 [0,1] 再存 fp16（否则 |负|>65504 变 -inf）
                    v = np.clip(np.nan_to_num(cams[j], nan=0.0, posinf=1.0, neginf=0.0),
                                0.0, 1.0)
                    arr[i, c] = v.astype(np.float16)
        if (s0 // 8) % 3 == 0:
            log(f"  cams {s0 + len(chunk)}/{N}")
    np.savez_compressed(CAMSF, gradcam_pp=pp, layercam=lc,
                        files=np.array(files), conds=np.array(conds))
    log(f"家族 CAM 缓存写入 {CAMSF}")


# ---------------- B. 家族内重训（与 repairer_v2.py 同协议） ----------------

def gate_probs(g, X):
    dev = next(g.parameters()).device
    with torch.no_grad():
        p = torch.softmax(g(torch.tensor(X, dtype=torch.float32, device=dev)),
                          dim=1).cpu().numpy()
    choice = p.argmax(1)
    margin = p[np.arange(len(p)), choice] - p[:, 0]
    return choice, margin


def apply_gate(g, X, out, deltas):
    choice, margin = gate_probs(g, X)
    pick = choice.copy()
    pick[(choice > 0) & (margin < deltas)] = 0
    iou = out["raw"].copy()
    iou[pick == 1] = out["prior"][pick == 1]
    iou[pick == 2] = out["cand"][pick == 2]
    return iou, pick, choice, margin


def train_family(fam, cams_f, masks, qs_base, files, conds, boxes_of,
                 folds, device) -> pd.DataFrame:
    N, C = len(files), len(conds)
    # 家族专属 q：前 7 维 q_vector 不变，后 2 维 cam_feats 用本家族 CAM 重算
    cf2 = np.zeros((N, C, 2), np.float32)
    for i in range(N):
        for c in range(C):
            cf2[i, c] = cam_feats(cams_f[i, c], masks[i, c])
    qs_f = np.concatenate([qs_base[..., :7], cf2], axis=2).astype(np.float32)

    Y = np.stack([box_mask(boxes_of[f]) for f in files])
    is_clean = np.array([1.0 if c == "clean" else 0.0 for c in conds])

    raw45 = np.zeros((N, C)); prior45 = np.zeros((N, C)); cand45 = np.zeros((N, C))
    rows = []
    torch.manual_seed(rp.SEED)
    rng = np.random.default_rng(rp.SEED)
    for k in range(NFOLD):
        te = folds[k]
        tr = sorted(i for j in range(NFOLD) if j != k for i in folds[j])
        log(f"[{fam}] fold{k}: train {len(tr)} / test {len(te)}")
        head = RepairHead().to(device)
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        tr_pairs = [(i, c) for i in tr for c in range(C)]
        for ep in range(EPOCHS):
            perm = rng.permutation(len(tr_pairs))
            for s0 in range(0, len(perm), BS):
                sel = perm[s0:s0 + BS]
                idx = [tr_pairs[s] for s in sel]
                ii = np.array([a for a, _ in idx]); cc = np.array([b for _, b in idx])
                h = torch.from_numpy(cams_f[ii, cc]).unsqueeze(1).to(device)
                m = torch.from_numpy(masks[ii, cc]).unsqueeze(1).to(device)
                y = torch.from_numpy(Y[ii]).unsqueeze(1).to(device)
                q = torch.from_numpy(qs_f[ii, cc]).to(device)
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
        torch.save(head.state_dict(), ART / f"{fam}_fold{k}_head.pt")

        # 折外 cand + 门控训练集
        y_all, X_tr = [], []
        with torch.no_grad():
            for i in tr:
                for c in range(C):
                    inp = torch.from_numpy(cams_f[i, c][None, None]).to(device)
                    mk = torch.from_numpy(masks[i, c][None, None]).to(device)
                    qt = torch.from_numpy(qs_f[i, c][None]).to(device)
                    cand = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                    bx = boxes_of[files[i]]
                    raw45[i, c] = rp.iou_at(cams_f[i, c], bx)
                    prior45[i, c] = rp.iou_at(cams_f[i, c] * masks[i, c], bx)
                    cand45[i, c] = rp.iou_at(cand, bx)
                    y_all.append([0.0, prior45[i, c] - raw45[i, c],
                                  cand45[i, c] - raw45[i, c]])
                    X_tr.append(qs_f[i, c])
            cand_te = {}
            for i in te:
                for c in range(C):
                    inp = torch.from_numpy(cams_f[i, c][None, None]).to(device)
                    mk = torch.from_numpy(masks[i, c][None, None]).to(device)
                    qt = torch.from_numpy(qs_f[i, c][None]).to(device)
                    cand = torch.clamp(inp + ALPHA * head(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                    bx = boxes_of[files[i]]
                    raw45[i, c] = rp.iou_at(cams_f[i, c], bx)
                    prior45[i, c] = rp.iou_at(cams_f[i, c] * masks[i, c], bx)
                    cand45[i, c] = rp.iou_at(cand, bx)
                    cand_te[(i, c)] = cand
        y_all = np.array(y_all).argmax(1)
        X_tr = np.stack(X_tr)

        gate = GateV2().to(device)
        gopt = torch.optim.Adam(gate.parameters(), lr=LR, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        for ep in range(300):
            perm = rng.permutation(len(y_all))
            for s0 in range(0, len(perm), 256):
                sel = perm[s0:s0 + 256]
                loss = lossf(gate(torch.from_numpy(X_tr[sel]).to(device)),
                             torch.from_numpy(y_all[sel]).to(device))
                gopt.zero_grad(set_to_none=True)
                loss.backward()
                gopt.step()
        gate.eval()
        torch.save(gate.state_dict(), ART / f"{fam}_fold{k}_gate.pt")

        # calib δ（患者级 80/20）
        rng_c = np.random.default_rng(rp.SEED + 100 + k)
        tr_pat = np.array([patient_of(files[i]) for i in tr for _ in range(C)])
        pts = np.array(sorted(set(tr_pat)))
        calib_p = set(pts[rng_c.permutation(len(pts))[: max(1, len(pts) // 5)]])
        is_cal = np.array([p in calib_p for p in tr_pat])
        X_cal, out_cal = X_tr[is_cal], {
            "raw": raw45[np.repeat(np.array(tr), C)[is_cal]],
            "prior": prior45[np.repeat(np.array(tr), C)[is_cal]],
            "cand": cand45[np.repeat(np.array(tr), C)[is_cal]]}
        best = (-1e9, 0.0)
        for dl in np.arange(0.0, 0.55, 0.05):
            iou_c, _, _, _ = apply_gate(gate, X_cal, out_cal,
                                        np.full(int(is_cal.sum()), dl))
            if iou_c.mean() > best[0]:
                best = (iou_c.mean(), float(dl))
        d_star = best[1]
        np.save(ART / f"{fam}_fold{k}_delta.npy", np.array([d_star]))

        # 折外记录
        if te:
            X_te = np.stack([qs_f[i, c] for i in te for c in range(C)])
            ch_te, mg_te = gate_probs(gate, X_te)
        m = 0
        for i in te:
            for c in range(C):
                pick = int(ch_te[m]) if (int(ch_te[m]) == 0 or mg_te[m] >= d_star) else 0
                variants = {"raw": cams_f[i, c], "prior": cams_f[i, c] * masks[i, c],
                            "cand": cand_te[(i, c)]}
                bx = boxes_of[files[i]]
                rows.append({
                    "fold": k, "nih_file": files[i], "condition": conds[c],
                    "family": fam, "choice": int(ch_te[m]), "picked": pick,
                    "margin": float(mg_te[m]),
                    "raw_45": raw45[i, c], "prior_45": prior45[i, c],
                    "cand_45": cand45[i, c],
                    "gated_45": rp.iou_at(variants[["raw", "prior", "cand"][pick]], bx),
                })
                m += 1
    return pd.DataFrame(rows)


# ---------------- 汇总 ----------------

def summarize(df: pd.DataFrame, tag: str) -> None:
    base = df.raw_45.to_numpy()
    print(f"\n== [{tag}] 折外（τ45，n={len(df)}） ==")
    print(f"  raw    {base.mean():.4f}")
    print(f"  prior  {df.prior_45.mean():.4f}  配对 Δ {np.mean(df.prior_45 - base):+.4f}  P(prior>raw) {(df.prior_45 > base).mean():.3f}")
    print(f"  cand   {df.cand_45.mean():.4f}  配对 Δ {np.mean(df.cand_45 - base):+.4f}  P(cand>raw) {(df.cand_45 > base).mean():.3f}")
    print(f"  gated  {df.gated_45.mean():.4f}  配对 Δ {np.mean(df.gated_45 - base):+.4f}")
    print(f"  pick 分布 raw/prior/cand: "
          f"{df.picked.value_counts(normalize=True).reindex([0,1,2]).round(3).to_dict()}")
    err = ((df.picked > 0) & (df.gated_45 < base - 1e-9)).mean()
    print(f"  错误修复率（picked>0 且 gated<raw）: {err:.3f}")
    # 按条件的 cand 配对增益方向
    byc = df.groupby("condition").apply(
        lambda g: pd.Series({"meanD": (g.cand_45 - g.raw_45).mean(),
                             "Pup": (g.cand_45 > g.raw_45).mean()}), include_groups=False)
    pos = byc[byc.meanD > 0]
    print(f"  cand 增益为正的条件: {len(pos)}/{len(byc)}（{' '.join(pos.index[:6])}…）")
    # risk-coverage：margin 降序的选择性收益（cand 固定 与 gated 两条）
    ord_ = df.sort_values("margin", ascending=False).reset_index()
    gains_c = (ord_.cand_45 - ord_.raw_45).to_numpy()
    gains_g = (ord_.gated_45 - ord_.raw_45).to_numpy()
    print("  risk-coverage（coverage→selective meanΔ）:")
    for f in [0.2, 0.4, 0.6, 0.8, 1.0]:
        n = max(1, int(len(ord_) * f))
        print(f"    cov={f:.1f}  cand {gains_c[:n].mean():+.4f}  gated {gains_g[:n].mean():+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    device = torch.device("cuda")
    ART.mkdir(exist_ok=True)

    d = np.load(OUT / "repairer" / "cache.npz", allow_pickle=True)
    files, conds = list(d["files"]), list(d["conds"])
    masks = d["masks"].astype(np.float32)
    qs_base = np.concatenate([d["qs"], d["cs"]], axis=2).astype(np.float32)
    N, C = len(files), len(conds)
    if args.limit:
        files, masks, qs_base = files[:args.limit], masks[:args.limit], qs_base[:args.limit]
        N = len(files)

    bbox = pd.read_csv(BBOX_CSV, skiprows=1, header=None, usecols=range(6),
                       names=["Image Index", "Finding Label", "x", "y", "w", "h"])
    bbox = bbox[bbox["Finding Label"] == "Cardiomegaly"]
    boxes_of = {f: gt_boxes_224(bbox, f) for f in files}

    if not SPLIT_CSV.exists():
        from patient_split import main as split_main
        split_main()
    sp = pd.read_csv(SPLIT_CSV)
    fold_of = {r.file: r.fold for r in sp[sp.finding == "Cardiomegaly"].itertuples()}
    folds = [[] for _ in range(NFOLD)]
    for i, f in enumerate(files):
        folds[fold_of[f]].append(i)

    if not CAMSF.exists() or args.limit:
        log("计算 gradcam_pp / layercam 缓存…")
        compute_family_cams(device)
    z = np.load(CAMSF)
    floc = {f: i for i, f in enumerate(list(z["files"]))}
    cloc = {c: i for i, c in enumerate(list(z["conds"]))}
    fl = np.array([floc[f] for f in files]); cl = np.arange(C)

    all_dfs = []
    for fam in fams:
        cams_f = z[fam][np.ix_(fl, cl)].astype(np.float32)
        # layercam 端 sum 在个别条件溢出（fp16 存了 -inf）：净化到 [0,1]
        n_bad = int((~np.isfinite(cams_f) | (cams_f < 0) | (cams_f > 1)).sum())
        log(f"[{fam}] 非法像素 {n_bad}（净化为 0/1 边界）")
        cams_f = np.clip(np.nan_to_num(cams_f, nan=0.0, posinf=1.0, neginf=0.0),
                         0.0, 1.0)
        df = train_family(fam, cams_f, masks, qs_base, files, conds, boxes_of,
                          folds, device)
        df.to_csv(ART / f"eval_{fam}.csv", index=False)
        all_dfs.append(df)
    pd.concat(all_dfs).to_csv(ART / "eval_all.csv", index=False)

    for fam in fams:
        summarize(pd.read_csv(ART / f"eval_{fam}.csv"), fam)
    # Grad-CAM 参照（同折同协议的 repairer_v2 折外）
    ref = pd.read_csv(OUT / "repairer_v2_artifacts" / "gate_v2_eval.csv")
    ref = ref.rename(columns={"gated2_45": "gated_45"})
    summarize(ref, "gradcam(ref: repairer_v2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
