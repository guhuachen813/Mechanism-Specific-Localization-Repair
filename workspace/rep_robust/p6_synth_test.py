r"""第五阶段 2：锁定合成外测——CheXpert test 154 阳性 × 合成采集轴。

定位：数据申请不能按期到位时的合法外测（用户 2026-09-23 批准方案）。
工作集：gt_segmentations_test.json Cardiomegaly 面积>0 的 frontal 154 图（从未参与
  任何训练/调参；DenseNet 只见 CheXpert train，修复器只见 NIH+valid photo）。
退化轴：degradations.py 17 条件（与修复器训练轴同款）+ clean，rng 口径与 stage_cache
  一致（dg.rng_for(SEED, key)）。
系统：raw / prior(MedSAM) / cand(v2 零样本 5 头中位) / gated_v2(5 门控中位+δ 中位) /
  cand_adapt(65 photo 银标签适配, 3 seed) / gated_v4(交叉拟合收益门控, seed 42) /
  oracle 三选一。全部零样本于 test 患者。

解读框架：转移链 A = 同退化轴 × 新患者（本实验）；转移链 B = 真实翻拍 × 新患者
  （P1/5.1a）。若 cand 在 A 成立而 B 失败 → 失效属退化域特异；A 也失败 → 更深。

用法：/root/miniconda3/bin/python p6_synth_test.py [--limit 6]
"""
from __future__ import annotations

import argparse
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


# ---------------- photo 适配（最终模型，65 患者全量，early-stop 协议） ----------------
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


class GateV4(torch.nn.Module):
    def __init__(self, n_in=12):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                       torch.nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def train_head_es(idxs, cal_idx, D, device, epochs_max=60, eval_every=5):
    import torch.nn.functional as F
    head = RepairHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    boxes = [mask_bbox(D["gts"][i])[0] for i in idxs]
    Y = np.zeros((len(idxs), 1, 224, 224), np.float32)
    for j, (i, b) in enumerate(zip(idxs, boxes)):
        Y[j, 0, int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1.0
    rng = np.random.default_rng(0)
    best = (-1e9, None, 0)

    def cal_iou():
        head.eval()
        with torch.no_grad():
            sc = []
            for i in cal_idx:
                h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
                m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
                q = torch.from_numpy(D["q9"][i][None]).to(device)
                c = torch.clamp(h + ALPHA * head(h, m, q), 0, 1)[0, 0].cpu().numpy()
                b = mask_bbox(D["gts"][i])
                sc.append(float(cam_iou_curve(c, b)[TAU45]))
        head.train()
        return float(np.mean(sc))

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
            loss = torch.nn.functional.binary_cross_entropy(cand, y) \
                + 0.25 * (cand * (1 - m)).mean() \
                + 0.05 * ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                          + (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % eval_every == 0 or ep == epochs_max - 1:
            sc = cal_iou()
            if sc > best[0]:
                best = (sc, {k: v.clone() for k, v in head.state_dict().items()}, ep + 1)
    head.load_state_dict(best[1])
    head.eval()
    return head, best[2]


def iou45(cam, gt):
    return float(cam_iou_curve(cam, gt)[TAU45])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)

    model = load_model(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    log("加载 MedSAM")
    from transformers import SamModel
    smodel = SamModel.from_pretrained(
        "/root/autodl-tmp/experiments/rep_robust/weights/medsam").to(device).eval()

    # ---- 工作集 ----
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
    log(f"锁定外测工作集 {len(keys)} 图 / {len(set(k.split('_')[0] for k in keys))} 患者")

    # ---- 零样本 v2 集成 ----
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

    # ---- photo 适配最终模型（3 seed） ----
    D = dict(np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True))
    pat_photo = np.array([k.split("_")[0] for k in D["keys"]])
    uphoto = sorted(set(pat_photo))
    adapt_heads = []
    for sd in (42, 7, 2024):
        rng = np.random.default_rng(sd)
        perm = rng.permutation(len(uphoto))
        cal_p = set(uphoto[j] for j in perm[: max(4, len(uphoto) // 5)])
        fit_idx = [i for i in range(len(pat_photo)) if pat_photo[i] not in cal_p]
        cal_idx = [i for i in range(len(pat_photo)) if pat_photo[i] in cal_p]
        h, bep = train_head_es(fit_idx, cal_idx, D, device)
        adapt_heads.append(h)
        log(f"  适配头 seed{sd} best_ep {bep}")

    # ---- 评估循环 ----
    rows, store = [], {}
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
            conds = [("clean", im224)] + [
                (c, dg.degrade(im224, c, dg.rng_for(SEED, zlib.crc32((k + c).encode()))))
                for c in dg.CONDITION_ORDER]
            for cond, deg in conds:
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
                    cs_ad = [torch.clamp(inp + ALPHA * hh(inp, mk, qt), 0, 1)[0, 0].cpu().numpy()
                             for hh in adapt_heads]
                    cand_ad = np.median(np.stack(cs_ad), 0)
                    pg = np.median(torch.softmax(torch.stack(
                        [g(qt) for g in gates_v2]), dim=2).cpu().numpy()[:, 0], 0)
                choice = int(pg.argmax())
                margin = float(pg[choice] - pg[0])
                pick = choice if (choice == 0 or margin >= d_med) else 0
                gated_v2 = [cam, cam * mm, cand_zs][pick]
                prior = cam * mm
                rec = {"key": k, "patient": patient, "cond": cond,
                       "prob": float(prob[0]), "field_ok": int(ok_r),
                       "raw45": iou45(cam, gt), "prior45": iou45(prior, gt),
                       "cand_zs45": iou45(cand_zs, gt),
                       "cand_ad45": iou45(cand_ad, gt),
                       "gated_v2_45": iou45(gated_v2, gt),
                       "oracle45": max(iou45(cam, gt), iou45(prior, gt),
                                       iou45(cand_zs, gt)),
                       "frac": float(qv[1]), "conflict": float(qv[8]),
                       "gt_area": float(m224.mean())}
                rows.append(rec)
                store[f"{k}|{cond}"] = (cam.astype(np.float16), qv.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            log(f"err {k}: {e}")
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(keys)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p6_synth_test.csv", index=False)
    np.savez_compressed(f"{OUT}/p6_synth_cams.npz",
                        keys=np.array(list(store)),
                        **{kk: vv[0] for kk, vv in list(store.items())[:0]})  # 占位省盘
    log("主循环完成，开始 gated_v4 训练（交叉拟合）")

    # ---- gated_v4：65 photo 患者交叉拟合收益门控（seed 42），外推到 test ----
    rng0 = np.random.default_rng(42)
    perm0 = rng0.permutation(len(uphoto))
    cal_p = set(uphoto[j] for j in perm0[: max(4, len(uphoto) // 5)])
    fit_p = [p for p in uphoto if p not in cal_p]
    cal_idx = [i for i in range(len(pat_photo)) if pat_photo[i] in cal_p]
    fit_idx = [i for i in range(len(pat_photo)) if pat_photo[i] in set(fit_p)]
    fp = sorted(set(fit_p))
    cf_folds = [set(fp[j::4]) for j in range(4)]
    X_lab, y_lab = [], []
    rawP = np.array([float(cam_iou_curve(D["cams_p"][i], mask_bbox(D["gts"][i]))[TAU45])
                     for i in range(len(pat_photo))])
    priorP = np.array([float(cam_iou_curve(D["cams_p"][i] * D["masks_p"][i],
                                           mask_bbox(D["gts"][i]))[TAU45])
                       for i in range(len(pat_photo))])
    C_NEG = 0.05
    for jf in range(4):
        hold_p = cf_folds[jf]
        t_p = [p for p in fp if p not in hold_p]
        t_idx = [i for i in fit_idx if pat_photo[i] in set(t_p)]
        cal_sub = [i for i in cal_idx if pat_photo[i] in set(t_p[: max(1, len(t_p) // 5)])]
        if not cal_sub:
            cal_sub = t_idx[: max(2, len(t_idx) // 5)]
        hj, _ = train_head_es(t_idx, cal_sub, D, device)
        with torch.no_grad():
            for i in range(len(pat_photo)):
                if pat_photo[i] in hold_p:
                    h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
                    m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
                    q = torch.from_numpy(D["q9"][i][None]).to(device)
                    c = torch.clamp(h + ALPHA * hj(h, m, q), 0, 1)[0, 0].cpu().numpy()
                    g_c = iou45(c, mask_bbox(D["gts"][i])) - rawP[i]
                    g_p = priorP[i] - rawP[i]
                    u = [0.0, g_p - C_NEG * (g_p < 0), g_c - C_NEG * (g_c < 0)]
                    X_lab.append(np.concatenate([D["q9"][i], [D["q9"][i][1], D["q9"][i][0],
                                                              D["q9"][i][8]]]).astype(np.float32))
                    y_lab.append(int(np.argmax(u)))
    gate4 = GateV4().to(device)
    gopt = torch.optim.Adam(gate4.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    X_lab, y_lab = np.stack(X_lab), np.array(y_lab)
    for ep in range(300):
        perm = np.random.default_rng(1).permutation(len(y_lab))
        for s0 in range(0, len(perm), 64):
            sel = perm[s0:s0 + 64]
            loss = lossf(gate4(torch.from_numpy(X_lab[sel]).to(device)),
                         torch.from_numpy(y_lab[sel]).to(device))
            gopt.zero_grad(set_to_none=True)
            loss.backward()
            gopt.step()
    gate4.eval()
    # δ 在 cal 池选（photo 域）
    best = (-1e9, 0.0)
    with torch.no_grad():
        for i in cal_idx:
            pass
    # 用 photo cal 池做 δ 网格
    Xc, rc = [], []
    for i in cal_idx:
        Xc.append(np.concatenate([D["q9"][i], [D["q9"][i][1], D["q9"][i][0], D["q9"][i][8]]]).astype(np.float32))
        rc.append((rawP[i], priorP[i], i))
    with torch.no_grad():
        pc = torch.softmax(gate4(torch.tensor(np.stack(Xc)).to(device)), 1).cpu().numpy()
    # cal 池上的 cand 用全池适配头（近似）
    cand_cal = {}
    with torch.no_grad():
        for j, i in enumerate(cal_idx):
            h = torch.from_numpy(D["cams_p"][i][None, None]).to(device)
            m = torch.from_numpy(D["masks_p"][i][None, None].astype(np.float32)).to(device)
            q = torch.from_numpy(D["q9"][i][None]).to(device)
            cand_cal[i] = torch.clamp(h + ALPHA * adapt_heads[0](h, m, q), 0, 1)[0, 0].cpu().numpy()
    for dl in np.arange(0.0, 0.55, 0.05):
        outs = []
        for j, i in enumerate(cal_idx):
            ch = int(pc[j].argmax())
            pick = ch if (ch == 0 or pc[j, ch] - pc[j, 0] >= dl) else 0
            outs.append({0: rawP[i], 1: priorP[i],
                         2: iou45(cand_cal[i], mask_bbox(D["gts"][i]))}[pick])
        if np.mean(outs) > best[0]:
            best = (np.mean(outs), float(dl))
    d4 = best[1]
    log(f"gated_v4 δ*={d4:.2f}")

    # 应用到 test 评估行
    g4_col, pick_col = [], []
    for r in df.itertuples():
        qv = np.zeros(9, np.float32)
        x4 = np.concatenate([qv, [r.frac, float(r.field_ok), r.conflict]]).astype(np.float32)
        with torch.no_grad():
            p4 = torch.softmax(gate4(torch.tensor(x4[None]).to(device)), 1).cpu().numpy()[0]
        ch = int(p4.argmax())
        pick = ch if (ch == 0 or p4[ch] - p4[0] >= d4) else 0
        pick_col.append(pick)
        g4_col.append({0: r.raw45, 1: r.prior45, 2: r.cand_ad45}[pick])
    df["gated_v4_45"] = g4_col
    df["v4_pick"] = pick_col
    df.to_csv(f"{OUT}/p6_synth_test.csv", index=False)

    # ---- 汇总 ----
    print("\n== 锁定合成外测（test 154 阳性 × 17 条件 + clean，τ45） ==")
    print(df.groupby("cond")[["raw45", "prior45", "cand_zs45", "cand_ad45",
                              "gated_v2_45", "gated_v4_45", "oracle45"]]
          .mean().round(3).to_string())
    rng = np.random.default_rng(42)
    up = df.patient.unique()
    fmap = df.patient.map({f: i for i, f in enumerate(up)}).to_numpy()
    print("\n== vs raw（患者整群 bootstrap 400） ==")
    for v in ["prior45", "cand_zs45", "cand_ad45", "gated_v2_45", "gated_v4_45", "oracle45"]:
        d = df[v] - df.raw45
        vals = [d[np.isin(fmap, rng.choice(len(up), len(up)))].mean() for _ in range(400)]
        print(f"  {v:12s} {d.mean():+.3f} CI [{np.percentile(vals,2.5):+.3f},"
              f"{np.percentile(vals,97.5):+.3f}]")
    print(f"\n修复覆盖率(v4): {(df.v4_pick>0).mean():.3f}  错误修复率: "
          f"{((df.gated_v4_45-df.raw45<-0.05)&(df.v4_pick>0)).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
