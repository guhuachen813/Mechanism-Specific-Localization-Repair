r"""P23 自主翻拍独立外测 —— 冻结终评（p21_final_eval.py 的 P23 变体，协议 v1.0 §7）。

仅替换（协议允许范围）：
  - photo 输入 = /root/autodl-tmp/p23_capture/ 的 pass1 文件（S###_1.jpg，sha256 已封存）
  - GT 银标签 = P23 注册传播（p23_register/p23_masks_224.npz）
  - 特征按需重算（新采集域，无缓存可复用）
逐字节不变：模型 ckpt（四病种各自冻结 backbone）、冻结工件
（pairloc_adapter_cardi_noid_frozen / pairloc_adapter_eff / rectnet_head_atel /
rectnet_head_ede）、路由表、WRONG_THR=-0.05、统计（患者级 boot 400 / seed 42、
sign-flip 20000、CP 精确上界、Holm）、Edema agree≥q50 门（集合内中位数，同 P21）。
排除（预注册 §6）：pass1 reg-fail 图像 = 标签缺失，raw 与 sys 双臂同图同删。
运行：CPU（无卡实例）——/root/miniconda3/bin/python p23_final_eval.py（cwd=rep_robust）
输出：p23_eval/p23_final_eval.csv、p23_eval/p23_final_summary.csv
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torchvision import transforms

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, load_model
from p18_new_findings import (Adapter, CornHead, VIEWS18, _apply_view,
                              _cam_binary, _cam_iou, cam_from_feat, iou45,
                              mask_bbox)

ROOT = "/root/autodl-tmp/experiments/rep_robust"
OUT = f"{ROOT}/p23_eval"
BASE = "/root/autodl-tmp"
CAPTURE = f"{BASE}/p23_capture"
MANIFEST = f"{BASE}/p23_capture/p23_shoot_manifest.csv"
REG_CSV = f"{ROOT}/p23_register/p23_register.csv"
REG2_CSV = f"{ROOT}/p23_register_pass2/p23_register_pass2.csv"
SEED = 42
WRONG_THR = -0.05
CORNERS = np.array([[0, 0], [224, 0], [224, 224], [0, 224]], np.float32)

SPEC = {
    "cardi": {"finding": "Cardiomegaly",
              "ckpt": "/root/project/outputs/repaired_seed42/densenet121/best.pt",
              "kind": "pairloc",
              "artifact": f"{ROOT}/chexlocalize/pairloc_adapter_cardi_noid_frozen.pt",
              "sys": "pairloc_noid_frozen"},
    "eff": {"finding": "Pleural Effusion",
            "ckpt": "/root/project/outputs/eff_seed42_densenet121/best.pt",
            "kind": "pairloc",
            "artifact": f"{ROOT}/chexlocalize/pairloc_adapter_eff.pt",
            "sys": "pairloc_full"},
    "atel": {"finding": "Atelectasis",
             "ckpt": "/root/project/outputs/atel_seed42_densenet121/best.pt",
             "kind": "rectnet",
             "artifact": f"{ROOT}/chexlocalize/rectnet_head_atel.pt",
             "sys": "rectnet"},
    "ede": {"finding": "Edema",
            "ckpt": "/root/project/outputs/ede_seed42_densenet121/best.pt",
            "kind": "rectnet",
            "artifact": f"{ROOT}/chexlocalize/rectnet_head_ede.pt",
            "sys": "rectnet_gated_q50"},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def boot_ci(d, key, n=400, seed=SEED):
    up = pd.unique(key)
    fmap = pd.Series(key).map({f: i for i, f in enumerate(up)}).to_numpy()
    r = np.random.default_rng(seed)
    vals = [d[np.isin(fmap, r.choice(len(up), len(up)))].mean() for _ in range(n)]
    return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def signflip_p(d, key, n=20000, seed=SEED):
    df = pd.DataFrame({"d": d, "p": key}).groupby("p")["d"].mean().to_numpy()
    obs = abs(df.mean())
    if obs == 0:
        return 1.0
    r = np.random.default_rng(seed)
    null = np.abs((r.choice([-1.0, 1.0], size=(n, len(df))) * df[None, :]).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[idx])
        adj[idx] = min(run, 1.0)
    return adj


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(min(64, os.cpu_count() or 8))
    device = torch.device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])

    man = pd.read_csv(MANIFEST).set_index("key")
    reg = pd.read_csv(REG_CSV).set_index("key")
    reg2 = pd.read_csv(REG2_CSV).set_index("key")
    ok_keys = [k for k in reg.index if reg.loc[k, "ok"] == 1]
    ok2 = {k for k in reg2.index if reg2.loc[k, "ok"] == 1}
    log(f"pass1 reg-ok 评估集：{len(ok_keys)}（排除 " +
        ",".join(sorted(set(reg.index) - set(ok_keys))) + "）")

    SM = np.load(f"{ROOT}/p23_register/p23_masks_224.npz", allow_pickle=True)
    SM2 = np.load(f"{ROOT}/p23_register_pass2/p23_masks_224.npz", allow_pickle=True)
    findings = [str(x) for x in SM["findings"]]

    # 病种阳性键（mask 存在且 >50 px，同 p18 口径）
    pos_keys = {tag: [] for tag in SPEC}
    for tag, cfg in SPEC.items():
        fi = findings.index(cfg["finding"])
        pos_keys[tag] = [k for k in ok_keys
                         if k in SM and SM[k][fi].sum() > 50]
        log(f"[{tag}] 阳性键 {len(pos_keys[tag])}")

    all_rows, cons_rows = [], []
    for tag, cfg in SPEC.items():
        log(f"===== {tag} / {cfg['sys']} =====")
        keys = pos_keys[tag]
        gts = {k: mask_bbox(SM[k][findings.index(cfg["finding"])]) for k in keys}
        gts2 = {k: (mask_bbox(SM2[k][findings.index(cfg["finding"])])
                    if k in ok2 and k in SM2 else [])
                for k in keys}

        cam_benchmark.CKPT = cfg["ckpt"]
        model = load_model(device)
        for p_ in model.parameters():
            p_.requires_grad_(False)
        w1 = model.classifier.weight[1].detach()
        ad = head = None
        if cfg["kind"] == "pairloc":
            ad = Adapter()
            ad.load_state_dict(torch.load(cfg["artifact"], map_location=device))
            ad = ad.to(device).eval()
        else:
            head = CornHead()
            head.load_state_dict(torch.load(cfg["artifact"], map_location=device))
            head = head.to(device).eval()

        # ---- photo 224 图像（pass1 / pass2）与特征（批量） ----
        shot = {k: man.loc[k, "file_pass1"] for k in keys}
        shot2 = {k: man.loc[k, "file_pass2"] for k in keys}
        pim1 = {k: Image.open(f"{CAPTURE}/{shot[k]}").convert("L")
                .resize((224, 224), Image.BILINEAR) for k in keys}

        def feats_batch(pils):
            x = torch.cat([tf(im.convert("RGB"))[None] for im in pils]).to(device)
            out = []
            with torch.no_grad():
                for s0 in range(0, len(x), 16):
                    out.append(model.features(x[s0:s0 + 16]))
            return torch.cat(out)  # (N,1024,7,7)

        fv = feats_batch([pim1[k] for k in keys])
        with torch.no_grad():
            raw_cam = cam_from_feat(fv, w1)
            sys_cam = (cam_from_feat(ad(fv), w1) if ad is not None else None)

        raw45, sys45, pl45, rect45 = [], [], [], []
        for i, k in enumerate(keys):
            gt = gts[k]
            raw45.append(iou45(raw_cam[i].cpu().numpy(), gt) if gt else np.nan)
            if ad is not None:
                pl45.append(iou45(sys_cam[i].cpu().numpy(), gt) if gt else np.nan)
                rect45.append(np.nan)
            else:
                pl45.append(np.nan)
                if not gt:
                    rect45.append(np.nan)
                    continue
                # rectnet：GAP → 角点 → 校正 → CAM → 扭回
                pred = head(fv[i][None].mean(dim=(2, 3))).reshape(4, 2).detach().cpu().numpy() * 224.0
                M = cv2.getPerspectiveTransform(pred.astype(np.float32), CORNERS)
                Mback = cv2.getPerspectiveTransform(CORNERS, pred.astype(np.float32))
                x = tf(pim1[k].convert("RGB"))[None].to(device)
                with torch.no_grad():
                    rect = F.grid_sample(x, _grid(M), align_corners=False)
                    fr = model.features(rect)
                    cam = torch.einsum("k,bkyx->byx", w1, fr).relu_()
                    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                    cam224 = F.interpolate(cam[:, None], size=IMAGE_SIZE,
                                           mode="bilinear", align_corners=False)[:, 0]
                    cam_back = F.grid_sample(cam224[:, None], _grid(Mback),
                                             align_corners=False)[0, 0].cpu().numpy()
                rect45.append(iou45(np.clip(cam_back, 0, 1), gt))

        # ---- 7 视图一致性 agree（pl 与 rect 两系统，同 p18/p11 视图组） ----
        views_pl, views_rc = [], []
        for v in VIEWS18:
            ims = [_apply_view(pim1[k], v) for k in keys]
            xv = torch.cat([tf(im.convert("RGB"))[None] for im in ims]).to(device)
            with torch.no_grad():
                fvv = torch.cat([model.features(xv[s0:s0 + 16])
                                 for s0 in range(0, len(xv), 16)])
                bp = (cam_from_feat(ad(fvv), w1) if ad is not None else None)
                br = None
                if head is not None:
                    pred_v = head(fvv.mean(dim=(2, 3))).reshape(-1, 4, 2).detach().cpu().numpy() * 224.0
                    rcs = []
                    for b in range(len(xv)):
                        Mv = cv2.getPerspectiveTransform(pred_v[b].astype(np.float32), CORNERS)
                        Mbv = cv2.getPerspectiveTransform(CORNERS, pred_v[b].astype(np.float32))
                        rv = F.grid_sample(xv[b][None], _grid(Mv), align_corners=False)
                        camv = torch.einsum("k,bkyx->byx", w1, model.features(rv)).relu_()
                        camv = camv / camv.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                        camv224 = F.interpolate(camv[:, None], size=IMAGE_SIZE,
                                                mode="bilinear", align_corners=False)[:, 0]
                        cbv = F.grid_sample(camv224[:, None], _grid(Mbv),
                                            align_corners=False)[0, 0].cpu().numpy()
                        rcs.append(torch.from_numpy(np.clip(cbv, 0, 1)))
                    br = torch.stack(rcs)
            views_pl.append([_cam_binary(bp[i].cpu().numpy()) for i in range(len(keys))]
                            if bp is not None else None)
            views_rc.append([_cam_binary(br[i].cpu().numpy()) for i in range(len(keys))]
                            if br is not None else None)

        agree_pl, agree_rect = [], []
        for i in range(len(keys)):
            ap = (float(np.mean([_cam_iou(views_pl[a][i], views_pl[b][i])
                                 for a in range(7) for b in range(a + 1, 7)]))
                  if views_pl[0] is not None else np.nan)
            ar = (float(np.mean([_cam_iou(views_rc[a][i], views_rc[b][i])
                                 for a in range(7) for b in range(a + 1, 7)]))
                  if views_rc[0] is not None else np.nan)
            agree_pl.append(ap)
            agree_rect.append(ar)

        # ---- pass1/pass2 拍摄内一致性（raw45 两侧，pass2 reg-ok 子集） ----
        keys2 = [k for k in keys if k in ok2]
        raw45_2 = {}
        if keys2:
            fv2 = feats_batch([Image.open(f"{CAPTURE}/{shot2[k]}").convert("L")
                               .resize((224, 224), Image.BILINEAR) for k in keys2])
            with torch.no_grad():
                rc2 = cam_from_feat(fv2, w1)
            for i, k in enumerate(keys2):
                gt2 = gts2[k]
                raw45_2[k] = iou45(rc2[i].cpu().numpy(), gt2) if gt2 else np.nan
            d12 = [abs(raw45_2[k] - raw45[keys.index(k)])
                   for k in keys2 if np.isfinite(raw45_2[k]) and np.isfinite(raw45[keys.index(k)])]
            v12 = [raw45_2[k] for k in keys2
                   if np.isfinite(raw45_2[k]) and np.isfinite(raw45[keys.index(k)])]
            v1 = [raw45[keys.index(k)] for k in keys2
                  if np.isfinite(raw45_2[k]) and np.isfinite(raw45[keys.index(k)])]
            cons_rows.append({"finding": tag, "n_pair": len(d12),
                              "median_abs_d_raw45": float(np.median(d12)) if d12 else np.nan,
                              "pearson_r": float(np.corrcoef(v1, v12)[0, 1]) if len(v12) > 2 else np.nan})

        # ---- 装配（路由规则逐字同 P21） ----
        for i, k in enumerate(keys):
            if cfg["kind"] == "pairloc":
                s45 = pl45[i]
            else:
                s45 = rect45[i]
            all_rows.append({"finding": tag, "key": k, "patient": k.split("_")[0],
                             "raw45": raw45[i], "sys45_raw_act": s45,
                             "pairloc45": pl45[i], "rectnet45": rect45[i],
                             "agree_p_pl": agree_pl[i], "agree_p_rect": agree_rect[i],
                             "raw45_pass2": raw45_2.get(k, np.nan),
                             "sys": cfg["sys"]})
        del model, fv, raw_cam, sys_cam
        log(f"[{tag}] 推理完成 n={len(keys)}")

    df = pd.DataFrame(all_rows)
    # Edema 预声明门：agree≥q50（集合内中位数，实现同 P21）→ rectnet，否则 raw
    for tag in ("ede",):
        g = df.finding == tag
        thr = np.nanquantile(df.loc[g, "agree_p_rect"], 0.5)
        df.loc[g, "gate_thr"] = thr
        df.loc[g, "sys45"] = np.where(df.loc[g, "agree_p_rect"] >= thr,
                                      df.loc[g, "sys45_raw_act"], df.loc[g, "raw45"])
        log(f"[ede] agree_q50 门限 = {thr:.4f}")
    df.loc[df.finding != "ede", "sys45"] = df.loc[df.finding != "ede", "sys45_raw_act"]
    df.to_csv(f"{OUT}/p23_final_eval.csv", index=False)

    # ---- 统计（boot 400/seed42 + signflip + CP + Holm，逐字同 P21 族口径） ----
    summary, ps = [], []
    tags = list(SPEC)
    for tag in tags:
        g = df[df.finding == tag]
        d = (g.sys45 - g.raw45).to_numpy()
        m = ~np.isnan(d)
        mean, lo, hi = boot_ci(d[m], g.patient[m].to_numpy())
        pv = signflip_p(d[m], g.patient[m].to_numpy())
        k = int((d[m] < WRONG_THR).sum())
        summary.append({"finding": tag, "system": SPEC[tag]["sys"], "n": int(m.sum()),
                        "raw": float(np.nanmean(g.raw45)), "delta": mean, "lo": lo,
                        "hi": hi, "p": pv, "wrong_k": k, "wrong_n": int(m.sum()),
                        "wrong_cp95ub": cp_upper(k, int(m.sum()))})
        ps.append(pv)
    padj = holm(np.array(ps))
    for s, pa in zip(summary, padj):
        s["p_holm"] = float(pa)
    d = (df.sys45 - df.raw45).to_numpy()
    m = ~np.isnan(d)
    mean, lo, hi = boot_ci(d[m], df.patient[m].to_numpy())
    k = int((d[m] < WRONG_THR).sum())
    summary.append({"finding": "pooled", "system": "routing(4 findings)",
                    "n": int(m.sum()), "raw": float(np.nanmean(df.raw45[m])),
                    "delta": mean, "lo": lo, "hi": hi,
                    "p": signflip_p(d[m], df.patient[m].to_numpy()), "p_holm": np.nan,
                    "wrong_k": k, "wrong_n": int(m.sum()),
                    "wrong_cp95ub": cp_upper(k, int(m.sum()))})
    sm = pd.DataFrame(summary)
    pd.DataFrame(cons_rows).to_csv(f"{OUT}/p23_consistency.csv", index=False)
    sm.to_csv(f"{OUT}/p23_final_summary.csv", index=False)

    print("\n==== P23 冻结终评（独立采集域复制，122→%d 患者） ====" % df.patient.nunique())
    for _, s in sm.iterrows():
        star = "*" if (s.finding != "pooled" and s.p_holm < 0.05) else ""
        print(f"  [{s.finding:5s}] {s.system:22s} n={int(s.n):3d} "
              f"Δ {s.delta:+.4f} [{s.lo:+.4f},{s.hi:+.4f}]{star} "
              f"wrong {int(s.wrong_k)}/{int(s.wrong_n)} CP95UB {s.wrong_cp95ub:.3f} "
              f"p={s.p:.2e}")
    log("完成")
    return 0


def _grid(M, size=224):
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
    src = np.linalg.inv(M) @ pts
    src = src[:2] / src[2:3]
    grid = src.T.reshape(size, size, 2)
    grid[..., 0] = grid[..., 0] / (size - 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / (size - 1) * 2 - 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(grid.astype(np.float32))[None].to(dev)


if __name__ == "__main__":
    raise SystemExit(main())
