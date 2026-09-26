r"""P21：valid natural 168 张 frontal 的冻结终评（dev-set frozen final eval）。

性质声明（写进协议，一次性执行）：本评估集 = CheXphoto valid natural oneplus
的 168 张 frontal（234 张中 66 张 lateral 无定位 GT）。它是开发集——q50 门
规则与病种路由在此集上形成；本评估为**开发集冻结终评**，非 external test
（external = test natural 668）。协议冻结点：
  1) 系统：Cardi→no_ident pairloc（主配置=无 early-stop 重训，零评估集接触；
     p08 early-stop 版为敏感性对照）；Effusion→pairloc(full)；Atel→rectnet；
     Edema→agree_p_rect≥q50（预声明分位规则）+raw fallback；其余→raw fallback。
  2) 一次评估：不调阈值、不换动作集、不按结果选病种。
  3) 指标：每病种 Δ vs raw 患者级 CI / wrong / CP 上界；患者级 pooled 汇总；
     基线 raw/prior/cand_zs/g2（有则报）；risk-coverage（四病种）。

输出 p21_final_eval.csv / p21_final_summary.csv / p21_rc_{tag}.csv。
用法：/root/miniconda3/bin/python p21_final_eval.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from scipy import stats

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, load_model
from p18_new_findings import (CLEAN_TRAIN, OUT, Adapter, W_FULL, cam_from_feat,
                              train_pairloc)

ROOT = "/root/autodl-tmp/experiments/rep_robust"
SEED = 42
WRONG_THR = -0.05
W_NOID = {"logit": 0.5, "cam": 1.0, "geo": 0.5, "anat": 0.1, "ident": 0.0}
SYS_OF = {"cardi": "pairloc_noid_frozen", "eff": "pairloc_full",
          "atel": "rectnet", "ede": "rectnet_gated_q50"}


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


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SEED)

    # ---------- Cardi 主配置：no_ident 无 early-stop 重训（零评估集接触） ----------
    cam_benchmark.CKPT = "/root/project/outputs/repaired_seed42/densenet121/best.pt"
    model = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    w1 = model.classifier.weight[1].detach()
    D = np.load(f"{OUT}/pair_cache_cardi.npz", allow_pickle=True)
    reg_ok = D["reg_ok"].astype(bool)
    mask_p = D["mask_p"].astype(np.float32)
    FE = np.load(f"{OUT}/p18_feat_cardi.npz", allow_pickle=True)
    keys = [str(k) for k in FE["keys"]]
    f_photo, f_clean = FE["f_photo"], FE["f_clean"]
    prob_t = FE["prob_t"].astype(np.float32)
    cam_t = np.zeros((len(keys), 224, 224), np.float32)
    C = np.load(f"{OUT}/pair_corners_pair_cache_cardi.npz", allow_pickle=True)
    Tpair, ok_pair = C["T"], C["ok"].astype(bool)
    import cv2
    with torch.no_grad():
        for i in np.where(ok_pair & reg_ok)[0]:
            fc_i = torch.from_numpy(np.asarray(f_clean[i], np.float32)).to(device)[None]
            cam_t[i] = cv2.warpPerspective(cam_from_feat(fc_i, w1)[0].cpu().numpy(),
                                           Tpair[i], (224, 224))
    log("训练 Cardi no_ident（40 epochs 固定，无 early stop）")
    ad_noid = train_pairloc(W_NOID, model, w1, device, reg_ok, f_photo, f_clean,
                            mask_p, prob_t, cam_t)
    torch.save(ad_noid.state_dict(), f"{OUT}/pairloc_adapter_cardi_noid_frozen.pt")

    f_val = FE["f_val"]
    p18c = pd.read_csv(f"{OUT}/p18_new_findings_cardi.csv").set_index("key")
    vkeys = list(p18c.index)
    with torch.no_grad():
        fv = torch.from_numpy(np.asarray(f_val, np.float32)).to(device)
        cam_noid = cam_from_feat(ad_noid(fv), w1).cpu().numpy()
    from cam_benchmark import cam_iou_curve, THRESHOLDS
    from p18_new_findings import mask_bbox
    Dv = np.load(f"{OUT}/cache_photo_adapt.npz", allow_pickle=True)
    gts_v = Dv["gts"]
    iou45 = lambda cam, gt: float(cam_iou_curve(cam, gt)[
        int(np.where(np.isclose(THRESHOLDS, 0.45))[0][0])])
    noid_frozen = {}
    for i, k in enumerate(vkeys):
        gt = mask_bbox(gts_v[i])
        noid_frozen[k] = iou45(cam_noid[i], gt) if gt else np.nan
    p20 = pd.read_csv(f"{OUT}/p20_perimage.csv")
    p20c = p20[p20.tag == "cardi"].set_index("key")

    # ---------- 逐病种装配 ----------
    rows = []
    SM = np.load(f"{OUT}/silver_masks_224.npz", allow_pickle=True)
    findings = [str(x) for x in SM["findings"]]
    spec = {
        "cardi": ("Cardiomegaly", "p18_new_findings_cardi.csv", "pairloc45"),
        "eff": ("Pleural Effusion", "p18_new_findings_eff.csv", "pairloc45"),
        "atel": ("Atelectasis", "p18_new_findings_atel.csv", "rectnet45"),
        "ede": ("Edema", "p18_new_findings_ede.csv", "rectnet45"),
    }
    for tag, (finding, csvname, syscol) in spec.items():
        df = pd.read_csv(f"{OUT}/{csvname}").set_index("key")
        df["sys45"] = df[syscol]
        df["agree"] = df.agree_p_rect if tag in ("atel", "ede") else df.agree_p_pl
        if tag == "ede":  # 预声明门规则：agree≥q50 → rectnet，否则 raw fallback
            thr = np.nanquantile(df.agree, 0.5)
            df["sys45"] = np.where(df.agree >= thr, df.rectnet45, df.raw45)
        if tag == "cardi":
            df["sys45"] = [noid_frozen.get(k, np.nan) for k in df.index]
            df["sys45_sens"] = p20c.loc[df.index, "noident45"]
            g2 = pd.read_csv(f"{OUT}/p62_photo_sage.csv").set_index("key")
            df["g2"] = g2.loc[df.index, "gated_v2_45"]
            df["prior45"] = g2.loc[df.index, "prior45"]
            df["cand_zs45"] = g2.loc[df.index, "cand_zs45"]
        elif tag == "atel":
            g2 = pd.read_csv(f"{OUT}/p65_atel_photo.csv").set_index("key")
            df["prior45"] = g2.loc[df.index, "prior45"]
            df["cand_zs45"] = g2.loc[df.index, "cand_zs45"]
        for k in df.index:
            rows.append({"finding": tag, "key": k, "patient": k.split("_")[0],
                         "raw45": df.loc[k, "raw45"], "sys45": df.loc[k, "sys45"],
                         "agree": df.loc[k, "agree"],
                         "prior45": df.loc[k, "prior45"] if "prior45" in df else np.nan,
                         "cand_zs45": df.loc[k, "cand_zs45"] if "cand_zs45" in df else np.nan,
                         "g2": df.loc[k, "g2"] if "g2" in df else np.nan,
                         "sys45_sens": df.loc[k, "sys45_sens"] if "sys45_sens" in df else np.nan})
    df_all = pd.DataFrame(rows)
    df_all.to_csv(f"{OUT}/p21_final_eval.csv", index=False)

    # ---------- 每病种统计 + pooled ----------
    summary = []
    for tag, g in df_all.groupby("finding"):
        d = (g.sys45 - g.raw45).to_numpy()
        m = ~np.isnan(d)
        mean, lo, hi = boot_ci(d[m], g.patient[m].to_numpy())
        k = int((d[m] < WRONG_THR).sum())
        summary.append({"finding": tag, "system": SYS_OF[tag], "n": int(m.sum()),
                        "raw": float(np.nanmean(g.raw45)), "delta": mean,
                        "lo": lo, "hi": hi, "p": signflip_p(d[m], g.patient[m].to_numpy()),
                        "wrong_k": k, "wrong_n": int(m.sum()),
                        "wrong_cp95ub": cp_upper(k, int(m.sum()))})
        print(f"  [{tag:5s}] {SYS_OF[tag]:20s} n={int(m.sum())} Δ {mean:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] wrong {k}/{int(m.sum())} "
              f"CP95UB {cp_upper(k, int(m.sum())):.3f}", flush=True)
    # 敏感性：cardi p08 early-stop 版
    gc = df_all[df_all.finding == "cardi"]
    ds = (gc.sys45_sens - gc.raw45).to_numpy()
    ms = ~np.isnan(ds)
    b = boot_ci(ds[ms], gc.patient[ms].to_numpy())
    print(f"  [cardi 敏感性] p08 early-stop 版 Δ {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]")
    # pooled（患者级，全 4 病种 image-finding 对）
    d = (df_all.sys45 - df_all.raw45).to_numpy()
    m = ~np.isnan(d)
    mean, lo, hi = boot_ci(d[m], df_all.patient[m].to_numpy())
    k = int((d[m] < WRONG_THR).sum())
    summary.append({"finding": "pooled", "system": "routing(4 findings)",
                    "n": int(m.sum()), "raw": float(np.nanmean(df_all.raw45[m])),
                    "delta": mean, "lo": lo, "hi": hi,
                    "p": signflip_p(d[m], df_all.patient[m].to_numpy()),
                    "wrong_k": k, "wrong_n": int(m.sum()),
                    "wrong_cp95ub": cp_upper(k, int(m.sum()))})
    print(f"  [pooled] n={int(m.sum())} Δ {mean:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"wrong {k}/{int(m.sum())} CP95UB {cp_upper(k, int(m.sum())):.3f}")

    # ---------- 风险-覆盖（四病种，agree 门扫描） ----------
    for tag, g in df_all.groupby("finding"):
        a = g.agree.to_numpy()
        d = (g.sys45 - g.raw45).to_numpy()
        m = ~np.isnan(a) & ~np.isnan(d)
        a, d = a[m], d[m]
        rc = []
        for q in np.linspace(0.0, 0.9, 19):
            thr = np.quantile(a, q)
            sel = a >= thr
            kk = int((d[sel] < WRONG_THR).sum())
            rc.append({"q": q, "thr": thr, "cov": sel.mean(),
                       "wrong": kk / max(sel.sum(), 1),
                       "wrong_cp95ub": cp_upper(kk, int(sel.sum())),
                       "sys_delta": d[sel].sum() / len(d)})
        pd.DataFrame(rc).to_csv(f"{OUT}/p21_rc_{tag}.csv", index=False)

    pd.DataFrame(summary).to_csv(f"{OUT}/p21_final_summary.csv", index=False)
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
