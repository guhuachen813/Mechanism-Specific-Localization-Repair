r"""P22：标签无关替代拒绝信号预研（预注册，最小成本，2-3 个信号族）。

背景：P21 证明零接触配置下 agree 门失效（Cardi q50 wrong 12.1% > 恒选 10.8%），
agree 门不得再写成有效安全机制。本预研按用户规范只预注册以下信号，
不再广泛搜索门控器：

  A1 视图一致性   修复 CAM 在 5 个预置扰动视图（hflip/cropTL/cropBR/
                  bright×0.85/bright×1.15）下的两两二值 IoU 均值。危险方向：低。
  A2 修复幅度     二值 IoU(raw CAM, 修复 CAM)。方向探索性（P4b 曾发现反向）。
  A3 视图输出方差 正类 prob 在 5 视图上的 std。危险方向：高。
  A4 seed 方差    3 个不同 seed 的 no_ident adapter 正类 prob std（仅 Cardi）。危险：高。
  B1 角点集成分差 3 个 seed 的 CornHead 预测角点逐角 std 均值（px）。危险：高。
  B2 四边形合理性 面积比偏离 + 对边平行角偏差 + 凸性违反。危险：高。

阈值纪律（冻结）：
  - 不用目标病种评估标签选阈值；
  - Cardi 校准集 = Effusion 评估键（同读出端机制、同信号管线，n=50）；
  - Edema 校准集 = Cardi + Effusion 评估键（B1/B2 用各自 rectnet 头）；
  - 规则只有两档：q50 / q70 校准分位，668 外测前冻结；
  - 只报告 risk-coverage 与 AUROC（诊断），不承诺满足 5% 严格风险。

预注册主终点：存在某预注册信号+档位，使目标病种 wrong < 恒选 wrong 且
cov ≥ 0.5。若无 → 按用户停止规则放弃门控主张，论文定位为
"机制特异的真实域定位修复"（非已完成严格风险控制的选择性部署）。

用法：/root/miniconda3/bin/python p22_reject_signals.py
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
from torchvision import transforms

import cam_benchmark
from cam_benchmark import IMAGE_SIZE, MEAN, STD, load_model
from p18_new_findings import (CLEAN_TRAIN, CornHead, OUT, Adapter, W_FULL,
                              _cam_binary, _cam_iou, cam_from_feat, train_pairloc)
from p21_final_eval import W_NOID, boot_ci, cp_upper

ROOT = "/root/autodl-tmp/experiments/rep_robust"
BASE = "/root/autodl-tmp"
PHOTO_ONEPLUS = f"{BASE}/chexphoto/valid/natural/oneplus"
SEED = 42
WRONG_THR = -0.05
PHOTO_ONEPLUS_DIR = PHOTO_ONEPLUS
CKPTS = {"cardi": "/root/project/outputs/repaired_seed42/densenet121/best.pt",
         "eff": "/root/project/outputs/eff_seed42_densenet121/best.pt",
         "ede": "/root/project/outputs/ede_seed42_densenet121/best.pt"}
# 各病种部署机制 adapter（p18/p21 产物；cardi=零接触 no_ident 冻结版）
ADAPT = {"cardi": "pairloc_adapter_cardi_noid_frozen.pt",
         "eff": "pairloc_adapter_eff.pt", "ede": "pairloc_adapter_ede.pt"}
HEAD = {"cardi": "rectnet_head_cardi.pt", "eff": "rectnet_head_eff.pt",
        "ede": "rectnet_head_ede.pt"}
FEAT = {"cardi": "p18_feat_cardi.npz", "eff": "p18_feat_eff.npz",
        "ede": "p18_feat_ede.npz"}
# LODO 校准池（预注册）：Cardi←{eff}；Edema←{cardi,eff}
CAL_POOL = {"cardi": ["eff"], "ede": ["cardi", "eff"]}
VIEW_SET = ["hflip", "crop_tl", "crop_br", "b085", "b115"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def apply_view224(p224, v):
    if v == "hflip":
        return p224.transpose(Image.FLIP_LEFT_RIGHT)
    if v == "crop_tl":
        return p224.crop((0, 0, 208, 208)).resize((224, 224), Image.BILINEAR)
    if v == "crop_br":
        return p224.crop((16, 16, 224, 224)).resize((224, 224), Image.BILINEAR)
    if v == "b085":
        return Image.fromarray(np.clip(np.asarray(p224, np.float32) * 0.85, 0, 255)
                               .astype(np.uint8))
    if v == "b115":
        return Image.fromarray(np.clip(np.asarray(p224, np.float32) * 1.15, 0, 255)
                               .astype(np.uint8))
    return p224


def quad_sanity(c):
    """c (4,2) 规范四角 → (面积比偏离, 对边平行角差均值 deg, 凸性违反 0/1)。"""
    area = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], -1))
                     - np.dot(c[:, 1], np.roll(c[:, 0], -1)))
    ar = abs(np.log(max(area, 1.0) / (224 * 224)))
    e = [c[1] - c[0], c[2] - c[1], c[2] - c[3], c[3] - c[0]]
    ang = []
    for a, b in [(e[0], e[2]), (e[1], e[3])]:
        cos = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        ang.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    cross = [float(np.cross(e[i], e[(i + 1) % 4])) for i in range(4)]
    convex = float(any(x < 0 for x in cross) != all(x < 0 for x in cross))
    return ar, float(np.mean(ang)), convex


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    df_out = pd.read_csv(f"{OUT}/p21_final_eval.csv")

    sig_rows = {}
    for tag in ["cardi", "eff", "ede"]:
        torch.manual_seed(SEED)
        cam_benchmark.CKPT = CKPTS[tag]
        model = load_model(device)
        for p in model.parameters():
            p.requires_grad_(False)
        w1 = model.classifier.weight[1].detach()
        FE = np.load(f"{OUT}/{FEAT[tag]}", allow_pickle=True)
        vkeys = [str(k) for k in FE["vkeys"]]
        f_val = FE["f_val"]

        # --- 机制 adapter（该病种部署动作对应的 pairloc） ---
        ad = Adapter().to(device)
        ad.load_state_dict(torch.load(f"{OUT}/{ADAPT[tag]}", map_location=device))
        ad.eval()
        # --- 3 seed CornHead（B1：ede/cardi/eff 都算，供校准池复用） ---
        C = np.load(f"{OUT}/pair_corners_pair_cache_cardi.npz", allow_pickle=True)
        heads = []
        for sd in (42, 143, 244):
            h = CornHead().to(device)
            if sd == 42:
                h.load_state_dict(torch.load(f"{OUT}/{HEAD[tag]}", map_location=device))
            else:
                torch.manual_seed(sd)
                h = train_head_quick(FE, C, device, sd)
            h.eval()
            heads.append(h)

        # --- A4：Cardi 的另 2 个 seed no_ident adapter（循环外只训一次） ---
        ad_seeds = ([train_seed_adapter(model, w1, device, sd) for sd in (143, 244)]
                    if tag == "cardi" else None)

        rows = []
        with torch.no_grad():
            for i, k in enumerate(vkeys):
                parts = k.split("_")
                stem = "_".join(parts[2:])
                p224 = Image.open(f"{PHOTO_ONEPLUS}/{parts[0]}/{parts[1]}/{stem}.jpg"
                                  ).convert("L").resize((224, 224), Image.BILINEAR)
                x = tf(p224.convert("RGB"))[None].to(device)
                fv = torch.from_numpy(np.asarray(f_val[i], np.float32)).to(device)[None]
                # raw / 修复 CAM 与 prob
                cam_raw = cam_from_feat(model.features(x), w1)[0].cpu().numpy()
                f_pl = ad(model.features(x))
                cam_pl = cam_from_feat(f_pl, w1)[0].cpu().numpy()
                prob_pl = float(torch.softmax(model.classifier(f_pl.mean(dim=(2, 3))), 1)[0, 1])
                # A1/A3：5 视图
                cams_v, probs_v = [], []
                for v in VIEW_SET:
                    xv = tf(apply_view224(p224, v).convert("RGB"))[None].to(device)
                    f_v = ad(model.features(xv))
                    cams_v.append(_cam_binary(cam_from_feat(f_v, w1)[0].cpu().numpy()))
                    probs_v.append(float(torch.softmax(
                        model.classifier(f_v.mean(dim=(2, 3))), 1)[0, 1]))
                A1 = float(np.mean([_cam_iou(cams_v[a], cams_v[b])
                                    for a in range(5) for b in range(a + 1, 5)]))
                A3 = float(np.std(probs_v))
                # A2 修复幅度
                A2 = _cam_iou(_cam_binary(cam_raw), _cam_binary(cam_pl))
                # A4 seed 方差（仅 cardi：3 个 no_ident adapter）
                A4 = np.nan
                if tag == "cardi":
                    ps = [prob_pl]
                    with torch.no_grad():
                        for ad2 in ad_seeds:
                            ps.append(float(torch.softmax(model.classifier(
                                ad2(fv).mean(dim=(2, 3))), 1)[0, 1]))
                    A4 = float(np.std(ps))
                # B1/B2：角点（3 seed 头）
                preds = np.stack([h(fv.mean(dim=(2, 3)))[0].cpu().numpy().reshape(4, 2)
                                  * 224.0 for h in heads])   # (3,4,2)
                B1 = float(preds.std(axis=0).mean())
                ar, ang, cv = quad_sanity(preds[0])
                B2 = ar + ang / 180.0 + 10.0 * cv
                rows.append({"finding": tag, "key": k, "patient": k.split("_")[0],
                             "A1": A1, "A2": A2, "A3": A3, "A4": A4,
                             "B1": B1, "B2": B2})
                if (i + 1) % 20 == 0:
                    log(f"  [{tag}] {i + 1}/{len(vkeys)}")
        sig_rows[tag] = pd.DataFrame(rows)

    # --- 装配结果（结局来自 p21 冻结终评） ---
    sig = pd.concat(sig_rows.values())
    df = df_out.merge(sig[["finding", "key", "A1", "A2", "A3", "A4", "B1", "B2"]],
                      on=["finding", "key"], how="left")
    df["d"] = df.sys45 - df.raw45
    df.to_csv(f"{OUT}/p22_signals.csv", index=False)

    def auc(y, x):
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.argsort(np.argsort(x[o]))
        n1, n0 = y.sum(), (1 - y).sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

    RULES = {  # 预注册：病种 → 信号 → 危险方向（True=信号低危险）
        "cardi": {"A1": True, "A2": False, "A3": False, "A4": False},
        "ede": {"B1": False, "B2": False, "A1": True},
    }
    verdict = []
    print("\n== P22 信号预研（LODO 校准，只报 risk-coverage） ==")
    for tag, rules in RULES.items():
        g = df[df.finding == tag].copy()
        pool = df[df.finding.isin(CAL_POOL[tag])]
        const_wrong = float((g.d < WRONG_THR).mean())
        const_k = int((g.d < WRONG_THR).sum())
        print(f"\n[{tag}] 恒选 wrong {const_k}/{len(g)}（{const_wrong * 100:.1f}%）")
        for s, low_danger in rules.items():
            y = (g.d < WRONG_THR).to_numpy()
            x = g[s].to_numpy()
            m = ~np.isnan(x) & ~np.isnan(g.d.to_numpy())
            a = auc(y[m], (x[m] if low_danger else -x[m]))
            line = f"  {s}: AUROC {a:.3f}"
            for q in (0.5, 0.7):
                thr = np.nanquantile(pool[s], q)  # LODO 校准
                if low_danger:
                    sel = g[s] >= thr
                else:
                    sel = g[s] <= thr
                sel = sel & g.d.notna()
                kk = int((g.d[sel] < WRONG_THR).sum())
                line += (f" | q{int(q*100)}(LODO) cov {sel.mean():.2f} "
                         f"wrong {kk}/{int(sel.sum())} sysΔ {g.d[sel].sum() / len(g):+.4f}")
                verdict.append({"finding": tag, "signal": s, "auroc": a, "q": q,
                                "cov": float(sel.mean()), "wrong_k": kk,
                                "wrong_n": int(sel.sum()),
                                "sys_delta": float(g.d[sel].sum() / len(g)),
                                "const_wrong": const_wrong})
            print(line, flush=True)
    pd.DataFrame(verdict).to_csv(f"{OUT}/p22_verdict.csv", index=False)

    # --- 停止规则判定 ---
    print("\n== 停止规则判定（主终点：wrong<恒选 且 cov≥0.5） ==")
    hit = False
    for v in verdict:
        if v["q"] in (0.5, 0.7) and v["cov"] >= 0.5 and v["wrong_k"] / max(v["wrong_n"], 1) < v["const_wrong"]:
            print(f"  ✅ {v['finding']}/{v['signal']} q{int(v['q']*100)}: "
                  f"wrong {v['wrong_k']}/{v['wrong_n']} < 恒选（cov {v['cov']:.2f}）")
            hit = True
    if not hit:
        print("  ❌ 无信号在 cov≥0.5 下稳定降低 wrong → 按预注册停止规则放弃门控主张")
    log("完成")
    return 0


def train_head_quick(FE, C, device, sd):
    """快速重训 CornHead（特征/角点缓存），用于 seed 集成。"""
    import numpy as np
    import torch.nn.functional as F
    keys = [str(k) for k in FE["keys"]]
    T, ok = C["T"], C["ok"].astype(bool)
    CORNERS = np.array([[0, 0], [224, 0], [224, 224], [0, 224]], np.float32)
    corners = np.zeros((len(keys), 8), np.float32)
    for i in np.where(ok)[0]:
        pts = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), T[i]).reshape(4, 2)
        corners[i] = (pts / 224.0).reshape(8)
    gap = FE["f_photo"].astype(np.float32).mean(axis=(2, 3))
    torch.manual_seed(sd)
    h = CornHead().to(device)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-5)
    idx = np.where(ok)[0]
    X = torch.from_numpy(gap).to(device)
    G = torch.from_numpy(corners).to(device)
    for ep in range(300):
        perm = np.random.default_rng(sd * 31 + ep).permutation(len(idx))
        for s0 in range(0, len(idx), 32):
            sel = perm[s0:s0 + 32]
            loss = F.mse_loss(h(X[idx][sel]), G[idx][sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return h.eval()


def train_seed_adapter(model, w1, device, sd):
    """Cardi no_ident adapter 的不同 seed 重训（A4 信号用）。"""
    D = np.load(f"{OUT}/pair_cache_cardi.npz", allow_pickle=True)
    FE = np.load(f"{OUT}/p18_feat_cardi.npz", allow_pickle=True)
    C = np.load(f"{OUT}/pair_corners_pair_cache_cardi.npz", allow_pickle=True)
    reg_ok = D["reg_ok"].astype(bool)
    mask_p = D["mask_p"].astype(np.float32)
    prob_t = FE["prob_t"].astype(np.float32)
    f_photo, f_clean = FE["f_photo"], FE["f_clean"]
    import cv2
    cam_t = np.zeros((len(f_photo), 224, 224), np.float32)
    Tpair, ok_pair = C["T"], C["ok"].astype(bool)
    with torch.no_grad():
        for i in np.where(ok_pair & reg_ok)[0]:
            fc_i = torch.from_numpy(np.asarray(f_clean[i], np.float32)).to(device)[None]
            cam_t[i] = cv2.warpPerspective(cam_from_feat(fc_i, w1)[0].cpu().numpy(),
                                           Tpair[i], (224, 224))
    torch.manual_seed(sd)
    return train_pairloc(W_NOID, model, w1, device, reg_ok, f_photo, f_clean,
                         mask_p, prob_t, cam_t)


if __name__ == "__main__":
    raise SystemExit(main())
