r"""P24 v3：center-field prior 基线（eff/ede/ede + cardi 管线验证臂）。

口径与冻结管线对齐：
  - CAM = p18 的 cam_from_feat(特征, w1)，特征按缓存口径 cast float16（p18 特征
    缓存为 fp16，raw45 自检必须走同一量化）；
  - prior = cam * MedSAM 场掩码（detect_field + segment + mask_224，与 p65 一致）；
  - iou45 = cam_iou_curve(...)[0.45]；GT = silver_masks_224.npz 对应病种。
cardi 臂用于验证：raw45 应逐位复现 p21（fp16 特征），prior45 应复现 p62
（≈0.0951）。eff/ede 的 cand_zs/g2 不适用（无 per-病种 repairer 工件）。
"""
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import cam_benchmark
from cam_benchmark import MEAN, STD, load_model
from cam_prior import detect_field, mask_224, medsam_prep, segment
from p18_new_findings import OUT, cam_from_feat, iou45, mask_bbox

ROOT = "/root/autodl-tmp/experiments/rep_robust"
CKPT = {"cardi": "/root/project/outputs/repaired_seed42/densenet121/best.pt",
        "eff": "/root/project/outputs/eff_seed42_densenet121/best.pt",
        "ede": "/root/project/outputs/ede_seed42_densenet121/best.pt"}
FIND = {"cardi": "Cardiomegaly", "eff": "Pleural Effusion", "ede": "Edema"}
PHOTO = "/root/autodl-tmp/chexphoto/valid/natural/oneplus"
# cardi 的 raw45 冻结值来自 p21（其特征也是 fp16 缓存口径）——同一量化下自检
D18 = {"cardi": "p21_final_eval.csv", "eff": "p18_new_findings_eff.csv",
       "ede": "p18_new_findings_ede.csv"}


def log(m):
    print("[" + time.strftime("%H:%M:%S") + "] " + m, flush=True)


def main() -> int:
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    from transformers import SamModel
    smodel = SamModel.from_pretrained(ROOT + "/weights/medsam").to(device).eval()
    SM = np.load(OUT + "/silver_masks_224.npz", allow_pickle=True)
    findings = [str(x) for x in SM["findings"]]
    for tag in ["cardi", "eff", "ede"]:
        torch.manual_seed(42)
        cam_benchmark.CKPT = CKPT[tag]
        model = load_model(device)
        for p in model.parameters():
            p.requires_grad_(False)
        w1 = model.classifier.weight[1].detach()
        fi = findings.index(FIND[tag])
        d18 = pd.read_csv(OUT + "/" + D18[tag])
        if tag == "cardi":
            d18 = d18[d18.finding == "cardi"]
        rows = []
        for n, k in enumerate(d18.key):
            parts = k.split("_")
            path = (PHOTO + "/" + parts[0] + "/" + parts[1] + "/"
                    + "_".join(parts[2:]) + ".jpg")
            im = Image.open(path).convert("L")
            p224 = im.resize((224, 224), Image.BILINEAR)
            x = tf(p224.convert("RGB"))[None].to(device)
            with torch.no_grad():
                feat32 = model.features(x)
                cam32 = cam_from_feat(feat32, w1)[0].cpu().numpy()      # float32：prior 口径（对齐 p62 缓存）
                cam16 = cam_from_feat(feat32.to(torch.float16).float(),
                                      w1)[0].cpu().numpy()              # fp16：raw45 自检口径（对齐 p18 缓存）
            fr, ok_r = detect_field(p224)
            mm = mask_224(segment(smodel, medsam_prep(im), fr, device)).astype(np.uint8)
            gt_box = mask_bbox(SM[k][fi])   # iou45 的 GT 口径 = mask_bbox（掩码外接框列表）
            rows.append({"key": k, "patient": parts[0],
                         "raw45_chk": iou45(cam16, gt_box),
                         "prior45": iou45(cam32 * mm, gt_box),
                         "field_ok": int(ok_r), "mm_frac": float(mm.mean())})
            if (n + 1) % 20 == 0:
                log("[" + tag + "] " + str(n + 1) + "/" + str(len(d18)))
        df = pd.DataFrame(rows)
        df.to_csv(OUT + "/p24_prior_" + tag + ".csv", index=False)
        m = df.merge(d18[["key", "raw45"]], on="key")
        d = (m.raw45_chk - m.raw45).abs().max()
        log("[" + tag + "] raw45 自检 max|diff|=" + format(d, ".2e")
            + " | prior45 mean=" + format(df.prior45.mean(), ".4f")
            + " raw=" + format(d18.raw45.mean(), ".4f")
            + " d_prior=" + format(df.prior45.mean() - d18.raw45.mean(), "+.4f")
            + " | field_ok " + str(int(df.field_ok.sum())) + "/" + str(len(df))
            + " | mm_frac median " + format(df.mm_frac.median(), ".3f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
