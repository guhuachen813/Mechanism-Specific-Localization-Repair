"""CheXphoto v1.0 真实降质推理：与 CheXpert/NIH 完全相同的管线，跑在配对清单上。

口径对齐（v1.5 设计 A）
----------------------
* 模型输入 224；QC 特征在**同一张 224 的模型输入图**上计算，与设计 A 一致。
* 因此 `quality_features.py` 的 `min_size=256` 守卫同样会让 `hard_fail` 恒触发；
  本实验结论不依赖该字段。
* 三域共用的冻结温度 T = 0.4943421483039856（densenet121 seed42）。

与 NIH 那次的差异（刻意为之）
---------------------------
NIH 镜像不含 AP/PA 元数据，`projection_unknown` 恒为 1，给 `quality_risk` 带来
+1/9 的常量偏移。CheXphoto **带真实 AP/PA 与 Frontal/Lateral**，故这里传入真值，
`projection_ap` / `projection_unknown` 是真实的，与 CheXpert 侧口径一致。

额外产出
--------
natural/oneplus 是真实手机翻拍，另两个是合成算子。为支持「QC 能否区分真实降质
与合成降质」的分析，额外从**原始彩色图**（转灰度之前）取色偏与饱和度统计；
这部分只做记录，不参与 `quality_risk`。

用法（gpu04）：
    /root/miniconda3/bin/python chexphoto_infer.py \
        --manifest /root/chexphoto/manifest.csv \
        --model1 /root/project/outputs/repaired_seed42/densenet121/best.pt \
        --model2 /root/project/outputs/repaired_seed42/resnet50/best.pt \
        --output /root/chexphoto/out/preds.csv

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/root/project/src")
from train_baseline import make_model  # noqa: E402

import quality_features as qf  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path: Path, arch_hint: str, device: torch.device):
    ck = torch.load(path, map_location="cpu")
    for k in ("model", "state_dict"):
        if k in ck:
            ck = ck if k == "model" else {"model": ck[k]}
            break
    nc = int(ck.get("num_classes", 3))
    arch = ck.get("arch", arch_hint)
    model = make_model(nc, arch)
    model.load_state_dict(ck["model"])
    log(f"    载入 {arch}（{nc} 类，epoch {ck.get('epoch','?')}，seed {ck.get('seed','?')}）")
    return model.to(device).eval()


@torch.no_grad()
def forward_batch(model, tensors: torch.Tensor, device: torch.device) -> np.ndarray:
    return model(tensors.to(device, non_blocking=True)).float().cpu().numpy()


def colour_stats(rgb: Image.Image) -> dict:
    """原始彩色图的色偏/饱和度统计，用于识别真实翻拍。

    手机翻拍会带来色温偏移和饱和度异常，而这些在转灰度后就丢了。
    """
    a = np.asarray(rgb.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "col_sat_mean": float(sat.mean()),
        "col_sat_p95": float(np.quantile(sat, 0.95)),
        "col_rb_gap": float(r.mean() - b.mean()),
        "col_rg_gap": float(r.mean() - g.mean()),
        "col_bg_gap": float(b.mean() - g.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model1", type=Path, required=True)
    ap.add_argument("--model2", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备 {device}")

    m1 = load_model(args.model1, "densenet121", device)
    m2 = load_model(args.model2, "resnet50", device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    man = pd.read_csv(args.manifest)
    if args.limit:
        man = man.head(args.limit)
    log(f"清单 {len(man)} 行，来自 {man['base_key'].nunique()} 张基础图像")

    rows = man.to_dict("records")
    out: list[dict] = []
    t0 = time.time()

    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s:s + args.batch_size]

        pil224: list[Image.Image] = []      # 224 灰度，模型与 QC 共用
        t_main: list[torch.Tensor] = []
        t_flip: list[torch.Tensor] = []
        cols: list[dict] = []

        for r in chunk:
            with Image.open(r["disk_path"]) as im:
                im = im.convert("RGB")
                cols.append(colour_stats(im))
                grey = im.convert("L").resize(
                    (args.image_size, args.image_size), Image.BILINEAR)
            pil224.append(grey)
            rgb = grey.convert("RGB")
            t_main.append(tf(rgb))
            t_flip.append(tf(rgb.transpose(Image.FLIP_LEFT_RIGHT)))

        T = torch.stack(t_main)
        Tf = torch.stack(t_flip)
        l1 = forward_batch(m1, T, device)
        l2 = forward_batch(m2, T, device)
        l1f = forward_batch(m1, Tf, device)

        for j, r in enumerate(chunk):
            buf = io.BytesIO()
            pil224[j].save(buf, format="PNG")
            buf.seek(0)
            stub = pd.Series({
                "Frontal/Lateral": "Frontal" if r["orient"] == "frontal" else "Lateral",
                "AP/PA": "" if pd.isna(r["AP_PA"]) else str(r["AP_PA"]),
            })
            q = qf.features(buf, stub)
            rec = {
                "Path": r["disk_path"],
                "base_key": r["base_key"],
                "cond": r["cond"],
                "orient": r["orient"],
                "Patient": r["Patient"],
                "Study": r["Study"],
                "View": r["View"],
                "label": int(r["label"]),
                "AP_PA": r["AP_PA"],
                "Sex": r["Sex"],
                "Age": r["Age"],
            }
            for c in range(3):
                rec[f"l1_{c}"] = float(l1[j, c])
                rec[f"l2_{c}"] = float(l2[j, c])
                rec[f"l1f_{c}"] = float(l1f[j, c])
            for k, v in q.items():
                rec[f"qc_{k}"] = v
            risk_terms = [
                float(q["dynamic_range"] < 80 / 255),
                float(q["foreground_ratio"] < 0.06 or q["foreground_ratio"] > 0.99),
                float(q["contrast"] < 0.12),
                float(q["blur_score"] < 0.0008),
                float(q["noise_score"] > 0.08),
                float(q["border_crop_score"] > 0.12),
                float(q["left_right_symmetry"] > 0.20),
            ]
            rec["qc_quality_risk_noproj"] = float(np.mean(risk_terms))
            for k, v in cols[j].items():
                rec[k] = v
            out.append(rec)

        n = min(s + args.batch_size, len(rows))
        if s % (args.batch_size * 5) == 0 or n == len(rows):
            el = time.time() - t0
            log(f"  {n}/{len(rows)}（{n/max(el,1e-9):.1f} 张/秒）")

    df = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log(f"完成：{len(df)} 行 -> {args.output}，耗时 {time.time()-t0:.0f}s")
    log(f"阳性率 {df['label'].mean():.4f}")
    log("各条件 quality_risk 均值：")
    for c, g in df.groupby("cond"):
        log(f"    {c:24s} n={len(g):4d}  mean={g['qc_quality_risk'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
