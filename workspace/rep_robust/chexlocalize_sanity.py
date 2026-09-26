r"""Sanity：把解码出的 Cardiomegaly/Atelectasis mask 叠回 CheXpert-small clean 图。"""
import numpy as np
from PIL import Image

NPZ = "/root/autodl-tmp/experiments/rep_robust/chexlocalize/gt_masks_val.npz"
ROOT = "/root/autodl-tmp/CheXpert-v1.0-small"
OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"

npz = np.load(NPZ, allow_pickle=True)
findings = list(npz["findings"])
targets = [("cardi", "Cardiomegaly"), ("atel", "Atelectasis")]
done = 0
for k in npz.files:
    if k == "findings":
        continue
    m = npz[k]
    for tag, f in targets:
        if f not in findings:
            continue
        mm = m[findings.index(f)]
        if mm.sum() <= 0:
            continue
        parts = k.split("_")  # patient64622_study1_view1_frontal
        path = f"{ROOT}/valid/{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}.jpg"
        try:
            im = Image.open(path).convert("L").resize((390, 390))
        except FileNotFoundError:
            print("miss", path)
            continue
        ov = np.stack([np.asarray(im)] * 3, -1)
        sel = mm > 0
        ov[sel, 0] = np.minimum(255, ov[sel, 0] * 0.4 + 110)
        Image.fromarray(ov).save(f"{OUT}/sanity_{tag}_{k}.png")
        print(tag, k, "area", round(float(mm.mean()), 3))
        done += 1
        break
    if done >= 6:
        break
