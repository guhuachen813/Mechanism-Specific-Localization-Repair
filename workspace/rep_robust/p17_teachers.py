r"""P17a：Pleural Effusion / Edema teacher 训练（新病种泛化的前置）。

复刻 train_baseline_atel.py 的配方（DenseNet-121 from scratch、3 类
{0,1,2=-1 uncertain}、加权 CE、AdamW 1e-4、320px、10 epochs、seed42），
标签列参数化。数据 = cardiomegaly_all.csv（CheXpert train 全表），
患者级抽样 train≈25k / val≈6k（与 atel teacher 同量级，~15min/个）。

输出 /root/project/outputs/{eff,ede}_seed42_densenet121/best.pt
（ckpt 格式与 load_model 兼容：model/epoch/seed/num_classes=3/arch）。

用法：/root/miniconda3/bin/python p17_teachers.py [eff|ede]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MANIFEST = "/root/project/data/manifests/cardiomegaly_all.csv"
DATA_ROOT = Path("/root/autodl-tmp")
COL = {"eff": "Pleural Effusion", "ede": "Edema"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DS(Dataset):
    def __init__(self, frame, train):
        self.f = frame.reset_index(drop=True)
        ops = [transforms.Resize((320, 320))]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        r = self.f.iloc[i]
        p = DATA_ROOT / str(r["Path"]).replace("CheXpert-v1.0-small/",
                                               "CheXpert-v1.0-small/")
        im = Image.open(p).convert("RGB")
        return self.tf(im), torch.tensor(int(r["y"]), dtype=torch.long)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "eff"
    col = COL[tag]
    seed_everything(42)
    df = pd.read_csv(MANIFEST)
    df = df[df[col].isin([0.0, 1.0, -1.0])].copy()
    df["y"] = df[col].astype(int).map({-1: 2, 0: 0, 1: 1})
    pats = sorted(df["Patient"].unique())
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pats))
    tr_p, va_p = set(), set()
    tr_rows = va_rows = 0
    for i in perm:
        n = int((df["Patient"] == pats[i]).sum())
        if tr_rows < 25000:
            tr_p.add(pats[i])
            tr_rows += n
        elif va_rows < 6000:
            va_p.add(pats[i])
            va_rows += n
        else:
            break
    tr = df[df["Patient"].isin(tr_p)]
    va = df[df["Patient"].isin(va_p)]
    print(f"[{tag}] train {len(tr)} rows / val {len(va)} rows", flush=True)

    device = torch.device("cuda")
    tl = DataLoader(DS(tr, True), batch_size=16, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(DS(va, False), batch_size=32, shuffle=False, num_workers=8,
                    pin_memory=True)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    model = model.to(device)
    counts = tr["y"].value_counts().reindex([0, 1, 2], fill_value=0).to_numpy(float)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(
        counts.sum() / np.maximum(counts * 3, 1), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    out = Path(f"/root/project/outputs/{tag}_seed42_densenet121")
    out.mkdir(parents=True, exist_ok=True)
    best = (float("inf"), None)
    hist = []
    t0 = time.time()
    for ep in range(1, 11):
        model.train()
        run = 0.0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            lo = model(x.to(device))
            loss = crit(lo, y.to(device))
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for x, y in vl:
                lo = model(x.to(device))
                vloss += crit(lo, y.to(device)).item() * len(y)
        vloss /= len(va)
        rec = {"epoch": ep, "train_loss": run / len(tr), "val_loss": vloss}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if vloss < best[0]:
            best = (vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    state = {"model": best[1], "epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1),
             "seed": 42, "num_classes": 3, "arch": "densenet121", "threshold": 0.5}
    torch.save(state, out / "best.pt")
    (out / "train_metadata.json").write_text(json.dumps(
        {"tag": tag, "col": col, "epochs": 10, "seed": 42, "train_rows": len(tr),
         "val_rows": len(va), "elapsed_seconds": time.time() - t0, "history": hist},
        indent=2))
    print(f"[{tag}] 完成 {time.time() - t0:.0f}s → {out}/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
