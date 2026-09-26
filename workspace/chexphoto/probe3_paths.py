"""解析 CheXphoto 路径结构，确认 valid 的配对设计与各变体规模。

Path 形如：
  CheXphoto-v1.0/{train|valid}/{natural|synthetic}/{device}/{patient}/{study}/{view}_{orient}.jpg

在 gpu04 上运行：/root/miniconda3/bin/python /root/chexphoto/probe3_paths.py

作者：ClawsGO Science Agent
"""
import pathlib
import re

import pandas as pd

META = pathlib.Path("/root/chexphoto/meta")


def parse(p):
    parts = p.split("/")
    # CheXphoto-v1.0 / split / kind / device / patient / study / file
    return pd.Series({
        "split": parts[1] if len(parts) > 1 else None,
        "kind": parts[2] if len(parts) > 2 else None,
        "device": parts[3] if len(parts) > 3 else None,
        "patient": parts[4] if len(parts) > 4 else None,
        "study": parts[5] if len(parts) > 5 else None,
        "img": parts[6] if len(parts) > 6 else None,
    })


for name in ("valid_csv", "train_csv"):
    df = pd.read_csv(META / f"{name}.csv")
    meta = df["Path"].apply(parse)
    df = pd.concat([df, meta], axis=1)
    df["variant"] = df["kind"] + "/" + df["device"]
    df["orient"] = df["img"].str.extract(r"_(frontal|lateral)\.jpg$")

    print("=" * 72)
    print(f"### {name}   n = {len(df)}")
    print("\n变体构成 (variant × orient):")
    print(pd.crosstab(df["variant"], df["orient"], margins=True).to_string())

    print(f"\n患者数: {df['patient'].nunique()}   检查数: {df.groupby(['patient','study']).ngroups}")
    print(f"唯一图像 (patient,study,img): {df.groupby(['patient','study','img']).ngroups}")

    n_img = df.groupby(["patient", "study", "img"]).ngroups
    print(f"每张唯一图像的平均变体数: {len(df)/max(n_img,1):.3f}")

    # 只有 frontal 的配对情况（我们的模型是单视图）
    fr = df[df["orient"] == "frontal"]
    print(f"\n仅 Frontal: n = {len(fr)}, 唯一图像 = "
          f"{fr.groupby(['patient','study','img']).ngroups}, "
          f"患者 = {fr['patient'].nunique()}")
    print("Frntal 下各变体的图像数（应为同一批基础图像）:")
    print(fr.groupby("variant").agg(行数=("Path", "size"),
                                   唯一图=("img", lambda s: s.size)).to_string())

    # 每个变体是否覆盖同一批基础图像
    base = fr.groupby("variant")["Path"].apply(
        lambda s: frozenset(fr.loc[s.index, "patient"] + "/" + fr.loc[s.index, "img"]))
    keys = list(base.index)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = base[keys[i]], base[keys[j]]
            print(f"  {keys[i]:22s} vs {keys[j]:22s}  "
                  f"交集 {len(a & b)}  仅左 {len(a - b)}  仅右 {len(b - a)}")

    if "Cardiomegaly" in df.columns:
        print("\nCardiomegaly 标签分布（按变体）：")
        print(pd.crosstab(df["variant"], df["Cardiomegaly"], dropna=False).to_string())
