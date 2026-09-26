"""拉取 CheXphoto 的两张小表（valid.csv / valid 文件索引），看清变换编码与标签分布。

在 gpu04 上运行：
    /root/miniconda3/bin/python /root/chexphoto/probe2_tables.py

作者：ClawsGO Science Agent
"""
import os
import pathlib

TOKEN_FILE = pathlib.Path("/root/.redivis_token")
os.environ["REDIVIS_API_TOKEN"] = TOKEN_FILE.read_text().strip()
os.environ.setdefault("REDIVIS_API_ENDPOINT", "https://stanford.redivis.com/api/v1")

import pandas as pd  # noqa: E402
import redivis  # noqa: E402

OUT = pathlib.Path("/root/chexphoto/meta")
OUT.mkdir(parents=True, exist_ok=True)

ds = redivis.dataset("aimi.chexphoto:2qwg:v1_0")

for name in ("valid_csv", "valid", "train_csv"):
    t = ds.table(name)
    print("=" * 70)
    print(f"表 {name}")
    df = t.to_pandas_dataframe()
    print(f"  shape = {df.shape}")
    print(f"  columns = {list(df.columns)}")
    print("  前 8 行：")
    print(df.head(8).to_string(max_colwidth=48))
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  -> 已存 {OUT / (name + '.csv')}")

    if name in ("valid_csv", "train_csv"):
        print("  标签列取值分布：")
        for c in ("Cardiomegaly", "No_Finding"):
            if c in df.columns:
                print(f"    {c}: {df[c].value_counts(dropna=False).to_dict()}")
        for c in ("Frontal_Lateral", "AP_PA", "Sex"):
            if c in df.columns:
                print(f"    {c}: {df[c].value_counts(dropna=False).to_dict()}")
        print(f"    Path 样例: {df['Path'].head(6).tolist()}")

    if name in ("valid", "train"):
        print(f"  file_name 样例: {df['file_name'].head(8).tolist()}")
        print(f"  文件总大小: {df['size'].sum() / 1e9:.3f} GB")
