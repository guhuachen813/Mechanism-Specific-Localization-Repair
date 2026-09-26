"""核对 CheXphoto valid 的基础图像是否与我们本地的 CheXpert-v1.0-small/valid 完全对齐。

若能对齐，就得到一个**配对设计**：同一张胸片，一个干净原图 + 三个降质变体，
标签相同。这比现有的非配对跨域比较强得多。

在 gpu04 上运行：/root/miniconda3/bin/python /root/chexphoto/probe4_pairing.py

作者：ClawsGO Science Agent
"""
import pathlib

import pandas as pd

CHEXPERT = pathlib.Path("/root/autodl-tmp/CheXpert-v1.0-small")
META = pathlib.Path("/root/chexphoto/meta")

# --- 本地 CheXpert valid 的 frontal 图像 ---
cx = sorted(p.relative_to(CHEXPERT).as_posix()
            for p in (CHEXPERT / "valid").rglob("*_frontal.jpg"))
print(f"CheXpert-v1.0-small/valid frontal 图像数: {len(cx)}")
print("  例:", cx[:3])
cx_set = set(cx)

# --- CheXphoto valid 的基础图像（用 natural/oneplus 这一变体代表）---
df = pd.read_csv(META / "valid_csv.csv")
nat = df[df["Path"].str.contains("/natural/oneplus/")]
nat_fr = nat[nat["Path"].str.endswith("_frontal.jpg")]
print(f"\nCheXphoto valid natural/oneplus frontal: {len(nat_fr)}")

mapped = []
for p in nat_fr["Path"]:
    # CheXphoto-v1.0/valid/natural/oneplus/patientX/studyN/viewM_frontal.jpg
    parts = p.split("/")
    rel = "/".join(["valid", parts[4], parts[5], parts[6]])
    mapped.append(rel)
mapped = sorted(mapped)
print("  映射到 CheXpert 路径例:", mapped[:3])

m_set = set(mapped)
print("\n=== 配对结果 ===")
print(f"  CheXphoto→CheXpert 命中: {len(m_set & cx_set)} / {len(m_set)}")
print(f"  仅在 CheXpert 侧: {sorted(cx_set - m_set)[:5]}  (共 {len(cx_set - m_set)})")
print(f"  仅在 CheXphoto 侧: {sorted(m_set - cx_set)[:5]}  (共 {len(m_set - cx_set)})")

if m_set == cx_set:
    print("\n  ✓ 两侧完全一致：202 张 frontal 图一一对应，可直接做配对实验")

# --- 标签一致性：CheXphoto 的 Cardiomegaly 对三个变体应当相同 ---
print("\n=== 三个变体的配对完整性 ===")
df["base"] = df["Path"].str.replace(
    r"^CheXphoto-v1\.0/valid/(natural|synthetic)/(oneplus|digital|photographic)/",
    "", regex=True)
piv = df.pivot_table(index="base", columns=df["Path"].str.extract(
    r"^CheXphoto-v1\.0/valid/(?:natural|synthetic)/(\w+)/")[0],
    values="Cardiomegaly", aggfunc="first")
print(f"  基础图像数: {len(piv)}   变体列: {list(piv.columns)}")
print(f"  任一变体缺失的行数: {piv.isna().any(axis=1).sum()}")
print(f"  变体之间标签不一致的行数: {(piv.nunique(axis=1, dropna=True) > 1).sum()}")
print(f"  Cardiomegaly 分布 (front+lat 全体): {piv.iloc[:,0].value_counts().to_dict()}")
