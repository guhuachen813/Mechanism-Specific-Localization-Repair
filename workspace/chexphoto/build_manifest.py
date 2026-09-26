"""核对 CheXphoto valid 下载完整性，并构建配对实验清单 manifest.csv。

输出 manifest 的每一行 = 一个（基础图像 × 条件）组合：
    cond = clean                  -> 本地 CheXpert-v1.0-small/valid 原图
    cond = natural/oneplus        -> CheXphoto 真实手机拍摄
    cond = synthetic/photographic -> 合成「翻拍」降质
    cond = synthetic/digital      -> 合成「数字」降质
四个条件共享同一 label（已验证 Cardiomegaly 在三变体间零分歧，且与 clean 侧一致）。

在 gpu04 上运行：/root/miniconda3/bin/python /root/chexphoto/build_manifest.py

作者：ClawsGO Science Agent
"""
import base64
import hashlib
import pathlib

import pandas as pd

META = pathlib.Path("/root/chexphoto/meta")
ROOT = pathlib.Path("/root/autodl-tmp/chexphoto/valid")
CHEXPERT = pathlib.Path("/root/autodl-tmp/CheXpert-v1.0-small/valid")
OUT = pathlib.Path("/root/chexphoto/manifest.csv")

idx = pd.read_csv(META / "valid.csv")
csv = pd.read_csv(META / "valid_csv.csv")

# ------------------------------------------------------------------ md5 校验
print("=== md5 完整性校验 ===")
bad = []
for r in idx.itertuples():
    p = ROOT / r.file_name
    h = hashlib.md5(p.read_bytes()).digest()
    if h != base64.b64decode(r.md5_hash):
        bad.append(r.file_name)
print(f"  校验 {len(idx)} 个文件，md5 不符 {len(bad)} 个")
if bad:
    for b in bad[:10]:
        print("   ", b)
    raise SystemExit("md5 校验失败，需重新下载")
print("  ✓ 全部 md5 与索引一致")

# ------------------------------------------------------------------ 构建清单
# CheXphoto 侧：Path = CheXphoto-v1.0/valid/<kind>/<device>/<patient>/<study>/<view>.jpg
cp = csv.copy()
cp["base_key"] = cp["Path"].str.extract(
    r"^CheXphoto-v1\.0/valid/(?:natural|synthetic)/\w+/(.+)$")[0]
cp["cond"] = cp["Path"].str.extract(
    r"^CheXphoto-v1\.0/valid/(natural|synthetic)/(\w+)/") \
    .apply(lambda s: f"{s[0]}/{s[1]}", axis=1)
cp["disk_path"] = ROOT.as_posix() + "/" + cp["cond"] + "/" + cp["base_key"]
cp["Patient"] = cp["base_key"].str.split("/").str[0]
cp["Study"] = cp["base_key"].str.split("/").str[1]
cp["View"] = cp["base_key"].str.split("/").str[2]
cp["orient"] = cp["View"].str.extract(r"_(frontal|lateral)\.jpg$")[0]
cp["label"] = (cp["Cardiomegaly"] == 1.0).astype(int)

# clean 侧：同一批基础图像，指向本地 CheXpert 原图
clean = cp[cp["cond"] == "natural/oneplus"][
    ["base_key", "Patient", "Study", "View", "orient", "label",
     "AP_PA", "Sex", "Age"]].copy()
clean["cond"] = "clean"
clean["disk_path"] = CHEXPERT.as_posix() + "/" + clean["base_key"]
clean["Cardiomegaly"] = clean["label"].astype(float)

keep = ["base_key", "Patient", "Study", "View", "orient", "cond",
        "disk_path", "label", "Cardiomegaly", "AP_PA", "Sex", "Age"]
man = pd.concat([clean[keep], cp[keep]], ignore_index=True)

# 存在性检查
missing = [p for p in man["disk_path"] if not pathlib.Path(p).is_file()]
print(f"\n=== 清单 ===")
print(f"  总行数 {len(man)}（基础图像 {man['base_key'].nunique()} × 条件 {man['cond'].nunique()}）")
print(f"  磁盘上缺失: {len(missing)}")
if missing:
    for m in missing[:5]:
        print("   ", m)
    raise SystemExit("清单指向了不存在的文件")

print(f"\n  条件分布:")
print(man["cond"].value_counts().to_string())
print(f"\n  朝向 × 条件:")
print(pd.crosstab(man["orient"], man["cond"]).to_string())

fr = man[man["orient"] == "frontal"]
print(f"\n  Frontal 子集: {len(fr)} 行, {fr['base_key'].nunique()} 张基础图, "
      f"阳性率 {fr['label'].mean():.4f}")
print(f"  AP/PA: {fr['AP_PA'].value_counts(dropna=False).to_dict()}")

man.to_csv(OUT, index=False)
print(f"\n  -> {OUT}")
