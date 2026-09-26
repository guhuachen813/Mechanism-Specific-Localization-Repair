"""探查 CheXphoto v1.0 的表结构与文件索引（只读元数据，不下载图像）。

在 gpu04 上运行：
    /root/miniconda3/bin/python /root/chexphoto/probe_chexphoto.py

token 从 /root/.redivis_token 读入环境变量，不落盘、不打印。

作者：ClawsGO Science Agent
"""
import os
import pathlib
import sys

TOKEN_FILE = pathlib.Path("/root/.redivis_token")
if not TOKEN_FILE.is_file():
    sys.exit(f"缺少 token 文件：{TOKEN_FILE}")
os.environ["REDIVIS_API_TOKEN"] = TOKEN_FILE.read_text().strip()
# 注意：REDIVIS_API_ENDPOINT 是完整 API 基址（客户端默认 https://redivis.com/api/v1），
# 不是裸主机名。只写主机名会让客户端去请求网页而不是 API，拿回一坨 HTML。
os.environ.setdefault("REDIVIS_API_ENDPOINT", "https://stanford.redivis.com/api/v1")

import redivis  # noqa: E402

REF = "AIMI.CheXphoto"

ds = redivis.dataset(REF)
print("=" * 62)
print("dataset :", ds.name)
print("version :", ds.version_tag)
props = getattr(ds, "properties", {}) or {}
for k in ("id", "name", "description", "createdAt", "updatedAt"):
    if k in props:
        v = str(props[k]).replace("\n", " ")[:120]
        print(f"  {k:12s}: {v}")
print("=" * 62)

tables = ds.list_tables()
print(f"表数量: {len(tables)}")
for t in tables:
    print(f"\n--- 表 {t.name!r} ---")
    try:
        variables = t.list_variables()
        print(f"  变量 {len(variables)} 个:")
        for v in variables:
            label = getattr(v, "label", None)
            print(f"    {v.name:32s} {str(getattr(v,'type','?')):10s} {label or ''}")
    except Exception as e:
        print(f"  变量读取失败: {type(e).__name__}: {e}")
    for attr in ("num_rows", "numRows", "properties"):
        try:
            val = getattr(t, attr)
            print(f"  {attr} = {str(val)[:200]}")
        except Exception:
            pass
