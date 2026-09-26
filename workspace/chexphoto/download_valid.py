"""下载 CheXphoto v1.0 的 valid 划分（702 个文件，约 1.4 GB）。

只下 valid：train 的 Cardiomegaly 有 79% 缺失（25,709/32,521 为 NA），
而本项目用的是冻结模型、不做再训练，train 对我们的分析没有用处；
且 train 约 65 GB，超出 gpu04 的可用空间。

valid 是完整的三变体配对设计：
  natural/oneplus       真实手机拍摄（真实物理降质）
  synthetic/photographic 合成「翻拍」降质
  synthetic/digital      合成「数字」降质
三者共享同一批基础图像与同一套标签。

在 gpu04 上运行：
    /root/miniconda3/bin/python /root/chexphoto/download_valid.py

支持断点续传：已存在且大小正确的文件会跳过。

作者：ClawsGO Science Agent
"""
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

TOKEN_FILE = pathlib.Path("/root/.redivis_token")
if not TOKEN_FILE.is_file():
    sys.exit(f"缺少 token 文件：{TOKEN_FILE}")

import os  # noqa: E402

os.environ["REDIVIS_API_TOKEN"] = TOKEN_FILE.read_text().strip()
os.environ.setdefault("REDIVIS_API_ENDPOINT", "https://stanford.redivis.com/api/v1")

import redivis  # noqa: E402
from redivis.classes.File import File  # noqa: E402

META = pathlib.Path("/root/chexphoto/meta")
DEST = pathlib.Path("/root/autodl-tmp/chexphoto/valid")
DEST.mkdir(parents=True, exist_ok=True)

WORKERS = 8
_lock = threading.Lock()
_done = {"n": 0, "bytes": 0, "skip": 0, "fail": 0}


def fetch(row, table, total, total_bytes):
    target = DEST / row.file_name
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size == int(row.size):
        with _lock:
            _done["n"] += 1
            _done["skip"] += 1
            _done["bytes"] += int(row.size)
        return None

    f = File(row.file_id, row.file_name, directory=None, table=table,
             properties={"size": int(row.size), "md5_hash": row.md5_hash})
    try:
        f.download(path=str(target), overwrite=True, progress=False)
    except Exception as e:
        with _lock:
            _done["n"] += 1
            _done["fail"] += 1
        return f"{row.file_name}: {type(e).__name__}: {e}"

    with _lock:
        _done["n"] += 1
        _done["bytes"] += int(row.size)
        if _done["n"] % 25 == 0 or _done["n"] == total:
            pct = 100 * _done["n"] / total
            gb = _done["bytes"] / 1e9
            print(f"  [{_done['n']:4d}/{total}] {pct:5.1f}%  {gb:5.2f}/{total_bytes/1e9:.2f} GB"
                  f"  跳过 {_done['skip']}  失败 {_done['fail']}", flush=True)
    return None


def main():
    ds = redivis.dataset("aimi.chexphoto:2qwg:v1_0")
    table = ds.table("valid")

    idx = pd.read_csv(META / "valid.csv")
    total = len(idx)
    total_bytes = int(idx["size"].sum())
    print(f"待下载 {total} 个文件，共 {total_bytes/1e9:.3f} GB -> {DEST}")
    print(f"并发 {WORKERS}\n")

    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, r, table, total, total_bytes)
                for r in idx.itertuples()]
        for fu in as_completed(futs):
            err = fu.result()
            if err:
                errors.append(err)

    print(f"\n完成：成功 {_done['n'] - _done['fail']}，跳过 {_done['skip']}，失败 {_done['fail']}")
    if errors:
        print("失败明细（前 10 条）：")
        for e in errors[:10]:
            print("  ", e)
        sys.exit(1)

    # 完整性核对
    print("\n=== 完整性核对 ===")
    bad = []
    for r in idx.itertuples():
        p = DEST / r.file_name
        if not p.exists() or p.stat().st_size != int(r.size):
            bad.append(r.file_name)
    print(f"  尺寸不符或缺失: {len(bad)} / {total}")
    if bad:
        for b in bad[:10]:
            print("   ", b)
        sys.exit(1)
    print("  ✓ 全部文件尺寸与索引一致")


if __name__ == "__main__":
    main()
