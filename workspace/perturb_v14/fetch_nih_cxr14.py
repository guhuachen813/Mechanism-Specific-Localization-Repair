"""从 HuggingFace 镜像拉取 NIH ChestX-ray14 的官方 test 划分。

背景
----
AutoDL 机器直连 huggingface.co 不通，需先 `source /etc/network_turbo`。
全量 45 GB 装不下（autodl-tmp 只剩 29 GB），故逐卷处理：
    下载 images_00N.zip → 解压 → 只保留官方 test_list 中的图像 → 删掉该卷
峰值占用 ≈ 单卷（最多 4 GB）+ 累积的 test 图像；全量 test 划分约 25,596 张，
按每卷约 23% 命中率估算，最终约占 10 GB。

这样得到的样本在**患者层面与 NIH 官方 train_val 划分不相交**，
可直接用于外部验证，不需要再自己切分。

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

BASE = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
DEST = "/root/autodl-tmp/nih_cxr14"
IMGDIR = os.path.join(DEST, "images")
TMPDIR = os.path.join(DEST, "_tmp")
ZIPDIR = os.path.join(DEST, "zips")

SMALL = ["Data_Entry_2017_v2020.csv", "test_list.txt", "train_val_list.txt"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url: str, out: str) -> bool:
    """带断点续传与重试的下载。"""
    for attempt in range(4):
        r = subprocess.run(
            ["curl", "-sSL", "--fail", "--retry", "2", "--retry-delay", "5",
             "-C", "-", "-o", out, "--max-time", "3600", url],
        )
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
            return True
        log(f"    下载失败（第 {attempt + 1} 次，返回码 {r.returncode}），重试")
        time.sleep(5)
    return False


def main() -> int:
    for d in (DEST, IMGDIR, TMPDIR, ZIPDIR):
        os.makedirs(d, exist_ok=True)

    log("拉取元数据 CSV / 划分清单")
    for f in SMALL:
        p = os.path.join(DEST, f)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
        if not fetch(f"{BASE}/{f}", p):
            log(f"关键文件 {f} 下载失败，终止")
            return 1

    with open(os.path.join(DEST, "test_list.txt")) as fh:
        wanted = {ln.strip() for ln in fh if ln.strip()}
    log(f"官方 test 划分共 {len(wanted)} 张图像")

    have = {f for f in os.listdir(IMGDIR) if f.endswith(".png")}
    log(f"本地已有 {len(have)} 张")

    for i in range(1, 13):
        name = f"images_{i:03d}.zip"
        zp = os.path.join(ZIPDIR, name)
        t0 = time.time()

        log(f"--- {name} 开始下载 ---")
        if not fetch(f"{BASE}/images/{name}", zp):
            log(f"{name} 下载失败，跳过")
            continue
        sz = os.path.getsize(zp) / 1e9
        log(f"    下载完成 {sz:.2f} GB，用时 {time.time() - t0:.0f}s "
            f"（{sz / max(time.time() - t0, 1) * 1000:.1f} MB/s）")

        if os.path.exists(TMPDIR):
            shutil.rmtree(TMPDIR, ignore_errors=True)
        os.makedirs(TMPDIR, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", zp, "-d", TMPDIR], check=False)

        moved = 0
        for root, _, files in os.walk(TMPDIR):
            for fn in files:
                if fn in wanted and fn.endswith(".png"):
                    src = os.path.join(root, fn)
                    try:
                        shutil.move(src, os.path.join(IMGDIR, fn))
                        moved += 1
                    except Exception as exc:  # noqa: BLE001
                        log(f"    移动 {fn} 失败：{exc}")
        shutil.rmtree(TMPDIR, ignore_errors=True)
        os.remove(zp)
        log(f"    从本卷取到 {moved} 张 test 图像，累计 "
            f"{len(os.listdir(IMGDIR))}/{len(wanted)}，本卷总耗时 {time.time() - t0:.0f}s")

        if len(os.listdir(IMGDIR)) >= len(wanted):
            log("test 划分已全部到齐")
            break

    final = len([f for f in os.listdir(IMGDIR) if f.endswith(".png")])
    log(f"完成：共 {final}/{len(wanted)} 张，位于 {IMGDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
