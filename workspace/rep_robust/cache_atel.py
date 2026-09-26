r"""Atelectasis cache 驱动：复用 repairer.stage_cache，输出到影子目录避免覆盖
Cardiomegaly cache.npz。MedSAM 权重经 symlink 共享。

用法：/root/miniconda3/bin/python cache_atel.py [--limit 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cam_benchmark
import repairer as rp

SHADOW = Path("/root/autodl-tmp/experiments/rep_robust_atel")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    (SHADOW / "repairer").mkdir(parents=True, exist_ok=True)
    if not (SHADOW / "weights").exists():
        (SHADOW / "weights").symlink_to(
            "/root/autodl-tmp/experiments/rep_robust/weights")
    rp.OUTDIR = SHADOW
    rp.FINDING = "Atelectasis"
    cam_benchmark.FINDING = "Atelectasis"
    cam_benchmark.CKPT = Path(
        "/root/project/outputs/atel_seed42_densenet121/best.pt")
    rp.stage_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
