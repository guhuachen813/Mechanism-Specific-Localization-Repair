"""把官方文件名与患者 ID 接回 NIH 推理结果表。

依据
----
镜像 `g-ronimo/NIH-Chest-X-ray-dataset_resized300px` 不保留文件名，但经验证
它按文件名排序：第 g 张图 = `sorted(test_list.txt)[g]`。验证方式是逐卷比对
全部 15 个类的计数（15 类 × 4 卷 = 60 项约束），全部逐项相同——卷内随机置换
不可能同时满足这些边际计数。

因此可以恢复每张图的官方文件名与 Patient ID，并把 Bootstrap 从图像级
升级为**患者级整群 Bootstrap**（官方 test 划分共 2,797 名患者、均 9.15 张/人、
最多 184 张，聚类效应不可忽略）。

本脚本同时做两项端到端校验：
  1. 结果表逐卷行号必须连续（证明 g = 行号这一前提）
  2. 由官方 CSV 独立推出的 Cardiomegaly 标签必须与推理时用的标签完全一致

作者：ClawsGO Science Agent
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

META = Path(__file__).resolve().parent / "nih_meta"
SHARD_SIZE = 6399


def main(csv_in: Path, csv_out: Path) -> int:
    df = pd.read_csv(csv_in)
    print(f"读入 {len(df)} 行")

    stem = df["Path"].str.split("/").str[1]
    stems = sorted(stem.unique())
    print(f"分卷：{stems}")
    if len(stems) != 4:
        print("分卷数不是 4，终止")
        return 1

    # 校验 1：逐卷行号连续 —— 这是 g = 行号 的前提
    for s in stems:
        idxs = df.index[stem == s]
        if list(idxs) != list(range(idxs.min(), idxs.max() + 1)):
            print(f"分卷 {s} 的行不连续，终止")
            return 1
        if len(idxs) != SHARD_SIZE:
            print(f"分卷 {s} 有 {len(idxs)} 行，预期 {SHARD_SIZE}，终止")
            return 1
    if [stems.index(s) for s in stem] != sorted([stems.index(s) for s in stem]):
        print("分卷未按顺序排列，终止")
        return 1

    # 恢复官方文件名与患者
    lst = sorted(ln.strip() for ln in open(META / "test_list.txt") if ln.strip())
    if len(lst) != len(df):
        print(f"清单 {len(lst)} 条与结果 {len(df)} 行不等，终止")
        return 1
    df["nih_file"] = [lst[i] for i in df.index]

    entry = pd.read_csv(META / "Data_Entry_2017_v2020.csv").set_index("Image Index")
    missing = [f for f in df["nih_file"] if f not in entry.index]
    if missing:
        print(f"{len(missing)} 个文件名不在官方 CSV 中，例如 {missing[:3]}，终止")
        return 1

    df["Patient"] = [str(entry.loc[f, "Patient ID"]) for f in df["nih_file"]]
    df["Study"] = [f.rsplit(".", 1)[0] for f in df["nih_file"]]
    df["nih_followup"] = [int(entry.loc[f, "Follow-up #"]) for f in df["nih_file"]]
    df["nih_view"] = [str(entry.loc[f, "View Position"]) for f in df["nih_file"]]

    # 校验 2：由官方 CSV 独立推出的标签必须与推理时用的标签一致
    ref = pd.Series(
        [int("Cardiomegaly" in str(entry.loc[f, "Finding Labels"]).split("|"))
         for f in df["nih_file"]], index=df.index)
    n_bad = int((ref != df["label"]).sum())
    print(f"标签一致性校验：{len(df) - n_bad}/{len(df)} 一致")
    if n_bad:
        print(f"  !! {n_bad} 行不一致，映射可能有误")
        return 1

    df["Path"] = df["nih_file"]

    # ---- 用官方 View Position 重算投影项与质量风险 ----
    # 推理时没有 AP/PA 元数据，只能把 projection_unknown 置 1、projection_ap 置 0，
    # 使 quality_risk 带一个 +1/9 的常量偏移。既然官方 CSV 提供 View Position，
    # 这里按真实投射位重算这两个布尔项与由它们定义的 quality_risk，
    # 消除该偏移，使跨数据集的质量风险水平可以直接比较。
    df["qc_projection_ap"] = [int(str(entry.loc[f, "View Position"]) == "AP")
                              for f in df["nih_file"]]
    df["qc_projection_unknown"] = 0

    def _risk(cols):
        return sum(cols) / len(cols)

    terms9 = [
        (df["qc_dynamic_range"] < 80 / 255).astype(float),
        ((df["qc_foreground_ratio"] < 0.06) | (df["qc_foreground_ratio"] > 0.99)).astype(float),
        (df["qc_contrast"] < 0.12).astype(float),
        (df["qc_blur_score"] < 0.0008).astype(float),
        (df["qc_noise_score"] > 0.08).astype(float),
        (df["qc_border_crop_score"] > 0.12).astype(float),
        (df["qc_left_right_symmetry"] > 0.20).astype(float),
        df["qc_projection_ap"].astype(float),
        df["qc_projection_unknown"].astype(float),
    ]
    df["qc_quality_risk"] = _risk(terms9)
    df["qc_quality_risk_noproj"] = _risk(terms9[:7])

    df.drop(columns=["nih_file"]).to_csv(csv_out, index=False)
    print(f"写出 {csv_out}")
    print(f"患者数 {df['Patient'].nunique()}，均 {len(df)/df['Patient'].nunique():.2f} 张/人，"
          f"最多 {df['Patient'].value_counts().max()} 张")
    print(f"视图分布 {dict(df['nih_view'].value_counts())}")
    print(f"Cardiomegaly 阳性率 {df['label'].mean():.4f}")
    print(f"quality_risk：均值 {df['qc_quality_risk'].mean():.4f}，"
          f"取值数 {df['qc_quality_risk'].nunique()}，AP 组均值 "
          f"{df.loc[df['qc_projection_ap'] == 1, 'qc_quality_risk'].mean():.4f}，PA 组均值 "
          f"{df.loc[df['qc_projection_ap'] == 0, 'qc_quality_risk'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
