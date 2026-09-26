"""
make_agent_split_fixed.py — 修正版患者级 Agent 划分

修复内容
--------
原版 src/make_agent_split.py 先按众数标签把患者排成 [组0, 组1, 组2] 的
拼接列表，再用**全局前缀切片**取 calibration / route_validation /
model_selection。由于众数标签=0 的患者占绝大多数（CheXpert 约 84% 行标签
为阴性），前 30% 的前缀完全落在组 0 内部 —— 三个子集全部只含阴性为主的
患者，患病率从 13.1% 塌缩到约 2.9%。

本版改为：在**每个众数标签组内部**按比例分配三个子集（与
make_patient_split.py 的正确做法一致），保证各子集类别构成与整体一致。

用法
----
    python3 src/make_agent_split_fixed.py --manifest <in.csv> --output-dir <dir>
    python3 src/make_agent_split_fixed.py --verify      # 跑合成数据自检

作者：ClawsGO Science Agent
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GROUP_FOR_ALLOCATION = 30   # 组太小则全部划入 model_train，避免过度切分


def allocate_splits(patient_label: pd.Series, fractions: dict, seed: int,
                    min_group: int = MIN_GROUP_FOR_ALLOCATION) -> dict:
    """
    在每个标签组内部按比例切分。返回 {split_name: set(patient_id)}。
    这是原版缺失的一步。
    """
    cal, route, sel = set(), set(), set()
    for value, g in patient_label.groupby(patient_label):
        shuffled = g.sample(frac=1.0, random_state=seed + int(value))
        ids = shuffled.index.astype(str).tolist()
        n = len(ids)
        if n < min_group:
            # 组太小：不切分，全部留给训练，避免某个子集被单一类别占据
            continue
        n_cal = max(1, round(n * fractions["calibration"]))
        n_route = max(1, round(n * fractions["route_validation"]))
        n_sel = max(1, round(n * fractions["model_selection"]))
        cal.update(ids[:n_cal])
        route.update(ids[n_cal:n_cal + n_route])
        sel.update(ids[n_cal + n_route:n_cal + n_route + n_sel])
    return {"calibration": cal, "route_validation": route,
            "model_selection": sel}


def build(manifest: Path, fractions: dict, seed: int):
    df = pd.read_csv(manifest)
    df["Cardiomegaly"] = df["Cardiomegaly"].fillna(0).replace(-1, 2).astype(int)
    lateral_rows = int((df["Frontal/Lateral"] != "Frontal").sum())
    train = df[df.split.eq("train") & df["Frontal/Lateral"].eq("Frontal")].copy()
    official = df[df.split.eq("valid") & df["Frontal/Lateral"].eq("Frontal")].copy()

    patient_label = train.groupby("Patient")["Cardiomegaly"].agg(
        lambda x: int(x.value_counts().index[0]))
    alloc = allocate_splits(patient_label, fractions, seed)
    cal, route, sel = alloc["calibration"], alloc["route_validation"], alloc["model_selection"]

    train["agent_split"] = train.Patient.astype(str).map(
        lambda x: "calibration" if x in cal
        else "route_validation" if x in route
        else "model_selection" if x in sel
        else "model_train")
    official["agent_split"] = "official_valid"
    out = pd.concat([train, official], ignore_index=True)
    return out, lateral_rows, patient_label


def composition_report(out: pd.DataFrame, patient_label: pd.Series) -> pd.DataFrame:
    rows = []
    for s, g in out.groupby("agent_split"):
        known = g[g["Cardiomegaly"] != 2]
        pl = patient_label.reindex(g["Patient"].astype(str).unique()).dropna()
        rows.append({
            "split": s,
            "patients": g["Patient"].nunique(),
            "rows": len(g),
            "prevalence_known%": round(100 * (known["Cardiomegaly"] == 1).mean(), 2)
            if len(known) else np.nan,
            "modal_label_0_patients": int((pl == 0).sum()),
            "modal_label_1_patients": int((pl == 1).sum()),
            "modal_label_2_patients": int((pl == 2).sum()),
        })
    return pd.DataFrame(rows).sort_values("split")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-fraction", type=float, default=.1)
    p.add_argument("--route-fraction", type=float, default=.1)
    p.add_argument("--model-selection-fraction", type=float, default=.1)
    p.add_argument("--verify", action="store_true",
                   help="在合成数据上自检修复效果")
    args = p.parse_args()

    fractions = {"calibration": args.calibration_fraction,
                 "route_validation": args.route_fraction,
                 "model_selection": args.model_selection_fraction}

    if args.verify:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from verify_split_bug import simulate_chexpert_patients
        sim = simulate_chexpert_patients()
        sim["Frontal/Lateral"] = "Frontal"
        sim["split"] = "train"
        sim["Path"] = [f"train/{p_}/{i}.jpg"
                       for i, p_ in enumerate(sim["Patient"])]
        tmp = Path("/tmp/_verify_manifest.csv")
        sim.to_csv(tmp, index=False)
        out, _, pl = build(tmp, fractions, args.seed)
        print("修复后各子集构成：")
        print(composition_report(out, pl).to_string(index=False))
        return

    if not args.manifest or not args.output_dir:
        p.error("需要 --manifest 与 --output-dir（或使用 --verify）")
    out, lateral_rows, pl = build(args.manifest, fractions, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "cardiomegaly_agent_split.csv", index=False)
    report = {
        "seed": args.seed, "fractions": fractions, "rows": len(out),
        "patient_counts": out.groupby("agent_split").Patient.nunique().to_dict(),
        "row_counts": out.agent_split.value_counts().to_dict(),
        "known_label_prevalence": {
            s: float((g.loc[g["Cardiomegaly"] != 2, "Cardiomegaly"] == 1).mean())
            for s, g in out.groupby("agent_split")},
        "modal_label_composition": {
            s: {int(k): int(v) for k, v in
                pl.reindex(g["Patient"].astype(str).unique()).dropna()
                  .value_counts().sort_index().items()}
            for s, g in out.groupby("agent_split") if s != "official_valid"},
        "lateral_rows_excluded_from_agent": lateral_rows,
        "patient_overlap": int(len(
            set(out.loc[out.agent_split.eq("model_train"), "Patient"]) &
            set(out.loc[out.agent_split.ne("model_train"), "Patient"]))),
        "note": "修正版：在每个众数标签组内部按比例切分，保证类别构成一致。",
    }
    (args.output_dir / "agent_split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
