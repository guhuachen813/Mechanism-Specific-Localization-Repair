"""
verify_split_bug.py — 复现 make_agent_split.py 的患者划分逻辑并检查其类别构成

背景
----
src/make_agent_split.py 第 10-12 行：

    patient_label = train.groupby("Patient")["Cardiomegaly"].agg(
        lambda x: int(x.value_counts().index[0]))          # 每个患者的众数标签
    groups = []
    for value, g in patient_label.groupby(patient_label):   # 按标签值分组 → 顺序 0,1,2
        groups.append(g.sample(frac=1, random_state=args.seed + int(value)))
    patients = pd.concat(groups).index.astype(str).tolist() # [组0打乱, 组1打乱, 组2打乱]
    n_cal   = round(len(patients)*0.10)
    n_route = round(len(patients)*0.10)
    n_sel   = round(len(patients)*0.10)
    cal     = set(patients[:n_cal])                         # ← 全局前缀切片
    route   = set(patients[n_cal:n_cal+n_route])
    selection = set(patients[n_cal+n_route : ...])

对照 src/make_patient_split.py 第 41-44 行——它从**每一个标签组内部**各取
val_fraction，是正确分层：

    for label_value, group in patient_labels.groupby(patient_labels):
        shuffled = group.sample(frac=1.0, random_state=...)
        val_count = max(1, round(len(shuffled) * args.val_fraction))
        val_patients.update(shuffled.iloc[:val_count].index)

本脚本的目的：在合成数据上执行**完全相同**的两种逻辑，检验
make_agent_split 的三个子集是否全部取自"众数标签=0"的患者组。

作者：ClawsGO Science Agent
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 42
FRAC = (0.10, 0.10, 0.10)   # calibration / route / model_selection


def simulate_chexpert_patients(n_patients: int = 64534, seed: int = 0) -> pd.DataFrame:
    """
    构造与 CheXpert Cardiomegaly 行级分布相符的合成患者表。
    行级标签占比（来自 data/manifests/cardiomegaly_summary.json，映射 -1→2、NaN→0）：
        0 : 160,935 / 191,027 = 84.2%
        1 :  23,385 / 191,027 = 12.2%
        2 :   6,707 / 191,027 =  3.5%
    """
    rng = np.random.default_rng(seed)
    p_row = np.array([0.842, 0.122, 0.035])
    rows = []
    for pid in range(n_patients):
        n_img = max(1, rng.poisson(3.0))          # CheXpert 每患者约 2-4 张
        n_study = max(1, int(round(n_img / 2)))
        # 患者级倾向：多数患者以阴性为主，少数以阳性/不确定为主
        theta = rng.dirichlet(1.2 * p_row + 1e-9)
        for _ in range(n_study):
            lab = int(rng.choice([0, 1, 2], p=theta))
            for _ in range(max(1, int(round(n_img / n_study)))):
                rows.append({"Patient": f"P{pid:06d}", "Cardiomegaly": lab})
    return pd.DataFrame(rows)


def modal_label(df: pd.DataFrame) -> pd.Series:
    """完全复刻 make_agent_split.py 第 10 行的众数标签计算。"""
    return df.groupby("Patient")["Cardiomegaly"].agg(
        lambda x: int(x.value_counts().index[0]))


def split_make_agent_split(train: pd.DataFrame, seed: int = SEED):
    """完全复刻 make_agent_split.py 的划分（有 bug 的版本）。"""
    patient_label = modal_label(train)
    groups = []
    for value, g in patient_label.groupby(patient_label):
        groups.append(g.sample(frac=1, random_state=seed + int(value)))
    patients = pd.concat(groups).index.astype(str).tolist()
    n = len(patients)
    n_cal = max(1, round(n * FRAC[0]))
    n_route = max(1, round(n * FRAC[1]))
    n_sel = max(1, round(n * FRAC[2]))
    cal = set(patients[:n_cal])
    route = set(patients[n_cal:n_cal + n_route])
    selection = set(patients[n_cal + n_route:n_cal + n_route + n_sel])
    return cal, route, selection, patient_label


def split_make_patient_split(train: pd.DataFrame, seed: int = SEED):
    """复刻 make_patient_split.py 的划分（正确分层版本），取其 10% 作为对照。"""
    patient_labels = train.groupby("Patient")["Cardiomegaly"].max()
    val_patients = set()
    for _, group in patient_labels.groupby(patient_labels):
        shuffled = group.sample(frac=1.0, random_state=seed)
        val_count = max(1, round(len(shuffled) * FRAC[0]))
        val_patients.update(shuffled.iloc[:val_count].index.astype(str))
    return val_patients, patient_labels


def describe(name, patient_ids, train, patient_label):
    sub = train[train["Patient"].isin(patient_ids)]
    if len(sub) == 0:
        return {"split": name, "patients": 0}
    known = sub[sub["Cardiomegaly"] != 2]
    return {
        "split": name,
        "patients": len(patient_ids),
        "rows": len(sub),
        "阴性%": 100 * (sub["Cardiomegaly"] == 0).mean(),
        "阳性%": 100 * (sub["Cardiomegaly"] == 1).mean(),
        "不确定%": 100 * (sub["Cardiomegaly"] == 2).mean(),
        "已知标签患病率%": 100 * (known["Cardiomegaly"] == 1).mean() if len(known) else np.nan,
        "患者众数标签分布": {
            int(k): int(v) for k, v in
            patient_label.loc[list(patient_ids)].value_counts().sort_index().items()},
    }


def main():
    print("=" * 78)
    print("合成 CheXpert 患者表（分布式复刻真实 manifest 的行级标签占比）")
    train = simulate_chexpert_patients()
    pl = modal_label(train)
    print(f"患者数 {train['Patient'].nunique()}，行数 {len(train)}")
    print("患者众数标签分布：",
          {int(k): int(v) for k, v in pl.value_counts().sort_index().items()})
    print(f"行级标签占比：阴性 {(train.Cardiomegaly==0).mean():.3f}  "
          f"阳性 {(train.Cardiomegaly==1).mean():.3f}  "
          f"不确定 {(train.Cardiomegaly==2).mean():.3f}")

    print("\n" + "=" * 78)
    print("【A】make_agent_split.py 的逻辑（全局前缀切片）")
    cal, route, sel, pl = split_make_agent_split(train)
    rows_a = [describe(n, s, train, pl) for n, s in
              [("calibration", cal), ("route_validation", route),
               ("model_selection", sel)]]
    train_patients = set(train["Patient"]) - cal - route - sel
    rows_a.append(describe("model_train", train_patients, train, pl))
    print(pd.DataFrame(rows_a).to_string(index=False))

    print("\n" + "=" * 78)
    print("【B】make_patient_split.py 的逻辑（组内分层）作为对照")
    val_patients, pl2 = split_make_patient_split(train)
    rows_b = [describe("对照-分层取10%", val_patients, train, pl2)]
    print(pd.DataFrame(rows_b).to_string(index=False))

    print("\n" + "=" * 78)
    print("结论：若【A】中 calibration/route_validation/model_selection 的"
          "\n      '患者众数标签分布' 只有 0、且与 model_train 明显不同 → bug 确认。")


if __name__ == "__main__":
    main()
