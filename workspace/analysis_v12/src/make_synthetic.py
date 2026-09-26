"""
make_synthetic.py — 合成预测表生成器（带已知真值结构）

用途
----
本机没有 CheXpert 数据也没有 GPU，无法跑真实推理。这个生成器造出一张
结构与真实导出表完全一致的预测表，并**人为注入已知的结构**：

  1. 质量 → 错误 的真实依赖（H1 应能检出）
  2. route_validation → official_valid 的校准漂移 + 先验漂移（exp4 应能检出）
  3. 患者内多图相关性（聚类 Bootstrap 应给出比朴素 Bootstrap 更宽的区间）
  4. 已知的最优分层策略（exp3 应能找回）

这样就能在真实数据到手之前，验证整条分析管线是**能工作的**——
而不是等真数据来了才发现脚本有 bug。

⚠️ 合成数据上的任何数值结果都没有科学意义，只能用于管线自检。

作者：ClawsGO Science Agent
"""

from __future__ import annotations
import numpy as np
import pandas as pd

QC_FEATURES = [
    "qc_foreground_ratio", "qc_dynamic_range", "qc_mean_intensity",
    "qc_intensity_std", "qc_contrast", "qc_blur_score", "qc_noise_score",
    "qc_black_ratio", "qc_white_ratio", "qc_border_crop_score",
    "qc_left_right_symmetry", "qc_center_offset",
]


def generate(
    n_patients_route: int = 6453,
    n_patients_official: int = 200,
    images_per_patient: float = 3.0,
    prevalence_route: float = 0.10,
    prevalence_official: float = 0.35,       # 官方集先验明显更高
    quality_error_effect: float = 0.9,       # 质量问题对错误的真实效应（logit 尺度）
    official_shift: float = 0.75,            # official 上的分布偏移强度
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def build(n_pat, prev, split, shift=0.0, extra_hardness=0.0):
        rows = []
        for pid in range(n_pat):
            n_img = max(1, rng.poisson(images_per_patient))
            # 患者级随机效应：同一患者的多张图相关
            pat_effect = rng.normal(0, 0.6)
            for _ in range(n_img):
                # ---- 质量风险（Beta 分布，偏斜向右）----
                qr = float(np.clip(rng.beta(2.0, 5.0) + shift * rng.beta(3, 3) * 0.5, 0, 1))
                # ---- 标签 ----
                y = int(rng.random() < prev)
                # ---- 真实 logit：难度由 患者效应 + 质量 + 病例固有难度 决定 ----
                hardness = pat_effect + extra_hardness + shift * 0.4
                logit = (
                    2.2 * (1 if y == 1 else -1)                 # 信号
                    - 2.6 * hardness                            # 难度
                    - quality_error_effect * 2.5 * qr           # 质检效应（真实存在）
                    + shift * 1.4                               # 分布偏移 → 系统性过自信
                )
                p_true = 1 / (1 + np.exp(-logit))
                # 温度 >1 表示欠自信，<1 表示过自信；official 上更过自信
                T = 0.85 if shift == 0 else 0.55
                logit_cal = logit / T
                p1 = float(1 / (1 + np.exp(-logit_cal)))
                # Model 2：与 M1 相关但误差去相关（rho 控制互补性）
                logit2 = logit * 0.92 + rng.normal(0, 0.9)
                p2 = float(1 / (1 + np.exp(-logit2 / T)))
                # p_uncertain：与质量弱相关、与错误弱相关（复现 v1 的负结果）
                p_unc = float(np.clip(
                    0.18 + 0.10 * qr + rng.normal(0, 0.06), 0.01, 0.9))
                # 软质控特征：由 qr 生成，带噪声
                feats = {}
                for c in QC_FEATURES:
                    feats[c] = float(np.clip(rng.normal(qr, 0.18), 0, 1))
                rows.append({
                    "patient_id": f"{split}_P{pid:06d}",
                    "split": split,
                    "label": y,
                    "p1_pos": np.clip(p1, 1e-4, 1 - 1e-4),
                    "p2_pos": np.clip(p2, 1e-4, 1 - 1e-4),
                    "p1_unc": p_unc,
                    "p1_neg": float(np.clip(1 - p1 - p_unc, 1e-4, 1 - 1e-4)),
                    "quality_risk": qr,
                    **feats,
                })
        return rows

    route = build(n_patients_route, prevalence_route, "route_validation",
                  shift=0.0, extra_hardness=0.0)
    official = build(n_patients_official, prevalence_official, "official_valid",
                     shift=official_shift, extra_hardness=official_shift * 0.5)
    df = pd.DataFrame(route + official)
    return df


GROUND_TRUTH = {
    "quality_error_effect": "质量风险对错误率有真实的正向效应（H1 为真）",
    "official_shift": "official_valid 上校准漂移 + 先验漂移（exp4 应检出 reliability↑、uncertainty↑）",
    "patient_clustering": "患者内多图相关（聚类 Bootstrap 应宽于朴素 Bootstrap）",
    "p_unc_weak": "p_uncertain 与错误只有弱关系（应复现 v1 的负结果）",
}


if __name__ == "__main__":
    df = generate()
    out = "out/synthetic_predictions.csv"
    df.to_csv(out, index=False)
    print(f"合成预测表已生成：{out}")
    print(f"  总行数 {len(df)}，患者数 {df['patient_id'].nunique()}")
    for s, g in df.groupby("split"):
        err = ((g["p1_pos"] >= 0.5).astype(int) != g["label"]).mean()
        print(f"  {s:18s} n={len(g):6d}  患病率={g['label'].mean():.3f}  "
              f"M1错误率={err:.3f}  平均质量风险={g['quality_risk'].mean():.3f}")
