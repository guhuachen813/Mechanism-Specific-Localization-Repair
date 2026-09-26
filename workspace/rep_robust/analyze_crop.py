"""第四阶段 1 分析：无学习基线的配对统计（vs raw，含与解剖先验的对比）。

输入：cam_crop/cam_crop_nih.csv（本脚本产出）+ cam_prior/cam_prior_nih.csv（第3步）
输出：cam_crop/cam_crop_paired.csv + 控制台表格
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
LADDER = ["clean", "jpeg_q50", "jpeg_q30", "jpeg_q10", "blur_1", "blur_3",
          "dark_05", "bright_16", "contrast_04", "ds_2", "ds_4",
          "noise_003", "noise_006", "noise_010", "noise_025",
          "combo_mild", "combo_sev"]


def boot_ci(a, n=400):
    a = np.asarray(a, float)
    bs = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return np.quantile(bs, [0.025, 0.975])


def main() -> int:
    crop = pd.read_csv("cam_crop/cam_crop_nih.csv")
    prior = pd.read_csv("cam_prior/cam_prior_nih.csv")

    def fixed_tau(df, tau=0.45):
        idx = int(round((tau - 0.05) / 0.05))
        return df.iou_curve.str.split(";").apply(lambda v: float(v[idx]))

    crop = crop.assign(iou45=fixed_tau(crop))
    prior = prior.assign(iou45=fixed_tau(prior))

    variants = ["center_fixed", "center_field", "field_crop", "prior_real_hard"]
    sources = {
        **{v: crop[crop.variant == v] for v in variants[:3]},
        "prior_real_hard": prior[prior.variant == "prior_real_hard"],
    }
    raws = {"crop": crop[crop.variant == "raw"], "prior": prior[prior.variant == "raw"]}

    rows = []
    for cond in LADDER:
        for v, src in sources.items():
            raw = raws["prior" if v == "prior_real_hard" else "crop"]
            s = src[src.condition == cond].set_index("nih_file")
            r = raw[raw.condition == cond].set_index("nih_file")
            j = s.join(r, rsuffix="_r")
            for metric, mcol, rcol in [("best", "best_iou", "best_iou_r"),
                                       ("tau45", "iou45", "iou45_r")]:
                d = (j[mcol] - j[rcol]).values
                lo, hi = boot_ci(d)
                rows.append({"condition": cond, "variant": v, "metric": metric,
                             "abs": d.mean() + j[rcol].values.mean(),  # 变体均值
                             "delta": d.mean(), "lo": lo, "hi": hi,
                             "star": "*" if (lo > 0 or hi < 0) else ""})
    out = pd.DataFrame(rows)
    out.to_csv("cam_crop/cam_crop_paired.csv", index=False)

    for metric in ["best", "tau45"]:
        print(f"\n== ΔIoU vs raw（{metric} 口径，* = 95% CI 不含零）==")
        piv = out[out.metric == metric].pivot(
            index="variant", columns="condition", values="delta")
        stars = out[out.metric == metric].pivot(
            index="variant", columns="condition", values="star")
        show = ["clean", "contrast_04", "combo_sev", "bright_16",
                "noise_006", "noise_010", "noise_025", "combo_mild", "jpeg_q10"]
        df = piv[show].astype(float).round(3).astype(str) + stars[show]
        print(df.to_string())

    # 关键对比：center_field vs prior_real_hard（同 raw 源，跨表配对）
    print("\n== center_field − prior_real_hard（同图配对）==")
    for cond in ["clean", "contrast_04", "combo_sev"]:
        a = crop[(crop.variant == "center_field") & (crop.condition == cond)
                 ].set_index("nih_file")
        b = prior[(prior.variant == "prior_real_hard") & (prior.condition == cond)
                  ].set_index("nih_file")
        j = a.join(b, rsuffix="_p")
        for metric, c1, c2 in [("best", "best_iou", "best_iou_p"),
                               ("tau45", "iou45", "iou45_p")]:
            d = (j[c1] - j[c2]).values
            lo, hi = boot_ci(d)
            print(f"  {cond:12s} {metric:5s} Δ={d.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]{'*' if lo>0 or hi<0 else ''}")
    return 0


if __name__ == "__main__":
    main()
