"""把各标签（seed × split × 设计）的结果表汇成一张对照表。"""

from __future__ import annotations

import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

TAGS = [t for t in sorted(os.listdir(RES))
        if t.startswith("table_conditions_") and t.endswith(".csv")]
TAGS = [t.replace("table_conditions_", "").replace(".csv", "") for t in TAGS]


def get(tag, kind):
    p = os.path.join(RES, f"table_{kind}_{tag}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    print("=" * 130)
    print("A. 逐标签总览")
    print("=" * 130)
    rows = []
    for tag in TAGS:
        c = get(tag, "conditions")
        if c is None:
            continue
        w = get(tag, "within_condition")
        i = get(tag, "incremental")
        op = get(tag, "operational")
        pooled = i[i["protocol"] == "按条件分折（跨降质泛化）"] if i is not None else None
        dq = pooled[pooled["model"] == "+quality"]["delta"].iloc[0] if pooled is not None and len(pooled) else float("nan")
        d_all = pooled[pooled["model"] == "+全部四路"]["delta"].iloc[0] if pooled is not None and len(pooled) else float("nan")
        rows.append(dict(
            tag=tag, n_pooled=int(c["n"].sum()),
            clean_auroc=c[c["cond"] == "clean"]["auroc"].iloc[0],
            worst_auroc=c["auroc"].min(),
            clean_err=c[c["cond"] == "clean"]["err"].iloc[0],
            worst_err=c["err"].max(),
            T_min=c["T_star_ece"].min(), T_max=c["T_star_ece"].max(),
            mean_corr_q=c["corr_q_err"].mean(),
            mean_auc_q=c["auc_err_quality"].mean(),
            within_delta=w["delta_vs_conf"].mean() if w is not None else float("nan"),
            pooled_dq=dq, pooled_d_all=d_all,
            op_d50=op[op["coverage"] == 0.5]["delta_cost"].iloc[0] if op is not None else float("nan"),
            op_ci50=("[%.4f,%.4f]" % (op[op["coverage"] == 0.5]["ci_lo"].iloc[0],
                                      op[op["coverage"] == 0.5]["ci_hi"].iloc[0])) if op is not None else "",
        ))
    d = pd.DataFrame(rows)
    print(d.round(4).to_string(index=False))

    print("\n" + "=" * 130)
    print("B. 信号对比（跨降质分折的池化增量 ΔAUC）")
    print("=" * 130)
    for tag in TAGS:
        i = get(tag, "incremental")
        if i is None:
            continue
        p = i[i["protocol"] == "按条件分折（跨降质泛化）"]
        if not len(p):
            continue
        print(f"\n-- {tag} --")
        for _, r in p.iterrows():
            star = " *" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "  "
            print(f"   {r['model']:<16} AUC {r['auc_conf']:.4f} → {r['auc_model']:.4f}  "
                  f"Δ={r['delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{star}")

    print("\n" + "=" * 130)
    print("C. 条件内 ΔAUC（学习式质量特征相对置信度），逐标签")
    print("=" * 130)
    for tag in TAGS:
        w = get(tag, "within_condition")
        if w is None:
            continue
        r = pd.Series(w["err"]).corr(pd.Series(w["delta_vs_conf"]))
        print(f"  {tag:<26} 平均 Δ={w['delta_vs_conf'].mean():+.4f}   "
              f"corr(错误率, Δ)={r:+.3f}   "
              f"Δ>0 的条件数 {int((w['delta_vs_conf'] > 0).sum())}/{len(w)}")


if __name__ == "__main__":
    main()
