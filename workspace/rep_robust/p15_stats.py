r"""P15（优先级2）：统计证据包。

输入（全部已有 CSV/npz）：
  p20_perimage.csv        tag×{raw45,full45,noident45,agree_p_noid,shift_noid} 逐图
  p18_new_findings_{cardi,atel}.csv  rectnet/pairloc/teach 逐图 + agree_p_rect
  p62_photo_sage.csv      Cardi gated_v2_45 逐图
  p65_atel_test.csv       Atel gated_v2_45（键能对上 valid 73 时才用）

主系统（最终形态）：
  Cardi = no_ident PAIR-Loc（恒选 = 强基线；agree 门 = 风险控制版）
  Atel  = rectnet（同上）

输出：
  p15_stats_main.csv      每行一个对比：均值/患者级 CI/p/Holm/wrong CP 上界/覆盖
  p15_risk_coverage_{cardi,atel}.csv  风险-覆盖扫描
预注册口径（写入报告）：主终点 = Δ IoU45 vs raw 患者级 CI>0（双病种）；
次终点 = wrong-repair CP 95% 上界、coverage、Δ vs gated_v2、clean 代价。

用法：/root/miniconda3/bin/python p15_stats.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "/root/autodl-tmp/experiments/rep_robust/chexlocalize"
SEED = 42
WRONG_THR = -0.05


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def boot_ci(d, key, n=400, seed=SEED):
    up = pd.unique(key)
    fmap = pd.Series(key).map({f: i for i, f in enumerate(up)}).to_numpy()
    r = np.random.default_rng(seed)
    vals = np.array([d[np.isin(fmap, r.choice(len(up), len(up)))].mean() for _ in range(n)])
    return d.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5), vals


def signflip_p(d, key, n=20000, seed=SEED):
    """患者级配对符号翻转检验（双侧）。d = 逐图差。"""
    df = pd.DataFrame({"d": d, "p": key}).groupby("p")["d"].mean().to_numpy()
    obs = abs(df.mean())
    if obs == 0:
        return 1.0
    r = np.random.default_rng(seed)
    s = r.choice([-1.0, 1.0], size=(n, len(df)))
    null = np.abs((s * df[None, :]).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def cp_upper(k, n, alpha=0.05):
    """Clopper-Pearson 精确二项 95% 置信上界。"""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def wilson(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(c - h, 0.0), min(c + h, 1.0)


def holm(pvals):
    """Holm-Bonferroni：返回校正后 p。"""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    cur = 0.0
    for rank, i in enumerate(order):
        cur = max(cur, (m - rank) * pvals[i])
        adj[i] = min(cur, 1.0)
    return adj


def main() -> int:
    pi = pd.read_csv(f"{OUT}/p20_perimage.csv")
    rows = []

    def add(family, disease, comparison, d, key, cov_mask=None, cov_n=None):
        d = np.asarray(d, float)
        key = np.asarray(key)
        m = ~np.isnan(d)
        d, key = d[m], key[m]
        mean, lo, hi, _ = boot_ci(d, key)
        p = signflip_p(d, key)
        wrong = int((d < WRONG_THR).sum())
        rows.append({"family": family, "disease": disease, "comparison": comparison,
                     "n": len(d), "mean": mean, "lo": lo, "hi": hi, "p": p,
                     "wrong_k": wrong, "wrong_n": len(d),
                     "wrong_cp95ub": cp_upper(wrong, len(d))})

    # ---------- 组装各病种数据 ----------
    data = {}
    for tag in ["cardi", "atel"]:
        g = pi[pi.tag == tag].set_index("key")
        p18 = pd.read_csv(f"{OUT}/p18_new_findings_{tag}.csv").set_index("key")
        df = g.join(p18[["rectnet45", "pairloc45", "agree_p_rect", "agree_p_pl"]],
                    how="left")
        if tag == "cardi":
            df["sys45"] = df.noident45
            df["agree"] = df.agree_p_noid
            g2 = pd.read_csv(f"{OUT}/p62_photo_sage.csv").set_index("key")
            df["g2"] = g2.loc[df.index, "gated_v2_45"]
        else:
            df["sys45"] = df.rectnet45
            df["agree"] = df.agree_p_rect
            g2 = None
            try:
                t = pd.read_csv(f"{OUT}/p65_atel_test.csv")
                if "cond" in t.columns:
                    # 取与 valid 73 键重合最大的 cond
                    best = None
                    for c, gc in t.groupby("cond"):
                        ov = len(set(gc.key) & set(df.index))
                        if best is None or ov > best[0]:
                            best = (ov, c, gc)
                    if best and best[0] >= 40:
                        g2 = best[2].set_index("key")
                        df["g2"] = g2.loc[df.index.intersection(g2.index), "gated_v2_45"]
                        log(f"atel g2 来源 p65_atel_test cond={best[1]}（{best[0]} 键）")
            except FileNotFoundError:
                pass
        data[tag] = df

    # ---------- 主终点：Δ vs raw ----------
    for tag in ["cardi", "atel"]:
        df = data[tag]
        sysname = "pairloc_no_ident" if tag == "cardi" else "rectnet"
        add("primary", tag, f"{sysname}_vs_raw", df.sys45 - df.raw45, df.patient)
    pv = [r["p"] for r in rows if r["family"] == "primary"]
    adj = holm(pv)
    for r, a in zip([r for r in rows if r["family"] == "primary"], adj):
        r["p_holm"] = a

    # ---------- 次终点：Δ vs gated_v2、门控版本 ----------
    for tag in ["cardi", "atel"]:
        df = data[tag]
        sysname = "pairloc_no_ident" if tag == "cardi" else "rectnet"
        if "g2" in df.columns and df.g2.notna().sum() > 30:
            m = df.g2.notna() & df.sys45.notna()
            add("secondary", tag, f"{sysname}_vs_g2", df.sys45[m] - df.g2[m],
                df.patient[m])
            add("secondary", tag, "g2_vs_raw", df.g2[m] - df.raw45[m], df.patient[m])
        # 门控版本（agree≥q50，分位规则不看标签）
        a = df.agree
        m = a.notna() & df.sys45.notna()
        thr = np.nanquantile(a[m], 0.5)
        gm = m & (a >= thr)
        d_sys = (df.sys45 - df.raw45).to_numpy()
        cov = int(gm.sum())
        w = int((d_sys[gm] < WRONG_THR).sum())
        wil = wilson(cov, int(m.sum()))
        rows.append({"family": "secondary", "disease": tag,
                     "comparison": f"{sysname}_gated_agree_q50",
                     "n": cov, "mean": float(np.nanmean(d_sys[gm])),
                     "lo": np.nan, "hi": np.nan, "p": np.nan, "p_holm": np.nan,
                     "wrong_k": w, "wrong_n": cov, "wrong_cp95ub": cp_upper(w, cov),
                     "cov": cov, "cov_total": int(m.sum()),
                     "cov_wilson_lo": wil[0], "cov_wilson_hi": wil[1]})
        # g2 本身的 wrong（背景）
        if "g2" in df.columns and df.g2.notna().sum() > 30:
            m2 = df.g2.notna()
            d_g2 = (df.g2 - df.raw45).to_numpy()
            wg = int((d_g2[m2] < WRONG_THR).sum())
            rows.append({"family": "context", "disease": tag, "comparison": "g2_wrong",
                         "n": int(m2.sum()), "mean": float(np.nanmean(d_g2[m2])),
                         "lo": np.nan, "hi": np.nan, "p": np.nan, "p_holm": np.nan,
                         "wrong_k": wg, "wrong_n": int(m2.sum()),
                         "wrong_cp95ub": cp_upper(wg, int(m2.sum())),
                         "cov": int(m2.sum()), "cov_total": int(m2.sum()),
                         "cov_wilson_lo": 1.0, "cov_wilson_hi": 1.0})

    # ---------- 风险-覆盖扫描 ----------
    for tag in ["cardi", "atel"]:
        df = data[tag]
        a = df.agree.to_numpy()
        d_sys = (df.sys45 - df.raw45).to_numpy()
        m = ~np.isnan(a) & ~np.isnan(d_sys)
        a, d_sys = a[m], d_sys[m]
        rc = []
        for q in np.linspace(0.0, 0.9, 19):
            thr = np.quantile(a, q)
            sel = a >= thr
            k = int((d_sys[sel] < WRONG_THR).sum())
            rc.append({"q": q, "thr": thr, "cov": sel.mean(),
                       "wrong": k / max(sel.sum(), 1),
                       "wrong_cp95ub": cp_upper(k, int(sel.sum())),
                       "sys_delta": d_sys[sel].sum() / len(d_sys)})
        pd.DataFrame(rc).to_csv(f"{OUT}/p15_risk_coverage_{tag}.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/p15_stats_main.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== P15 统计证据包 ==")
    print(out.to_string(index=False))
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
