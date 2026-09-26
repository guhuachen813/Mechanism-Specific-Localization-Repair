r"""P15 配图：风险-覆盖曲线（Cardi/Atel 双面板，英文标注）。

输入 results/p6/p15_risk_coverage_{cardi,atel}.csv（gpu17 p15_stats.py 产出）。
每面板：wrong-repair 率 vs 覆盖率的门控扫描曲线 + 恒选点（cov=1）+ q50 操作点。
输出 results/p6/risk_coverage.png（300dpi）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OK = "#0072B2"      # Okabe-Ito blue
ACC = "#D55E00"     # vermillion

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, tag, name, const_wrong in [
        (axes[0], "cardi", "Cardiomegaly (n=65)", None),
        (axes[1], "atel", "Atelectasis (n=73)", None)]:
    df = pd.read_csv(f"/home/user/ClawsGO/project-20260824-a/results/p6/"
                     f"p15_risk_coverage_{tag}.csv").sort_values("q")
    ax.plot(df["cov"], df.wrong * 100, "-o", ms=3, lw=1.2, color=OK,
            label="agree-gated sweep")
    ax.axhline(5, color="0.4", lw=0.8, ls="--")
    ax.text(0.02, 5.4, r"$\alpha$ = 5%", fontsize=7, color="0.3")
    # 恒选点 = q=0
    c0 = df.iloc[0]
    ax.plot([c0["cov"]], [c0.wrong * 100], "s", ms=6, color=ACC,
            label="always-repair (cov=1)")
    # q50 操作点
    r50 = df[df.q == 0.5].iloc[0]
    ax.plot([r50["cov"]], [r50.wrong * 100], "^", ms=7, color="#009E73",
            label="gate at q50 (operating point)")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Coverage", fontsize=8)
    ax.set_xlim(-0.03, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("Wrong-repair rate (%)", fontsize=8)
axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("/home/user/ClawsGO/project-20260824-a/results/p6/risk_coverage.png",
            dpi=300)
print("saved risk_coverage.png")
