"""make_split_bug_figure.py — 生成划分 bug 的对比图"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from rc_lib import setup_matplotlib, OKABE_ITO
from verify_split_bug import simulate_chexpert_patients, split_make_agent_split, modal_label
from make_agent_split_fixed import build, allocate_splits, MIN_GROUP_FOR_ALLOCATION
from pathlib import Path as P

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

sim = simulate_chexpert_patients()
sim["Frontal/Lateral"] = "Frontal"; sim["split"] = "train"
sim["Path"] = [f"train/{p_}/{i}.jpg" for i, p_ in enumerate(sim["Patient"])]
tmp = P("/tmp/_fig_manifest.csv"); sim.to_csv(tmp, index=False)

pl = modal_label(sim)
cal, route, sel, _ = split_make_agent_split(sim)
buggy = {}
for name, ids in [("calibration", cal), ("route_validation", route), ("model_selection", sel)]:
    sub = sim[sim["Patient"].isin(ids)]
    known = sub[sub["Cardiomegaly"] != 2]
    buggy[name] = 100 * (known["Cardiomegaly"] == 1).mean()
train_ids = set(sim["Patient"]) - cal - route - sel
sub = sim[sim["Patient"].isin(train_ids)]; known = sub[sub["Cardiomegaly"] != 2]
buggy["model_train"] = 100 * (known["Cardiomegaly"] == 1).mean()

out, _, plf = build(tmp, {"calibration": .1, "route_validation": .1, "model_selection": .1}, 42)
fixed = {}
for s, g in out.groupby("agent_split"):
    known = g[g["Cardiomegaly"] != 2]
    fixed[s] = 100 * (known["Cardiomegaly"] == 1).mean()
true_prev = 100 * (sim.loc[sim.Cardiomegaly != 2, "Cardiomegaly"] == 1).mean()

order = ["calibration", "route_validation", "model_selection", "model_train"]
plt = setup_matplotlib()
fig, ax = plt.subplots(figsize=(8.2, 4.4))
x = np.arange(len(order))
ax.bar(x - 0.2, [buggy.get(s, np.nan) for s in order], 0.4,
       label="原版（前缀切片）", color=OKABE_ITO["vermillion"])
ax.bar(x + 0.2, [fixed.get(s, np.nan) for s in order], 0.4,
       label="修正版（组内分层）", color=OKABE_ITO["blue"])
ax.axhline(true_prev, color=OKABE_ITO["black"], ls="--", lw=1.2,
           label=f"真实患病率 {true_prev:.1f}%")
for i, s in enumerate(order):
    b = buggy.get(s, np.nan); f = fixed.get(s, np.nan)
    if not np.isnan(b):
        ax.text(i - 0.2, b + 0.4, f"{b:.1f}", ha="center", fontsize=8.5)
    if not np.isnan(f):
        ax.text(i + 0.2, f + 0.4, f"{f:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
ax.set_ylabel("已知标签样本的患病率 (%)")
ax.set_title("患者级划分的类别构成：原版 vs 修正版", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.25, lw=0.6)
fig.savefig(os.path.join(OUT, "fig_split_bug.png"))
print("saved fig_split_bug.png")
print("buggy:", {k: round(v, 2) for k, v in buggy.items()})
print("fixed:", {k: round(v, 2) for k, v in fixed.items()})
print("true :", round(true_prev, 2))
