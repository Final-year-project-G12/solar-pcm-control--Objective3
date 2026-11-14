"""Two plots for the final-review deck (Rajasthan, regime 1). Reads only results/ files; no model is run.

  o3_rj_tradeoff.png : delivered energy vs shield hours, validation years (left) and the locked test (right)
  o3_rj_frontier.png : DQN, % of B0 delivery and shield-hour cut vs shield weight (3 seeds)
Style follows tamilnadu/scripts/make_plots.py plot 2 (delivery vs shield trade-off).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "plots" / "deck"
OUT.mkdir(parents=True, exist_ok=True)
MAROON = "#8B1A3A"

dqn = pd.read_csv(R / "gen2" / "gen2_summary_dqn.csv")
sac = pd.read_csv(R / "gen2" / "gen2_summary_sac.csv")
dep = json.load(open(R / "deployment_metrics_r1.json"))["policies"]
test = json.load(open(R / "phase6_summary_dqn.json"))

d1 = dqn[(dqn.regime_id == 1) & (dqn.control_step_min == 60) & (dqn.shield_weight == 1.7)
         & dqn.tag.str.contains("V3_60min_dr")]
s1 = sac[(sac.regime_id == 1) & (sac.shield_weight == 1.7)]
b0 = d1.iloc[0]

# ---------- plot 1: trade-off ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
a = ax[0]
a.scatter(dep["TWIN_water_only"]["shield_h_yr"], dep["TWIN_water_only"]["E_del45_kWh_yr"], c="grey", s=60, marker="s", label="Water-only twin")
a.scatter(b0.B0_shield_h, b0.B0_E45_kWh, c="black", s=70, marker="D", label="B0 always charge (= PPO, 3 seeds)", zorder=3)
a.scatter(b0.B1_shield_h, b0.B1_E45_kWh, c="tab:orange", s=60, marker="^", label="B1 reactive rule")
a.scatter(d1.shield_h, d1.E45_kWh, c=MAROON, s=55, label="DQN (3 seeds)")
a.scatter(s1.shield_h, s1.E45_kWh, c="tab:blue", s=55, marker="v", label="SAC (3 seeds)")
sel = d1[d1.seed == 42].iloc[0]
a.annotate("selected DQN", (sel.shield_h, sel.E45_kWh), xytext=(sel.shield_h + 60, sel.E45_kWh + 18), fontsize=8, color=MAROON,
           arrowprops=dict(arrowstyle="-", color=MAROON))
a.set_title("Validation years 2021-2023", fontsize=10)
a.set_xlabel("Shield hours per year (fewer is better)", fontsize=9)
a.set_ylabel("Heat delivered to 45 °C (kWh/yr)", fontsize=9)
a.grid(alpha=0.3)

b = ax[1]
P = {"Water-only twin": ("TWIN_water_only", "grey", "s"), "B0 always charge": ("B0_fixed_charge", "black", "D"),
     "B1 reactive rule": ("B1_reactive_rule", "tab:orange", "^"), "DQN (selected)": ("DQN", MAROON, "o")}
for lab, (k, c, m) in P.items():
    b.scatter(test[k]["shield_h_mean"], test[k]["delivered_kWh_mean"], c=c, s=70, marker=m, label=lab, zorder=3)
b.set_title("Locked test, fresh years 2006-2010", fontsize=10)
b.set_xlabel("Shield hours per year (fewer is better)", fontsize=9)
b.grid(alpha=0.3)
fig.suptitle("Rajasthan regime 1 (Nagaur): top-left is better", fontsize=10)
h, l = a.get_legend_handles_labels()
h2, l2 = b.get_legend_handles_labels()
for hh, ll in zip(h2, l2):
    if ll == "DQN (selected)":
        pass
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "o3_rj_tradeoff.png", dpi=200)
plt.close(fig)

# ---------- plot 2: frontier over shield weight ----------
f = dqn[(dqn.regime_id == 1) & (dqn.control_step_min == 60) & dqn.tag.str.contains("V3_60min_dr")]
g = f.groupby("shield_weight").agg(pct=("pct_of_B0_delivery", "mean"), cut=("shield_cut_vs_B0_pct", "mean"),
                                   lo=("pct_of_B0_delivery", "min"), hi=("pct_of_B0_delivery", "max"))
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for w, r in f.groupby("shield_weight"):
    ax.scatter([w] * len(r), r.pct_of_B0_delivery, c=MAROON, alpha=0.35, s=22)
ax.plot(g.index, g.pct, c=MAROON, marker="o", label="Delivery, % of B0 (mean of 3 seeds)")
ax.axhline(95, c="grey", ls="--", lw=1)
ax.text(0.05, 95.6, "95 % annual-delivery rule", ha="left", fontsize=7, color="grey")
ax.set_xlabel("Shield weight in the reward", fontsize=9)
ax.set_ylabel("Delivery, % of B0", fontsize=9, color=MAROON)
ax2 = ax.twinx()
ax2.plot(g.index, g.cut, c="tab:blue", marker="s", label="Shield-hour cut vs B0 (%)")
ax2.set_ylabel("Shield-hour cut vs B0 (%)", fontsize=9, color="tab:blue")
ax.set_title("DQN, regime 1: more penalty cuts shield time but costs delivery", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "o3_rj_frontier.png", dpi=200)
print(g.round(1))
