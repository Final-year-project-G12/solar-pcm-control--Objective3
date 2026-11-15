"""
Objective 3 (Tamil Nadu) - one interactive page with the key result plots.

Run from the objective3 root:
    python scripts/make_interactive_o3.py
Output:
    results/plots/O3_interactive_tamilnadu.html   (open in any browser; needs internet once for plotly.js)

Data sources (all post weather-year-fix, 2026-11-13):
    results/phase4_comparison_r{0,1,2}.csv      validation 2021-2023, V1 models (no forecast)
    results/phase9_sensor_faults_r0.csv         regime 0, year 2021
    results/phase5_robustness_r0_sac.csv        regime 0, validation years
    locked-test numbers (V1 models) are copied from README.md / docs/METHODOLOGY_AND_FINDINGS.md
"""
import pathlib

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).parents[1]
RES = ROOT / "results"
OUT = RES / "plots" / "O3_interactive_tamilnadu.html"

COL = {"B0 always charge": "#555555", "B1 reactive rules": "#2a7ab0", "B2 forecast rules": "#7fb3d5",
       "DQN": "#e08a1e", "SAC": "#A4123F"}
NAMES = {"B0_fixed_charge": "B0 always charge", "B1_reactive_rule": "B1 reactive rules",
         "B2_forecast_rule": "B2 forecast rules", "DQN": "DQN", "SAC": "SAC"}
LAY = dict(paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Segoe UI, Arial", size=13),
           margin=dict(l=60, r=30, t=60, b=60))


def load(regime):
    d = pd.read_csv(RES / f"phase4_comparison_r{regime}.csv")
    d["guard"] = d.total_shield_water + d.total_shield_pcm
    d["name"] = d.policy.map(NAMES)
    return d


D = {r: load(r) for r in (0, 1, 2)}

# ---- 1. trade-off scatter with a regime dropdown ----------------------------
f1 = go.Figure()
for r in (0, 1, 2):
    g = D[r].groupby("name").agg(kwh=("total_delivered_kWh", "mean"), guard=("guard", "mean"),
                                 viol=("n_safety_violations", "sum")).reset_index()
    for _, row in g.iterrows():
        f1.add_trace(go.Scatter(
            x=[row.kwh], y=[row.guard], mode="markers+text", text=[row["name"]], textposition="top center",
            marker=dict(size=18 if row["name"] == "SAC" else 13, color=COL[row["name"]], line=dict(width=1, color="black")),
            name=row["name"], visible=(r == 0), showlegend=False,
            hovertemplate=f"<b>{row['name']}</b><br>%{{x:,.0f}} kWh/yr<br>%{{y:,.0f}} guard hours/yr<br>"
                          f"safety violations: {int(row.viol)}<extra>regime {r}</extra>"))
n = [len(D[r].groupby("name")) for r in (0, 1, 2)]
f1.update_layout(**LAY, title="1. Heat delivered vs guard hours (validation 2021-23). Bottom-right is better",
                 xaxis_title="Hot water delivered (kWh/yr)", yaxis_title="Guard hours per year", height=480,
                 updatemenus=[dict(type="dropdown", x=0.0, y=1.18, buttons=[
                     dict(label=f"Regime {r}", method="update",
                          args=[{"visible": sum([[k == r] * n[k] for k in range(3)], [])}]) for r in range(3)])])

# ---- 2. per-year bars, metric + regime dropdown -----------------------------
f2 = go.Figure()
combos = []
for r in (0, 1, 2):
    for metric, lab in (("total_delivered_kWh", "Hot water delivered (kWh/yr)"), ("guard", "Guard hours per year")):
        combos.append((r, metric, lab))
for i, (r, metric, lab) in enumerate(combos):
    for p, c in COL.items():
        s = D[r][D[r].name == p]
        f2.add_trace(go.Bar(x=s.year.astype(str), y=s[metric], name=p, marker_color=c, visible=(i == 0),
                            legendgroup=p, showlegend=(i == 0 or True),
                            hovertemplate=f"{p}<br>%{{x}}: %{{y:,.0f}}<extra></extra>"))
per = len(COL)
btn = []
for i, (r, metric, lab) in enumerate(combos):
    vis = [False] * (len(combos) * per)
    for k in range(per):
        vis[i * per + k] = True
    btn.append(dict(label=f"Regime {r}: {lab}", method="update", args=[{"visible": vis}, {"yaxis.title": lab}]))
f2.update_layout(**LAY, barmode="group", title="2. Every validation year, every controller", height=480,
                 xaxis_title="Year", yaxis_title=combos[0][2], updatemenus=[dict(type="dropdown", x=0.0, y=1.18, buttons=btn)])

# ---- 3. regimes, locked test 2011-15 (V1 models; README / METHODOLOGY) ------
reg = ["Regime 0", "Regime 1", "Regime 2"]
b0 = [2096.9, 2094.8, 2068.8]
dqn = [2018.0, 2094.7, 2068.8]
sac = [2044.6, 2067.5, 2012.6]
b0g = [46969, 24815, 34796]
dqng = [17838, 24298, 34796]
sacg = [21406, 15569, 23138]
f3 = make_subplots(rows=1, cols=2, subplot_titles=("Heat delivered (kWh/yr)", "Guard hours per year"))
for nm, kv, gv, c in (("B0 always charge", b0, b0g, COL["B0 always charge"]), ("DQN", dqn, dqng, COL["DQN"]),
                      ("SAC", sac, sacg, COL["SAC"])):
    f3.add_bar(x=reg, y=kv, name=nm, marker_color=c, legendgroup=nm, row=1, col=1,
               text=[f"{v:,.0f}" for v in kv], textposition="outside")
    f3.add_bar(x=reg, y=gv, name=nm, marker_color=c, legendgroup=nm, showlegend=False, row=1, col=2,
               text=[f"{v:,.0f}" for v in gv], textposition="outside")
f3.update_yaxes(range=[1900, 2150], row=1, col=1)
f3.update_layout(**LAY, barmode="group", height=470,
                 title="3. Locked test 2011-15: DQN only helps in regime 0, SAC helps in all three")

# ---- 4. sensor faults (regime 0, 2021) --------------------------------------
sf = pd.read_csv(RES / "phase9_sensor_faults_r0.csv")
lab = {"nominal": "no fault", "ghi_dropout": "sun sensor dead", "ghi_stuck": "sun sensor stuck",
       "tw_bias_+2C": "water +2 C", "tw_noise": "water noise", "tpcm_dropout": "PCM sensor dead",
       "combined_fault": "all combined"}
sf = sf[sf.fault.isin(lab)]
f4 = make_subplots(rows=1, cols=2, subplot_titles=("Heat delivered (kWh)", "Guard activations"))
for a, c in (("DQN", COL["DQN"]), ("SAC", COL["SAC"])):
    s = sf[sf.algo == a].set_index("fault").loc[list(lab)]
    f4.add_bar(x=list(lab.values()), y=s.delivered_kWh, name=a, marker_color=c, legendgroup=a, row=1, col=1)
    f4.add_bar(x=list(lab.values()), y=s.shields, name=a, marker_color=c, legendgroup=a, showlegend=False, row=1, col=2)
f4.update_yaxes(range=[1600, 2100], row=1, col=1)
f4.update_xaxes(tickangle=-30)
f4.update_layout(**LAY, barmode="group", height=480,
                 title="4. Sensor faults, regime 0, year 2021 (zero safety violations everywhere)")

# ---- 5. perturbation scenarios for SAC --------------------------------------
rb = pd.read_csv(RES / "phase5_robustness_r0_sac.csv")
rb["guard"] = rb.total_shield_water + rb.total_shield_pcm
g = rb.groupby("scenario").agg(kwh=("total_delivered_kWh", "mean"), guard=("guard", "mean"),
                               viol=("n_safety_violations", "sum")).reset_index()
f5 = make_subplots(specs=[[{"secondary_y": True}]])
f5.add_bar(x=g.scenario, y=g.kwh, name="Heat delivered (kWh)", marker_color="#e9a3b3", secondary_y=False)
f5.add_trace(go.Scatter(x=g.scenario, y=g.guard, name="Guard hours", mode="lines+markers",
                        line=dict(color="#A4123F")), secondary_y=True)
f5.update_yaxes(range=[1800, 2100], title_text="kWh/yr", secondary_y=False)
f5.update_yaxes(title_text="guard hours/yr", secondary_y=True)
f5.update_xaxes(tickangle=-30)
f5.update_layout(**LAY, height=470,
                 title=f"5. SAC robustness to weather noise, temperature offsets and PCM start state "
                       f"(violations: {int(g.viol.sum())})")

# ---- 6. forecast conditions --------------------------------------------------
conds = ["no forecast", "zeroed forecast", "realistic 1-3 h forecast"]
dat = {"SAC regime 0": [54.0, 49.8, 51.8], "SAC regime 1": [37.0, 29.0, 26.2], "SAC regime 2": [33.5, 41.0, 35.9],
       "DQN regime 0": [62.0, 0.0, 18.3]}
colr = ["#A4123F", "#d6657f", "#e9a3b3", "#e08a1e"]
f6 = go.Figure()
for (k, v), c in zip(dat.items(), colr):
    f6.add_bar(x=conds, y=v, name=k, marker_color=c, text=[f"{x:.0f}%" for x in v], textposition="outside")
f6.update_layout(**LAY, barmode="group", height=450, yaxis_title="Guard-hour reduction vs always-charge (%)",
                 title="6. Does the controller need a forecast? SAC does not, DQN does (test 2011-15)")

figs = [f1, f2, f3, f4, f5, f6]
notes = [
    "Each dot is a controller averaged over 2021-2023. Use the dropdown to switch regime. Hover for safety violations.",
    "Pick regime and metric in the dropdown. Click legend entries to hide controllers.",
    "Locked test years, never seen in training. Numbers from README.md and docs/METHODOLOGY_AND_FINDINGS.md (no-forecast models).",
    "Dead sun sensor doubles SAC's guard activations (about 18k to 38k) but heat stays about the same, and no limit is broken.",
    "Ten perturbation scenarios over three validation years and three PCM start states.",
    "Guard-hour cut = fewer hours of safety-guard intervention than always charging.",
]
html = ['<!doctype html><html><head><meta charset="utf-8"><title>Objective 3 Tamil Nadu</title>',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f5f5f7;margin:0}"
        "header{background:#A4123F;color:#fff;padding:14px 22px}h1{margin:0;font-size:20px}"
        "header p{margin:4px 0 0;font-size:13px;opacity:.9}.card{background:#fff;border-radius:10px;margin:14px 22px;"
        "padding:10px;box-shadow:0 1px 3px #0002}.n{font-size:13px;color:#555;margin:0 8px 4px}</style></head><body>",
        "<header><h1>Objective 3 - Tamil Nadu: adaptive PCM controller</h1>"
        "<p>100 L tank, regimes 0-2. All numbers recomputed after the 2026-11-13 weather-year fix.</p></header>"]
for i, (f, nt) in enumerate(zip(figs, notes)):
    html.append(f'<div class="card"><p class="n">{nt}</p>')
    html.append(pio.to_html(f, include_plotlyjs="cdn" if i == 0 else False, full_html=False))
    html.append("</div>")
html.append("</body></html>")
OUT.write_text("\n".join(html), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
