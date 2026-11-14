"""
scripts/make_plots.py -- self-contained results dashboard (Tamil Nadu section 16.6 equivalent), Uttarakhand.
Writes results/plots/dashboard.html (Plotly inlined: opens offline). Every chart has a hover layer, a legend, direct
labels on the key marks and a table view; light and dark mode use the validated reference palette steps.

  1. Delivery vs shield-hours trade-off (validation 2021-23), one panel per PCM regime: B0, B1, B2, every PPO seed,
     water-only twin, with the G2 random policies and the B1 / B2 threshold grids as grey context
  2. Locked O3 test 2006-10 (M6): window delivery by controller and regime (PPO = mean of 3 seeds, +- 1 SD)
  3. PPO learning curves: validation window delivery vs training steps, chosen weight set, 3 seeds, B0 reference
  4. Locked test per year (M6), one panel per regime
  5. Sensor-fault robustness (validation): change in annual delivery vs nominal, by fault scenario
  6. Regime map: the 45 O1 points by regime, medoids marked (O1 cluster assignment, hash-checked)
Usage: python scripts/make_plots.py
"""
import hashlib
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3 import roles_o3                            # noqa: E402

R = o3c.RESULTS
OUT = R / "plots"
ASSIGN = Path(__file__).resolve().parents[1] / "data" / "o1" / "cluster_assignments_uttarakhand.csv"
ASSIGN_SHA = "7722f6bdf471e19aaf300572d243fd2b015b516ef06f3e3423a2f431de31b811"
# validated reference palette (dataviz references/palette.md), slots 1-5 in fixed order; light -> dark steps
LIGHT = {"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "PPO": "#eda100", "water_only_twin": "#e87ba4",
         "DQN": "#008300", "PPO_V2DR": "#6250d6", "SAC": "#e87ba4", "PPO_smart": "#6250d6"}
DARK = {"#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500", "#e87ba4": "#d55181",
        "#008300": "#008300", "#6250d6": "#9085e9"}
REGIME_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CONTEXT = "#b9b8b0"
LABEL = {"B0": "B0 always-on", "B1": "B1 rule", "B2": "B2 forecast rule", "PPO": "PPO", "water_only_twin": "Water-only twin",
         "DQN": "DQN", "PPO_V2DR": "PPO forecast obs + DR", "SAC": "SAC", "PPO_smart": "PPO + smart forecast",
         "PPO_frozen": "PPO frozen (M6)"}
PCM = [0, 1, 2]


def base_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=60, r=20, t=50, b=50), font=dict(family="system-ui, sans-serif", size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_size=12),
                      legend=dict(orientation="h", y=-0.18, x=0))
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False, linewidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, zeroline=False, linewidth=1)
    return fig


def table(df, caption):
    return (f"<details><summary>Table view: {html.escape(caption)}</summary>"
            + df.to_html(index=False, float_format=lambda v: f"{v:,.2f}", border=0, classes="tbl") + "</details>")


def ctrl_group(name):
    return name.split("_seed")[0] if name.startswith("PPO") else name


def chart_tradeoff():
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Regime {k}" for k in PCM], shared_yaxes=False,
                        horizontal_spacing=0.07)
    rows = []
    for i, k in enumerate(PCM, start=1):
        ctx = []
        for f, lab in ((f"g2_random_r{k}.csv", "G2 random policy"), (f"m3_b1_grid_r{k}.csv", "B1 grid point"),
                       (f"m9_b2_grid_r{k}.csv", "B2 grid point")):
            if (R / f).exists():
                d = pd.read_csv(R / f)
                ctx.append(d.assign(kind=lab)[["overheat_h", "E_del_window_kWh", "kind"]])
        c = pd.concat(ctx)
        fig.add_trace(go.Scatter(x=c.overheat_h, y=c.E_del_window_kWh, mode="markers", name="Rule / random search",
                                 marker=dict(size=7, color=CONTEXT, opacity=0.6), legendgroup="ctx", showlegend=i == 1,
                                 text=c.kind, hovertemplate="%{text}<br>shield %{x:.0f} h/yr<br>window %{y:.1f} kWh/yr"
                                                            "<extra></extra>"), 1, i)
        m5 = pd.read_csv(R / f"m5_robustness_r{k}.csv")
        m5 = m5[(m5.demand == "deterministic") & (m5.scenario == "fully_solid")]
        pts = m5.groupby("controller")[["overheat_h", "E_del_window_kWh", "g_window_kWh"]].mean().reset_index()
        tw = m5.groupby("year")[["twin_E_del_window_kWh"]].mean().mean()
        twin_sh = m5.groupby("year").twin_shield_water_h.mean().mean()
        pts = pd.concat([pts, pd.DataFrame([{"controller": "water_only_twin", "overheat_h": twin_sh,
                                             "E_del_window_kWh": tw.iloc[0], "g_window_kWh": 0.0}])])
        b2 = R / f"m9_b2_summary_r{k}.csv"
        if b2.exists():
            d = pd.read_csv(b2)
            d = d[d.controller == "B2_perfect"].groupby("controller")[["overheat_h", "E_del_window_kWh", "g_window_kWh"]].mean()
            pts = pd.concat([pts, d.reset_index().assign(controller="B2")])
        for _, r in pts.iterrows():
            g = ctrl_group(r.controller)
            lab = LABEL[g] + (f" seed {r.controller[-1]}" if g == "PPO" else "")
            fig.add_trace(go.Scatter(
                x=[r.overheat_h], y=[r.E_del_window_kWh], mode="markers+text" if g != "PPO" or r.controller.endswith("0") else "markers",
                name=LABEL[g], legendgroup=g, showlegend=(i == 1 and (g != "PPO" or r.controller.endswith("0"))),
                marker=dict(size=12, color=LIGHT[g], line=dict(width=2, color="#fcfcfb"),
                            symbol="diamond" if g == "PPO" else "circle"),
                text=[LABEL[g].split(" ")[0] if g != "water_only_twin" else "Twin"],
                textposition="bottom center" if g == "B1" else "top center",
                hovertemplate=f"{html.escape(lab)}<br>shield %{{x:.0f}} h/yr<br>window %{{y:.1f}} kWh/yr<extra></extra>"), 1, i)
            rows.append({"regime": k, "controller": lab, "shield_h_per_yr": r.overheat_h,
                         "window_delivery_kWh_per_yr": r.E_del_window_kWh})
        fig.update_xaxes(title_text="Shield-active hours per year", row=1, col=i)
    fig.update_yaxes(title_text="Window delivery to 50 C (kWh/yr)", row=1, col=1)
    base_layout(fig, 470)
    return fig, table(pd.DataFrame(rows), "trade-off points (validation 2021-23)")


def chart_test():
    d = pd.read_csv(R / "m6_test_results.csv")
    d["g"] = d.controller.map(ctrl_group)
    per = d.groupby(["regime", "g", "controller"]).E_del_window_kWh.mean().reset_index()
    agg = per.groupby(["regime", "g"]).E_del_window_kWh.agg(["mean", "std"]).reset_index()
    tw = d.groupby("regime").twin_E_del_window_kWh.mean().reset_index().rename(columns={"twin_E_del_window_kWh": "mean"})
    agg = pd.concat([agg, tw.assign(g="water_only_twin", std=float("nan"))])
    fig = go.Figure()
    for g in ["B0", "B1", "PPO", "water_only_twin"]:
        a = agg[agg.g == g]
        fig.add_trace(go.Bar(x=[f"Regime {k}" for k in a.regime], y=a["mean"], name=LABEL[g], marker_color=LIGHT[g],
                             marker_line=dict(width=2, color="#fcfcfb"),
                             error_y=dict(type="data", array=a["std"].fillna(0), visible=g == "PPO", thickness=1.5),
                             hovertemplate=f"{LABEL[g]}<br>%{{x}}: %{{y:.1f}} kWh/yr<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text="Window delivery to 50 C (kWh/yr)", rangemode="tozero")
    base_layout(fig, 420)
    t = agg.rename(columns={"g": "controller", "mean": "window_kWh_mean", "std": "PPO_seed_sd"})
    return fig, table(t.assign(controller=t.controller.map(LABEL)), "locked test 2006-10, mean of 5 years")


def chart_learning():
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Regime {k}" for k in PCM], horizontal_spacing=0.07)
    rows = []
    for i, k in enumerate(PCM, start=1):
        fz = frozen.load_frozen(k)
        b0 = pd.read_csv(R / f"m3_baselines_r{k}.csv")
        b0 = b0[(b0.controller == "B0") & (b0.purpose == "validate")].E_del_window_kWh.mean()
        for j, seed in enumerate(sorted(fz["policies"])):
            v = pd.read_csv(R / "ppo" / f"r{k}" / fz["weight_set"] / seed / "val_scores.csv")
            fig.add_trace(go.Scatter(x=v.steps, y=v.E_del_window_kWh, mode="lines+markers", name=f"PPO {seed}",
                                     legendgroup=seed, showlegend=i == 1,
                                     line=dict(width=2, color=LIGHT["PPO"], dash=["solid", "dash", "dot"][j]),
                                     marker=dict(size=8, color=LIGHT["PPO"]),
                                     hovertemplate=f"PPO {seed}<br>%{{x:,}} steps<br>%{{y:.1f}} kWh/yr<extra></extra>"), 1, i)
            rows += [{"regime": k, "seed": seed, "steps": s, "window_kWh": e} for s, e in zip(v.steps, v.E_del_window_kWh)]
        fig.add_hline(y=b0, line=dict(color=LIGHT["B0"], width=2), annotation_text="B0", row=1, col=i)
        fig.update_xaxes(title_text="Training steps", row=1, col=i)
    fig.update_yaxes(title_text="Validation window delivery (kWh/yr)", row=1, col=1)
    base_layout(fig, 420)
    return fig, table(pd.DataFrame(rows), "validation score at each checkpoint (chosen weight set)")


def chart_peryear():
    d = pd.read_csv(R / "m6_test_results.csv")
    d["g"] = d.controller.map(ctrl_group)
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"Regime {k}" for k in range(4)], horizontal_spacing=0.05)
    rows = []
    for i, k in enumerate(range(4), start=1):
        dk = d[d.regime == k]
        y = dk.groupby(["g", "year"]).E_del_window_kWh.mean().reset_index()
        tw = dk.groupby("year").twin_E_del_window_kWh.mean().reset_index().rename(columns={"twin_E_del_window_kWh": "E_del_window_kWh"})
        y = pd.concat([y, tw.assign(g="water_only_twin")])
        for g in ["B0", "B1", "PPO", "water_only_twin"]:
            s = y[y.g == g]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=s.year, y=s.E_del_window_kWh, mode="lines+markers", name=LABEL[g], legendgroup=g,
                                     showlegend=i == 1, line=dict(width=2, color=LIGHT[g]), marker=dict(size=8),
                                     hovertemplate=f"{LABEL[g]}<br>%{{x}}: %{{y:.1f}} kWh<extra></extra>"), 1, i)
            rows += [{"regime": k, "controller": LABEL[g], "year": int(a), "window_kWh": b} for a, b in zip(s.year, s.E_del_window_kWh)]
        fig.update_xaxes(dtick=1, row=1, col=i)
    fig.update_yaxes(title_text="Window delivery (kWh)", row=1, col=1)
    base_layout(fig, 400)
    return fig, table(pd.DataFrame(rows), "locked test per year")


def chart_faults():
    fs = [pd.read_csv(p) for p in sorted(R.glob("m10_sensor_faults_r*.csv"))]
    if not fs:
        return None, ""
    f = pd.concat(fs)
    f = f[f.scenario != "none"]
    regs = sorted(f.regime.unique())
    fig = make_subplots(rows=1, cols=len(regs), subplot_titles=[f"Regime {k}" for k in regs], shared_yaxes=True,
                        horizontal_spacing=0.04)
    rows = []
    for i, k in enumerate(regs, start=1):
        a = f[f.regime == k].groupby(["controller", "scenario"]).d_annual_vs_nominal_pct.mean().reset_index()
        for g in ["B1", "B2", "PPO"]:
            s = a[a.controller == g]
            if s.empty:
                continue
            fig.add_trace(go.Bar(y=s.scenario, x=s.d_annual_vs_nominal_pct, orientation="h", name=LABEL[g], legendgroup=g,
                                 showlegend=i == 1, marker_color=LIGHT[g], marker_line=dict(width=2, color="#fcfcfb"),
                                 hovertemplate=f"{LABEL[g]}<br>%{{y}}: %{{x:+.2f}} %<extra></extra>"), 1, i)
            rows += [{"regime": k, "controller": LABEL[g], "scenario": sc, "annual_delivery_change_pct": v}
                     for sc, v in zip(s.scenario, s.d_annual_vs_nominal_pct)]
        fig.update_xaxes(title_text="Annual delivery vs fault-free (%)", row=1, col=i)
    fig.update_layout(barmode="group")
    base_layout(fig, 440)
    return fig, table(pd.DataFrame(rows), "sensor-fault scenarios (validation 2021-23)")


def chart_map():
    if hashlib.sha256(ASSIGN.read_bytes()).hexdigest() != ASSIGN_SHA:
        raise RuntimeError("O1 cluster assignment file differs from the O1 manifest hash")
    a = pd.read_csv(ASSIGN)
    fig = go.Figure()
    for k in range(4):
        s = a[a.cluster_id == k]
        fig.add_trace(go.Scattergeo(lon=s.lon, lat=s.lat, mode="markers", name=f"Regime {k} ({len(s)} points)",
                                    marker=dict(size=9, color=REGIME_COL[k], line=dict(width=1, color="#fcfcfb")),
                                    text=s.point_id, hovertemplate="%{text}<br>%{lat:.3f} N, %{lon:.3f} E<extra></extra>"))
    med = a[a.point_id.isin(roles_o3.MEDOIDS.values())]
    fig.add_trace(go.Scattergeo(lon=med.lon, lat=med.lat, mode="markers+text", name="Medoid (simulated point)",
                                marker=dict(size=17, symbol="star", color="rgba(0,0,0,0)", line=dict(width=2, color="#52514a")),
                                text=[f"r{c} {p}" for c, p in zip(med.cluster_id, med.point_id)], textposition="top right",
                                hovertemplate="%{text}<extra></extra>"))
    fig.update_geos(fitbounds="locations", showcountries=True, showsubunits=True, showland=True, landcolor="#f1f0ec",
                    bgcolor="rgba(0,0,0,0)", resolution=50)
    base_layout(fig, 520)
    return fig, table(a[["point_id", "cluster_id", "lat", "lon"]].rename(columns={"cluster_id": "regime"}),
                      "O1 point-to-regime assignment")


def chart_extensions():
    p = R / "m12_extensions.csv"
    if not p.exists():
        return None, ""
    d = pd.read_csv(p)
    d = d[~d.diagnostic.astype(bool)]
    fz = {k: json.loads((R / "frozen" / f"r{k}" / "frozen.json").read_text()) for k in PCM}

    def pick(r):
        if r.study == "baseline":
            return r.algo
        if r.study == "m4" and r.w2 == fz[r.regime]["w2"] and r.w4 == fz[r.regime]["w4"]:
            return "PPO"
        if r.study == "dqn":
            return "DQN"
        if r.study == "v2dr":
            return "PPO_V2DR"
        return None
    d["g"] = d.apply(pick, axis=1)
    d = d[d.g.notna() & d.regime.isin(PCM)]
    order = ["B0", "B1", "B2", "PPO", "DQN", "PPO_V2DR"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Regime {k}" for k in PCM], horizontal_spacing=0.06)
    rows = []
    for i, k in enumerate(PCM, start=1):
        for g in order:
            s = d[(d.regime == k) & (d.g == g)]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=[LABEL[g]] * len(s), y=s.E_del_window_kWh, mode="markers", name=LABEL[g],
                                     legendgroup=g, showlegend=i == 1 or (g == "PPO_V2DR"),
                                     marker=dict(size=11, color=LIGHT[g], line=dict(width=2, color="#fcfcfb"),
                                                 symbol="diamond" if s.study.iloc[0] != "baseline" else "circle"),
                                     customdata=s[["seed", "overheat_h"]],
                                     hovertemplate=f"{LABEL[g]}<br>seed %{{customdata[0]}}<br>window %{{y:.1f}} kWh/yr"
                                                   f"<br>shield %{{customdata[1]:.1f}} h/yr<extra></extra>"), 1, i)
            rows.append({"regime": k, "controller": LABEL[g], "n": len(s), "window_kWh_mean": s.E_del_window_kWh.mean(),
                         "window_kWh_sd": s.E_del_window_kWh.std(), "shield_h_mean": s.overheat_h.mean()})
        fig.update_xaxes(showticklabels=False, row=1, col=i)
    fig.update_yaxes(title_text="Validation window delivery (kWh/yr)", row=1, col=1)
    base_layout(fig, 430)
    return fig, table(pd.DataFrame(rows), "extension study vs baselines (validation 2021-23), one dot per seed")


def chart_frontier():
    p = R / "m12_extensions.csv"
    if not p.exists():
        return None, ""
    d = pd.read_csv(p)
    d = d[(d.regime == 1)]
    w4 = json.loads((R / "frozen" / "r1" / "frozen.json").read_text())["w4"]
    sweep = d[d.study.isin(["m4", "pareto"]) & (d.w4 == w4)]
    diag = d[d.study == "w3diag"]
    base = d[d.study == "baseline"]
    fig = go.Figure()
    rows = []
    for name, s, col, sym in (("PPO, w2 sweep (w3 by dominance rule)", sweep, LIGHT["PPO"], "diamond"),
                              ("PPO, w3 below dominance (DIAGNOSTIC)", diag, "#6250d6", "triangle-up")):
        if s.empty:
            continue
        lab = (s.w2.map(lambda v: f"w2={v:g}") if name.startswith("PPO, w2") else s.w3.map(lambda v: f"w3={v:g}"))
        fig.add_trace(go.Scatter(x=s.overheat_h, y=s.E_del_window_kWh, mode="markers", name=name,
                                 marker=dict(size=11, color=col, symbol=sym, line=dict(width=2, color="#fcfcfb")),
                                 text=lab + s.seed.map(lambda v: f", seed {v}"),
                                 hovertemplate="%{text}<br>shield %{x:.1f} h/yr<br>window %{y:.1f} kWh/yr<extra></extra>"))
        rows += [{"series": name, "point": t, "shield_h": x, "window_kWh": y} for t, x, y in zip(lab, s.overheat_h, s.E_del_window_kWh)]
    for g in ("B0", "B1", "B2"):
        b = base[base.algo == g]
        fig.add_trace(go.Scatter(x=b.overheat_h, y=b.E_del_window_kWh, mode="markers+text", name=LABEL[g],
                                 marker=dict(size=13, color=LIGHT[g], line=dict(width=2, color="#fcfcfb")),
                                 text=[g], textposition="bottom center" if g == "B1" else "top center",
                                 hovertemplate=f"{LABEL[g]}<br>shield %{{x:.0f}} h/yr<br>window %{{y:.1f}} kWh/yr<extra></extra>"))
        rows += [{"series": LABEL[g], "point": "", "shield_h": x, "window_kWh": y} for x, y in zip(b.overheat_h, b.E_del_window_kWh)]
    fig.update_xaxes(title_text="Shield-active hours per year")
    fig.update_yaxes(title_text="Validation window delivery (kWh/yr)")
    base_layout(fig, 460)
    return fig, table(pd.DataFrame(rows), "regime 1 frontier points, one per seed")


def chart_agents():
    """Regime 1: PPO / DQN / SAC under perfect, smart-persistence and no forecast (m13 study), one dot per seed."""
    p = R / "m13_forecast_algo_r1.csv"
    if not p.exists():
        return None, ""
    d = pd.read_csv(p)
    d = d[~((d.source == "fcalgo") & (d.forecast == "none"))]          # the single leftover seed; V1 runs stand for 'none'
    fc_lab = {"smart": "smart forecast", "perfect": "perfect forecast", "none (V1)": "no forecast",
              "trained perfect, scored smart": "trained perfect, scored smart", "-": ""}
    sym = {"smart": "diamond", "perfect": "square", "none (V1)": "circle-open",
           "trained perfect, scored smart": "square-open", "-": "circle"}
    order = [("B0", "-"), ("B1", "-"), ("PPO", "none (V1)"), ("PPO", "perfect"), ("PPO", "trained perfect, scored smart"),
             ("PPO", "smart"), ("DQN", "none (V1)"), ("DQN", "smart"), ("SAC", "smart")]
    fig = go.Figure()
    rows = []
    for algo, fc in order:
        s = d[(d.algo == algo) & (d.forecast == fc)]
        if s.empty:
            continue
        cat = f"{algo}<br>{fc_lab[fc]}" if fc != "-" else LABEL[algo]
        fig.add_trace(go.Scatter(
            x=[cat] * len(s), y=s.E_del_window_kWh, mode="markers", name=f"{LABEL[algo]} {fc_lab[fc]}".strip(),
            legendgroup=algo, showlegend=False,
            marker=dict(size=12, color=LIGHT[algo], symbol=sym[fc], line=dict(width=2, color="#fcfcfb")),
            customdata=s[["seed", "overheat_h", "bypass_switches"]],
            hovertemplate=f"{html.escape(LABEL[algo])} {html.escape(fc_lab[fc])}<br>seed %{{customdata[0]}}"
                          "<br>window %{y:.1f} kWh/yr<br>overheat %{customdata[1]:.1f} h/yr"
                          "<br>switches %{customdata[2]:.0f}/yr<extra></extra>"))
        rows.append({"agent": LABEL[algo], "forecast": fc_lab[fc] or "-", "n_seeds": len(s),
                     "window_kWh_mean": s.E_del_window_kWh.mean(), "window_kWh_sd": s.E_del_window_kWh.std(),
                     "overheat_h_mean": s.overheat_h.mean(), "switches_mean": s.bypass_switches.mean(),
                     "hours_over_limit_max": s.hours_over_limit.max()})
    best = d[(d.algo == "PPO") & (d.forecast == "smart")].E_del_window_kWh.mean()
    fig.add_hline(y=best, line=dict(width=1, dash="dot", color=LIGHT["PPO"]),
                  annotation_text=f"PPO + smart forecast, mean {best:.1f}", annotation_position="bottom left")
    fig.update_yaxes(title_text="Validation window delivery (kWh/yr)")
    base_layout(fig, 450)
    return fig, table(pd.DataFrame(rows), "regime 1 agent and forecast comparison (validation 2021-23)")


def chart_recommended():
    """Recommended agent (PPO + smart forecast) vs B0 and the frozen PPO on the test years, EXPLORATORY."""
    p = R / "m13b_postlock_test_summary_EXPLORATORY.csv"
    if not p.exists():
        return None, ""
    s = pd.read_csv(p)
    s = s[s.controller.isin(["B0", "PPO_frozen", "PPO_smart"])]
    fig = go.Figure()
    for g, col in (("B0", LIGHT["B0"]), ("PPO_frozen", LIGHT["PPO"]), ("PPO_smart", LIGHT["PPO_smart"])):
        a = s[s.controller == g]
        fig.add_trace(go.Bar(x=[f"Regime {k}" for k in a.regime], y=a.E_del_window_kWh_mean, name=LABEL[g],
                             marker_color=col, marker_line=dict(width=2, color="#fcfcfb"),
                             error_y=dict(type="data", array=a.E_del_window_kWh_sd.fillna(0), visible=g != "B0",
                                          thickness=1.5),
                             customdata=a[["delivery_pct_of_B0", "overheat_h_mean", "bypass_switches_mean"]],
                             hovertemplate=f"{LABEL[g]}<br>%{{x}}: %{{y:.1f}} kWh/yr (%{{customdata[0]:.1f}} % of B0)"
                                           "<br>overheat %{customdata[1]:.1f} h/yr<br>switches %{customdata[2]:.0f}/yr"
                                           "<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text="Window delivery to 50 C (kWh/yr)", rangemode="tozero")
    base_layout(fig, 420)
    t = s[["regime", "controller", "source", "E_del_window_kWh_mean", "E_del_window_kWh_sd", "delivery_pct_of_B0",
           "overheat_h_mean", "bypass_switches_mean", "hours_over_limit_max"]]
    return fig, table(t.assign(controller=t.controller.map(LABEL)), "test years 2006-10, mean of 5 years (exploratory)")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>O3 Uttarakhand results</title>
<style>
:root{{--surface:#fcfcfb;--ink:#21201c;--ink2:#52514a;--grid:#e4e3dd;--line:#cfcec6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34332f;--line:#4a4944}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34332f;--line:#4a4944}}
body{{margin:0;background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:16px}}
.banner{{border:1px solid var(--line);border-left:4px solid #e34948;padding:8px 12px;margin:8px 0 16px;font-weight:600}}
h1{{font-size:24px;margin:8px 0}} h2{{font-size:18px;margin:28px 0 4px}} p{{color:var(--ink2);margin:4px 0 8px;max-width:900px}}
details{{margin:4px 0 18px}} summary{{cursor:pointer;color:var(--ink2)}}
.tbl{{border-collapse:collapse;font-size:13px;margin-top:6px;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}}
.tbl th{{color:var(--ink2);font-weight:600}}
</style></head><body><main>
<div class="banner">Software-in-the-loop simulation. Not field validation.</div>
<h1>Objective 3 - Uttarakhand: hourly bypass controller results</h1>
<p>O2's frozen 50 L / 1.5 m&sup2; PCM water heater, unchanged physics. Train 2016-20, validate 2021-23, locked O3 test
2006-10 (run once). Window = 18:00-09:00 IST, delivery target 50 C. Shield = O2's in-kernel 72 C / 62 C protection.
Config <code>{cfg}</code>.</p>
{body}
</main>
<script>
const DARK = {dark};
function theme(){{const d = document.documentElement.dataset.theme==='dark' ||
  (document.documentElement.dataset.theme!=='light' && matchMedia('(prefers-color-scheme: dark)').matches);
 const ink=d?'#f0efec':'#21201c', grid=d?'#34332f':'#e4e3dd', ring=d?'#1a1a19':'#fcfcfb';
 document.querySelectorAll('.js-plotly-plot').forEach(el=>{{
  if(!el._light){{el._light=el.data.map(t=>({{m:t.marker&&t.marker.color, l:t.line&&t.line.color}}));}}
  const upd={{'marker.color':[], 'line.color':[], 'marker.line.color':[]}};
  el.data.forEach((t,i)=>{{const o=el._light[i];
   upd['marker.color'].push(d&&typeof o.m==='string'&&DARK[o.m]?DARK[o.m]:o.m);
   upd['line.color'].push(d&&typeof o.l==='string'&&DARK[o.l]?DARK[o.l]:o.l);
   upd['marker.line.color'].push(t.type==='scattergeo'&&t.marker&&t.marker.symbol==='star'?ink:ring);}});
  Plotly.restyle(el, upd);
  Plotly.relayout(el, {{'font.color':ink,'xaxis.gridcolor':grid,'yaxis.gridcolor':grid,'xaxis2.gridcolor':grid,
   'yaxis2.gridcolor':grid,'xaxis3.gridcolor':grid,'yaxis3.gridcolor':grid,'xaxis4.gridcolor':grid,'yaxis4.gridcolor':grid,
   'geo.landcolor':d?'#2a2a27':'#f1f0ec','geo.countrycolor':grid}});}});}}
window.addEventListener('load', theme);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', theme);
</script></body></html>"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sections = [
        ("1. Delivery vs shield hours (validation 2021-23)",
         "Up is more hot water in the evening/night window; left is less shield activity. Grey = every rule and random "
         "on/off policy tried (G2 search, B1 and B2 grids). No policy sits above B0: on/off control can only trade "
         "delivery for fewer shield hours. B2 (perfect 2-4 h forecast) lands on B1: the forecast adds nothing.", chart_tradeoff),
        ("2. Locked O3 test, 2006-2010 (run once)",
         "Mean of 5 test years, fully solid start. PPO bar = mean of 3 frozen seeds, error bar = seed SD. Regime 3 has "
         "no PPO (O2: PCM gives no gain there).", chart_test),
        ("3. PPO learning curves (chosen weight set)",
         "Validation window delivery at each checkpoint. PPO settles below B0 because the contract reward penalises "
         "reaching the 72 C / 62 C guard band.", chart_learning),
        ("4. Locked test per year", "Window delivery per test year; the ranking B0 > B1 > PPO holds in every year.",
         chart_peryear),
        ("5. Sensor-fault robustness (validation)",
         "Observation corrupted, physics untouched, so the shield still acts on the true state: no fault produced a "
         "hard-limit hour.", chart_faults),
        ("6. Post-lock extension study (validation 2021-23)",
         "DQN (same reward as PPO) and PPO with a 1-3 h GHI forecast in its observation plus training-time domain "
         "randomisation, against the baselines; one dot per seed. None reaches B0.", chart_extensions),
        ("7. Delivery vs shield frontier, regime 1",
         "Sweeping the overheat weight w2 changes little: the safety-guard weight w3 (dominance rule) already removes "
         "shield time. Lowering w3 (diagnostic, never selectable) moves PPO along the trade-off toward B0.",
         chart_frontier),
        ("8. Which agent, with a forecast a real controller can make? (regime 1, validation 2021-23)",
         "PPO, DQN and SAC trained with the same reward and budget, 3 seeds each; one dot per seed. Diamond = smart-"
         "persistence forecast (clear-sky index), square = perfect forecast, open circle = no forecast. PPO with the "
         "smart forecast is highest and most consistent; SAC bypasses far more and loses heat. The forecast is worth "
         "only a few kWh/yr here.", chart_agents),
        ("9. Recommended agent on the test years, 2006-2010 (EXPLORATORY)",
         "PPO with the smart forecast against B0 and the frozen PPO from the locked test. Exploratory: the agent was "
         "chosen after the one-shot test had been run, so section 2 stays the official result. Error bars = seed SD. "
         "No run spent an hour over a hard limit.", chart_recommended),
        ("10. Regimes and simulated points",
         "O1's 45 Uttarakhand grid points by regime; stars = the medoids O2 and O3 simulate.", chart_map),
    ]
    body = []
    first = True
    for title, text, fn in sections:
        fig, tbl = fn()
        if fig is None:
            continue
        div = fig.to_html(full_html=False, include_plotlyjs=True if first else False,
                          config={"displaylogo": False, "responsive": True})
        first = False
        body.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>{div}{tbl}")
    page = PAGE.format(body="\n".join(body), dark=json.dumps(DARK), cfg=o3c.config_hash()[:16])
    (OUT / "dashboard.html").write_text(page, encoding="utf-8")
    print(OUT / "dashboard.html", f"{len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
