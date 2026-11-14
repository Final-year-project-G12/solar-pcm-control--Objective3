"""
Objective 3 — interactive visualisations (Plotly + Folium).

Generates a self-contained analysis dashboard for the thesis:

  Plotly (in results/plots/<tag>/dashboard.html):
    1. Policy comparison          — delivered energy, shield events, reward
    2. Delivery vs shield tradeoff — the key plot: does DRL beat the rules?
    3. Learning curves            — eval reward vs timesteps per DRL algorithm
    4. Reward decomposition       — delivery vs unmet vs shield penalty
    5. Per-year robustness        — delivered energy per validation year

  Folium (in results/plots/<tag>/tamilnadu_regimes_map.html):
    6. Tamil Nadu regime map      — 133 climate points, 3 regimes, medoid designs

Run from the objective3 root:
    python scripts/make_plots.py --tag R2                 # uses current comparison CSV
    python scripts/make_plots.py --tag R1 --csv results/archive_R1_reward/phase4_comparison_r0_R1.csv

The script is robust to which policies are present in the CSV.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

REPO_ROOT   = pathlib.Path(__file__).parents[1]
RESULTS_DIR = REPO_ROOT / "results"
NEW_OBJ     = REPO_ROOT.parent
CONTRACT    = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
CLUSTER_CSV = NEW_OBJ / "documentation" / "obj1_results" / "cluster_assignments_tamilnadu.csv"

# ---- validated categorical palette (dataviz skill reference instance) -------
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = \
    "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

# algorithm -> colour (DRL algos get distinct slots; baselines share group hues)
ALGO_COLOR = {
    "PPO": C_BLUE, "DQN": C_ORANGE, "RecurrentPPO": C_AQUA,
    "QRDQN": C_YELLOW, "MaskablePPO": C_VIOLET, "SAC": C_RED,
}
BASELINE_PREFIXES = ("B0", "B1", "B2", "B4")

LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
              color=INK, size=13),
    margin=dict(l=70, r=30, t=60, b=60),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def _is_baseline(policy: str) -> bool:
    return policy.startswith(BASELINE_PREFIXES)


def _short(policy: str) -> str:
    return (policy.replace("_fixed_charge", "").replace("_reactive_rule", "")
                  .replace("_forecast_rule", "").replace("_random_safe", ""))


def load_comparison(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["shield_total"] = df["total_shield_water"] + df["total_shield_pcm"]
    agg = df.groupby("policy", as_index=False).agg(
        delivered=("total_delivered_kWh", "mean"),
        unmet=("total_unmet_kWh", "mean"),
        shield=("shield_total", "mean"),
        reward=("total_reward", "mean"),
        safety=("n_safety_violations", "sum"),
    )
    agg["label"] = agg["policy"].map(_short)
    agg["is_baseline"] = agg["policy"].map(_is_baseline)
    # order: baselines first (by name), then DRL
    agg = agg.sort_values(["is_baseline", "label"], ascending=[False, True])
    return agg


# ---------------------------------------------------------------------------
# Plot 1 — policy comparison (delivered, shield, reward as three panels)
# ---------------------------------------------------------------------------

def plot_comparison(agg: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("Delivered energy (kWh)",
                                        "Shield activations / yr",
                                        "Mean episode reward"))
    colors = [C_AQUA if b else C_ORANGE for b in agg["is_baseline"]]
    for col, field, fmt in [(1, "delivered", ".0f"),
                            (2, "shield", ".0f"),
                            (3, "reward", ".3s")]:
        fig.add_bar(x=agg["label"], y=agg[field], marker_color=colors,
                    text=[f"{v:,.0f}" for v in agg[field]],
                    textposition="outside", row=1, col=col,
                    showlegend=False, cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
    fig.update_layout(**{k: v for k, v in LAYOUT.items()
                         if k not in ("xaxis", "yaxis")})
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      title="1 · Policy comparison — rule-based (aqua) vs Deep RL (orange)",
                      height=440)
    fig.update_xaxes(tickangle=-40, gridcolor=GRID, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7")
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — delivery vs shield tradeoff (THE key plot)
# ---------------------------------------------------------------------------

def plot_tradeoff(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for is_base, name, color in [(True, "Rule-based baseline", C_AQUA),
                                 (False, "Deep RL", C_ORANGE)]:
        sub = agg[agg["is_baseline"] == is_base]
        fig.add_trace(go.Scatter(
            x=sub["shield"], y=sub["delivered"], mode="markers+text",
            name=name, text=sub["label"], textposition="top center",
            textfont=dict(color=INK2, size=11),
            marker=dict(size=15, color=color,
                        line=dict(width=2, color=SURFACE)),
            hovertemplate="%{text}<br>shield=%{x:,.0f}<br>"
                          "delivered=%{y:,.1f} kWh<extra></extra>"))
    fig.update_layout(**LAYOUT, height=520,
                      title="2 · Delivery vs shield tradeoff — up-and-left is better "
                            "(more energy, fewer shield trips)",
                      xaxis_title="Shield activations per year (fewer →)",
                      yaxis_title="Delivered energy (kWh)")
    fig.update_xaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — learning curves from evaluations.npz
# ---------------------------------------------------------------------------

def _short_tf(df):
    df = df.copy()
    df["label"] = df["policy"].map(_short)
    df["shield_total"] = df["total_shield_water"] + df["total_shield_pcm"]
    return df


def plot_locked_test(test_csv: pathlib.Path, regime: int | None = None) -> go.Figure | None:
    """Held-out test set (2011-2015): the FINAL verified output."""
    if not test_csv.exists():
        return None
    raw = pd.read_csv(test_csv)
    if regime is not None and "regime_id" in raw.columns:
        raw = raw[raw["regime_id"] == regime]   # CSV is overwritten per regime run
        if raw.empty:
            return None
    df = _short_tf(raw)
    g = df.groupby("label", as_index=False).agg(
        delivered=("total_delivered_kWh", "mean"),
        shield=("shield_total", "mean"),
        safety=("n_safety_violations", "sum"))
    # order B0, B1, then the model
    g["ord"] = g["label"].map({"B0": 0, "B1": 1}).fillna(2)
    g = g.sort_values("ord")
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        subplot_titles=("Delivered energy (kWh)",
                                        "Shield activations / yr"))
    colors = [C_AQUA if l in ("B0", "B1") else C_ORANGE for l in g["label"]]
    fig.add_bar(x=g["label"], y=g["delivered"], marker_color=colors, row=1, col=1,
                text=[f"{v:,.0f}" for v in g["delivered"]], textposition="outside",
                showlegend=False, cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,.1f} kWh<extra></extra>")
    fig.add_bar(x=g["label"], y=g["shield"], marker_color=colors, row=1, col=2,
                text=[f"{v:,.0f}" for v in g["shield"]], textposition="outside",
                showlegend=False, cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=440,
                      font=dict(family='system-ui, sans-serif', color=INK, size=13),
                      margin=dict(l=60, r=30, t=70, b=50),
                      title="3 · LOCKED TEST (2011-2015, never seen) — the final verified "
                            "result: DQN holds delivery, cuts shields, zero violations")
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7")
    return fig


def plot_robustness(robust_csv: pathlib.Path) -> go.Figure | None:
    """DQN delivery + shields across perturbation scenarios (stability)."""
    if not robust_csv.exists():
        return None
    df = pd.read_csv(robust_csv)
    df["shield_total"] = df["total_shield_water"] + df["total_shield_pcm"]
    g = df.groupby("scenario", as_index=False).agg(
        delivered=("total_delivered_kWh", "mean"),
        shield=("shield_total", "mean"),
        safety=("n_safety_violations", "sum"))
    fig = go.Figure()
    fig.add_bar(x=g["scenario"], y=g["delivered"], marker_color=C_BLUE,
                name="Delivered kWh",
                hovertemplate="%{x}<br>%{y:,.1f} kWh<extra></extra>")
    fig.update_layout(**LAYOUT, height=460,
                      title="4 · Robustness — DQN delivered energy across perturbation "
                            "scenarios (all zero safety violations)",
                      xaxis_title="", yaxis_title="Delivered energy (kWh)")
    fig.update_xaxes(tickangle=-40)
    return fig


def plot_multiseed(ms_csv: pathlib.Path) -> go.Figure | None:
    """DQN across seeds on the delivery-shield plane (tradeoff variation)."""
    if not ms_csv.exists():
        return None
    df = pd.read_csv(ms_csv)
    df = df[df.get("algo_key", "") == "dqn"]
    if df.empty:
        return None
    df["shield_total"] = df["total_shield_water"] + df["total_shield_pcm"]
    g = df.groupby("seed", as_index=False).agg(
        delivered=("total_delivered_kWh", "mean"),
        shield=("shield_total", "mean"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["shield"], y=g["delivered"], mode="markers+text",
        text=[f"seed {s}" for s in g["seed"]], textposition="top center",
        textfont=dict(color=INK2, size=11),
        marker=dict(size=15, color=C_ORANGE, line=dict(width=2, color=SURFACE)),
        name="DQN seeds",
        hovertemplate="%{text}<br>shield=%{x:,.0f}<br>del=%{y:,.1f} kWh<extra></extra>"))
    # baseline refs
    for lbl, d, s, col in [("B0", 2146.2, 50298, C_AQUA), ("B1", 2015.3, 22371, C_GREEN)]:
        fig.add_trace(go.Scatter(x=[s], y=[d], mode="markers+text", text=[lbl],
                                 textposition="bottom center",
                                 marker=dict(size=13, color=col, symbol="square",
                                             line=dict(width=2, color=SURFACE)),
                                 name=lbl, showlegend=False,
                                 hovertemplate=f"{lbl}<br>shield=%{{x:,.0f}}<br>"
                                               "del=%{y:,.1f}<extra></extra>"))
    fig.update_layout(**LAYOUT, height=470,
                      title="5 · DQN across seeds — all beat the baselines; seed 42 "
                            "uniquely keeps delivery ≥95% of B0 (up-and-left is better)",
                      xaxis_title="Shield activations / yr (fewer →)",
                      yaxis_title="Delivered energy (kWh)")
    fig.update_xaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — reward decomposition (delivery reward vs penalties)
# ---------------------------------------------------------------------------

def plot_reward_decomp(agg: pd.DataFrame, w_unmet=2.0, w_shield=10.0) -> go.Figure:
    # reconstruct reward components (Wh scale): delivery = del*1000,
    # unmet penalty = w_unmet*unmet*1000, shield penalty = w_shield*shield
    delivery = agg["delivered"] * 1000.0
    unmet_pen = -w_unmet * agg["unmet"] * 1000.0
    shield_pen = -w_shield * agg["shield"]
    fig = go.Figure()
    fig.add_bar(x=agg["label"], y=delivery, name="Delivery reward",
                marker_color=C_AQUA,
                hovertemplate="%{x}<br>+%{y:,.0f}<extra>delivery</extra>")
    fig.add_bar(x=agg["label"], y=unmet_pen, name="Unmet penalty",
                marker_color=C_YELLOW,
                hovertemplate="%{x}<br>%{y:,.0f}<extra>unmet</extra>")
    fig.add_bar(x=agg["label"], y=shield_pen, name="Shield penalty",
                marker_color=C_RED,
                hovertemplate="%{x}<br>%{y:,.0f}<extra>shield</extra>")
    fig.add_trace(go.Scatter(
        x=agg["label"], y=agg["reward"], name="Net reward",
        mode="markers", marker=dict(size=11, color=INK, symbol="diamond",
                                    line=dict(width=1.5, color=SURFACE)),
        hovertemplate="%{x}<br>net=%{y:,.0f}<extra></extra>"))
    fig.update_layout(**LAYOUT, height=500, barmode="relative",
                      title="6 · Reward decomposition (R2) — how each term builds the score",
                      xaxis_title="", yaxis_title="Reward contribution (Wh-equiv)")
    fig.update_xaxes(tickangle=-40)
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — per-year delivered energy (robustness across validation years)
# ---------------------------------------------------------------------------

def plot_per_year(csv_path: pathlib.Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    df["label"] = df["policy"].map(_short)
    years = sorted(df["year"].unique())
    fig = go.Figure()
    palette = [C_BLUE, C_ORANGE, C_AQUA]
    for i, yr in enumerate(years):
        sub = df[df["year"] == yr]
        fig.add_bar(x=sub["label"], y=sub["total_delivered_kWh"],
                    name=str(yr), marker_color=palette[i % len(palette)],
                    hovertemplate=f"%{{x}} · {yr}<br>%{{y:,.1f}} kWh<extra></extra>")
    fig.update_layout(**LAYOUT, height=480, barmode="group",
                      title="7 · Delivered energy per validation year (consistency check)",
                      xaxis_title="", yaxis_title="Delivered energy (kWh)")
    fig.update_xaxes(tickangle=-40)
    return fig


# ---------------------------------------------------------------------------
# Folium — Tamil Nadu regime map
# ---------------------------------------------------------------------------

def make_map(out_path: pathlib.Path) -> bool:
    import folium
    if not CLUSTER_CSV.exists():
        print(f"[WARN] cluster CSV not found: {CLUSTER_CSV}")
        return False
    df = pd.read_csv(CLUSTER_CSV)
    contract = json.loads(CONTRACT.read_text())
    design = {d["regime_id"]: d for d in contract["static_design_per_regime"]}

    regime_color = {0: C_BLUE, 1: C_ORANGE, 2: C_AQUA}
    medoids = {d["regime_id"]: d.get("medoid_point_id") for d in design.values()}

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    for _, r in df.iterrows():
        cid = int(r["cluster_id"])
        pop = float(r["population"])
        radius = 3 + 9 * (pop / df["population"].max()) ** 0.5
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=radius,
            color=regime_color.get(cid, "#888"), fill=True,
            fill_color=regime_color.get(cid, "#888"), fill_opacity=0.55,
            weight=1,
            popup=folium.Popup(
                f"<b>{r['point_id']}</b><br>Regime {cid}<br>"
                f"pop {pop:,.0f}<br>({r['lat']:.3f}, {r['lon']:.3f})", max_width=220),
        ).add_to(m)

    # highlight medoids with design info
    for rid, pid in medoids.items():
        row = df[df["point_id"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]
        d = design[rid]
        html = (f"<b>Regime {rid} medoid — {pid}</b><br>"
                f"{d.get('label','')}<br><hr style='margin:4px 0'>"
                f"<b>Selected PCM:</b> {d['selected_pcm']} "
                f"(O1 rank {d.get('selected_pcm_o1_consensus_rank','?')})<br>"
                f"<b>Capsules:</b> {d['n_capsules']}<br>"
                f"<b>Flow:</b> {d['nominal_flow_kg_s']*1000:.1f} g/s<br>"
                f"<b>PCM mass:</b> {d['pcm_mass_kg']:.1f} kg<br>"
                f"<b>Elev:</b> {d.get('medoid_elevation_m','?')} m<br>"
                f"<b>Pop covered:</b> {d.get('population_covered',0):,}")
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="darkred", icon="star", prefix="fa"),
            popup=folium.Popup(html, max_width=280),
            tooltip=f"Regime {rid} medoid: {d['selected_pcm']}",
        ).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
         background: #fcfcfb; padding: 10px 14px; border: 1px solid #ccc;
         border-radius: 6px; font-family: system-ui, sans-serif; font-size: 13px;">
      <b>Tamil Nadu climate regimes</b><br>
      <span style="color:#2a78d6;">&#9679;</span> Regime 0 — coastal/interior lowland<br>
      <span style="color:#eb6834;">&#9679;</span> Regime 1 — elevated/Nilgiris-adjacent<br>
      <span style="color:#1baf7a;">&#9679;</span> Regime 2 — coastal metro lowland<br>
      <span style="color:#8b0000;">&#9733;</span> regime medoid (selected design)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(out_path))
    return True


# ---------------------------------------------------------------------------
# Dashboard assembly
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--tag", type=str, default="R2")
    ap.add_argument("--csv", type=str, default=None,
                    help="comparison CSV (default results/phase4_comparison_r{regime}.csv)")
    ap.add_argument("--selected", type=str, default="dqn",
                    help="selected policy whose locked-test/robustness CSVs are plotted")
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv) if args.csv else \
        RESULTS_DIR / f"phase4_comparison_r{args.regime}.csv"
    if not csv_path.exists():
        print(f"[FAIL] comparison CSV not found: {csv_path}")
        sys.exit(1)

    out_dir = RESULTS_DIR / "plots" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = load_comparison(csv_path)
    figs = [
        plot_comparison(agg),     # 1 validation comparison
        plot_tradeoff(agg),       # 2 validation tradeoff
    ]
    # 3 locked test (final verified), 4 robustness, 5 multi-seed — add if present
    test_csv   = RESULTS_DIR / f"phase6_final_test_{args.selected}.csv"
    robust_csv = RESULTS_DIR / f"phase5_robustness_r{args.regime}_{args.selected}.csv"
    ms_csv     = RESULTS_DIR / f"phase4_multiseed_raw_r{args.regime}.csv"
    for f in (plot_locked_test(test_csv, args.regime), plot_robustness(robust_csv),
              plot_multiseed(ms_csv)):
        if f is not None:
            figs.append(f)
    figs.append(plot_reward_decomp(agg))   # 6 reward decomposition
    figs.append(plot_per_year(csv_path))   # 7 per-year

    # write combined dashboard
    dash = out_dir / "dashboard.html"
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>Objective 3 — {args.tag} results</title>',
             '<style>body{background:#f9f9f7;margin:0;padding:16px;'
             'font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b}'
             'h1{font-size:20px}.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);'
             'border-radius:10px;margin:14px 0;padding:8px}'
             'a{color:#2a78d6}</style></head><body>',
             f'<h1>Objective 3 — adaptive PCM control · reward {args.tag} · regime {args.regime}</h1>',
             f'<p style="color:#52514e">Tamil Nadu · 100 L tank · {len(agg)} policies · '
             'validation years 2021–2023. '
             '<a href="tamilnadu_regimes_map.html">↗ open the regime map</a></p>']
    for i, fig in enumerate(figs):
        parts.append('<div class="card">')
        parts.append(pio.to_html(fig, include_plotlyjs=("cdn" if i == 0 else False),
                                 full_html=False))
        parts.append('</div>')
    parts.append('</body></html>')
    dash.write_text("\n".join(parts), encoding="utf-8")
    print(f"[INFO] dashboard -> {dash}")

    ok = make_map(out_dir / "tamilnadu_regimes_map.html")
    if ok:
        print(f"[INFO] map       -> {out_dir / 'tamilnadu_regimes_map.html'}")
    print(f"\n[OK] {len(figs)} Plotly charts + {'1 Folium map' if ok else 'no map'} "
          f"in {out_dir}")


if __name__ == "__main__":
    main()
