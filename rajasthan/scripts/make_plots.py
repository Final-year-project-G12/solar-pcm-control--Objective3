"""
Objective 3 — interactive visualisations (Plotly + optional Folium), Rajasthan.

Generates a self-contained analysis dashboard in results/plots/<tag>/:

  Plotly (dashboard.html):
    1. Policy comparison            — delivered energy, shield hours, reward
    2. Delivery vs shield tradeoff  — the key plot: does DRL beat the rules?
    3. Learning curves              — validation reward vs training steps per DRL algorithm
    4. Reward decomposition         — delivery vs unmet vs shield penalty (weights read from the environment)
    5. Per-year delivered energy    — consistency across validation years
    6. Window gain vs water-only twin (Rajasthan addition: the plan's primary success metric)

  Folium (rajasthan_regimes_map.html), only if `folium` is installed:
    7. Rajasthan map of the three O2 medoid points and their released designs.

Run from the rajasthan O3 root:
    python scripts/make_plots.py --regime 1 --tag R2
    python scripts/make_plots.py --regime 1 --tag smoke --csv <tmp>/phase4_comparison_r1.csv --results-dir <tmp>

The script is robust to which policies are present in the CSV. Plotting is one process and takes
seconds; --n-jobs / --n-envs are accepted for uniformity and unused.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args  # noqa: E402
from o3.environment import pcm_env  # noqa: E402
from o3.demo.live_adapter import MEDOIDS  # noqa: E402

RESULTS_DIR = config_O2.RESULTS_DIR
CONTRACT = config_O2.CONTRACT_FILE

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

ALGO_COLOR = {"PPO": C_BLUE, "DQN": C_ORANGE, "RecurrentPPO": C_AQUA, "QRDQN": C_YELLOW, "MaskablePPO": C_VIOLET, "SAC": C_RED}
BASELINE_PREFIXES = ("B0", "B1", "B2", "B4", "TWIN")

LAYOUT = dict(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
              font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13),
              margin=dict(l=70, r=30, t=60, b=60),
              xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7"),
              yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7"),
              legend=dict(bgcolor="rgba(0,0,0,0)"))


def _is_baseline(policy: str) -> bool:
    return policy.startswith(BASELINE_PREFIXES)


def _short(policy: str) -> str:
    return (policy.replace("_fixed_charge", "").replace("_reactive_rule", "").replace("_forecast_rule", "")
            .replace("_random_safe", "").replace("_water_only", ""))


def load_comparison(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["shield_total"] = df["total_shield_water"] + df["total_shield_pcm"]
    agg = df.groupby("policy", as_index=False).agg(
        delivered=("total_delivered_kWh", "mean"), unmet=("total_unmet_kWh", "mean"), shield=("shield_total", "mean"),
        reward=("total_reward", "mean"), safety=("n_safety_violations", "sum"),
        shield_h=("shield_water_h", "mean"), window=("E_del45_window_kWh", "mean"),
        gain=("gain_window_kWh", "mean") if "gain_window_kWh" in df else ("total_reward", "mean"))
    if "gain_window_kWh" not in df:
        agg["gain"] = np.nan
    agg["label"] = agg["policy"].map(_short)
    agg["is_baseline"] = agg["policy"].map(_is_baseline)
    return agg.sort_values(["is_baseline", "label"], ascending=[False, True])


def plot_comparison(agg: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("Delivered to 45 C (kWh/yr)", "Shield activations / yr", "Mean episode reward"))
    colors = [C_AQUA if b else C_ORANGE for b in agg["is_baseline"]]
    for col, field in [(1, "delivered"), (2, "shield"), (3, "reward")]:
        fig.add_bar(x=agg["label"], y=agg[field], marker_color=colors, text=[f"{v:,.0f}" for v in agg[field]],
                    textposition="outside", row=1, col=col, showlegend=False, cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
    fig.update_layout(**{k: v for k, v in LAYOUT.items() if k not in ("xaxis", "yaxis")})
    fig.update_layout(title="1 · Policy comparison — rule-based / twin (aqua) vs Deep RL (orange)", height=440)
    fig.update_xaxes(tickangle=-40, gridcolor=GRID, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7")
    return fig


def plot_tradeoff(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for is_base, name, color in [(True, "Rule-based / twin", C_AQUA), (False, "Deep RL", C_ORANGE)]:
        sub = agg[agg["is_baseline"] == is_base]
        fig.add_trace(go.Scatter(x=sub["shield"], y=sub["delivered"], mode="markers+text", name=name, text=sub["label"],
                                 textposition="top center", textfont=dict(color=INK2, size=11),
                                 marker=dict(size=15, color=color, line=dict(width=2, color=SURFACE)),
                                 hovertemplate="%{text}<br>shield=%{x:,.0f}<br>delivered=%{y:,.1f} kWh<extra></extra>"))
    fig.update_layout(**LAYOUT, height=520, title="2 · Delivery vs shield tradeoff — up-and-left is better",
                      xaxis_title="Shield activations per year (fewer →)", yaxis_title="Delivered to 45 C (kWh)")
    fig.update_xaxes(autorange="reversed")
    return fig


_ALGO_DIR = {"PPO": "ppo", "DQN": "dqn", "RecurrentPPO": "recurrent_ppo", "QRDQN": "qrdqn", "MaskablePPO": "maskable_ppo", "SAC": "sac"}


def plot_learning_curves(regime: int, results_dir: pathlib.Path = RESULTS_DIR) -> go.Figure | None:
    fig = go.Figure()
    any_data = False
    for algo, subdir in _ALGO_DIR.items():
        npz = results_dir / subdir / f"r{regime}" / "eval_logs" / "evaluations.npz"
        if not npz.exists():
            continue
        try:
            d = np.load(npz)
            ts, res = d["timesteps"], d["results"].mean(axis=1)
        except Exception:
            continue
        any_data = True
        fig.add_trace(go.Scatter(x=ts, y=res, mode="lines+markers", name=algo, line=dict(width=2, color=ALGO_COLOR.get(algo, C_BLUE)),
                                 marker=dict(size=6), hovertemplate=f"{algo}<br>step=%{{x:,}}<br>reward=%{{y:,.0f}}<extra></extra>"))
    if not any_data:
        return None
    fig.update_layout(**LAYOUT, height=480, title="3 · Learning curves — validation reward vs training steps",
                      xaxis_title="Training timesteps", yaxis_title="Mean validation reward")
    return fig


def plot_reward_decomp(agg: pd.DataFrame, w_unmet: float = pcm_env.W_UNMET, w_shield: float = 1.7) -> go.Figure:
    # delivery reward = delivered*1000 (Wh); unmet penalty = w_unmet*unmet*1000; shield penalty = w_shield*shield count
    # (the TN script hard-coded w_shield=20 although its environment default was 10; here main() passes the
    #  regime's actual default from pcm_env.default_w_shield)
    delivery, unmet_pen, shield_pen = agg["delivered"] * 1000.0, -w_unmet * agg["unmet"] * 1000.0, -w_shield * agg["shield"]
    fig = go.Figure()
    fig.add_bar(x=agg["label"], y=delivery, name="Delivery reward", marker_color=C_AQUA)
    fig.add_bar(x=agg["label"], y=unmet_pen, name="Unmet penalty", marker_color=C_YELLOW)
    fig.add_bar(x=agg["label"], y=shield_pen, name="Shield penalty", marker_color=C_RED)
    fig.add_trace(go.Scatter(x=agg["label"], y=agg["reward"], name="Net reward", mode="markers",
                             marker=dict(size=11, color=INK, symbol="diamond", line=dict(width=1.5, color=SURFACE))))
    fig.update_layout(**LAYOUT, height=500, barmode="relative", title="4 · Reward decomposition (R2, default weights)",
                      yaxis_title="Reward contribution (Wh-equiv)")
    fig.update_xaxes(tickangle=-40)
    return fig


def plot_per_year(csv_path: pathlib.Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    df["label"] = df["policy"].map(_short)
    palette = [C_BLUE, C_ORANGE, C_AQUA]
    fig = go.Figure()
    for i, yr in enumerate(sorted(df["year"].unique())):
        sub = df[df["year"] == yr]
        fig.add_bar(x=sub["label"], y=sub["total_delivered_kWh"], name=str(yr), marker_color=palette[i % len(palette)])
    fig.update_layout(**LAYOUT, height=480, barmode="group", title="5 · Delivered energy per validation year",
                      yaxis_title="Delivered to 45 C (kWh)")
    fig.update_xaxes(tickangle=-40)
    return fig


def plot_gain(agg: pd.DataFrame) -> go.Figure | None:
    sub = agg.dropna(subset=["gain"])
    sub = sub[~sub["policy"].str.startswith("TWIN")]
    if sub.empty:
        return None
    fig = go.Figure(go.Bar(x=sub["label"], y=sub["gain"], marker_color=[C_AQUA if b else C_ORANGE for b in sub["is_baseline"]],
                           text=[f"{v:+.1f}" for v in sub["gain"]], textposition="outside"))
    fig.update_layout(**LAYOUT, height=440, title="6 · 18:00–09:00 delivered energy to 45 C, gain vs the water-only twin (kWh/yr)",
                      yaxis_title="kWh per year")
    fig.update_xaxes(tickangle=-40)
    return fig


def make_map(out_path: pathlib.Path) -> bool:
    try:
        import folium
    except ImportError:
        print("[INFO] folium not installed: skipping the Rajasthan map (pip install folium)")
        return False
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    m = folium.Map(location=[25.9, 74.8], zoom_start=7, tiles="cartodbpositron")
    colors = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
    for rid, p in MEDOIDS.items():
        d = next(x for x in c["static_design_per_regime"] if x["regime_id"] == rid)
        html = (f"<b>Regime {rid} — {p['place']} ({p['point']})</b><br>PCM: {d['selected_pcm']}<br>{d['n_capsules']} capsules, "
                f"flow {d['nominal_flow_kg_s']:.4f} kg/s<br>{c['fixed_system']['tank_volume_L']:.0f} L / {c['fixed_system']['collector_area_m2']} m2")
        folium.CircleMarker([p["lat"], p["lon"]], radius=10, color=colors[rid], fill=True, fill_opacity=0.8,
                            popup=folium.Popup(html, max_width=260), tooltip=f"Regime {rid} {p['place']}").add_to(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--tag", type=str, default="R2")
    ap.add_argument("--csv", type=str, default=None, help="comparison CSV (default results/phase4_comparison_r{regime}.csv)")
    ap.add_argument("--results-dir", type=str, default=None)
    add_parallel_args(ap)
    args = ap.parse_args()
    res_dir = pathlib.Path(args.results_dir) if args.results_dir else RESULTS_DIR

    csv_path = pathlib.Path(args.csv) if args.csv else res_dir / f"phase4_comparison_r{args.regime}.csv"
    if not csv_path.exists():
        print(f"[FAIL] comparison CSV not found: {csv_path}")
        sys.exit(1)
    out_dir = res_dir / "plots" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = load_comparison(csv_path)
    figs = [plot_comparison(agg), plot_tradeoff(agg), plot_reward_decomp(agg, w_shield=pcm_env.default_w_shield(args.regime)),
            plot_per_year(csv_path)]
    g = plot_gain(agg)
    if g is not None:
        figs.append(g)
    lc = plot_learning_curves(args.regime, res_dir)
    if lc is not None:
        figs.insert(2, lc)

    dash = out_dir / "dashboard.html"
    parts = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>Objective 3 — Rajasthan {args.tag} results</title>',
             '<style>body{background:#f9f9f7;margin:0;padding:16px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b}'
             'h1{font-size:20px}.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:10px;margin:14px 0;padding:8px}'
             '.banner{background:#fff4d6;border:1px solid #eda100;padding:8px;border-radius:8px}a{color:#2a78d6}</style></head><body>',
             '<div class="banner">Software-in-the-loop simulation. Not field validation.</div>',
             f'<h1>Objective 3 — adaptive PCM control · reward {args.tag} · regime {args.regime}</h1>',
             f'<p style="color:#52514e">Rajasthan · 50 L tank / 1.5 m² collector · {len(agg)} policies · validation years 2021–2023. '
             '<a href="rajasthan_regimes_map.html">open the regime map</a> (if generated)</p>']
    for i, fig in enumerate(figs):
        parts += ['<div class="card">', pio.to_html(fig, include_plotlyjs=("cdn" if i == 0 else False), full_html=False), '</div>']
    parts.append('</body></html>')
    dash.write_text("\n".join(parts), encoding="utf-8")
    print(f"[INFO] dashboard -> {dash}")
    ok = make_map(out_dir / "rajasthan_regimes_map.html")
    print(f"\n[OK] {len(figs)} Plotly charts + {'1 Folium map' if ok else 'no map'} in {out_dir}")


if __name__ == "__main__":
    main()
