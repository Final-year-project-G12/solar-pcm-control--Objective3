"""
Objective 3 — Streamlit demo (plan 15.9, M7).   Run:  streamlit run app/streamlit_app.py

    Software-in-the-loop simulation. Not field validation.

Replay mode: pick a regime (O2 medoid), an allowed year (train / validation roles only — test years are never offered
and the loader refuses them without the O3 lock) and a date range; B0, B1, B2 (and any trained agent found under results/)
run side by side through the same environment step; temperatures, melt fraction, draw, action, shield status,
cumulative delivered energy and over-limit time are shown; the log can be downloaded.
Live mode: an hourly adapter (Open-Meteo at the chosen medoid) advances the simulator one step per click; provider,
timestamp, latency and missing-data flags are logged; stale or failed data falls back visibly to persistence / stub.
The policy shown is a frozen checkpoint; nothing here trains and the O3 test years are never touched.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from o3.demo.live_adapter import BANNER, MEDOIDS, LiveDemoRunner, DemoMode  # noqa: E402
from o3.demo.replay import run_replay, available_models, POLICY_LABELS  # noqa: E402
from o3.utils import weather_roles  # noqa: E402

COLORS = {"twin": "#898781", "b0": "#2a78d6", "b1": "#eb6834", "b2": "#1baf7a"}
PALETTE = ["#6250d6", "#e34948", "#eda100", "#e87ba4", "#008300", "#52514e"]

st.set_page_config(page_title="O3 PCM controller demo", layout="wide")


def banner() -> None:
    """Persistent label — rendered at the top of every view."""
    st.error(f"**{BANNER}** All tank, PCM and melt-fraction values come from the frozen Objective-2 simulator, not from sensors.",
             icon=":material/warning:")


@st.cache_data(show_spinner="Simulating…")
def _replay(regime: int, year: int, key: str, n_hours: int, model: tuple | None) -> pd.DataFrame:
    return run_replay(regime, year, key, n_hours, model)


def replay_view() -> None:
    models = available_models(st.session_state.regime)
    c1, c2, c3 = st.columns(3)
    years = weather_roles.TRAIN_YEARS + weather_roles.VAL_YEARS           # allowed roles only
    year = c1.selectbox("Year (train / validation only)", years, index=years.index(2021))
    start_day = c2.number_input("Start day of year", 1, 365, 150)
    n_days = c3.number_input("Days shown", 1, 30, 7)
    keys = st.multiselect("Policies (run side by side)", list(POLICY_LABELS), default=["twin", "b0", "b1"],
                          format_func=POLICY_LABELS.get)
    chosen_models = st.multiselect("Trained agents found under results/", list(models)) if models else []
    if not models:
        st.caption("No trained agents under results/ yet (full training has not been run); baselines only.")
    end_hour = int((start_day - 1 + n_days) * 24)
    runs = {POLICY_LABELS[k]: (_replay(st.session_state.regime, year, k, end_hour, None), COLORS[k]) for k in keys}
    for i, label in enumerate(chosen_models):
        runs[label] = (_replay(st.session_state.regime, year, "model", end_hour, models[label]), PALETTE[i % len(PALETTE)])
    if not runs:
        st.info("Pick at least one policy.")
        return
    lo, hi = int((start_day - 1) * 24), end_hour

    # --- summary table over the shown range
    rows = []
    for label, (df, _) in runs.items():
        d = df[(df.hour >= lo) & (df.hour < hi)]
        rows.append({"policy": label, "E to 45 C (kWh)": d.E_del45_Wh.sum() / 1000, "window 18–09 (kWh)": d[d.in_window].E_del45_Wh.sum() / 1000,
                     "water-shield (h)": d.shield_water_s.sum() / 3600, "PCM-shield (h)": d.shield_pcm_s.sum() / 3600,
                     "hours over limit": d.over_limit_s.sum() / 3600, "peak T_top (C)": d.T_top_C.max(), "bypass/idle hours": int((d.action != "CHARGE").sum())})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

    # --- curves
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[1.2, 1, 0.7, 0.8, 0.9, 0.9],
                        subplot_titles=("Top water temperature (C) — shield guard band 72 C", "PCM temperature (C) — guard band 62 C",
                                        "Melt fraction", "Hot-water draw (kg/h)", "Cumulative delivered to 45 C (kWh) · cumulative hours over limit (dashed)",
                                        "Action (CHARGE = loop on) and shield-active hours"))
    for label, (df, col) in runs.items():
        d = df[(df.hour >= lo) & (df.hour < hi)]
        fig.add_scatter(x=d.hour, y=d.T_top_C, name=label, line=dict(color=col), legendgroup=label, row=1, col=1)
        fig.add_scatter(x=d.hour, y=d.T_pcm_C, name=label, line=dict(color=col), legendgroup=label, showlegend=False, row=2, col=1)
        fig.add_scatter(x=d.hour, y=d.f_melt, name=label, line=dict(color=col), legendgroup=label, showlegend=False, row=3, col=1)
        fig.add_scatter(x=d.hour, y=d.draw_kg, name=label, line=dict(color=col), legendgroup=label, showlegend=False, row=4, col=1)
        fig.add_scatter(x=d.hour, y=d.cum_E_del45_kWh - d.cum_E_del45_kWh.iloc[0], name=label, line=dict(color=col), legendgroup=label, showlegend=False, row=5, col=1)
        fig.add_scatter(x=d.hour, y=d.cum_over_limit_h - d.cum_over_limit_h.iloc[0], line=dict(color=col, dash="dash"), legendgroup=label, showlegend=False, row=5, col=1)
        fig.add_scatter(x=d.hour, y=(d.action == "CHARGE").astype(int) * 0.8 + (d.shield_active.astype(int)) * 0.2, name=label, mode="lines",
                        line=dict(color=col, shape="hv"), legendgroup=label, showlegend=False, row=6, col=1)
    fig.add_hline(y=72, line_dash="dot", line_color="#e34948", row=1, col=1)
    fig.add_hline(y=62, line_dash="dot", line_color="#e34948", row=2, col=1)
    fig.update_layout(height=1000, margin=dict(l=40, r=10, t=40, b=30), legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, width="stretch")

    log = pd.concat([df.assign(policy=label) for label, (df, _) in runs.items()])
    log = log[(log.hour >= lo) & (log.hour < hi)]
    st.download_button("Download log (CSV)", log.to_csv(index=False).encode(), f"o3_replay_r{st.session_state.regime}_{year}.csv", "text/csv")
    st.caption(f"{BANNER} Replay of a stored {year} year, regime {st.session_state.regime} ({MEDOIDS[st.session_state.regime]['place']}).")


def live_view() -> None:
    st.write("One simulator hour per click, fed by the provider's current-hour value (a nowcast; the stored-year GHI is an hourly "
             "mean). Needs internet; failures fall back to persistence, then to a synthetic day, and are flagged below.")
    algo = st.selectbox("Policy", ["b1", "b0", "b2"], format_func=lambda k: POLICY_LABELS[k])
    if "live" not in st.session_state or st.session_state.get("live_key") != (algo, st.session_state.regime):
        r = LiveDemoRunner(algo=algo, regime_id=st.session_state.regime, mode=DemoMode.LIVE, log_dir=ROOT / "results" / "demo_logs")
        r.reset()
        st.session_state.live, st.session_state.live_key, st.session_state.live_rows = r, (algo, st.session_state.regime), []
    c1, c2 = st.columns(2)
    if c1.button("Advance one hour (fetch weather)", type="primary"):
        st.session_state.live_rows.append(st.session_state.live.step_once())
    n = c2.number_input("Fast-forward: replay the last N hours of stub weather", 0, 168, 0)
    if n and c2.button("Fast-forward"):
        for _ in range(int(n)):
            st.session_state.live_rows.append(st.session_state.live.step_once())
    rows = st.session_state.live_rows
    if not rows:
        st.info("No steps yet.")
        return
    df = pd.DataFrame(rows)
    if (~df.quality_ok).any():
        st.warning("Some steps used a fallback (stale / failed provider). See the `note` column.")
    st.line_chart(df.set_index("step")[["T_top_C", "T_bottom_C", "T_pcm_C"]])
    st.dataframe(df[["timestamp", "source", "quality_ok", "latency_s", "ghi_wm2", "t_amb_C", "action", "E_del45_Wh", "T_top_C", "T_pcm_C", "f_melt", "note"]].tail(24),
                 hide_index=True, width="stretch")
    st.download_button("Download live log (CSV)", df.to_csv(index=False).encode(), "o3_live_log.csv", "text/csv")


banner()
st.title("Objective 3 — adaptive PCM solar water heating controller")
st.sidebar.header("Setup")
st.session_state.regime = st.sidebar.radio("Regime / medoid", [0, 1, 2], index=1,
                                           format_func=lambda r: f"{r} — {MEDOIDS[r]['place']} ({MEDOIDS[r]['point']})")
mode = st.sidebar.radio("Mode", ["Replay", "Live"])
st.sidebar.caption("50 L tank · 1.5 m² collector · O2's released design for this regime. Test years (2006–2010) are never shown.")
(replay_view if mode == "Replay" else live_view)()
banner()
