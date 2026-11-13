"""
app/streamlit_app.py -- Objective 3 Uttarakhand, milestone M7: software-in-the-loop demo (replay + live weather).

    streamlit run app/streamlit_app.py          (from the uttarakhand/ folder, with the .venv active)

Every view carries the banner "Software-in-the-loop simulation. Not field validation." The tank and PCM state are
simulated by the unchanged O2 kernel; only the weather is real. Replay is limited to train / validate years (the O3
test years and O2's used years are refused by o3/roles_o3.py). Live uses Open-Meteo at the chosen medoid.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c          # noqa: E402
from o3 import live                   # noqa: E402

st.set_page_config(page_title="O3 Uttarakhand SIL demo", layout="wide")
st.error(f"**{live.BANNER}** Tank and PCM temperatures, melt fraction and stored energy come from the O2 simulator, "
         "not from sensors.", icon="⚠️")
st.title("Objective 3 - DRL bypass controller, Uttarakhand")
st.caption("Hourly decision: enable or bypass the collector loop. O2's in-kernel shield (top node 72 C, PCM 62 C) "
           "always applies; hard limits 75 C water / 65 C PCM. Delivery target 50 C, window 18:00-09:00 IST.")

COLORS = {"B0": "#6b7280", "B1": "#d97706", "PPO": "#2563eb", "water_only_twin": "#9ca3af"}


@st.cache_resource(show_spinner=False)
def get_controllers(regime: int):
    return live.controllers(regime)


@st.cache_data(show_spinner="Simulating...")
def run_replay(regime: int, year: int, scenario: str):
    frame = live.replay_frame(regime, year)
    return live.simulate(regime, frame, get_controllers(regime), scenario)


def run_frame(regime: int, frame: pd.DataFrame, scenario: str, nowcast: bool = True):
    """Live sessions and re-runs of saved live logs: the policy sees a persistence nowcast (plan 3.2)."""
    return live.simulate(regime, frame, get_controllers(regime), scenario, nowcast=nowcast)


def plot_runs(runs: dict, t0=None, t1=None):
    """Plan 15.9 panels: top / bottom / PCM temperature, melt fraction, draw, action, shield status, cumulative delivered
    energy, over-limit time (+ weather). PCM temperature and melt fraction are simulator state, not sensor values."""
    names = [n for n in runs if n != "water_only_twin"]
    titles = ["Weather: GHI (W/m2)", "Water temperature (C): top solid, bottom dashed",
              "PCM temperature (C) - simulator state", "PCM melt fraction (-) - simulator state",
              "Hot-water draw (kg/h)", "Action (1 enable / 0 bypass)",
              "Shield active (h/h) and time over hard limit (h/h, red dotted)", "Cumulative delivered energy to 50 C (kWh)"]
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.025, subplot_titles=titles,
                        row_heights=[0.10, 0.17, 0.13, 0.11, 0.09, 0.11, 0.13, 0.16])
    ref = runs[names[0]]
    sel = slice(None) if t0 is None else (ref["local_time"] >= t0) & (ref["local_time"] < t1)
    r0 = ref.loc[sel]
    fig.add_trace(go.Scatter(x=r0.local_time, y=r0.GHI_Wm2, name="GHI", line=dict(color="#f59e0b"), showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=r0.local_time, y=r0.draw_kg, name="draw", marker_color="#0ea5e9", showlegend=False), 5, 1)
    for n, df in runs.items():
        d = df.loc[sel]
        c = COLORS.get(n.split("_seed")[0], "#10b981")
        twin = n == "water_only_twin"
        dash = "dot" if twin else None
        fig.add_trace(go.Scatter(x=d.local_time, y=d.T_top, name=n, legendgroup=n, line=dict(color=c, dash=dash)), 2, 1)
        fig.add_trace(go.Scatter(x=d.local_time, y=d.T_bottom, name=n, legendgroup=n, showlegend=False,
                                 line=dict(color=c, dash="dash", width=1)), 2, 1)
        if not twin:
            fig.add_trace(go.Scatter(x=d.local_time, y=d.T_pcm, name=n, legendgroup=n, showlegend=False,
                                     line=dict(color=c)), 3, 1)
            fig.add_trace(go.Scatter(x=d.local_time, y=d.f_melt, name=n, legendgroup=n, showlegend=False,
                                     line=dict(color=c)), 4, 1)
            fig.add_trace(go.Scatter(x=d.local_time, y=d.action, name=n, legendgroup=n, showlegend=False,
                                     line=dict(color=c, shape="hv")), 6, 1)
            fig.add_trace(go.Scatter(x=d.local_time, y=(d.shield_water_s + d.shield_pcm_s) / 3600, name=n,
                                     legendgroup=n, showlegend=False, line=dict(color=c, shape="hv")), 7, 1)
            fig.add_trace(go.Scatter(x=d.local_time, y=d.over_limit_s / 3600, name=n, legendgroup=n, showlegend=False,
                                     line=dict(color="#dc2626", shape="hv", dash="dot")), 7, 1)
        fig.add_trace(go.Scatter(x=d.local_time, y=d.Edel_Wh.cumsum() / 1000, name=n, legendgroup=n, showlegend=False,
                                 line=dict(color=c, dash=dash)), 8, 1)
    for y, r in ((72, 2), (75, 2), (62, 3), (65, 3)):
        fig.add_hline(y=y, line=dict(color="#dc2626", width=1, dash="dash" if y in (72, 62) else "solid"), row=r, col=1)
    fig.update_layout(height=1250, margin=dict(l=40, r=20, t=40, b=20), legend=dict(orientation="h", y=1.03))
    return fig


def show_results(runs: dict, regime: int, key: str, t0=None, t1=None):
    tbl = live.metrics_table(runs).rename(columns={"E_del_annual_kWh": "E_del_total_kWh"})
    st.subheader("Metrics over the simulated period")
    st.dataframe(tbl.set_index("controller").round(3), use_container_width=True)
    st.plotly_chart(plot_runs(runs, t0, t1), use_container_width=True)
    log = pd.concat([df.assign(controller=n) for n, df in runs.items()], ignore_index=True)
    keep = ["controller", "timestamp_utc", "local_time", "GHI_Wm2", "T_amb_C", "action", "loop_available", "T_top",
            "T_bottom", "T_mean", "T_pcm", "f_melt", "draw_kg", "Edel_Wh", "Q_coll_Wh", "shield_water_s",
            "shield_pcm_s", "over_limit_s", "guard_hit", "bypass_switch"] + [c for c in ("missing", "stale_fill") if c in log]
    st.download_button("Download hourly log (CSV)", log[keep].to_csv(index=False).encode(), f"o3_uk_r{regime}_{key}.csv",
                       "text/csv", key=f"dl_{key}")


# ---------------------------------------------------------------------------------------------------- sidebar
cfg = o3c.load()
regime = st.sidebar.selectbox("Regime (medoid point)", [0, 1, 2, 3], index=cfg["regimes"]["primary"],
                              format_func=lambda k: live.REGIME_LABELS[k])
scenario = st.sidebar.selectbox("Reset state", cfg["reset_scenarios"], index=0)
ctrls = get_controllers(regime)
names = [c.name for c in ctrls]
st.sidebar.write("Controllers:", ", ".join(names) + " + water-only twin")
if regime in cfg["regimes"]["pcm"] and "PPO" not in names:
    st.sidebar.warning("No frozen PPO policy for this regime yet (M4 not run).")
if regime in cfg["regimes"]["water_only"]:
    st.sidebar.info("Regime 3: O2 found PCM gives no gain here (4 capsules). No PPO is trained; baselines only.")

tab_replay, tab_live, tab_log = st.tabs(["Replay (recorded weather)", "Live weather", "Re-run a saved log"])

with tab_replay:
    allowed = cfg["years"]["train"] + cfg["years"]["validate"]
    c1, c2 = st.columns(2)
    year = c1.selectbox("Year (train / validate roles only)", allowed, index=len(cfg["years"]["train"]))
    rng = c2.date_input("Show dates", (date(year, 6, 1), date(year, 6, 7)), min_value=date(year, 1, 1),
                        max_value=date(year, 12, 31))
    st.caption("The whole year is simulated from 1 January (so the state on the shown dates is warmed up); the metrics "
               "table is for the full year and matches the M3 / M5 validation numbers for the same controller.")
    runs = run_replay(regime, year, scenario)
    t0, t1 = (pd.Timestamp(rng[0]), pd.Timestamp(rng[-1]) + pd.Timedelta(days=1)) if rng else (None, None)
    show_results(runs, regime, f"replay_{year}", t0, t1)

with tab_live:
    lat, lon = live.MEDOID_COORDS[regime]
    st.markdown(f"Provider **Open-Meteo** forecast API (hourly `shortwave_radiation`, `temperature_2m`) at the medoid "
                f"grid point {lat} N, {lon} E. Only completed hours are used. Missing or stale data (> "
                f"{live.STALE_AFTER_H} h old) are filled by **persistence** of the last value and flagged.")
    st.caption("Decisions use a **nowcast**: the policy sees the last completed hour's GHI and air temperature, as a "
               "real controller would; the simulator then runs the hour with its actual weather (plan 3.2).")
    st.caption("Open-Meteo radiation is a model product on a ~11 km grid, not NASA POWER (the training data); a live "
               "run is a pipeline demonstration, not a performance claim.")
    st.markdown("[Weather data by Open-Meteo.com](https://open-meteo.com/), licensed "
                "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Changes made:** hour-end timestamps "
                "shifted to hour-start, negative GHI clipped to 0, missing or stale hours filled by persistence "
                "(flagged). Free tier, non-commercial / research use only.")
    c1, c2, c3 = st.columns(3)
    past = c1.slider("Fast-forward: hours of history to start from", 24, 24 * 7, 72, 24)
    auto = c2.toggle("Auto-advance each hour", value=False)
    go_btn = c3.button("Fetch latest weather", type="primary")
    ss = st.session_state
    ss.setdefault("live_fetches", [])

    def do_fetch():
        fr, rec = live.fetch_live(regime, past_days=max(1, -(-past // 24)))
        ss.live_fetches.append(rec)
        if not rec["ok"]:
            if "live_frame" in ss and ss.get("live_regime") == regime:
                fr, n = live.extend_stale(ss.live_frame)
                ss.live_frame, ss.live_stale = fr, n
            return
        fr = fr.iloc[-past:].reset_index(drop=True)
        fr, n = live.extend_stale(fr)
        ss.live_frame, ss.live_regime, ss.live_stale = fr, regime, n

    if go_btn or (auto and "live_frame" not in ss):
        do_fetch()

    @st.fragment(run_every="15m" if auto else None)
    def live_view():
        if auto and "live_frame" in ss:
            last = pd.Timestamp(ss.live_frame.timestamp_utc.iloc[-1])
            if pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1) > last:
                do_fetch()                                   # a new completed hour exists -> advance
        if "live_frame" not in ss or ss.get("live_regime") != regime:
            st.info("Press 'Fetch latest weather' to start a live session for this regime.")
            return
        rec = ss.live_fetches[-1]
        fr = ss.live_frame
        m = st.columns(5)
        m[0].metric("Fetch", "OK" if rec["ok"] else "FAILED")
        m[1].metric("Latency (s)", rec.get("latency_s"))
        m[2].metric("Hours simulated", len(fr))
        m[3].metric("Missing / filled hours", int(fr["missing"].sum()))
        m[4].metric("Last hour (IST)", pd.Timestamp(fr.timestamp_utc.iloc[-1]).tz_convert(live.IST).strftime("%d %b %H:00"))
        if not rec["ok"]:
            st.warning(f"Live fetch failed ({rec.get('error')}). Showing the last session extended by persistence.")
        if ss.get("live_stale"):
            st.warning(f"Data stale: {ss.live_stale} hour(s) appended by persistence fallback (flagged in the log).")
        runs = run_frame(regime, fr, scenario)
        show_results(runs, regime, "live")
        st.caption(f"Raw provider JSON of every fetch is stored unmodified in results/demo_logs/raw/ "
                   f"(last: {Path(rec['raw_file']).name if rec.get('raw_file') else 'none - fetch failed'}).")
        if st.button("Save session log to results/demo_logs/"):
            p = live.save_live_log(regime, fr, ss.live_fetches)
            st.success(f"Saved {p.name} (+ weather CSV) for an exact re-run.")
        with st.expander("Fetch records (provider, timestamps, timezone, latency)"):
            st.json([{k: v for k, v in r.items() if k != "raw_json"} for r in ss.live_fetches])

    live_view()

with tab_log:
    st.markdown("Upload a weather CSV saved by the live tab (`results/demo_logs/live_*_weather.csv`). It is pushed "
                "through exactly the same code path, so the numbers reproduce the live session.")
    up = st.file_uploader("Weather log CSV", type="csv")
    if up is not None:
        fr = live.load_weather_log(up)
        show_results(run_frame(regime, fr, scenario), regime, "rerun")
