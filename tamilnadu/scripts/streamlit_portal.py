"""
Phase O3-7 (portal) — Streamlit demo portal for the Objective 3 controller.

Two tabs:
  - Live Controller: pick a policy / regime / weather window, run it through
    the frozen physics, and watch what the pump actually did hour by hour,
    compared against an always-on pump on the same weather.
  - Deployment & ML Models: thermal efficiency, comparison against the
    baseline controllers, inference latency/memory, sensor-fault robustness
    and an overview of the six algorithms — for any regime.

Every chart and table is clickable: click a point / bar / row and a short
2-line explanation appears under it.

SOFTWARE-IN-THE-LOOP ONLY — thermal state is simulated, never measured.

Run from the tamilnadu folder:
    streamlit run scripts/streamlit_portal.py
"""

import datetime
import json
import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = pathlib.Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# importing this module also puts the Objective-2 code on sys.path
from scripts.make_live_dashboard import (
    run_rollout, run_rollout_live, fetch_open_meteo, REGIME_COORDS,
    TEST_YEARS, model_variant_available,
)

RESULTS_DIR = REPO_ROOT / "results"

# validated, colourblind-safe palette
C_SAC, C_DQN = "#1baf7a", "#2a78d6"
C_B0, C_B1 = "#8f8e88", "#b5b4ae"
C_BYPASS = "#eb6834"

# what the pump PHYSICALLY did (CHARGE only permits it; the simulator cuts the
# pump itself at night / when the collector has no net gain)
MODE_COLOR = {"collect": "#2a78d6", "wait": "#b4b7bd", "hold": "#1baf7a", "bypass": "#eb6834"}
MODE_LABEL = {"collect": "COLLECTING SOLAR HEAT", "wait": "STANDBY — PUMP OFF",
              "hold": "HELD OFF (in sun)", "bypass": "BYPASS — OVERHEAT PROTECTION"}
MODE_SHORT = {"collect": "Collecting", "wait": "Standby", "hold": "Held off", "bypass": "Bypass"}
MODE_DEF = {
    "collect": "The pump runs and moves solar heat from the collector into the tank and PCM. "
               "This only happens when the collector is hotter than the tank.",
    "wait": "The pump is off because there is no useful sun (night / cloud) or the collector "
            "cannot add net heat. The stored heat keeps serving the hot-water draw.",
    "hold": "Sun is available but the controller chose to pause the pump to keep the tank and "
            "PCM safely below their limits.",
    "bypass": "The collector is isolated to stop overheating — either chosen by the controller "
              "or forced by the safety shield at the 72 °C / 62 °C guard bands.",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]      # 2021-2023: no leap years
MONTH_START = [sum(_MDAYS[:i]) for i in range(12)]


def pump_mode(r) -> str:
    if r["exe"] == 2 and r["overridden"]:
        return "bypass"
    if r["ghi"] < 50 and r["pump"] < 0.5:
        return "wait"
    if r["exe"] == 2:
        return "bypass"
    if r["exe"] == 1:
        return "hold"
    return "collect" if r["pump"] >= 0.5 else "wait"


def hour_reason(r) -> str:
    m, tw, g = r["mode"], f"{r['tw']:.0f}", f"{r['ghi']:.0f}"
    if m == "collect":
        return (f"Sun is {g} W/m² and the collector is hotter than the tank ({tw} °C), "
                f"so the pump runs and stores the heat.")
    if m == "bypass" and r["overridden"]:
        return (f"Water/PCM reached the guard band ({tw} °C water), so the safety shield "
                f"isolated the collector.")
    if m == "bypass":
        return (f"The controller isolated the collector early: tank {tw} °C with {g} W/m² of "
                f"sun means it is heading for overheating.")
    if m == "hold":
        return (f"Sun is available ({g} W/m²) but the controller paused the pump to keep the "
                f"tank ({tw} °C) and PCM ({r['tpcm']:.0f} °C) below their limits.")
    if r["ghi"] < 50:
        return (f"No useful sun ({g} W/m²), so the pump is off by itself and stored heat serves "
                f"the {r['demand_L']:.1f} L draw.")
    return f"Sun is {g} W/m² but the collector cannot add net heat to a {tw} °C tank."


def show_chart(fig, key, explain_fn,
               hint=":material/ads_click: Click any point or bar for a short explanation."):
    """Render a Plotly chart; clicking a point shows explain_fn(point) below."""
    ev = st.plotly_chart(fig, width="stretch", key=key,
                         on_select="rerun", selection_mode="points")
    try:
        pts = ev.selection.points
    except Exception:
        pts = []
    if pts:
        try:
            st.info(explain_fn(pts[0]))
        except Exception:
            st.info("Clicked point has no explanation attached.")
    else:
        st.caption(hint)


st.set_page_config(page_title="O3 Controller Portal", layout="wide")

st.title("Objective 3 — Climate-Adaptive PCM Controller Portal")
st.caption(
    "Tamil Nadu solar water heating · safety-shielded DRL · "
    "**SOFTWARE-IN-THE-LOOP** — thermal state is simulated, not measured. "
    "Field validation is Objective 4."
)

tab_live, tab_deploy = st.tabs(["Live Controller", "Deployment & ML Models"])

# =============================================================================
# TAB 1 — Live Controller
# =============================================================================
with tab_live:
    with st.sidebar:
        st.header("Run configuration")
        algo = st.selectbox("Policy", ["sac", "dqn"], index=0,
                            format_func=lambda a: {"sac": "SAC (selected)",
                                                   "dqn": "DQN (max-delivery alt.)"}[a])
        regime = st.selectbox("Regime", [0, 1, 2], index=0,
                              format_func=lambda r: {
                                  0: "0 — coastal/interior (PureTemp 58)",
                                  1: "1 — elevated/Nilgiris-adjacent (RT57HC)",
                                  2: "2 — coastal metro (RT57HC)"}[r])

        smartfc_ready = model_variant_available(algo, regime, "smartfc")
        mode = st.radio("Weather source", ["Historical replay", "LIVE (Open-Meteo)"])

        variant_options = ["Original (no forecast, V1)"]
        if smartfc_ready:
            variant_options.append("Realistic forecast (smart persistence, V3)")
        variant = st.radio("Model variant", variant_options, index=0)
        use_smartfc = variant.startswith("Realistic forecast")

        if not smartfc_ready:
            st.caption(":material/hourglass_empty: Realistic-forecast model for this policy/regime is not available.")
        if mode == "LIVE (Open-Meteo)" and use_smartfc:
            st.warning("Smart-persistence forecast needs a clear-sky irradiance "
                       "column that Open-Meteo doesn't provide — not supported in "
                       "LIVE mode yet. Falling back to zero forecast fields.")

        year, start_day, days, past_days, forecast_days = 2021, 90, 10, 2, 3
        if mode == "Historical replay":
            year = st.selectbox("Year", [2021, 2022, 2023], index=0,
                                help="Validation years only — test years "
                                     "(2011-2015) are locked and refused.")
            month = st.selectbox("Start month", MONTHS, index=3)
            mi = MONTHS.index(month)
            dom = st.slider("Start day of month", 1, _MDAYS[mi], 1)
            days = st.slider("Window length (days)", 1, 30, 10)
            start_day = min(MONTH_START[mi] + dom - 1, 365 - days)
            d0 = datetime.date(year, 1, 1) + datetime.timedelta(days=start_day)
            d1 = d0 + datetime.timedelta(days=days - 1)
            st.caption(f"Window: **{d0:%d %b %Y} → {d1:%d %b %Y}**")
        else:
            past_days = st.slider("Past days", 0, 5, 2)
            forecast_days = st.slider("Forecast days", 1, 7, 3)

        compare = st.checkbox("Compare SAC vs DQN side-by-side", value=False,
                              help="Runs both policies over the same window.")
        autoplay_speed = st.select_slider(
            "Scrub step size (for the Hour slider below)", [1, 3, 6, 12, 24], value=1)

        run_btn = st.button("Run", type="primary", icon=":material/play_arrow:", width="stretch")

    if "recs" not in st.session_state:
        st.session_state.recs = None
        st.session_state.recs2 = None
        st.session_state.base = None
        st.session_state.meta = None
        st.session_state.live_ctx = None

    if run_btn:
        if mode == "Historical replay" and year in TEST_YEARS:
            st.error("That year is a locked test year and is refused.")
            st.stop()
        with st.spinner("Running the policy (and an always-on pump for comparison) "
                        "through the frozen physics..."):
            try:
                live_ctx = None
                if mode == "LIVE (Open-Meteo)":
                    # fetch ONCE and reuse for every run so all see identical weather
                    lat, lon, label = REGIME_COORDS.get(regime, REGIME_COORDS[0])
                    wx_df, iso_times, fetched = fetch_open_meteo(
                        lat, lon, past_days, forecast_days)
                    live_ctx = {"wx_df": wx_df, "iso_times": iso_times,
                               "fetched": fetched, "lat": lat, "lon": lon,
                               "label": label, "now_idx": past_days * 24}

                def _run_one(a, use_sfc, force_charge=False):
                    if use_sfc:
                        tag, obsv, fcm = "smartfc", "V3", "smart"
                    else:
                        tag, obsv, fcm = None, "V1", "perfect"
                    if live_ctx is not None:
                        return run_rollout_live(
                            a, regime, live_ctx["wx_df"], live_ctx["iso_times"],
                            model_tag=tag, obs_version=obsv, forecast_mode=fcm,
                            force_charge=force_charge)
                    return run_rollout(a, regime, year, start_day * 24, days * 24,
                                       model_tag=tag, obs_version=obsv, forecast_mode=fcm,
                                       force_charge=force_charge)

                st.session_state.recs = _run_one(algo, use_smartfc)
                st.session_state.base = _run_one(algo, False, force_charge=True)
                st.session_state.recs2 = (_run_one("dqn" if algo == "sac" else "sac", False)
                                          if compare else None)
                st.session_state.live_ctx = live_ctx
            except Exception as e:
                st.error(f"Run failed: {e}. If using LIVE mode, check network "
                         f"access to api.open-meteo.com, or try Historical replay.")
                st.stop()
        desc = (f"historical year {year}, {d0:%d %b} → {d1:%d %b}"
                if mode == "Historical replay" else "present-day weather")
        st.session_state.meta = {"mode": mode, "algo": algo, "regime": regime,
                                 "use_smartfc": use_smartfc, "desc": desc,
                                 "compare": compare}

    recs = st.session_state.recs
    if recs is None:
        st.info("Configure a run in the sidebar and press **Run**.")
    else:
        meta = st.session_state.meta
        live_ctx = st.session_state.live_ctx
        df = pd.DataFrame(recs)
        df["mode"] = df.apply(pump_mode, axis=1)
        df["why"] = df.apply(hour_reason, axis=1)
        algo_name = {"sac": "SAC", "dqn": "DQN"}[meta["algo"]]
        variant_label = ("Realistic forecast (smart persistence)" if meta.get("use_smartfc")
                         else "Original (no forecast)")
        is_live = meta["mode"].startswith("LIVE") and live_ctx is not None

        if is_live:
            st.success(f"LIVE — {algo_name} ({variant_label}) · regime {meta['regime']} · "
                       f"{live_ctx['label']} · weather fetched {live_ctx['fetched']}")
        else:
            st.success(f"REPLAY — {algo_name} ({variant_label}) · regime {meta['regime']} · "
                       f"{meta['desc']}")

        last = df.iloc[-1]
        mix = df["mode"].value_counts()
        n_over = int(df["overridden"].sum())
        base = st.session_state.base
        b = pd.DataFrame(base).iloc[-1] if base else None

        c1, c2, c3, c4, c5 = st.columns(5)
        if b is not None:
            d_del = 100 * (last["cum_del"] - b["cum_del"]) / b["cum_del"]
            cut = (100 * (b["cum_shield"] - last["cum_shield"]) / b["cum_shield"]
                   if b["cum_shield"] else 0.0)
            c1.metric("Hot water delivered", f"{last['cum_del']:.1f} kWh",
                      f"{d_del:+.1f}% vs always-on ({b['cum_del']:.1f})", delta_color="off",
                      help="Useful heat delivered to the hot-water draw over the window.")
            c2.metric("Overheat shield hits", f"{int(last['cum_shield']):,}",
                      f"{cut:.0f}% fewer than always-on ({int(b['cum_shield']):,})",
                      help="Simulator sub-steps where the safety shield had to isolate the "
                           "collector. Fewer = less wasted heat and less thermal stress.")
        else:
            c1.metric("Hot water delivered", f"{last['cum_del']:.1f} kWh")
            c2.metric("Overheat shield hits", f"{int(last['cum_shield']):,}")
        c3.metric("Hard-limit breaches", int(last["cum_safety"]),
                  help="Water above 75 °C or PCM above 65 °C. Target is 0.")
        c4.metric("Pump actually ran", f"{df['pump'].sum():.0f} / {len(df)} h",
                  help="Real pump run-time from the simulator, not the command requested.")
        c5.metric("Controller stepped in",
                  f"{int(mix.get('hold', 0) + mix.get('bypass', 0))} h",
                  f"{n_over} forced by shield", delta_color="off",
                  help="Hours the pump was held off or the collector bypassed in sunshine.")
        st.caption(
            "CHARGE only *permits* the pump; the simulator switches it off itself at "
            "night or when the collector has no net gain. Hours are therefore coloured "
            "by what the pump physically did. 'Efficient' = the same hot water with far "
            "fewer overheat shield hits, not extra kWh.")

        st.subheader("Hour-by-hour playback")
        k = st.slider("Hour", 0, len(df) - 1, min(24, len(df) - 1),
                      step=int(autoplay_speed))
        row = df.iloc[k]

        badge_color = MODE_COLOR[row["mode"]]
        badge_text = MODE_LABEL[row["mode"]]
        text_color = "#1a1a1a" if row["mode"] == "wait" else "white"
        shield_html = (
            '<div style="background:#fff3e0;color:#8a4b00;font-weight:600;padding:8px 14px;'
            'border-radius:8px;border:1px solid #f0c38a;">SAFETY SHIELD FORCED THIS</div>'
            if row["overridden"] else "")
        # Single-line HTML: an empty shield_html used to leave a whitespace-only line, which
        # Markdown treats as a blank line, so the indented rest rendered as a code block.
        st.markdown(
            '<div style="display:flex;gap:16px;align-items:center;margin:8px 0;flex-wrap:wrap;">'
            f'<div style="background:{badge_color};color:{text_color};font-weight:700;'
            f'font-size:20px;padding:10px 22px;border-radius:10px;">{badge_text}</div>'
            f'{shield_html}'
            f'<div style="color:#8a8a85;font-weight:600;">{row["ts"]}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write(row["why"])

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Sun (GHI)", f"{row['ghi']:.0f} W/m²")
        k2.metric("Outside", f"{row['tamb']:.1f} °C")
        k3.metric("Tank water", f"{row['tw']:.1f} °C")
        k4.metric("PCM", f"{row['tpcm']:.1f} °C")
        k5.metric("PCM melted", f"{row['melt']:.0f} %")
        k6.metric("Hot water drawn", f"{row['demand_L']:.1f} L")

        now_idx = min(live_ctx["now_idx"], len(df) - 1) if is_live else None

        # ---- temperature chart ----
        fig_t = go.Figure()
        if is_live:
            fig_t.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor="#2a78d6",
                            opacity=0.06, line_width=0,
                            annotation_text="forecast →", annotation_position="top left",
                            annotation_font_size=10, annotation_font_color="#5c5b57")
        fig_t.add_scatter(x=df["k"], y=df["tw"], name="Tank water", line=dict(color=C_DQN),
                          hovertemplate="%{customdata}<br>water %{y:.1f} °C<extra></extra>",
                          customdata=df["ts"])
        fig_t.add_scatter(x=df["k"], y=df["tpcm"], name="PCM", line=dict(color=C_BYPASS),
                          hovertemplate="%{customdata}<br>PCM %{y:.1f} °C<extra></extra>",
                          customdata=df["ts"])
        for y, c, d, lab in [(45, "#1baf7a", "dot", "45° delivery target"),
                             (72, "#eb6834", "dash", "72° water guard"),
                             (75, "#b01d0f", "solid", "75° water limit")]:
            fig_t.add_hline(y=y, line=dict(color=c, dash=d, width=1),
                            annotation_text=lab, annotation_position="top right",
                            annotation_font_size=10, annotation_font_color=c)
        if is_live:
            fig_t.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"))
        fig_t.add_vline(x=k, line=dict(color="black", width=2))
        fig_t.update_layout(
            title="Tank water and PCM temperature (°C) — stay under the guard lines"
            + ("  (shaded = forecast)" if is_live else ""),
            height=320, template="plotly_white", margin=dict(l=40, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.14), xaxis_title="hour in window")

        # ---- pump-activity strip (one coloured bar per hour) ----
        fig_r = go.Figure(go.Bar(
            x=df["k"], y=[1] * len(df), marker_color=[MODE_COLOR[m] for m in df["mode"]],
            marker_line_width=0, customdata=df[["ts", "mode"]].values,
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        fig_r.add_vline(x=k, line=dict(color="black", width=2))
        fig_r.update_layout(title="What the pump did each hour (blue collecting · grey standby · "
                                  "green held off · orange bypass)",
                            height=120, template="plotly_white", bargap=0,
                            margin=dict(l=40, r=20, t=34, b=10),
                            yaxis=dict(visible=False), showlegend=False)

        # ---- GHI chart ----
        fig_g = go.Figure()
        if is_live:
            fig_g.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor="#2a78d6",
                            opacity=0.06, line_width=0)
        fig_g.add_scatter(x=df["k"], y=df["ghi"], fill="tozeroy",
                          line=dict(color="#e0a21a"), name="GHI",
                          hovertemplate="%{customdata}<br>%{y:.0f} W/m²<extra></extra>",
                          customdata=df["ts"])
        if is_live:
            fig_g.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"),
                            annotation_text="now", annotation_position="top")
        fig_g.add_vline(x=k, line=dict(color="black", width=2))
        fig_g.update_layout(title="Sunshine available (GHI, W/m²)", height=220,
                            template="plotly_white", margin=dict(l=40, r=20, t=40, b=30),
                            xaxis_title="hour in window")

        # ---- per-day pump hours ----
        df["daylab"] = df["ts"].apply(lambda s: s.rsplit(" ", 1)[0])
        day_order = list(dict.fromkeys(df["daylab"]))
        fig_d = go.Figure()
        for m in ["collect", "hold", "bypass", "wait"]:
            cnts = df[df["mode"] == m].groupby("daylab").size()
            fig_d.add_bar(name=MODE_SHORT[m], x=day_order,
                          y=[int(cnts.get(d, 0)) for d in day_order],
                          marker_color=MODE_COLOR[m])
        fig_d.update_layout(barmode="stack", title="What the pump did each day (hours)",
                            height=300, template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=30),
                            legend=dict(orientation="h", y=1.15))

        def _ex_hour(p):
            r = df.iloc[int(p["x"])]
            return (f"**{r['ts']}** — {MODE_SHORT[r['mode']]}: water {r['tw']:.1f} °C, "
                    f"PCM {r['tpcm']:.1f} °C, sun {r['ghi']:.0f} W/m².\n\n{r['why']}")

        def _ex_ghi(p):
            r = df.iloc[int(p["x"])]
            if r["ghi"] < 50:
                return (f"**{r['ts']}** — sun is only {r['ghi']:.0f} W/m².\n\n"
                        f"Below ~50 W/m² the collector can't add heat, so the pump idles.")
            return (f"**{r['ts']}** — sun is {r['ghi']:.0f} W/m².\n\n"
                    f"More sun means faster heating; around midday it pushes the tank toward "
                    f"the guard band, which is when the controller must step in.")

        def _ex_day(p):
            d = p["x"]
            sub = df[df["daylab"] == d]
            parts = ", ".join(f"{MODE_SHORT[m]} {int((sub['mode'] == m).sum())} h"
                              for m in ["collect", "hold", "bypass", "wait"])
            return (f"**{d}** — {parts}.\n\nThe pump only needs to be held off or bypassed "
                    f"for a few hot midday hours; the rest is normal collecting or standby.")

        col1, col2 = st.columns([2, 1])
        with col1:
            show_chart(fig_t, "fig_t", _ex_hour,
                       ":material/ads_click: Click a point on a line to see what the controller did that hour and why.")
            show_chart(fig_r, "fig_r", _ex_hour,
                       ":material/ads_click: Click a coloured bar to see what the pump did that hour.")
            show_chart(fig_g, "fig_g", _ex_ghi)
            if is_live:
                import plotly.express as px
                st.caption(
                    f":material/location_on: Live weather source: **{live_ctx['label']}** "
                    f"({live_ctx['lat']:.2f}, {live_ctx['lon']:.2f}) — Open-Meteo, "
                    f"fetched {live_ctx['fetched']}. Dotted line = now; shaded "
                    f"region = the forecast portion of the fetched window."
                )
                fig_map = px.scatter_map(
                    pd.DataFrame({"lat": [live_ctx["lat"]], "lon": [live_ctx["lon"]],
                                 "label": [live_ctx["label"]]}),
                    lat="lat", lon="lon", hover_name="label", zoom=6, height=220)
                fig_map.update_traces(marker=dict(size=14, color="#eb6834"))
                fig_map.update_layout(map_style="open-street-map",
                                      margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_map, width="stretch")
        with col2:
            fig_a = go.Figure(go.Bar(
                x=[MODE_SHORT[m] for m in mix.index], y=list(mix.values),
                marker_color=[MODE_COLOR[m] for m in mix.index]))
            fig_a.update_layout(title="Pump hours, whole window", height=300,
                                template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

            def _ex_mix(p):
                m = next(k_ for k_, v in MODE_SHORT.items() if v == p["x"])
                return (f"**{MODE_SHORT[m]}** — {int(mix.get(m, 0))} of {len(df)} hours.\n\n"
                        f"{MODE_DEF[m]}")
            show_chart(fig_a, "fig_a", _ex_mix)

            st.markdown("**Hours around the selected one** (click a row)")
            lo, hi = max(0, k - 5), min(len(df), k + 6)
            tbl = df.iloc[lo:hi][["ts", "mode", "pump", "tw", "tpcm", "ghi"]].copy()
            tbl["mode"] = tbl["mode"].map(MODE_SHORT)
            tbl = tbl.rename(columns={"ts": "time", "mode": "pump did", "pump": "pump on (fraction)",
                                      "tw": "water °C", "tpcm": "PCM °C", "ghi": "sun W/m²"})
            sel = st.dataframe(tbl, hide_index=True, width="stretch",
                               on_select="rerun", selection_mode="single-row", key="tbl_hours")
            try:
                rows = sel.selection.rows
            except Exception:
                rows = []
            if rows:
                st.info(f"**{df.iloc[lo + rows[0]]['ts']}** — {df.iloc[lo + rows[0]]['why']}")

        show_chart(fig_d, "fig_d", _ex_day)

        if meta.get("compare") and st.session_state.recs2 is not None:
            st.divider()
            other = "DQN" if meta["algo"] == "sac" else "SAC"
            st.subheader(f"Side-by-side: {other} (Original) over the same window")
            df2 = pd.DataFrame(st.session_state.recs2)
            last2 = df2.iloc[-1]
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric(f"{algo_name} delivered", f"{last['cum_del']:.1f} kWh")
            cc2.metric(f"{other} delivered", f"{last2['cum_del']:.1f} kWh")
            cc3.metric(f"{algo_name} shield hits", int(last["cum_shield"]))
            cc4.metric(f"{other} shield hits", int(last2["cum_shield"]))
            fig_cmp = go.Figure()
            fig_cmp.add_scatter(x=df["k"], y=df["cum_del"], name=f"{algo_name} delivered",
                                line=dict(color=C_SAC if algo_name == "SAC" else C_DQN))
            fig_cmp.add_scatter(x=df2["k"], y=df2["cum_del"], name=f"{other} delivered",
                                line=dict(color=C_DQN if algo_name == "SAC" else C_SAC, dash="dash"))
            if b is not None:
                fig_cmp.add_scatter(x=[r["k"] for r in base], y=[r["cum_del"] for r in base],
                                    name="always-on pump", line=dict(color=C_B0, dash="dot"))
            if is_live:
                fig_cmp.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"),
                                  annotation_text="now", annotation_position="top")
            fig_cmp.update_layout(title="Cumulative delivered energy (kWh)",
                                  height=280, template="plotly_white",
                                  margin=dict(l=40, r=20, t=40, b=30), xaxis_title="hour in window")

            def _ex_cmp(p):
                i = int(p["x"])
                a_, b_ = df.iloc[min(i, len(df) - 1)], df2.iloc[min(i, len(df2) - 1)]
                return (f"By **{a_['ts']}**: {algo_name} has delivered {a_['cum_del']:.1f} kWh and "
                        f"{other} {b_['cum_del']:.1f} kWh.\n\nThe curves rise in the daytime and "
                        f"flatten at night; a bigger gap means one policy collects or stores heat better.")
            show_chart(fig_cmp, "fig_cmp", _ex_cmp)

        st.caption(
            "This portal reuses the same rollout engine as `make_live_dashboard.py`: the "
            "frozen O2 physics + the in-environment safety shield, run with the selected "
            "policy's `best_model` checkpoint. All results are simulation."
        )

# =============================================================================
# TAB 2 — Deployment & ML Models
# =============================================================================
CTRL_DESC = {
    "B0": "Always-on pump (the Objective-2 operating point); only the safety shield protects it from overheating.",
    "B1": "Reactive rules: bypass at 70 °C / 60 °C, let PCM discharge in the evening, otherwise charge when sunny.",
    "B2": "Rule-based controller that also uses a forecast of upcoming sun.",
    "DQN": "Deep Q-network (value-based RL), shielded for safety.",
    "SAC": "Soft actor-critic (entropy-regularised RL) — the selected controller, shielded for safety.",
}

with tab_deploy:
    st.header("Deployment metrics and the six algorithms")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        d_regime = st.selectbox("Regime", [0, 1, 2], key="d_regime",
                                format_func=lambda r: f"Regime {r}")
    variants = {"Original models": ""}
    for lab, suf in (("Trained without forecast input", "_none"),
                     ("Realistic forecast (smart persistence)", "_smartfc")):
        if (RESULTS_DIR / f"deployment_metrics_r{d_regime}{suf}.json").exists():
            variants[lab] = suf
    with dcol2:
        d_variant = st.selectbox("Model set", list(variants.keys()), key="d_variant")

    metrics_path = RESULTS_DIR / f"deployment_metrics_r{d_regime}{variants[d_variant]}.json"
    if not metrics_path.exists():
        st.warning("Run `python scripts/phase9_deployment_metrics.py --regime "
                   f"{d_regime}` to generate this page's data.")
    else:
        metrics = json.loads(metrics_path.read_text())
        eff = metrics["efficiency_and_savings"]
        lat = metrics["latency_memory"]
        faults = metrics["sensor_faults"]
        ctr = eff["controllers"]
        st.caption("All numbers below are one simulated validation year (2021) for the chosen "
                   "regime. B0 = always-on pump, B1 = reactive rules, B2 = forecast rules.")

        # ---------------------------------------------------------- 1. efficiency
        st.subheader("1 · Thermal efficiency & demand coverage")
        order = [c for c in ("B0", "B1", "B2", "DQN", "SAC") if c in ctr]
        colmap = {"B0": C_B0, "B1": C_B1, "B2": "#d2d1cb", "DQN": C_DQN, "SAC": C_SAC}
        fig1 = go.Figure()
        fig1.add_bar(name="thermal efficiency %", x=order,
                     y=[ctr[c]["thermal_efficiency_pct"] for c in order],
                     marker_color=[colmap[c] for c in order])
        fig1.add_bar(name="demand coverage %", x=order,
                     y=[ctr[c]["demand_coverage_pct"] for c in order],
                     marker_color=[colmap[c] for c in order], opacity=0.45)
        fig1.update_layout(barmode="group", template="plotly_white", height=340,
                           yaxis_title="%",
                           title=f"Solar→load efficiency (GHI {eff['annual_ghi_kwh_m2']} "
                                 f"kWh/m²·yr × {eff['collector_area_m2']} m² collector)")

        def _ex1(p):
            c = p["x"]
            if p["curve_number"] == 0:
                return (f"**{c}** — {CTRL_DESC[c]}\n\nThermal efficiency {p['y']:.1f}% = hot-water "
                        f"energy delivered ÷ solar energy falling on the collector.")
            return (f"**{c}** — {CTRL_DESC[c]}\n\nDemand coverage {p['y']:.1f}% = share of the "
                    f"hot-water demand met by solar; the rest needs electric backup.")
        show_chart(fig1, "fig1", _ex1)

        # ---------------------------------------------------------- 2. comparison
        st.subheader("2 · How the DRL controllers compare with the baselines")
        st.caption("Positive = better than that baseline. Compare the two charts together: "
                   "the DRL controllers deliver slightly less than the always-on pump (B0, so "
                   "a small negative backup saving) but cut overheat shield hits sharply. "
                   "Shield hits are counted per simulator sub-step, not per hour.")
        drls = [c for c in ("DQN", "SAC") if c in ctr]
        scol1, scol2 = st.columns(2)
        with scol1:
            fig2a = go.Figure()
            for base_name, c in [("B0", C_B0), ("B1", C_B1)]:
                fig2a.add_bar(name=f"vs {base_name}", x=drls,
                              y=[ctr[d].get(f"backup_energy_saved_vs_{base_name}_kWh", 0) for d in drls],
                              marker_color=c)
            fig2a.update_layout(title="Electric backup saved (kWh/yr) — negative = needs more backup",
                                barmode="group", template="plotly_white", height=320)

            def _ex2a(p):
                base_name = ["B0", "B1"][p["curve_number"]]
                v = p["y"]
                word = "less" if v >= 0 else "more"
                return (f"**{p['x']} vs {base_name}** — needs {abs(v):.1f} kWh/yr {word} electric backup.\n\n"
                        + ("Negative here only means it delivers a little less solar heat than "
                           "an always-on pump; check the shield chart for what it gains in return."
                           if v < 0 else
                           "It delivers more solar hot water than that baseline, so less backup is needed."))
            show_chart(fig2a, "fig2a", _ex2a)
        with scol2:
            fig2b = go.Figure()
            for base_name, c in [("B0", C_B0), ("B1", C_B1)]:
                fig2b.add_bar(name=f"vs {base_name}", x=drls,
                              y=[ctr[d].get(f"shield_reduction_vs_{base_name}_pct", 0) for d in drls],
                              marker_color=c)
            fig2b.update_layout(title="Overheat shield hits reduced (%) — negative = more hits",
                                barmode="group", template="plotly_white", height=320)

            def _ex2b(p):
                base_name = ["B0", "B1"][p["curve_number"]]
                v = p["y"]
                return (f"**{p['x']} vs {base_name}** — {abs(v):.1f}% {'fewer' if v >= 0 else 'more'} "
                        f"overheat shield hits.\n\nFewer hits means less heat dumped and less "
                        f"thermal stress on the tank and PCM.")
            show_chart(fig2b, "fig2b", _ex2b)

        m1, m2 = st.columns(2)
        for col, name in ((m1, "DQN"), (m2, "SAC")):
            with col:
                if name in ctr and "delivery_vs_B0_pct" in ctr[name]:
                    st.metric(f"{name} delivery vs B0 / B1",
                              f"{ctr[name]['delivery_vs_B0_pct'] - 100:+.1f}% / "
                              f"{ctr[name]['delivery_vs_B1_pct'] - 100:+.1f}%",
                              help="Hot-water delivery relative to the always-on pump (B0) "
                                   "and the reactive-rule controller (B1).")

        # ---------------------------------------------------------- 3. latency
        st.subheader("3 · Inference latency & memory footprint (embedded-target proxy)")
        algos_lat = [a for a in ("DQN", "SAC") if a in lat]
        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            fig3a = go.Figure(go.Bar(
                x=algos_lat, y=[lat[a]["latency_mean_us"] for a in algos_lat],
                marker_color=[colmap[a] for a in algos_lat],
                error_y=dict(type="data", array=[lat[a]["latency_p95_us"] - lat[a]["latency_mean_us"]
                                                 for a in algos_lat], visible=True)))
            fig3a.update_layout(title="Mean latency per decision (µs)", height=300,
                                template="plotly_white")
            show_chart(fig3a, "fig3a", lambda p: (
                f"**{p['x']}** — {p['y']:.0f} µs per decision (error bar = 95th percentile).\n\n"
                f"The controller decides once per hour, so even a slow microcontroller has "
                f"enormous headroom."))
        with lcol2:
            fig3b = go.Figure(go.Bar(
                x=algos_lat, y=[lat[a]["weights_fp32_kB"] for a in algos_lat],
                marker_color=[colmap[a] for a in algos_lat]))
            fig3b.update_layout(title="Model weights (kB, fp32)", height=300,
                                template="plotly_white")
            show_chart(fig3b, "fig3b", lambda p: (
                f"**{p['x']}** — {p['y']:.0f} kB of weights at 32-bit precision.\n\n"
                f"Small enough to fit on a low-cost embedded controller."))
        with lcol3:
            st.markdown("**Parameters**")
            for a in algos_lat:
                st.metric(a, f"{lat[a]['n_params']:,}",
                          help="Number of trainable weights in the policy network.")
        st.caption(lat.get("note", ""))

        # ---------------------------------------------------------- 4. sensor faults
        st.subheader("4 · Robustness under sensor faults (observation corrupted, physics true)")
        fdf = pd.DataFrame(faults)
        fig4 = go.Figure()
        for a, c in [("DQN", C_DQN), ("SAC", C_SAC)]:
            sub = fdf[fdf["algo"] == a]
            if len(sub):
                fig4.add_bar(name=a, x=sub["fault"], y=sub["delivered_kWh"], marker_color=c)
        fig4.update_layout(title="Delivered energy under sensor faults (kWh/yr)",
                           barmode="group", template="plotly_white", height=360)
        fig4.update_xaxes(tickangle=25)

        def _ex4(p):
            a = fig4.data[p["curve_number"]].name
            nom = fdf[(fdf["algo"] == a) & (fdf["fault"] == "nominal")]
            nom_txt = (f" (clean sensors: {nom['delivered_kWh'].iloc[0]:.0f} kWh)"
                       if len(nom) else "")
            return (f"**{a} with fault '{p['x']}'** — delivers {p['y']:.0f} kWh/yr{nom_txt}.\n\n"
                    f"The policy is fed wrong readings, but the safety shield uses the true "
                    f"state, so safety holds and only efficiency can slip.")
        show_chart(fig4, "fig4", _ex4)
        n_viol = int(fdf["safety_violations"].sum()) if len(fdf) else 0
        st.success(f"Zero hard-safety violations across all {len(fdf)} "
                   f"(policy × fault) scenarios — the shield acts on the true "
                   f"physical state, so safety holds even on corrupted sensor "
                   f"readings." if n_viol == 0 else
                   f":material/warning: {n_viol} safety violations recorded — investigate.")
        with st.expander("Raw sensor-fault results"):
            st.dataframe(fdf, hide_index=True, width="stretch")

        st.divider()

        # ---------------------------------------------------------- ML models overview
        st.subheader("The six algorithms compared, and why two were selected")
        algo_rows = [
            ("PPO", "On-policy policy gradient", "Collapsed to always-charge (B0) — entropy bonus insufficient to keep exploring"),
            ("DQN", "Value-based, ε-greedy", "Learns anticipatory control on regime 0 only (−59% shields); collapses to B0 on regimes 1–2"),
            ("RecurrentPPO", "On-policy + LSTM", "Collapsed to always-charge — memory doesn't fix an exploration problem"),
            ("QRDQN", "Distributional value-based", "Collapsed to always-charge at this seed/weight"),
            ("MaskablePPO", "On-policy + action masking", "Essentially always-charge — mask forces a few non-charge steps only"),
            ("SAC", "Off-policy, entropy-regularised", "Selected — only controller beating always-on on all 3 regimes (97–99% of B0 delivery, −33 to −54% shields)"),
        ]
        adf = pd.DataFrame(algo_rows, columns=["Algorithm", "Family", "Outcome"])
        st.caption("Click a row for a short explanation.")
        asel = st.dataframe(adf, hide_index=True, width="stretch",
                            on_select="rerun", selection_mode="single-row", key="tbl_algos")
        try:
            arows = asel.selection.rows
        except Exception:
            arows = []
        if arows:
            r = algo_rows[arows[0]]
            st.info(f"**{r[0]}** ({r[1]}) — {r[2]}.\n\n"
                    f"\"Always-charge\" means it learned nothing better than leaving the pump on; "
                    f"it only counts as a success if it beats that with fewer shield hits.")
        st.caption(
            "Selection rule (lexicographic): zero hard safety violations → "
            "delivery ≥ 95% of B0 → maximum shield reduction, consistent across regimes. "
            "Both pass on regime 0; only SAC keeps the advantage on regimes 1–2, so SAC is selected. "
            "The two winners both use **sustained exploration** (ε-greedy for "
            "DQN, entropy regularisation + replay for SAC) — the separating "
            "factor is exploration mechanism, not algorithm family."
        )
