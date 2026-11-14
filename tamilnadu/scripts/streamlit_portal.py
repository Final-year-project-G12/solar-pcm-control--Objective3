"""
Phase O3-7 (portal) — Streamlit demo portal for the Objective 3 controller.

This is the interactive "portal" the implementation plan's section 15.9 called
for. It has two tabs:
  - Live Controller: pick a policy/regime/mode, run it through the frozen
    physics, and watch CHARGE / IDLE / BYPASS decisions, shield overrides,
    and tank state update hour by hour.
  - Deployment & ML Models: thermal efficiency, energy savings vs the
    baselines, inference latency/memory, sensor-fault robustness, an
    overview of all six algorithms and why two were selected, and a live
    status panel for the Generation-2 research-extension study (forecast
    observation, sub-hourly control ablation, reward-weight Pareto sweep,
    regime 1/2 coverage, training-time domain randomisation).

SOFTWARE-IN-THE-LOOP ONLY — thermal state is simulated, never measured.

Run from the objective3 root:
    streamlit run scripts/streamlit_portal.py

Requires: streamlit (already in this environment; `pip install streamlit`
if it is ever missing).
"""

import json
import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.make_live_dashboard import (
    run_rollout, run_rollout_live, fetch_open_meteo, REGIME_COORDS,
    TEST_YEARS, ACTION_NAME, ACTION_SHORT, model_variant_available,
)

RESULTS_DIR = REPO_ROOT / "results"

# validated, colourblind-safe palette (dataviz skill)
C_CHARGE, C_IDLE, C_BYPASS = "#2a78d6", "#1baf7a", "#eb6834"
C_SAC, C_DQN = "#1baf7a", "#2a78d6"
C_B0, C_B1 = "#8f8e88", "#b5b4ae"

st.set_page_config(page_title="O3 Controller Portal", layout="wide")

st.title("Objective 3 — Climate-Adaptive PCM Controller Portal")
st.caption(
    "Tamil Nadu solar water heating · safety-shielded DRL · "
    "**SOFTWARE-IN-THE-LOOP** — thermal state is simulated, not measured. "
    "Field validation is Objective 4."
)

tab_live, tab_deploy = st.tabs(["🔴 Live Controller", "📊 Deployment & ML Models"])

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

        # Generation-2 model variant (forecast obs + domain randomisation),
        # offered only once that background training study has produced it
        # Generation-2 tags, as actually written by scripts/train_one.py's
        # launch lanes — NOT a uniform pattern across (algo, regime):
        #   sac, regime 0 -> part of the Pareto sweep ("pareto_w10_s42")
        #   everything else -> "gen2_<algo>_r<regime>_w10_s42"
        gen2_tag = ("pareto_w10_s42" if (algo == "sac" and regime == 0)
                   else f"gen2_{algo}_r{regime}_w10_s42")
        gen2_ready = model_variant_available(algo, regime, gen2_tag)
        smartfc_ready = model_variant_available(algo, regime, "smartfc")

        mode = st.radio("Weather source", ["Historical replay", "LIVE (Open-Meteo)"])

        variant_options = ["Original (no forecast, V1)"]
        if smartfc_ready:
            variant_options.append("Realistic forecast (smart persistence, V3)")
        # Generation-2 used an unrealistic perfect-lookahead forecast and was
        # trained on pre-weather-year-fix data; retired, kept only for LIVE
        # mode where smart-forecast isn't available yet (no clear-sky column
        # from Open-Meteo — see CORRECTIONS_OBJECTIVE3.md "Still open" #3).
        if gen2_ready and mode == "LIVE (Open-Meteo)":
            variant_options.append("Generation-2 (forecast + domain-rand., retired)")

        variant = st.radio("Model variant", variant_options, index=0)
        use_smartfc = variant.startswith("Realistic forecast")
        use_gen2 = variant.startswith("Generation-2")

        if mode == "Historical replay" and smartfc_ready:
            st.caption("✅ Realistic-forecast (smart persistence) model available "
                       "for historical replay.")
        elif not smartfc_ready:
            st.caption("⏳ Realistic-forecast (smartfc) model for this policy/regime "
                       "is not available.")
        if mode == "LIVE (Open-Meteo)" and use_smartfc:
            st.warning("Smart-persistence forecast needs a clear-sky irradiance "
                       "column that Open-Meteo doesn't provide — not supported in "
                       "LIVE mode yet. Falling back to zero forecast fields "
                       "(equivalent to forecast_mode='none') for this run.")

        if mode == "Historical replay":
            year = st.selectbox("Year", [2021, 2022, 2023], index=0,
                                help="Validation years only — test years "
                                     "(2011-2015) are locked and refused.")
            start_day = st.slider("Start day of year", 1, 350, 160)
            days = st.slider("Window length (days)", 1, 30, 10)
        else:
            past_days = st.slider("Past days", 0, 5, 2)
            forecast_days = st.slider("Forecast days", 1, 7, 3)

        compare = st.checkbox("Compare SAC vs DQN side-by-side", value=False,
                              help="Runs both policies over the same window.")
        autoplay_speed = st.select_slider(
            "Scrub step size (for the Hour slider below)", [1, 3, 6, 12, 24], value=1)

        run_btn = st.button("▶ Run", type="primary", use_container_width=True)

    if "recs" not in st.session_state:
        st.session_state.recs = None
        st.session_state.recs2 = None
        st.session_state.meta = None
        st.session_state.live_ctx = None

    if run_btn:
        with st.spinner("Running the policy through the frozen physics..."):
            if mode == "Historical replay" and year in TEST_YEARS:
                st.error("That year is a locked test year and is refused.")
                st.stop()
            try:
                live_ctx = None
                if mode == "LIVE (Open-Meteo)":
                    # fetch ONCE and reuse for both runs when comparing, so
                    # the primary and comparison policy see identical weather
                    lat, lon, label = REGIME_COORDS.get(regime, REGIME_COORDS[0])
                    wx_df, iso_times, fetched = fetch_open_meteo(
                        lat, lon, past_days, forecast_days)
                    live_ctx = {"wx_df": wx_df, "iso_times": iso_times,
                               "fetched": fetched, "lat": lat, "lon": lon,
                               "label": label, "now_idx": past_days * 24}

                def _run_one(a, use_g2, use_sfc):
                    if use_g2:
                        tag, obsv, fcm = gen2_tag, "V2", "perfect"
                    elif use_sfc:
                        tag, obsv, fcm = "smartfc", "V3", "smart"
                    else:
                        tag, obsv, fcm = None, "V1", "perfect"
                    if live_ctx is not None:
                        return run_rollout_live(
                            a, regime, live_ctx["wx_df"], live_ctx["iso_times"],
                            model_tag=tag, obs_version=obsv, forecast_mode=fcm)
                    return run_rollout(a, regime, year, start_day * 24, days * 24,
                                       model_tag=tag, obs_version=obsv, forecast_mode=fcm)

                st.session_state.recs = _run_one(algo, use_gen2, use_smartfc)
                st.session_state.recs2 = _run_one(
                    "dqn" if algo == "sac" else "sac", False, False) if compare else None
                st.session_state.live_ctx = live_ctx
            except Exception as e:
                st.error(f"Run failed: {e}. If using LIVE mode, check network "
                         f"access to api.open-meteo.com, or try Historical replay.")
                st.stop()
        st.session_state.meta = {"mode": mode, "algo": algo, "regime": regime,
                                 "use_gen2": use_gen2, "use_smartfc": use_smartfc}

    recs = st.session_state.recs
    if recs is None:
        st.info("Configure a run in the sidebar and press **Run**.")
    else:
        meta = st.session_state.meta
        live_ctx = st.session_state.live_ctx
        df = pd.DataFrame(recs)
        algo_name = {"sac": "SAC", "dqn": "DQN"}[meta["algo"]]
        variant_label = ("Generation-2 (retired)" if meta["use_gen2"] else
                         "Realistic forecast (smart persistence)" if meta.get("use_smartfc")
                         else "Original (no forecast)")
        is_live = meta["mode"].startswith("LIVE") and live_ctx is not None

        if is_live:
            st.success(f"LIVE — {algo_name} ({variant_label}) · "
                      f"{live_ctx['label']} · present-day weather "
                      f"fetched {live_ctx['fetched']}")
        else:
            st.success(f"REPLAY — {algo_name} ({variant_label}) · "
                      f"historical year {year}, {days} days")

        last = df.iloc[-1]
        mix = df["exe"].map(ACTION_SHORT).value_counts()
        n_over = int(df["overridden"].sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Delivered", f"{last['cum_del']:.1f} kWh")
        c2.metric("Shield fires", f"{int(last['cum_shield']):,}")
        c3.metric("Safety violations", int(last["cum_safety"]))
        c4.metric("Shield overrode policy", f"{n_over} / {len(df)} hrs")
        c5.metric("Action mix", ", ".join(f"{k} {v}" for k, v in mix.items()))

        st.subheader("Hour-by-hour playback")
        k = st.slider("Hour", 0, len(df) - 1, min(24, len(df) - 1),
                      step=int(autoplay_speed))
        row = df.iloc[k]

        badge_color = {0: C_CHARGE, 1: C_IDLE, 2: C_BYPASS}[int(row["exe"])]
        badge_text  = ACTION_NAME[int(row["exe"])]
        st.markdown(
            f"""<div style="display:flex;gap:16px;align-items:center;margin:8px 0;">
            <div style="background:{badge_color};color:white;font-weight:700;
                 font-size:20px;padding:10px 22px;border-radius:10px;">{badge_text}</div>
            {'<div style="background:#fff3e0;color:#8a4b00;font-weight:600;'
             'padding:8px 14px;border-radius:8px;border:1px solid #f0c38a;">'
             f'SHIELD OVERRODE POLICY (asked {ACTION_SHORT[int(row["req"])]})</div>'
             if row['overridden'] else ''}
            <div style="color:#5c5b57;">{row.get('ts', f"hour {k}")}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("GHI", f"{row['ghi']:.0f} W/m²")
        k2.metric("Ambient", f"{row['tamb']:.1f} °C")
        k3.metric("Water T", f"{row['tw']:.1f} °C")
        k4.metric("PCM T", f"{row['tpcm']:.1f} °C")
        k5.metric("Melt fraction", f"{row['melt']:.0f} %")
        k6.metric("Demand now", f"{row['demand_L']:.1f} L")

        now_idx = None
        if is_live:
            now_idx = min(live_ctx["now_idx"], len(df) - 1)

        fig_t = go.Figure()
        if is_live:
            fig_t.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor="#2a78d6",
                            opacity=0.06, line_width=0,
                            annotation_text="forecast →", annotation_position="top left",
                            annotation_font_size=10, annotation_font_color="#5c5b57")
        fig_t.add_scatter(x=df["k"], y=df["tw"], name="Water T", line=dict(color=C_DQN))
        fig_t.add_scatter(x=df["k"], y=df["tpcm"], name="PCM T", line=dict(color=C_BYPASS))
        for y, c, d in [(45, "#1baf7a", "dot"), (72, "#eb6834", "dash"), (75, "#b01d0f", "solid"),
                       (62, "#eb6834", "dash"), (65, "#b01d0f", "solid")]:
            fig_t.add_hline(y=y, line=dict(color=c, dash=d, width=1))
        if is_live:
            fig_t.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"))
        fig_t.add_vline(x=k, line=dict(color="black", width=2))
        fig_t.update_layout(title="Temperature vs guard bands / hard limits (°C)"
                            + ("  (shaded = Open-Meteo forecast, beyond 'now')" if is_live else ""),
                            height=320, template="plotly_white",
                            margin=dict(l=40, r=20, t=40, b=30))

        fig_g = go.Figure()
        if is_live:
            fig_g.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor="#2a78d6",
                            opacity=0.06, line_width=0)
        fig_g.add_scatter(x=df["k"], y=df["ghi"], fill="tozeroy",
                          line=dict(color="#e0a21a"), name="GHI")
        if is_live:
            fig_g.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"),
                            annotation_text="now", annotation_position="top")
        fig_g.add_vline(x=k, line=dict(color="black", width=2))
        fig_g.update_layout(title="Solar irradiance (W/m²)", height=220,
                            template="plotly_white", margin=dict(l=40, r=20, t=40, b=30))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_t, use_container_width=True)
            st.plotly_chart(fig_g, use_container_width=True)
            if is_live:
                import plotly.express as px
                st.caption(
                    f"📍 Live weather source: **{live_ctx['label']}** "
                    f"({live_ctx['lat']:.2f}, {live_ctx['lon']:.2f}) — Open-Meteo, "
                    f"fetched {live_ctx['fetched']}. Dotted line = now; shaded "
                    f"region = the forecast portion of the fetched window."
                )
                fig_map = px.scatter_mapbox(
                    pd.DataFrame({"lat": [live_ctx["lat"]], "lon": [live_ctx["lon"]],
                                 "label": [live_ctx["label"]]}),
                    lat="lat", lon="lon", hover_name="label", zoom=6, height=220)
                fig_map.update_traces(marker=dict(size=14, color="#eb6834"))
                fig_map.update_layout(mapbox_style="open-street-map",
                                      margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_map, use_container_width=True)
        with col2:
            fig_a = go.Figure(go.Bar(
                x=list(mix.index), y=list(mix.values),
                marker_color=[{"CHARGE": C_CHARGE, "IDLE": C_IDLE,
                              "BYPASS": C_BYPASS}[a] for a in mix.index]))
            fig_a.update_layout(title="Action mix (whole window)", height=300,
                                template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_a, use_container_width=True)
            st.dataframe(df[["ts", "exe", "overridden", "tw", "tpcm", "ghi"]]
                        .rename(columns={"exe": "action", "ts": "time"})
                        .assign(action=lambda d: d["action"].map(ACTION_SHORT))
                        .tail(10), hide_index=True, use_container_width=True)

        if compare and st.session_state.recs2 is not None:
            st.divider()
            other = "DQN" if meta["algo"] == "sac" else "SAC"
            st.subheader(f"Side-by-side: {other} (Original) over the same window")
            df2 = pd.DataFrame(st.session_state.recs2)
            last2 = df2.iloc[-1]
            mix2 = df2["exe"].map(ACTION_SHORT).value_counts()
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric(f"{algo_name} delivered", f"{last['cum_del']:.1f} kWh")
            cc2.metric(f"{other} delivered", f"{last2['cum_del']:.1f} kWh")
            cc3.metric(f"{algo_name} shields", int(last["cum_shield"]))
            cc4.metric(f"{other} shields", int(last2["cum_shield"]))
            fig_cmp = go.Figure()
            fig_cmp.add_scatter(x=df["k"], y=df["cum_del"], name=f"{algo_name} delivered",
                                line=dict(color=C_SAC if algo_name == "SAC" else C_DQN))
            fig_cmp.add_scatter(x=df2["k"], y=df2["cum_del"], name=f"{other} delivered",
                                line=dict(color=C_DQN if algo_name == "SAC" else C_SAC, dash="dash"))
            if is_live:
                fig_cmp.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"),
                                  annotation_text="now", annotation_position="top")
            fig_cmp.update_layout(title="Cumulative delivered energy (kWh)"
                                  + ("  — both policies ran on the identical fetched weather"
                                     if is_live else ""),
                                  height=280, template="plotly_white",
                                  margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.caption(
            "This portal reuses the same rollout engine as "
            "`make_live_dashboard.py`: the frozen O2 physics + the "
            "in-environment safety shield, run with the selected policy's "
            "`best_model` checkpoint. All results are simulation."
        )

# =============================================================================
# TAB 2 — Deployment & ML Models
# =============================================================================
with tab_deploy:
    st.header("Deployment metrics, the six algorithms, and what's training now")

    metrics_path = RESULTS_DIR / "deployment_metrics_r0.json"
    if not metrics_path.exists():
        st.warning("Run `python scripts/phase9_deployment_metrics.py` to "
                  "generate this page's data.")
    else:
        metrics = json.loads(metrics_path.read_text())
        eff = metrics["efficiency_and_savings"]
        lat = metrics["latency_memory"]
        faults = metrics["sensor_faults"]

        # ---------------------------------------------------------- 1. efficiency
        st.subheader("1 · Thermal efficiency & demand coverage")
        order = [c for c in ("B0", "B1", "B2", "DQN", "SAC") if c in eff["controllers"]]
        ctr = eff["controllers"]
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
        st.plotly_chart(fig1, use_container_width=True)

        # ---------------------------------------------------------- 2. savings
        st.subheader("2 · Energy savings vs baseline controllers")
        drls = [c for c in ("DQN", "SAC") if c in ctr]
        scol1, scol2 = st.columns(2)
        with scol1:
            fig2a = go.Figure()
            for base, c in [("B0", C_B0), ("B1", C_B1)]:
                fig2a.add_bar(name=f"vs {base}", x=drls,
                             y=[ctr[d].get(f"backup_energy_saved_vs_{base}_kWh", 0) for d in drls],
                             marker_color=c)
            fig2a.update_layout(title="Backup-heater energy saved (kWh/yr)",
                               barmode="group", template="plotly_white", height=320)
            st.plotly_chart(fig2a, use_container_width=True)
        with scol2:
            fig2b = go.Figure()
            for base, c in [("B0", C_B0), ("B1", C_B1)]:
                fig2b.add_bar(name=f"vs {base}", x=drls,
                             y=[ctr[d].get(f"shield_reduction_vs_{base}_pct", 0) for d in drls],
                             marker_color=c)
            fig2b.update_layout(title="Shield-activation reduction (%)",
                               barmode="group", template="plotly_white", height=320)
            st.plotly_chart(fig2b, use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("DQN vs B1", f"+{ctr['DQN'].get('delivery_vs_B1_pct', 0)-100:.1f}% delivery"
                      if 'delivery_vs_B1_pct' in ctr.get('DQN', {}) else "—")
        with m2:
            st.metric("SAC vs B1", f"+{ctr['SAC'].get('delivery_vs_B1_pct', 0)-100:.1f}% delivery"
                      if 'delivery_vs_B1_pct' in ctr.get('SAC', {}) else "—")

        # ---------------------------------------------------------- 3. latency
        st.subheader("3 · Inference latency & memory footprint (embedded-target proxy)")
        algos_lat = [a for a in ("DQN", "SAC") if a in lat]
        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            fig3a = go.Figure(go.Bar(
                x=algos_lat, y=[lat[a]["latency_mean_us"] for a in algos_lat],
                marker_color=[colmap[a] for a in algos_lat],
                error_y=dict(type="data", array=[lat[a]["latency_p95_us"]-lat[a]["latency_mean_us"]
                                                 for a in algos_lat], visible=True)))
            fig3a.update_layout(title="Mean latency per decision (µs)", height=300,
                               template="plotly_white")
            st.plotly_chart(fig3a, use_container_width=True)
        with lcol2:
            fig3b = go.Figure(go.Bar(
                x=algos_lat, y=[lat[a]["weights_fp32_kB"] for a in algos_lat],
                marker_color=[colmap[a] for a in algos_lat]))
            fig3b.update_layout(title="Model weights (kB, fp32)", height=300,
                               template="plotly_white")
            st.plotly_chart(fig3b, use_container_width=True)
        with lcol3:
            st.markdown("**Params**")
            for a in algos_lat:
                st.metric(a, f"{lat[a]['n_params']:,}")
        st.caption(lat.get("note", ""))

        # ---------------------------------------------------------- 4. sensor faults
        st.subheader("4 · Robustness under sensor faults (observation corrupted, physics true)")
        fdf = pd.DataFrame(faults)
        fig4 = go.Figure()
        for a, c in [("DQN", C_DQN), ("SAC", C_SAC)]:
            sub = fdf[fdf["algo"] == a]
            fig4.add_bar(name=a, x=sub["fault"], y=sub["delivered_kWh"], marker_color=c)
        fig4.update_layout(title="Delivered energy under sensor faults (kWh/yr)",
                          barmode="group", template="plotly_white", height=360)
        fig4.update_xaxes(tickangle=25)
        st.plotly_chart(fig4, use_container_width=True)
        n_viol = int(fdf["safety_violations"].sum())
        st.success(f"Zero hard-safety violations across all {len(fdf)} "
                  f"(policy × fault) scenarios — the shield acts on the true "
                  f"physical state, so safety holds even on corrupted sensor "
                  f"readings." if n_viol == 0 else
                  f"⚠ {n_viol} safety violations recorded — investigate.")
        with st.expander("Raw sensor-fault results"):
            st.dataframe(fdf, hide_index=True, use_container_width=True)

        st.divider()

        # ---------------------------------------------------------- ML models overview
        st.subheader("The six algorithms compared, and why two were selected")
        algo_rows = [
            ("PPO", "On-policy policy gradient", "Collapsed to always-charge (B0) — entropy bonus insufficient to keep exploring"),
            ("DQN", "Value-based, ε-greedy", "✅ Learns anticipatory control on regime 0 only (−59% shields); collapses to B0 on regimes 1–2"),
            ("RecurrentPPO", "On-policy + LSTM", "Collapsed to always-charge — memory doesn't fix an exploration problem"),
            ("QRDQN", "Distributional value-based", "Collapsed to always-charge at this seed/weight"),
            ("MaskablePPO", "On-policy + action masking", "Essentially always-charge — mask forces a few non-charge steps only"),
            ("SAC", "Off-policy, entropy-regularised", "✅ Selected — only controller beating always-on on all 3 regimes (97–99% of B0 delivery, −33 to −54% shields)"),
        ]
        st.dataframe(pd.DataFrame(algo_rows, columns=["Algorithm", "Family", "Outcome"]),
                    hide_index=True, use_container_width=True)
        st.caption(
            "Selection rule (lexicographic): zero hard safety violations → "
            "delivery ≥ 95% of B0 → maximum shield reduction, consistent across regimes. "
            "Both pass on regime 0; only SAC keeps the advantage on regimes 1–2, so SAC is selected. "
            "The two winners both use **sustained exploration** (ε-greedy for "
            "DQN, entropy regularisation + replay for SAC) — the separating "
            "factor is exploration mechanism, not algorithm family."
        )

        st.divider()

        # ---------------------------------------------------------- GEN2 live status
        st.subheader("🔬 Generation-2 research extensions — live status")
        st.warning("Gen-2 (Phase 10) results below were produced BEFORE the 2026-11-13 weather-year fix and have not been re-run; treat them as indicative only.")
        st.markdown(
            "Five gaps from advisor feedback, each with an environment change "
            "and a training study. **Domain randomisation** (GHI noise, "
            "ambient-temperature bias, and demand multipliers sampled per "
            "training episode and applied to the physics, not just the "
            "observation) is active in *every* run below — it is a property "
            "of the training environment, not a separate experiment."
        )

        def _scan_tag(tag_prefix):
            hits = list(RESULTS_DIR.glob(f"phase10_{tag_prefix}*.csv"))
            return len(hits)

        pareto_done = _scan_tag("pareto_w")
        status_rows = [
            ("1. Sub-hourly control ablation (60 vs 15 min)",
             "✅ done" if (RESULTS_DIR/"phase10_subhourly15_w10_s42.csv").exists() else "⏳ training"),
            ("2. Forecast observation (V2: +3h GHI, +1h demand)",
             "✅ implemented, used in every Gen-2 run below"),
            ("3. Pareto sweep — SAC, 5 weights × 3 seeds (15 runs)",
             f"{'✅ all 15 done' if pareto_done>=15 else f'⏳ {pareto_done}/15 done'}"),
            ("4. Regimes 1 & 2 coverage (SAC + DQN)",
             "✅ done" if _scan_tag("gen2_sac_r1")+_scan_tag("gen2_dqn_r1")+_scan_tag("gen2_sac_r2")+_scan_tag("gen2_dqn_r2") >= 4 else "⏳ training"),
            ("5. Training-time domain randomisation",
             "✅ implemented, active in every run below"),
        ]
        st.table(pd.DataFrame(status_rows, columns=["Feedback point", "Status"]))

        pareto_csvs = sorted(RESULTS_DIR.glob("phase10_pareto_*.csv"))
        if pareto_csvs:
            st.markdown("**Pareto sweep results so far:**")
            rows = []
            for p in pareto_csvs:
                d = pd.read_csv(p)
                rows.append({"weight": d["shield_weight"].iloc[0], "seed": d["seed"].iloc[0],
                            "delivered_kWh": round(d["total_delivered_kWh"].mean(), 1),
                            "shields": int((d["total_shield_water"]+d["total_shield_pcm"]).mean()),
                            "reward": round(d["total_reward"].mean())})
            pdf = pd.DataFrame(rows).sort_values(["weight", "seed"])
            st.dataframe(pdf, hide_index=True, use_container_width=True)
            if len(pdf) >= 2:
                fig5 = go.Figure(go.Scatter(
                    x=pdf["shields"], y=pdf["delivered_kWh"], mode="markers+text",
                    text=[f"w{w}/s{s}" for w, s in zip(pdf["weight"], pdf["seed"])],
                    textposition="top center",
                    marker=dict(size=10, color=pdf["weight"], colorscale="Viridis",
                               showscale=True, colorbar=dict(title="weight"))))
                fig5.update_layout(title="Delivery-vs-shields Pareto frontier (live, partial)",
                                  xaxis_title="shield trips/yr (lower better)",
                                  yaxis_title="delivered kWh/yr (higher better)",
                                  template="plotly_white", height=420)
                st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No Pareto-sweep runs have finished yet — this chart "
                   "populates automatically as results land.")

        st.caption("This tab re-reads result files on every page load — "
                  "refresh the browser any time to see training progress.")
