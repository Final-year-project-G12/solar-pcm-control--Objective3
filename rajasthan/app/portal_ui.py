"""
app/portal_ui.py - the Objective-3 portal user interface, ported from tamilnadu/scripts/streamlit_portal.py.

The same file lives in rajasthan/app/ and uttarakhand/app/ (each state folder stays self-contained; keep the two copies identical).
The state's own app (app/streamlit_app.py) supplies an *adapter* dict `P`, so every state shows the same portal on its own simulator.

What the portal shows (the Tamil Nadu design):
  * Tab 1 "Live Controller": pick policy / regime / weather window (month + day + length picker, IST clock) or LIVE Open-Meteo weather,
    run it through the frozen physics and see what the pump physically did hour by hour, against an always-on pump (B0) on the same
    weather: delivered heat, overheat-shield hits, hard-limit breaches, pump run-time, hours the controller stepped in, a per-hour
    reason, a pump-activity strip, a per-day chart, and every chart/table is clickable for a short explanation.
  * Tab 2 "Deployment & ML Models": thermal efficiency and demand coverage, comparison with the baselines, inference cost,
    sensor-fault robustness and an overview of the algorithms, for any regime that has the data.

Adapter P (all keys documented here):
  state, title, caption                 strings
  regimes                               {regime_id: label}
  policies(regime)                      {policy_key: label}   (always includes P["always_on"])
  default_policy(regime)                policy_key
  always_on                             policy_key of the always-charge baseline B0
  years                                 allowed replay years (train / validation only)
  default_year, default_month           ints
  replay(regime, key, year, start_h, n_h)           -> list[record]   (records below; cumulative columns count inside the window)
  live(regime, keys, past_days, forecast_days)      -> ({key: [record]}, ctx)   ctx: label, lat, lon, fetched, now_idx   (or None to hide LIVE)
  deployment(regime)                    -> dict | None   (see render_deployment)
  guards                                 (water guard C, water hard limit C, target C)
  shield_unit                            "sub-step hits" or "hours" (what cum_shield counts)
  pump_note                              one sentence on how 'pump ran' is measured
  algo_rows, selection_note             optional text for the algorithm overview
Record: k, ts, hod, ghi, tamb, demand_L, exe (0 charge /1 hold /2 bypass), overridden (shield forced it), tw, tpcm, melt (%),
        pump (0..1), cum_del (kWh), cum_shield, cum_safety.
"""
from __future__ import annotations

import datetime
import json  # noqa: F401  (adapters may pass JSON-derived dicts)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

C_SAC, C_DQN, C_B0, C_B1 = "#1baf7a", "#2a78d6", "#8f8e88", "#b5b4ae"
C_BYPASS = "#eb6834"
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
              "or forced by the safety shield at the guard bands.",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _month_days(year: int) -> list[int]:
    """The weather series has 8760 hours in every year (29 Feb is removed from leap years), so the window picker uses a 365-day calendar."""
    return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


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
        return f"Sun is {g} W/m² and the collector is hotter than the tank ({tw} °C), so the pump runs and stores the heat."
    if m == "bypass" and r["overridden"]:
        return f"Water/PCM reached the guard band ({tw} °C water), so the safety shield isolated the collector."
    if m == "bypass":
        return (f"The controller isolated the collector early: tank {tw} °C with {g} W/m² of sun means it is heading for overheating.")
    if m == "hold":
        return (f"Sun is available ({g} W/m²) but the controller paused the pump to keep the tank ({tw} °C) and PCM "
                f"({r['tpcm']:.0f} °C) below their limits.")
    if r["ghi"] < 50:
        return f"No useful sun ({g} W/m²), so the pump is off by itself and stored heat serves the {r['demand_L']:.1f} L draw."
    return f"Sun is {g} W/m² but the collector cannot add net heat to a {tw} °C tank."


def show_chart(fig, key, explain_fn, hint=":material/ads_click: Click any point or bar for a short explanation."):
    """Render a Plotly chart; clicking a point shows explain_fn(point) below."""
    ev = st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode="points")
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


# =============================================================================================== tab 1
def render_live(P: dict) -> None:
    regimes, years = P["regimes"], P["years"]
    g_guard, g_limit, g_target = P["guards"]
    unit = P["shield_unit"]
    with st.sidebar:
        st.header("Run configuration")
        regime = st.selectbox("Regime", list(regimes), index=0, format_func=lambda r: regimes[r], key="p_regime")
        pols = P["policies"](regime)
        keys = list(pols)
        dflt = P["default_policy"](regime)
        policy = st.selectbox("Policy", keys, index=keys.index(dflt) if dflt in keys else 0, format_func=pols.get, key=f"p_policy_{regime}")   # per-regime key: the option list changes with the regime
        live_ok = P.get("live") is not None
        mode = st.radio("Weather source", ["Historical replay"] + (["LIVE (Open-Meteo)"] if live_ok else []), key="p_mode")
        year, start_h, days, past_days, forecast_days = years[0], 0, 7, 2, 3
        d0 = d1 = None
        if mode == "Historical replay":
            year = st.selectbox("Year", years, index=years.index(P["default_year"]) if P["default_year"] in years else 0,
                                help="Train / validation years only — held-out test years are never offered.", key="p_year")
            md = _month_days(year)
            month = st.selectbox("Start month", MONTHS, index=P["default_month"], key="p_month")
            mi = MONTHS.index(month)
            dom = st.slider("Start day of month", 1, md[mi], 1, key="p_dom")
            days = st.slider("Window length (days)", 1, 30, 7, key="p_days")
            doy = sum(md[:mi]) + dom - 1
            doy = min(doy, sum(md) - days)
            start_h = doy * 24
            d0 = (datetime.date(2001, 1, 1) + datetime.timedelta(days=doy)).replace(year=year)      # 2001 = a non-leap calendar
            d1 = (datetime.date(2001, 1, 1) + datetime.timedelta(days=doy + days - 1)).replace(year=year)
            st.caption(f"Window: **{d0:%d %b %Y} → {d1:%d %b %Y}**")
        else:
            past_days = st.slider("Past days", 0, 5, 2, key="p_past")
            forecast_days = st.slider("Forecast days", 1, 7, 3, key="p_fc")
        other_keys = ["(none)"] + [k for k in keys if k != policy]
        other = st.selectbox("Compare with another policy", other_keys, format_func=lambda k: "(none)" if k == "(none)" else pols[k],
                             key=f"p_other_{regime}", help="Runs it over the same window.")
        step = st.select_slider("Scrub step size (for the Hour slider below)", [1, 3, 6, 12, 24], value=1, key="p_step")
        run_btn = st.button("Run", type="primary", icon=":material/play_arrow:", width="stretch", key="p_run")

    ss = st.session_state
    for k in ("p_recs", "p_base", "p_recs2", "p_meta", "p_ctx"):
        ss.setdefault(k, None)

    if run_btn:
        with st.spinner("Running the policy (and an always-on pump for comparison) through the frozen physics..."):
            try:
                ctx = None
                want = [policy, P["always_on"]] + ([other] if other != "(none)" else [])
                if mode.startswith("LIVE"):
                    runs, ctx = P["live"](regime, want, past_days, forecast_days)
                else:
                    runs = {k: P["replay"](regime, k, year, start_h, days * 24) for k in dict.fromkeys(want)}
                ss.p_recs, ss.p_base = runs[policy], runs[P["always_on"]]
                ss.p_recs2 = runs[other] if other != "(none)" else None
                ss.p_ctx = ctx
            except Exception as e:                                  # noqa: BLE001
                st.error(f"Run failed: {e}. If using LIVE mode, check network access to api.open-meteo.com, or try Historical replay.")
                st.stop()
        desc = (f"historical year {year}, {d0:%d %b} → {d1:%d %b}" if mode == "Historical replay" else "present-day weather")
        ss.p_meta = {"mode": mode, "policy": policy, "label": pols[policy], "regime": regime, "desc": desc,
                     "other": other, "other_label": pols.get(other, "")}

    recs = ss.p_recs
    if recs is None:
        st.info("Configure a run in the sidebar and press **Run**.")
        return
    meta, ctx = ss.p_meta, ss.p_ctx
    df = pd.DataFrame(recs)
    df["mode"] = df.apply(pump_mode, axis=1)
    df["why"] = df.apply(hour_reason, axis=1)
    is_live = meta["mode"].startswith("LIVE") and ctx is not None
    if is_live:
        st.success(f"LIVE — {meta['label']} · regime {meta['regime']} · {ctx['label']} · weather fetched {ctx['fetched']}")
    else:
        st.success(f"REPLAY — {meta['label']} · regime {meta['regime']} · {meta['desc']}")

    last = df.iloc[-1]
    mix = df["mode"].value_counts()
    n_over = int(df["overridden"].sum())
    base = ss.p_base
    b = pd.DataFrame(base).iloc[-1] if base else None
    c1, c2, c3, c4, c5 = st.columns(5)
    if b is not None:
        d_del = 100 * (last["cum_del"] - b["cum_del"]) / b["cum_del"] if b["cum_del"] else 0.0
        cut = 100 * (b["cum_shield"] - last["cum_shield"]) / b["cum_shield"] if b["cum_shield"] else 0.0
        c1.metric("Hot water delivered", f"{last['cum_del']:.1f} kWh", f"{d_del:+.1f}% vs always-on ({b['cum_del']:.1f})",
                  delta_color="off", help="Useful heat delivered to the hot-water draw over the window.")
        c2.metric("Overheat shield " + ("hours" if unit == "hours" else "hits"), f"{last['cum_shield']:,.1f}" if unit == "hours" else f"{int(last['cum_shield']):,}",
                  f"{cut:.0f}% fewer than always-on ({b['cum_shield']:,.1f})" if unit == "hours" else f"{cut:.0f}% fewer than always-on ({int(b['cum_shield']):,})",
                  help="Time the safety shield had to isolate the collector. Fewer = less wasted heat and less thermal stress.")
    else:
        c1.metric("Hot water delivered", f"{last['cum_del']:.1f} kWh")
        c2.metric("Overheat shield", f"{last['cum_shield']:,.1f}")
    c3.metric("Hard-limit breaches", f"{last['cum_safety']:g}", help=f"Water above {g_limit:g} °C or PCM above its limit. Target is 0.")
    c4.metric("Pump actually ran", f"{df['pump'].sum():.0f} / {len(df)} h", help=P["pump_note"])
    c5.metric("Controller stepped in", f"{int(mix.get('hold', 0) + mix.get('bypass', 0))} h", f"{n_over} forced by shield",
              delta_color="off", help="Hours the pump was held off or the collector bypassed in sunshine.")
    st.caption("CHARGE only *permits* the pump; the simulator switches it off itself at night or when the collector has no net gain. "
               "Hours are therefore coloured by what the pump physically did. 'Efficient' = the same hot water with far fewer "
               "overheat shield hits, not extra kWh. " + P["pump_note"])

    st.subheader("Hour-by-hour playback")
    k = st.slider("Hour", 0, len(df) - 1, min(24, len(df) - 1), step=int(step), key="p_hour")
    row = df.iloc[k]
    text_color = "#1a1a1a" if row["mode"] == "wait" else "white"
    shield_html = ('<div style="background:#fff3e0;color:#8a4b00;font-weight:600;padding:8px 14px;border-radius:8px;'
                   'border:1px solid #f0c38a;">SAFETY SHIELD FORCED THIS</div>' if row["overridden"] else "")
    st.markdown(f"""<div style="display:flex;gap:16px;align-items:center;margin:8px 0;flex-wrap:wrap;">
        <div style="background:{MODE_COLOR[row['mode']]};color:{text_color};font-weight:700;font-size:20px;padding:10px 22px;
        border-radius:10px;">{MODE_LABEL[row['mode']]}</div>{shield_html}
        <div style="color:#5c5b57;font-weight:600;">{row['ts']}</div></div>""", unsafe_allow_html=True)
    st.write(row["why"])
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Sun (GHI)", f"{row['ghi']:.0f} W/m²")
    k2.metric("Outside", f"{row['tamb']:.1f} °C")
    k3.metric("Tank water", f"{row['tw']:.1f} °C")
    k4.metric("PCM", f"{row['tpcm']:.1f} °C")
    k5.metric("PCM melted", f"{row['melt']:.0f} %")
    k6.metric("Hot water drawn", f"{row['demand_L']:.1f} L")

    now_idx = min(ctx["now_idx"], len(df) - 1) if is_live else None
    fig_t = go.Figure()
    if is_live:
        fig_t.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor="#2a78d6", opacity=0.06, line_width=0,
                        annotation_text="forecast →", annotation_position="top left", annotation_font_size=10)
    fig_t.add_scatter(x=df["k"], y=df["tw"], name="Tank water", line=dict(color=C_DQN),
                      hovertemplate="%{customdata}<br>water %{y:.1f} °C<extra></extra>", customdata=df["ts"])
    fig_t.add_scatter(x=df["k"], y=df["tpcm"], name="PCM", line=dict(color=C_BYPASS),
                      hovertemplate="%{customdata}<br>PCM %{y:.1f} °C<extra></extra>", customdata=df["ts"])
    for y, c, d, lab in [(g_target, "#1baf7a", "dot", f"{g_target:g}° delivery target"), (g_guard, "#eb6834", "dash", f"{g_guard:g}° water guard"),
                         (g_limit, "#b01d0f", "solid", f"{g_limit:g}° water limit")]:
        fig_t.add_hline(y=y, line=dict(color=c, dash=d, width=1), annotation_text=lab, annotation_position="top right",
                        annotation_font_size=10, annotation_font_color=c)
    if is_live:
        fig_t.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"))
    fig_t.add_vline(x=k, line=dict(color="black", width=2))
    fig_t.update_layout(title="Tank water and PCM temperature (°C) — stay under the guard lines" + ("  (shaded = forecast)" if is_live else ""),
                        height=320, template="plotly_white", margin=dict(l=40, r=20, t=50, b=30), legend=dict(orientation="h", y=1.14),
                        xaxis_title="hour in window")
    fig_r = go.Figure(go.Bar(x=df["k"], y=[1] * len(df), marker_color=[MODE_COLOR[m] for m in df["mode"]], marker_line_width=0,
                             customdata=df[["ts", "mode"]].values, hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig_r.add_vline(x=k, line=dict(color="black", width=2))
    fig_r.update_layout(title="What the pump did each hour (blue collecting · grey standby · green held off · orange bypass)",
                        height=120, template="plotly_white", bargap=0, margin=dict(l=40, r=20, t=34, b=10),
                        yaxis=dict(visible=False), showlegend=False)
    fig_g = go.Figure()
    if is_live:
        fig_g.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor="#2a78d6", opacity=0.06, line_width=0)
    fig_g.add_scatter(x=df["k"], y=df["ghi"], fill="tozeroy", line=dict(color="#e0a21a"), name="GHI",
                      hovertemplate="%{customdata}<br>%{y:.0f} W/m²<extra></extra>", customdata=df["ts"])
    if is_live:
        fig_g.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"), annotation_text="now", annotation_position="top")
    fig_g.add_vline(x=k, line=dict(color="black", width=2))
    fig_g.update_layout(title="Sunshine available (GHI, W/m²)", height=220, template="plotly_white",
                        margin=dict(l=40, r=20, t=40, b=30), xaxis_title="hour in window")

    df["daylab"] = df["ts"].apply(lambda s: s.rsplit(" ", 1)[0])
    day_order = list(dict.fromkeys(df["daylab"]))
    fig_d = go.Figure()
    for m in ["collect", "hold", "bypass", "wait"]:
        cnts = df[df["mode"] == m].groupby("daylab").size()
        fig_d.add_bar(name=MODE_SHORT[m], x=day_order, y=[int(cnts.get(d, 0)) for d in day_order], marker_color=MODE_COLOR[m])
    fig_d.update_layout(barmode="stack", title="What the pump did each day (hours)", height=300, template="plotly_white",
                        margin=dict(l=20, r=20, t=40, b=30), legend=dict(orientation="h", y=1.15))

    def _ex_hour(p):
        r = df.iloc[int(p["x"])]
        return (f"**{r['ts']}** — {MODE_SHORT[r['mode']]}: water {r['tw']:.1f} °C, PCM {r['tpcm']:.1f} °C, sun {r['ghi']:.0f} W/m².\n\n{r['why']}")

    def _ex_ghi(p):
        r = df.iloc[int(p["x"])]
        if r["ghi"] < 50:
            return f"**{r['ts']}** — sun is only {r['ghi']:.0f} W/m².\n\nBelow ~50 W/m² the collector can't add heat, so the pump idles."
        return (f"**{r['ts']}** — sun is {r['ghi']:.0f} W/m².\n\nMore sun means faster heating; around midday it pushes the tank toward "
                f"the guard band, which is when the controller must step in.")

    def _ex_day(p):
        d = p["x"]
        sub = df[df["daylab"] == d]
        parts = ", ".join(f"{MODE_SHORT[m]} {int((sub['mode'] == m).sum())} h" for m in ["collect", "hold", "bypass", "wait"])
        return (f"**{d}** — {parts}.\n\nThe pump only needs to be held off or bypassed for a few hot midday hours; the rest is "
                f"normal collecting or standby.")

    col1, col2 = st.columns([2, 1])
    with col1:
        show_chart(fig_t, "fig_t", _ex_hour, ":material/ads_click: Click a point on a line to see what the controller did that hour and why.")
        show_chart(fig_r, "fig_r", _ex_hour, ":material/ads_click: Click a coloured bar to see what the pump did that hour.")
        show_chart(fig_g, "fig_g", _ex_ghi)
        if is_live:
            import plotly.express as px
            st.caption(f":material/location_on: Live weather source: **{ctx['label']}** ({ctx['lat']:.2f}, {ctx['lon']:.2f}) — Open-Meteo, fetched "
                       f"{ctx['fetched']}. Dotted line = now; shaded region = the forecast portion of the fetched window.")
            fig_map = px.scatter_map(pd.DataFrame({"lat": [ctx["lat"]], "lon": [ctx["lon"]], "label": [ctx["label"]]}),
                                     lat="lat", lon="lon", hover_name="label", zoom=6, height=220)
            fig_map.update_traces(marker=dict(size=14, color="#eb6834"))
            fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map, width="stretch")
    with col2:
        fig_a = go.Figure(go.Bar(x=[MODE_SHORT[m] for m in mix.index], y=list(mix.values), marker_color=[MODE_COLOR[m] for m in mix.index]))
        fig_a.update_layout(title="Pump hours, whole window", height=300, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

        def _ex_mix(p):
            m = next(k_ for k_, v in MODE_SHORT.items() if v == p["x"])
            return f"**{MODE_SHORT[m]}** — {int(mix.get(m, 0))} of {len(df)} hours.\n\n{MODE_DEF[m]}"
        show_chart(fig_a, "fig_a", _ex_mix)
        st.markdown("**Hours around the selected one** (click a row)")
        lo, hi = max(0, k - 5), min(len(df), k + 6)
        tbl = df.iloc[lo:hi][["ts", "mode", "pump", "tw", "tpcm", "ghi"]].copy()
        tbl["mode"] = tbl["mode"].map(MODE_SHORT)
        tbl = tbl.rename(columns={"ts": "time", "mode": "pump did", "pump": "pump on (fraction)", "tw": "water °C", "tpcm": "PCM °C", "ghi": "sun W/m²"})
        sel = st.dataframe(tbl, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="tbl_hours")
        try:
            rows = sel.selection.rows
        except Exception:
            rows = []
        if rows:
            st.info(f"**{df.iloc[lo + rows[0]]['ts']}** — {df.iloc[lo + rows[0]]['why']}")
    show_chart(fig_d, "fig_d", _ex_day)

    if meta.get("other") not in (None, "(none)") and ss.p_recs2 is not None:
        st.divider()
        other = meta["other_label"]
        st.subheader(f"Side-by-side: {other} over the same window")
        df2 = pd.DataFrame(ss.p_recs2)
        last2 = df2.iloc[-1]
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric(f"{meta['label']} delivered", f"{last['cum_del']:.1f} kWh")
        cc2.metric(f"{other} delivered", f"{last2['cum_del']:.1f} kWh")
        cc3.metric(f"{meta['label']} shield", f"{last['cum_shield']:,.1f}")
        cc4.metric(f"{other} shield", f"{last2['cum_shield']:,.1f}")
        fig_cmp = go.Figure()
        fig_cmp.add_scatter(x=df["k"], y=df["cum_del"], name=f"{meta['label']} delivered", line=dict(color=C_SAC))
        fig_cmp.add_scatter(x=df2["k"], y=df2["cum_del"], name=f"{other} delivered", line=dict(color=C_DQN, dash="dash"))
        if b is not None:
            fig_cmp.add_scatter(x=[r["k"] for r in base], y=[r["cum_del"] for r in base], name="always-on pump", line=dict(color=C_B0, dash="dot"))
        if is_live:
            fig_cmp.add_vline(x=now_idx, line=dict(color="#8a4b00", width=2, dash="dot"), annotation_text="now", annotation_position="top")
        fig_cmp.update_layout(title="Cumulative delivered energy (kWh)", height=280, template="plotly_white",
                              margin=dict(l=40, r=20, t=40, b=30), xaxis_title="hour in window")

        def _ex_cmp(p):
            i = int(p["x"])
            a_, b_ = df.iloc[min(i, len(df) - 1)], df2.iloc[min(i, len(df2) - 1)]
            return (f"By **{a_['ts']}**: {meta['label']} has delivered {a_['cum_del']:.1f} kWh and {other} {b_['cum_del']:.1f} kWh.\n\n"
                    f"The curves rise in the daytime and flatten at night; a bigger gap means one policy collects or stores heat better.")
        show_chart(fig_cmp, "fig_cmp", _ex_cmp)
    st.caption(f"{P['state']}: the frozen Objective-2 physics plus the in-environment safety shield, run with the selected policy. "
               "All results are simulation.")


# =============================================================================================== tab 2
CTRL_DESC_DEFAULT = {
    "B0": "Always-on pump (the Objective-2 operating point); only the safety shield protects it from overheating.",
    "B1": "Reactive rules: bypass near the guard bands, otherwise charge when sunny.",
    "B2": "Rule-based controller that also uses a forecast of upcoming sun.",
    "Water-only": "Tank without PCM (the reference the PCM must beat).",
}


def render_deployment(P: dict) -> None:
    """P['deployment'](regime) -> {"controllers": {name: {delivered_kWh, shield_h, violations, thermal_efficiency_pct, demand_coverage_pct}},
    "inference": {name: {ms, params, kB}}, "faults": DataFrame[policy, fault, delivered_kWh, shield_h, violations], "note": str,
    "assumptions": str} or None"""
    st.header("Deployment metrics and algorithms")
    d_regime = st.selectbox("Regime", list(P["regimes"]), key="d_regime", format_func=lambda r: P["regimes"][r])
    dep = P["deployment"](d_regime) if P.get("deployment") else None
    if not dep:
        st.warning("No deployment-metrics file exists for this regime yet (it is produced by the phase-9 / M10 script).")
    else:
        ctr = dep["controllers"]
        st.caption(dep.get("assumptions", "") or "Numbers are simulated validation years for the chosen regime.")
        order = list(ctr)
        palette = {n: (C_B0 if n.startswith("B0") else C_B1 if n.startswith(("B1", "B2")) else "#d2d1cb" if "twin" in n.lower() or "water" in n.lower() else C_DQN)
                   for n in order}
        has_eff = all("thermal_efficiency_pct" in ctr[c] for c in order)
        st.subheader("1 · Delivered heat" + (" and efficiency" if has_eff else ""))
        fig1 = go.Figure()
        fig1.add_bar(name="delivered kWh/yr", x=order, y=[ctr[c]["delivered_kWh"] for c in order], marker_color=[palette[c] for c in order])
        fig1.update_layout(template="plotly_white", height=320, yaxis_title="kWh per year", title=f"Heat delivered to {P['guards'][2]:g} °C (kWh/yr)")
        show_chart(fig1, "dfig1", lambda p: f"**{p['x']}** — {p['y']:.0f} kWh/yr. {CTRL_DESC_DEFAULT.get(str(p['x'])[:2], '')}")
        if has_eff:
            fig1b = go.Figure()
            fig1b.add_bar(name="thermal efficiency %", x=order, y=[ctr[c]["thermal_efficiency_pct"] for c in order], marker_color=[palette[c] for c in order])
            if all("demand_coverage_pct" in ctr[c] for c in order):
                fig1b.add_bar(name="demand coverage %", x=order, y=[ctr[c]["demand_coverage_pct"] for c in order],
                              marker_color=[palette[c] for c in order], opacity=0.45)
            fig1b.update_layout(barmode="group", template="plotly_white", height=320, yaxis_title="%", title="Solar→load efficiency and demand coverage")
            show_chart(fig1b, "dfig1b", lambda p: (f"**{p['x']}** — {p['y']:.1f}%. Efficiency = hot-water energy delivered ÷ solar energy falling on the collector; "
                                                   f"coverage = share of the hot-water demand met by solar (the rest needs backup)."))
        b0 = next((c for c in order if c.startswith("B0")), None)
        if all("shield_h" in ctr[c] for c in order):
            st.subheader("2 · Overheat-shield hours against the baselines")
            fig2 = go.Figure(go.Bar(x=order, y=[ctr[c]["shield_h"] for c in order], marker_color=[palette[c] for c in order]))
            fig2.update_layout(template="plotly_white", height=320, yaxis_title="hours per year", title="Hours the safety shield had to act (fewer is better)")

            def _ex2(p):
                msg = f"**{p['x']}** — {p['y']:.0f} shield hours per year."
                if b0 and ctr[b0]["shield_h"]:
                    msg += f" That is {100 * (1 - p['y'] / ctr[b0]['shield_h']):.0f}% fewer than the always-on pump."
                return msg + "\n\nFewer hours means less heat dumped and less thermal stress on the tank and PCM."
            show_chart(fig2, "dfig2", _ex2)
        inf = dep.get("inference")
        if inf:
            st.subheader("3 · Inference latency & memory footprint (embedded-target proxy)")
            names = list(inf)
            l1, l2, l3 = st.columns(3)
            with l1:
                f3a = go.Figure(go.Bar(x=names, y=[inf[a]["ms"] * 1000 for a in names], marker_color=C_DQN))
                f3a.update_layout(title="Latency per decision (µs)", height=300, template="plotly_white")
                show_chart(f3a, "dfig3a", lambda p: f"**{p['x']}** — {p['y']:.0f} µs per decision. The controller decides once per hour, so a slow microcontroller has enormous headroom.")
            with l2:
                f3b = go.Figure(go.Bar(x=names, y=[inf[a]["kB"] for a in names], marker_color=C_SAC))
                f3b.update_layout(title="Model weights (kB, fp32)", height=300, template="plotly_white")
                show_chart(f3b, "dfig3b", lambda p: f"**{p['x']}** — {p['y']:.0f} kB of weights; small enough for a low-cost embedded controller.")
            with l3:
                st.markdown("**Parameters**")
                for a in names:
                    st.metric(a, f"{inf[a]['params']:,}", help="Number of weights in the acting network.")
            if dep.get("note"):
                st.caption(dep["note"])
        faults = dep.get("faults")
        if faults is not None and len(faults):
            st.subheader("4 · Robustness under sensor faults (observation corrupted, physics true)")
            fdf = faults
            f4 = go.Figure()
            for i, a in enumerate(dict.fromkeys(fdf["policy"])):
                sub = fdf[fdf["policy"] == a]
                f4.add_bar(name=a, x=sub["fault"], y=sub["delivered_kWh"], marker_color=[C_DQN, C_SAC, C_B0, C_B1][i % 4])
            f4.update_layout(title="Delivered energy under sensor faults (kWh/yr)", barmode="group", template="plotly_white", height=360)
            f4.update_xaxes(tickangle=25)

            def _ex4(p):
                a = f4.data[p["curve_number"]].name
                nom = fdf[(fdf["policy"] == a) & (fdf["fault"].isin(["nominal", "none"]))]
                nom_txt = f" (clean sensors: {nom['delivered_kWh'].iloc[0]:.0f} kWh)" if len(nom) else ""
                return (f"**{a} with fault '{p['x']}'** — delivers {p['y']:.0f} kWh/yr{nom_txt}.\n\nThe policy is fed wrong readings, but the safety "
                        f"shield uses the true state, so safety holds and only efficiency can slip.")
            show_chart(f4, "dfig4", _ex4)
            n_viol = int(fdf["violations"].sum())
            (st.success if n_viol == 0 else st.warning)(
                f"Zero hard-limit violations across all {len(fdf)} (policy × fault) scenarios." if n_viol == 0
                else f"{n_viol} hard-limit violations recorded — investigate.")
            with st.expander("Raw sensor-fault results"):
                st.dataframe(fdf, hide_index=True, width="stretch")
    rows = P.get("algo_rows")
    if rows:
        st.divider()
        st.subheader("The algorithms compared")
        adf = pd.DataFrame(rows, columns=["Algorithm", "Family", "Outcome"])
        st.caption("Click a row for a short explanation.")
        asel = st.dataframe(adf, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="tbl_algos")
        try:
            arows = asel.selection.rows
        except Exception:
            arows = []
        if arows:
            r = rows[arows[0]]
            st.info(f"**{r[0]}** ({r[1]}) — {r[2]}.\n\n\"Always-charge\" means it learned nothing better than leaving the pump on; it only counts "
                    f"as a success if it beats that with fewer shield hits.")
        if P.get("selection_note"):
            st.caption(P["selection_note"])


def run_portal(P: dict) -> None:
    st.title(P["title"])
    st.caption(P["caption"])
    tab_live, tab_dep = st.tabs(["Live Controller", "Deployment & ML Models"])
    with tab_live:
        render_live(P)
    with tab_dep:
        render_deployment(P)
