"""
app/portal_ui.py - the Objective-3 portal user interface (design taken from tamilnadu/scripts/streamlit_portal.py).

The same file lives in rajasthan/app/ and uttarakhand/app/ (each state folder stays self-contained; keep the two copies identical).
The state's own app (app/streamlit_app.py) supplies an *adapter* dict `P`, so every state shows the same portal on its own simulator.

  Tab "Live controller":  pick regime / policy / weather window (month, day, length) or LIVE Open-Meteo weather, run it through the frozen
    physics, and see delivered heat, shield time and hard-limit breaches against an always-on pump (B0) on the same weather, plus what the
    pump physically did each hour. Every chart and the two small tables are clickable for a short explanation.
  Tab "Deployment and models":  controller table, delivered heat and shield hours, inference cost, sensor-fault robustness and an
    algorithm overview, for any regime that has the data.

Look and feel: colours, font, radius and chart colour order live in .streamlit/config.toml (repo root). The values below (PALETTE, sizes)
are the only other place that holds a colour or a pixel size; everything in this file reads them from here.

Adapter P:
  state, title, caption                 strings (title = the state name; caption = one line about the system)
  regimes                               {regime_id: label}
  policies(regime)                      {policy_key: label}   (always includes P["always_on"])
  default_policy(regime)                policy_key
  always_on                             policy_key of the always-charge baseline B0
  years                                 allowed replay years (train / validation only)
  default_year, default_month           ints
  replay(regime, key, year, start_h, n_h)           -> list[record]   (cumulative columns count inside the window)
  live(regime, keys, past_days, forecast_days)      -> ({key: [record]}, ctx)   ctx: label, lat, lon, fetched, now_idx   (None hides LIVE)
  live_forecast_days                    False when the live fetch ignores the forecast-days setting (the slider is then hidden)
  deployment(regime)                    -> dict | None   (see render_deployment)
  guards                                 (water guard C, water hard limit C, delivery target C)
  shield_unit                            "hours" or "hits" (what cum_shield counts)
  pump_note                              one sentence on how 'pump ran' is measured
  algo_rows, selection_note             optional text for the algorithm overview
Record: k, ts, hod, ghi, tamb, demand_L, exe (0 charge /1 hold /2 bypass), overridden (shield forced it), tw, tpcm, melt (%),
        pump (0..1), cum_del (kWh), cum_shield, cum_safety.
"""
from __future__ import annotations

import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------------------------- style: the one place for chart styling
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
H_MAIN, H_STRIP, H_SMALL, H_BAR, H_MAP = 300, 150, 200, 280, 240          # chart heights (px); Plotly needs a number
HOUR_CURSOR_PX = 1
# Validated categorical set (dataviz skill, 4 slots) + neutrals + status steps. Light and dark are separate selections, not a flip.
PALETTE = {
    "light": dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#6250d6", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9",
                  axis="#c3c2b7", base="#898781", base1="#52514e", base2="#b4b3ab", standby="#c3c2b7", warn="#ec835a", bad="#d03b3b"),
    "dark": dict(blue="#3987e5", orange="#d95926", aqua="#199e70", violet="#9085e9", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a",
                 axis="#383835", base="#898781", base1="#c3c2b7", base2="#6d6c68", standby="#4a4945", warn="#ec835a", bad="#d03b3b"),
}


def pal() -> dict:
    try:
        return PALETTE[st.context.theme.type if st.context.theme.type in PALETTE else "light"]
    except Exception:                                                   # theme not available (tests, old runtimes)
        return PALETTE["light"]


def ctrl_color(name: str) -> str:
    """One colour per controller, the same in every chart: baselines are greys, learned agents carry a hue."""
    p, s = pal(), str(name).lower()
    if "sac" in s or "ppo" in s:
        return p["aqua"]
    if "dqn" in s:
        return p["violet"]
    if "b0" in s or "always" in s:
        return p["base"]
    if "b1" in s or "reactive" in s:
        return p["base1"]
    if "b2" in s or "forecast" in s:
        return p["base2"]
    return p["standby"]                                                  # water-only twin and anything else


def style_fig(fig: go.Figure, title: str, height: int, x: str | None = None, y: str | None = None, legend: bool = True) -> go.Figure:
    p = pal()
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=13, color=p["ink"])),
        height=height, margin=dict(l=8, r=8, t=44, b=8), template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=p["ink2"]), showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    fig.update_xaxes(title=x, showgrid=False, zeroline=False, linecolor=p["axis"], ticks="outside", tickcolor=p["axis"], automargin=True)
    fig.update_yaxes(title=y, showgrid=True, gridcolor=p["grid"], gridwidth=1, zeroline=False, showline=False, automargin=True)
    return fig


def note(text: str) -> None:
    """A short answer to a click: plain text with one icon, not a coloured box."""
    st.markdown(f":material/info: {text}")


def show_chart(fig, key, explain_fn, hint=":material/ads_click: Click a point for a short explanation."):
    """Render a Plotly chart; clicking a point shows explain_fn(point) below."""
    ev = st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode="points", config={"displayModeBar": False})
    try:
        pts = ev.selection.points
    except Exception:
        pts = []
    if pts:
        try:
            note(explain_fn(pts[0]))
        except Exception:
            note("This point has no explanation attached.")
    else:
        st.caption(hint)


def table(df, **kw):
    """The one table style: no index, full width, as tall as its rows (no inner scrolling for small tables)."""
    return st.dataframe(df, hide_index=True, width="stretch", height="content", **kw)


def selectable_table(df: pd.DataFrame, key: str, **kw):
    """st.dataframe with single-row selection; returns the selected positional row index or None."""
    sel = table(df, on_select="rerun", selection_mode="single-row", key=key, **kw)
    try:
        rows = sel.selection.rows
    except Exception:
        rows = []
    return rows[0] if rows else None


# ----------------------------------------------------------------------------------------------- pump modes
MODE_ORDER = ["collect", "hold", "bypass", "wait"]
MODE_LABEL = {"collect": "Collecting solar heat", "wait": "Standby, pump off", "hold": "Held off in sun", "bypass": "Bypass, overheat protection"}
MODE_SHORT = {"collect": "Collecting", "wait": "Standby", "hold": "Held off", "bypass": "Bypass"}
MODE_DEF = {
    "collect": "The pump runs and moves solar heat from the collector into the tank and PCM. This only happens when the collector is hotter than the tank.",
    "wait": "The pump is off because there is no useful sun (night or cloud) or the collector cannot add net heat. Stored heat keeps serving the draw.",
    "hold": "Sun is available but the controller paused the pump to keep the tank and PCM safely below their limits.",
    "bypass": "The collector is isolated to stop overheating, either chosen by the controller or forced by the safety shield at the guard bands.",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def mode_color(mode: str) -> str:
    p = pal()
    return {"collect": p["blue"], "hold": p["aqua"], "bypass": p["orange"], "wait": p["standby"]}[mode]


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
        return f"Water or PCM reached the guard band ({tw} °C water), so the safety shield isolated the collector."
    if m == "bypass":
        return f"The controller isolated the collector early: tank {tw} °C with {g} W/m² of sun means it is heading for overheating."
    if m == "hold":
        return (f"Sun is available ({g} W/m²) but the controller paused the pump to keep the tank ({tw} °C) and PCM "
                f"({r['tpcm']:.0f} °C) below their limits.")
    if r["ghi"] < 50:
        return f"No useful sun ({g} W/m²), so the pump is off by itself and stored heat serves the {r['demand_L']:.1f} L draw."
    return f"Sun is {g} W/m² but the collector cannot add net heat to a {tw} °C tank."


# ================================================================================================ tab 1: live controller
def _sidebar(P: dict) -> dict:
    """All run settings. Returns the chosen values; every widget here changes what the main area shows."""
    regimes, years = P["regimes"], P["years"]
    s = {}
    with st.sidebar:
        st.subheader("Run configuration")
        s["regime"] = regime = st.selectbox("Regime", list(regimes), index=0, format_func=lambda r: regimes[r], key="p_regime")
        pols = s["pols"] = P["policies"](regime)
        keys = list(pols)
        dflt = P["default_policy"](regime)
        # per-regime key: the option list changes with the regime
        s["policy"] = policy = st.selectbox("Policy", keys, index=keys.index(dflt) if dflt in keys else 0, format_func=pols.get, key=f"p_policy_{regime}")
        s["mode"] = mode = st.radio("Weather source", ["Historical replay"] + (["LIVE (Open-Meteo)"] if P.get("live") else []), key="p_mode")
        s.update(year=years[0], start_h=0, days=7, past_days=2, forecast_days=3, d0=None, d1=None)
        if mode == "Historical replay":
            s["year"] = year = st.selectbox("Year", years, index=years.index(P["default_year"]) if P["default_year"] in years else 0,
                                            help="Train and validation years only. Held-out test years are never offered.", key="p_year")
            md = _month_days(year)
            mi = MONTHS.index(st.selectbox("Start month", MONTHS, index=P["default_month"], key="p_month"))
            dom = st.slider("Start day of month", 1, md[mi], 1, key="p_dom")
            s["days"] = days = st.slider("Window length (days)", 1, 30, 7, key="p_days")
            doy = min(sum(md[:mi]) + dom - 1, sum(md) - days)
            s["start_h"] = doy * 24
            base = datetime.date(2001, 1, 1)                                   # 2001 = a non-leap calendar
            s["d0"] = (base + datetime.timedelta(days=doy)).replace(year=year)
            s["d1"] = (base + datetime.timedelta(days=doy + days - 1)).replace(year=year)
            st.caption(f"Window: {s['d0']:%d %b %Y} to {s['d1']:%d %b %Y}")
        else:
            s["past_days"] = st.slider("Past days", 0, 5, 2, key="p_past")
            if P.get("live_forecast_days", True):
                s["forecast_days"] = st.slider("Forecast days", 1, 7, 3, key="p_fc")
        others = ["(none)"] + [k for k in keys if k != policy]
        s["other"] = st.selectbox("Compare with", others, format_func=lambda k: "(none)" if k == "(none)" else pols[k],
                                  key=f"p_other_{regime}", help="Runs a second policy over the same window.")
        s["step"] = st.select_slider("Hour slider step", [1, 3, 6, 12, 24], value=1, key="p_step")
        s["run"] = st.button("Run", type="primary", icon=":material/play_arrow:", width="stretch", key="p_run")
    return s


def _run(P: dict, s: dict) -> None:
    ss = st.session_state
    want = [s["policy"], P["always_on"]] + ([s["other"]] if s["other"] != "(none)" else [])
    with st.spinner("Running the policy and an always-on pump through the frozen physics..."):
        try:
            ctx = None
            if s["mode"].startswith("LIVE"):
                runs, ctx = P["live"](s["regime"], want, s["past_days"], s["forecast_days"])
            else:
                runs = {k: P["replay"](s["regime"], k, s["year"], s["start_h"], s["days"] * 24) for k in dict.fromkeys(want)}
        except Exception as e:                                              # noqa: BLE001
            st.error(f"Run failed: {e}")
            st.caption("For LIVE mode check network access to api.open-meteo.com, or switch the weather source to Historical replay.")
            st.stop()
    ss.p_recs, ss.p_base, ss.p_ctx = runs[s["policy"]], runs[P["always_on"]], ctx
    ss.p_recs2 = runs[s["other"]] if s["other"] != "(none)" else None
    desc = f"{s['d0']:%d %b} to {s['d1']:%d %b} {s['year']}" if s["mode"] == "Historical replay" else "present-day weather"
    ss.p_meta = {"mode": s["mode"], "label": s["pols"][s["policy"]], "regime": s["regime"], "desc": desc,
                 "other": s["other"], "other_label": s["pols"].get(s["other"], "")}


def _headline(P: dict, df: pd.DataFrame, base: list | None) -> None:
    """The three results that matter, at the top. Deltas are real comparisons against the always-on pump on the same window."""
    last, hours = df.iloc[-1], P["shield_unit"] == "hours"
    b = pd.DataFrame(base).iloc[-1] if base else None
    fmt = (lambda v: f"{v:,.1f}") if hours else (lambda v: f"{int(v):,}")
    c1, c2, c3 = st.columns(3)
    unit = "hours" if hours else "hits"
    if b is not None:
        d_del = 100 * (last["cum_del"] - b["cum_del"]) / b["cum_del"] if b["cum_del"] else 0.0
        cut = 100 * (b["cum_shield"] - last["cum_shield"]) / b["cum_shield"] if b["cum_shield"] else 0.0
        c1.metric("Hot water delivered", f"{last['cum_del']:.1f} kWh", f"{d_del:+.1f}% vs always-on ({b['cum_del']:.1f} kWh)", delta_color="off",
                  help="Useful heat delivered to the hot-water draw over the window.")
        c2.metric(f"Overheat shield {unit}", fmt(last["cum_shield"]), f"{cut:.0f}% fewer than always-on ({fmt(b['cum_shield'])})", delta_color="off",
                  help="Time the safety shield had to isolate the collector. Fewer means less wasted heat and less thermal stress.")
    else:
        c1.metric("Hot water delivered", f"{last['cum_del']:.1f} kWh")
        c2.metric(f"Overheat shield {unit}", fmt(last["cum_shield"]))
    c3.metric("Hard-limit breaches", f"{last['cum_safety']:g}", help=f"Water above {P['guards'][1]:g} °C or PCM above its limit. The target is 0.")


def _selected_hour(P: dict, df: pd.DataFrame, step: int) -> int:
    k = st.slider("Hour in window", 0, len(df) - 1, min(24, len(df) - 1), step=int(step), key="p_hour")
    row = df.iloc[k]
    forced = " · :orange[safety shield forced this hour]" if row["overridden"] else ""
    st.markdown(f"**{MODE_LABEL[row['mode']]}** · {row['ts']}{forced}")
    st.write(row["why"])
    table(pd.DataFrame([{"Sun (W/m²)": row["ghi"], "Outside (°C)": row["tamb"], "Tank water (°C)": row["tw"], "PCM (°C)": row["tpcm"],
                                "PCM melted (%)": row["melt"], "Hot water drawn (L)": row["demand_L"]}]), column_config={c: st.column_config.NumberColumn(c, format="%.1f") for c in
                                ["Sun (W/m²)", "Outside (°C)", "Tank water (°C)", "PCM (°C)", "PCM melted (%)", "Hot water drawn (L)"]})
    return k


def _charts(P: dict, df: pd.DataFrame, k: int, ctx: dict | None, is_live: bool) -> None:
    p = pal()
    g_guard, g_limit, g_target = P["guards"]
    now_idx = min(ctx["now_idx"], len(df) - 1) if is_live and ctx else None

    def cursor(fig):
        fig.add_vline(x=k, line=dict(color=p["ink"], width=HOUR_CURSOR_PX))
        if now_idx is not None:
            fig.add_vline(x=now_idx, line=dict(color=p["ink2"], width=1, dash="dot"), annotation_text="now", annotation_position="top")
            fig.add_vrect(x0=now_idx, x1=len(df) - 1, fillcolor=p["blue"], opacity=0.06, line_width=0)

    fig_t = go.Figure()
    for col, name, colour in (("tw", "Tank water", p["blue"]), ("tpcm", "PCM", p["orange"])):
        fig_t.add_scatter(x=df["k"], y=df[col], name=name, line=dict(color=colour, width=2),
                          hovertemplate="%{customdata}<br>" + name.lower() + " %{y:.1f} °C<extra></extra>", customdata=df["ts"])
    for y, colour, dash, label in ((g_target, p["aqua"], "dot", "delivery target"), (g_guard, p["warn"], "dash", "water guard"),
                                   (g_limit, p["bad"], "solid", "water limit")):
        fig_t.add_hline(y=y, line=dict(color=colour, dash=dash, width=1), annotation_text=f"{label} {y:g} °C", annotation_position="top right",
                        annotation_font=dict(size=11, color=p["ink2"]))
    cursor(fig_t)
    style_fig(fig_t, "Tank water and PCM temperature", H_MAIN, x="Hour in window", y="Temperature (°C)")

    fig_r = go.Figure()
    for m in MODE_ORDER:
        sub = df[df["mode"] == m]
        fig_r.add_bar(x=sub["k"], y=[1] * len(sub), name=MODE_SHORT[m], marker_color=mode_color(m), marker_line_width=0,
                      customdata=sub["ts"], hovertemplate="%{customdata}<br>" + MODE_SHORT[m] + "<extra></extra>")
    fig_r.update_layout(barmode="overlay", bargap=0)
    fig_r.update_yaxes(visible=False, showgrid=False)
    cursor(fig_r)
    style_fig(fig_r, "What the pump did each hour", H_STRIP, x="Hour in window")
    fig_r.update_yaxes(visible=False, showgrid=False)

    fig_g = go.Figure(go.Scatter(x=df["k"], y=df["ghi"], fill="tozeroy", line=dict(color=p["orange"], width=2), fillcolor="rgba(235,104,52,0.15)",
                                 name="GHI", hovertemplate="%{customdata}<br>%{y:.0f} W/m²<extra></extra>", customdata=df["ts"]))
    cursor(fig_g)
    style_fig(fig_g, "Solar irradiance", H_SMALL, x="Hour in window", y="GHI (W/m²)", legend=False)

    df["daylab"] = df["ts"].apply(lambda s: s.rsplit(" ", 1)[0])
    day_order = list(dict.fromkeys(df["daylab"]))
    fig_d = go.Figure()
    for m in MODE_ORDER:
        cnts = df[df["mode"] == m].groupby("daylab").size()
        fig_d.add_bar(name=MODE_SHORT[m], x=day_order, y=[int(cnts.get(d, 0)) for d in day_order], marker_color=mode_color(m), marker_line_width=0)
    fig_d.update_layout(barmode="stack")
    style_fig(fig_d, "Pump hours per day", H_BAR, x="Day", y="Hours")

    def ex_hour(pt):
        r = df.iloc[int(pt["x"])]
        return f"**{r['ts']}** · {MODE_SHORT[r['mode']]}: water {r['tw']:.1f} °C, PCM {r['tpcm']:.1f} °C, sun {r['ghi']:.0f} W/m². {r['why']}"

    def ex_ghi(pt):
        r = df.iloc[int(pt["x"])]
        if r["ghi"] < 50:
            return f"**{r['ts']}** · sun is only {r['ghi']:.0f} W/m². Below about 50 W/m² the collector cannot add heat, so the pump idles."
        return (f"**{r['ts']}** · sun is {r['ghi']:.0f} W/m². More sun heats faster; around midday it pushes the tank toward the guard band, "
                f"which is when the controller must step in.")

    def ex_day(pt):
        sub = df[df["daylab"] == pt["x"]]
        parts = ", ".join(f"{MODE_SHORT[m]} {int((sub['mode'] == m).sum())} h" for m in MODE_ORDER)
        return f"**{pt['x']}** · {parts}. The pump is held off or bypassed only for a few hot midday hours; the rest is collecting or standby."

    left, right = st.columns([2, 1])
    with left:
        show_chart(fig_t, "fig_t", ex_hour, ":material/ads_click: Click a point to see what the controller did that hour and why.")
        show_chart(fig_r, "fig_r", ex_hour, ":material/ads_click: Click a bar to see what the pump did that hour.")
        show_chart(fig_g, "fig_g", ex_ghi)
    with right:
        st.markdown("**Pump hours by mode**")
        mix = df["mode"].value_counts()
        table(pd.DataFrame([{"Mode": MODE_SHORT[m], "Hours": int(mix.get(m, 0)), "Share (%)": 100 * mix.get(m, 0) / len(df), "Meaning": MODE_DEF[m]}
                                   for m in MODE_ORDER]), column_config={"Hours": st.column_config.NumberColumn(format="%d"), "Share (%)": st.column_config.NumberColumn(format="%.0f"),
                                    "Meaning": st.column_config.TextColumn(width="large")})
        st.caption(P["pump_note"])
        st.markdown("**Hours around the selected one**")
        lo, hi = max(0, k - 5), min(len(df), k + 6)
        tbl = df.iloc[lo:hi][["ts", "mode", "pump", "tw", "tpcm", "ghi"]].copy()
        tbl["mode"] = tbl["mode"].map(MODE_SHORT)
        tbl = tbl.rename(columns={"ts": "Time", "mode": "Pump", "pump": "Pump on", "tw": "Water (°C)", "tpcm": "PCM (°C)", "ghi": "Sun (W/m²)"})
        i = selectable_table(tbl, "tbl_hours", column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                                              ["Pump on", "Water (°C)", "PCM (°C)", "Sun (W/m²)"]})
        if i is not None:
            note(f"**{df.iloc[lo + i]['ts']}** · {df.iloc[lo + i]['why']}")
        else:
            st.caption(":material/ads_click: Select a row for the reason.")
    show_chart(fig_d, "fig_d", ex_day)

    if is_live and ctx:
        import plotly.express as px
        st.caption(f"Live weather: {ctx['label']} ({ctx['lat']:.2f}, {ctx['lon']:.2f}), Open-Meteo, fetched {ctx['fetched']}. "
                   f"The dotted line marks now; the shaded region is the forecast part of the window.")
        fig_map = px.scatter_map(pd.DataFrame({"lat": [ctx["lat"]], "lon": [ctx["lon"]], "label": [ctx["label"]]}), lat="lat", lon="lon",
                                 hover_name="label", zoom=6, height=H_MAP)
        fig_map.update_traces(marker=dict(size=12, color=p["blue"]))
        fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, width="stretch", config={"displayModeBar": False})


def _comparison(P: dict, meta: dict, df: pd.DataFrame, base: list, df2: pd.DataFrame, k: int, ctx: dict | None, is_live: bool) -> None:
    p = pal()
    other = meta["other_label"]
    st.subheader(f"Comparison with {other}")
    unit = "hours" if P["shield_unit"] == "hours" else "hits"
    rows = [(meta["label"], df), (other, df2)] + ([("B0 always-on pump", pd.DataFrame(base))] if base else [])
    table(pd.DataFrame([{"Policy": n, "Delivered (kWh)": d.iloc[-1]["cum_del"], f"Shield ({unit})": d.iloc[-1]["cum_shield"]} for n, d in rows]), column_config={"Delivered (kWh)": st.column_config.NumberColumn(format="%.1f"), f"Shield ({unit})": st.column_config.NumberColumn(format="%.1f")})
    fig = go.Figure()
    for name, d in rows:
        fig.add_scatter(x=d["k"], y=d["cum_del"], name=name, line=dict(color=ctrl_color(name), width=2, dash="dot" if name.startswith("B0") else "solid"))
    if is_live and ctx:
        fig.add_vline(x=min(ctx["now_idx"], len(df) - 1), line=dict(color=p["ink2"], width=1, dash="dot"), annotation_text="now", annotation_position="top")
    style_fig(fig, "Cumulative delivered heat", H_BAR, x="Hour in window", y="Delivered heat (kWh)")

    def ex(pt):
        i = int(pt["x"])
        a, b = df.iloc[min(i, len(df) - 1)], df2.iloc[min(i, len(df2) - 1)]
        return (f"By **{a['ts']}**: {meta['label']} has delivered {a['cum_del']:.1f} kWh and {other} {b['cum_del']:.1f} kWh. "
                f"A bigger gap means one policy collects or stores heat better.")
    show_chart(fig, "fig_cmp", ex)


def render_live(P: dict) -> None:
    s = _sidebar(P)
    ss = st.session_state
    for k in ("p_recs", "p_base", "p_recs2", "p_meta", "p_ctx"):
        ss.setdefault(k, None)
    if s["run"]:
        _run(P, s)
    if ss.p_recs is None:
        st.subheader("No run yet")
        st.write("Choose a regime, policy and weather window in the sidebar, then press **Run**. The portal replays the policy through the "
                 "frozen physics and compares it with an always-on pump on the same weather.")
        return
    meta, ctx = ss.p_meta, ss.p_ctx
    df = pd.DataFrame(ss.p_recs)
    df["mode"] = df.apply(pump_mode, axis=1)
    df["why"] = df.apply(hour_reason, axis=1)
    is_live = meta["mode"].startswith("LIVE") and ctx is not None
    st.caption(f"{'Live' if is_live else 'Replay'} · {meta['label']} · regime {meta['regime']} · "
               f"{ctx['label'] + ' · fetched ' + ctx['fetched'] if is_live else meta['desc']}")
    _headline(P, df, ss.p_base)
    st.subheader("Selected hour")
    k = _selected_hour(P, df, s["step"])
    st.subheader("Charts")
    _charts(P, df, k, ctx, is_live)
    if meta.get("other") not in (None, "(none)") and ss.p_recs2 is not None:
        _comparison(P, meta, df, ss.p_base, pd.DataFrame(ss.p_recs2), k, ctx, is_live)


# ================================================================================================ tab 2: deployment and models
CTRL_DESC = {
    "B0": "Always-on pump (the Objective-2 operating point); only the safety shield protects it from overheating.",
    "B1": "Reactive rules: bypass near the guard bands, otherwise charge when sunny.",
    "B2": "Rule-based controller that also uses a forecast of upcoming sun.",
    "Wa": "Tank without PCM (the reference the PCM must beat).",
}


def _controller_table(dep: dict) -> pd.DataFrame:
    ctr = dep["controllers"]
    cols = {"delivered_kWh": "Delivered (kWh/yr)", "thermal_efficiency_pct": "Thermal efficiency (%)", "demand_coverage_pct": "Demand coverage (%)",
            "shield_h": "Shield (h/yr)", "violations": "Hard-limit breaches"}
    rows = [{"Controller": n, **{label: v[key] for key, label in cols.items() if key in v}} for n, v in ctr.items()]
    return pd.DataFrame(rows)


def render_deployment(P: dict) -> None:
    """P['deployment'](regime) -> {"controllers": {name: {delivered_kWh, shield_h, violations, thermal_efficiency_pct, demand_coverage_pct}},
    "inference": {name: {ms, params, kB}}, "faults": DataFrame[policy, fault, delivered_kWh, shield_h, violations], "note": str,
    "assumptions": str} or None"""
    d_regime = st.selectbox("Regime", list(P["regimes"]), key="d_regime", format_func=lambda r: P["regimes"][r])
    with st.spinner("Loading deployment metrics..."):
        dep = P["deployment"](d_regime) if P.get("deployment") else None
    if not dep:
        st.subheader("No deployment data for this regime")
        st.write("The deployment metrics file for this regime has not been generated yet. It is produced by the phase-9 (or M10) script; "
                 "the other regimes that have data are available in the selector above.")
    else:
        ctr = dep["controllers"]
        st.caption(dep.get("assumptions", "") or "Simulated validation years for the chosen regime.")
        st.subheader("Controllers")
        table(_controller_table(dep), column_config={"Delivered (kWh/yr)": st.column_config.NumberColumn(format="%.0f"),
                                    "Thermal efficiency (%)": st.column_config.NumberColumn(format="%.1f"),
                                    "Demand coverage (%)": st.column_config.NumberColumn(format="%.1f"),
                                    "Shield (h/yr)": st.column_config.NumberColumn(format="%.0f"),
                                    "Hard-limit breaches": st.column_config.NumberColumn(format="%d")})
        order = list(ctr)
        colours = [ctrl_color(c) for c in order]
        b0 = next((c for c in order if c.startswith("B0")), None)
        left, right = st.columns(2)
        with left:
            f1 = go.Figure(go.Bar(x=order, y=[ctr[c]["delivered_kWh"] for c in order], marker_color=colours, marker_line_width=0))
            style_fig(f1, f"Heat delivered to {P['guards'][2]:g} °C", H_BAR, y="kWh per year", legend=False)
            show_chart(f1, "dfig1", lambda pt: f"**{pt['x']}** · {pt['y']:.0f} kWh/yr. {CTRL_DESC.get(str(pt['x'])[:2], '')}")
        if all("shield_h" in ctr[c] for c in order):
            with right:
                f2 = go.Figure(go.Bar(x=order, y=[ctr[c]["shield_h"] for c in order], marker_color=colours, marker_line_width=0))
                style_fig(f2, "Hours the safety shield acted (fewer is better)", H_BAR, y="Hours per year", legend=False)

                def ex2(pt):
                    msg = f"**{pt['x']}** · {pt['y']:.0f} shield hours per year."
                    if b0 and ctr[b0]["shield_h"]:
                        msg += f" That is {100 * (1 - pt['y'] / ctr[b0]['shield_h']):.0f}% fewer than the always-on pump."
                    return msg + " Fewer hours means less heat dumped and less thermal stress."
                show_chart(f2, "dfig2", ex2)
        inf = dep.get("inference")
        if inf:
            st.subheader("Inference cost")
            table(pd.DataFrame([{"Policy": n, "Latency per decision (µs)": v["ms"] * 1000, "Weights, fp32 (kB)": v["kB"], "Parameters": v["params"]}
                                       for n, v in inf.items()]), column_config={"Latency per decision (µs)": st.column_config.NumberColumn(format="%.0f"),
                                        "Weights, fp32 (kB)": st.column_config.NumberColumn(format="%.0f"),
                                        "Parameters": st.column_config.NumberColumn(format="%d")})
            st.caption((dep.get("note") or "") + " The controller decides once per hour, so even a slow microcontroller has large headroom.")
        faults = dep.get("faults")
        if faults is not None and len(faults):
            st.subheader("Sensor-fault robustness")
            st.caption("The observation is corrupted; the physics and the safety shield use the true state.")
            fdf = faults
            f4 = go.Figure()
            for a in dict.fromkeys(fdf["policy"]):
                sub = fdf[fdf["policy"] == a]
                f4.add_bar(name=a, x=sub["fault"], y=sub["delivered_kWh"], marker_color=ctrl_color(a), marker_line_width=0)
            f4.update_layout(barmode="group")
            style_fig(f4, "Delivered heat under sensor faults", H_BAR, y="kWh per year")
            f4.update_xaxes(tickangle=25)

            def ex4(pt):
                a = f4.data[pt["curve_number"]].name
                nom = fdf[(fdf["policy"] == a) & (fdf["fault"].isin(["nominal", "none"]))]
                nom_txt = f" (clean sensors: {nom['delivered_kWh'].iloc[0]:.0f} kWh)" if len(nom) else ""
                return (f"**{a} with fault '{pt['x']}'** · delivers {pt['y']:.0f} kWh/yr{nom_txt}. The policy is fed wrong readings, but the safety "
                        f"shield uses the true state, so safety holds and only efficiency can slip.")
            show_chart(f4, "dfig4", ex4)
            n_viol = int(fdf["violations"].sum())
            if n_viol == 0:
                st.caption(f":green[No hard-limit violations] in any of the {len(fdf)} policy and fault scenarios.")
            else:
                st.warning(f"{n_viol} hard-limit violations recorded. Investigate before relying on this controller.")
            with st.expander("All sensor-fault results"):
                table(fdf, column_config={"delivered_kWh": st.column_config.NumberColumn("Delivered (kWh/yr)", format="%.1f"),
                                            "shield_h": st.column_config.NumberColumn("Shield (h/yr)", format="%.0f"),
                                            "violations": st.column_config.NumberColumn("Hard-limit breaches", format="%d")})
    rows = P.get("algo_rows")
    if rows:
        st.subheader("Algorithms compared")
        i = selectable_table(pd.DataFrame(rows, columns=["Algorithm", "Family", "Outcome"]), "tbl_algos")
        if i is not None:
            r = rows[i]
            note(f"**{r[0]}** ({r[1]}): {r[2]}. Always-charge means it learned nothing better than leaving the pump on; a policy counts as a success "
                 f"only if it beats that with fewer shield hours.")
        else:
            st.caption(":material/ads_click: Select a row for a short explanation.")
        if P.get("selection_note"):
            st.caption(P["selection_note"])


def run_portal(P: dict) -> None:
    st.title(P["title"])
    st.caption(P["caption"])
    tab_live, tab_dep = st.tabs([":material/sensors: Live controller", ":material/monitoring: Deployment and models"])
    with tab_live:
        render_live(P)
    with tab_dep:
        render_deployment(P)
