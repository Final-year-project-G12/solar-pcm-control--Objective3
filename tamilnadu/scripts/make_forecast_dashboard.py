"""Forecast-realism dashboard: no-forecast (V1) vs realistic smart-forecast (V3)
locked-test comparison, all three Tamil Nadu regimes, DQN + SAC.

Usage: python scripts/make_forecast_dashboard.py
Writes: results/plots/forecast_comparison/dashboard.html
"""
import pathlib
import plotly.graph_objects as go
import plotly.io as pio

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "plots" / "forecast_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_NO_FC = "#4C78A8"   # blue   — no forecast field at all (V1)
COLOR_NONE  = "#54A24B"   # green  — V3 shape, forecast zeroed ("none")
COLOR_SMART = "#F58518"   # orange — realistic smart-persistence forecast (V3)
INK = "#0b0b0b"
MUTED = "#52514e"

# locked-test (2011-2015) results, see docs/CORRECTIONS_OBJECTIVE3.md O3-5/O3-6
# regime, policy, delivery_V1, shield_V1, delivery_none, shield_none, delivery_smart, shield_smart
ROWS = [
    (0, "DQN", 96.2, 62.0, 100.0, 0.0, 99.6, 18.3),
    (0, "SAC", 97.5, 54.0,  99.1, 49.8, 98.4, 51.8),
    (1, "SAC", 98.7, 37.0,  99.3, 29.0, 99.7, 26.2),
    (2, "SAC", 97.3, 33.5,  98.9, 41.0, 98.8, 35.9),
]

labels = [f"R{r} {p}" for r, p, *_ in ROWS]

fig_shield = go.Figure()
fig_shield.add_bar(name="No forecast field at all (V1)", x=labels,
                    y=[row[3] for row in ROWS], marker_color=COLOR_NO_FC,
                    text=[f"{row[3]:.1f}%" for row in ROWS], textposition="outside")
fig_shield.add_bar(name="V3 shape, forecast zeroed (\"none\")", x=labels,
                    y=[row[5] for row in ROWS], marker_color=COLOR_NONE,
                    text=[f"{row[5]:.1f}%" for row in ROWS], textposition="outside")
fig_shield.add_bar(name="Realistic 3h forecast (V3, smart persistence)", x=labels,
                    y=[row[7] for row in ROWS], marker_color=COLOR_SMART,
                    text=[f"{row[7]:.1f}%" for row in ROWS], textposition="outside")
fig_shield.update_layout(
    title="Shield-activation reduction vs. B0 — across all three forecast conditions",
    yaxis_title="Shield reduction vs. B0 (%)", barmode="group",
    plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    font=dict(color=INK), legend=dict(orientation="h", y=-0.2),
    margin=dict(t=60, b=80),
)

fig_delivery = go.Figure()
fig_delivery.add_bar(name="No forecast field at all (V1)", x=labels,
                      y=[row[2] for row in ROWS], marker_color=COLOR_NO_FC,
                      text=[f"{row[2]:.1f}%" for row in ROWS], textposition="outside")
fig_delivery.add_bar(name="V3 shape, forecast zeroed (\"none\")", x=labels,
                      y=[row[4] for row in ROWS], marker_color=COLOR_NONE,
                      text=[f"{row[4]:.1f}%" for row in ROWS], textposition="outside")
fig_delivery.add_bar(name="Realistic 3h forecast (V3, smart persistence)", x=labels,
                      y=[row[6] for row in ROWS], marker_color=COLOR_SMART,
                      text=[f"{row[6]:.1f}%" for row in ROWS], textposition="outside")
fig_delivery.update_layout(
    title="Delivered energy — % of B0 — across all three forecast conditions",
    yaxis_title="Delivery (% of B0)", yaxis_range=[90, 101], barmode="group",
    plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    font=dict(color=INK), legend=dict(orientation="h", y=-0.2),
    margin=dict(t=60, b=80),
)

# sensor-fault re-evaluation: GHI dropout/stuck pyranometer, % delivery drop
fig_sensor = go.Figure()
sensor_rows = [
    ("R0 SAC", 0.0, 9.1, 14.0),
    ("R1 SAC", 99.5, 13.2, 1.2),
    ("R2 SAC", 15.0, 2.2, 2.4),
]
fig_sensor.add_bar(name="V1 (original checkpoint)", x=[r[0] for r in sensor_rows],
                    y=[r[1] for r in sensor_rows], marker_color=COLOR_NO_FC,
                    text=[f"{r[1]:.1f}%" for r in sensor_rows], textposition="outside")
fig_sensor.add_bar(name="V3 + none", x=[r[0] for r in sensor_rows],
                    y=[r[2] for r in sensor_rows], marker_color=COLOR_NONE,
                    text=[f"{r[2]:.1f}%" for r in sensor_rows], textposition="outside")
fig_sensor.add_bar(name="V3 + smart", x=[r[0] for r in sensor_rows],
                    y=[r[3] for r in sensor_rows], marker_color=COLOR_SMART,
                    text=[f"{r[3]:.1f}%" for r in sensor_rows], textposition="outside")
fig_sensor.update_layout(
    title="Sensor-fault re-evaluation — % delivery drop under GHI dropout/stuck pyranometer",
    yaxis_title="Delivery drop (%)", barmode="group",
    plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    font=dict(color=INK), legend=dict(orientation="h", y=-0.2),
    margin=dict(t=60, b=80),
)

html_parts = [
    '<!doctype html><html><head><meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    '<title>Objective 3 — forecast realism</title>',
    '<style>body{background:#f9f9f7;margin:0;padding:16px;'
    'font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;'
    'max-width:980px;margin-left:auto;margin-right:auto}'
    'h1{font-size:20px}h2{font-size:15px;color:#52514e;font-weight:600;margin-top:28px}'
    '.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);'
    'border-radius:10px;margin:14px 0;padding:8px}'
    '.note{background:#fff8e6;border:1px solid #e6c766;border-radius:8px;'
    'padding:12px 16px;font-size:14px;line-height:1.5;margin:14px 0}'
    'table{border-collapse:collapse;width:100%;font-size:14px}'
    'td,th{padding:6px 10px;border-bottom:1px solid rgba(11,11,11,.08);text-align:left}'
    'a{color:#2a78d6}code{background:#eee;padding:1px 5px;border-radius:4px}'
    '</style></head><body>',
    '<h1>Objective 3 — does the controller advantage survive a realistic weather forecast?</h1>',
    '<p style="color:#52514e">Tamil Nadu PCM-SWH, locked test years 2011–2015. '
    'Three forecast conditions compared: <b>no forecast field at all</b> '
    '(<code>obs_version="V1"</code>), the same 15-dim shape with the forecast '
    '<b>zeroed</b> (<code>obs_version="V3"</code>, <code>forecast_mode="none"</code> — '
    'the ablation floor), and a <b>realistic 1–3h GHI forecast</b> '
    '(<code>obs_version="V3"</code>, <code>forecast_mode="smart"</code> — smart '
    'persistence on the clear-sky index, not a perfect look-ahead). See '
    '<a href="../../../docs/CORRECTIONS_OBJECTIVE3.md">CORRECTIONS_OBJECTIVE3.md §O3-5–O3-6</a> '
    'for the full method and tests.</p>',
    '<div class="note"><b>Caveat on the DQN row (regime 0), resolved by a controlled check:</b> '
    'the chart’s DQN bars compare “no forecast” to “realistic forecast” — '
    'not a controlled, same-seed ablation. DQN has substantially higher training variance in this '
    'project than SAC, so a same-seed, same-obs-shape check (<code>obs_version="V3"</code>, '
    '<code>forecast_mode="perfect"</code> instead of <code>"smart"</code>) was run to isolate the '
    'forecast effect. Result: that same seed, given a strictly <i>better</i> (perfect) forecast than '
    'the original no-forecast run, only reaches <b>28.3%</b> shield cut — not 62%. So most of the '
    'originally-reported 62%→18.3% gap is retrain variance, and the true, isolated '
    'forecast-realism effect is <b>28.3%→18.3%</b> (10 points) — real, but far smaller than '
    'first appeared. SAC is not subject to this caveat: all four SAC rows use the same framing and SAC '
    'has low retrain variance in this project, so its near-unchanged shield reduction is good evidence '
    'the result is real.</div>',
    '<div class="card">', pio.to_html(fig_shield, include_plotlyjs="cdn", full_html=False), '</div>',
    '<div class="card">', pio.to_html(fig_delivery, include_plotlyjs=False, full_html=False), '</div>',
    '<h2>Numbers behind the charts</h2>',
    '<div class="card"><table><tr><th>Regime</th><th>Policy</th>'
    '<th>Delivery (V1)</th><th>Shield cut (V1)</th>'
    '<th>Delivery (none)</th><th>Shield cut (none)</th>'
    '<th>Delivery (smart)</th><th>Shield cut (smart)</th></tr>',
]
for r, p, d0, s0, dn, sn, d1, s1 in ROWS:
    html_parts.append(f"<tr><td>{r}</td><td>{p}</td><td>{d0:.1f}%</td>"
                       f"<td>{s0:.1f}%</td><td>{dn:.1f}%</td><td>{sn:.1f}%</td>"
                       f"<td>{d1:.1f}%</td><td>{s1:.1f}%</td></tr>")
html_parts.append("</table></div>")
html_parts.append(
    '<div class="card"><h2 style="margin-top:0">Sensor-fault re-evaluation</h2>'
    + pio.to_html(fig_sensor, include_plotlyjs=False, full_html=False)
    + '<p style="font-size:13px;color:#52514e">The original V1 checkpoint showed a severe '
      '~99.5% delivery collapse for SAC regime 1 under a dead/stuck pyranometer '
      '(CORRECTIONS_OBJECTIVE3.md §O3-4). Re-running the sensor-fault suite on the newer V3 '
      'checkpoints (none/smart) does not reproduce this — both show mild degradation '
      '(1.2%–13.2%), in the same range as regimes 0 and 2. This suggests the severe regime-1 '
      'vulnerability was specific to that one training run, not an inherent property of '
      '“SAC on regime 1,” though per-checkpoint sensor-fault testing before deployment '
      'remains necessary either way. Zero hard-safety violations in every cell (shield acts on '
      'true state regardless of what the policy observes).</p></div>'
)
html_parts.append(
    '<div class="card"><h2 style="margin-top:0">Controlled DQN r0 ablation result</h2>'
    '<table><tr><th>Variant (same seed=42, same steps)</th><th>Delivery</th><th>Shield cut</th></tr>'
    '<tr><td>V1, no forecast (original)</td><td>96.2%</td><td>62%</td></tr>'
    '<tr><td>V3 + perfect forecast (controlled)</td><td>97.7%</td><td><b>28.3%</b></td></tr>'
    '<tr><td>V3 + smart forecast (realistic)</td><td>99.6%</td><td>18.3%</td></tr>'
    '</table>'
    '<p style="font-size:13px;color:#52514e;margin-bottom:0">Since the perfect-forecast variant '
    'objectively has <i>more</i> information than the no-forecast run, the 62%→28.3% portion '
    'of the gap cannot be a forecast effect — it is retrain variance. The genuine, isolated '
    'forecast-realism effect is 28.3%→18.3%: real, but far smaller than the naive 62%→18.3% '
    'comparison implied.</p></div>'
)
html_parts.append("</body></html>")

out = OUT_DIR / "dashboard.html"
out.write_text("\n".join(html_parts), encoding="utf-8")
print(f"[INFO] forecast-realism dashboard -> {out}")
