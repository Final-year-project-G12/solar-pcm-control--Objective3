"""
make_gen2_plots.py — the plan-19.5 visualisations from the Generation-2 study (Plotly, one self-contained HTML).

  12 Pareto frontier   : delivered kWh vs shield hours, one point per (weight, seed), weight = colour, seed = marker; mean curve
  13 Seed robustness   : delivery and shield hours across seeds at each weight
  14 Sub-hourly ablation: 60 vs 15 minute control, same weight and seed
  15 Regime comparison : % of B0 delivery and shield cut, per regime
  13b SAC vs DQN seed spread, 14b ingredient ablation, 16 randomisation overlay: need gen2_study.py --algo sac / gen2_ablation.py first;
  each is drawn only when its input exists.

    python scripts/make_gen2_plots.py --algo dqn        -> results/plots/gen2/gen2_<algo>.html
"""

import argparse
import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()
from o3.utils.cli import add_parallel_args  # noqa: E402

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"]
SYM = {42: "circle", 0: "square", 123: "diamond"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="dqn")
    ap.add_argument("--regime", type=int, default=1)
    add_parallel_args(ap)
    a = ap.parse_args()
    g = config_O2.RESULTS_DIR / "gen2"
    s = pd.read_csv(g / f"gen2_summary_{a.algo}.csv")
    s = s[~s.tag.str.contains("_V2_|_V3_60min$")]
    r = s[(s.regime_id == a.regime) & (s.control_step_min == 60)]
    b0 = r.iloc[0]
    figs = []

    f = go.Figure()
    ws = sorted(r.shield_weight.unique())
    for i, w in enumerate(ws):
        d = r[r.shield_weight == w]
        f.add_scatter(x=d.shield_h, y=d.E45_kWh, mode="markers", name=f"w = {w:g}", marker=dict(size=13, color=COL[i % 6],
                      symbol=[SYM.get(int(x), "circle") for x in d.seed], line=dict(width=1, color="white")),
                      text=[f"seed {int(x)}" for x in d.seed])
    m = r.groupby("shield_weight")[["shield_h", "E45_kWh"]].mean()
    f.add_scatter(x=m.shield_h, y=m.E45_kWh, mode="lines", name="mean per weight", line=dict(color="#52514e", dash="dot"))
    f.add_scatter(x=[b0.B0_shield_h], y=[b0.B0_E45_kWh], mode="markers+text", text=["B0"], textposition="top center", name="B0",
                  marker=dict(size=15, color="black", symbol="star"))
    f.add_scatter(x=[b0.B1_shield_h], y=[b0.B1_E45_kWh], mode="markers+text", text=["B1"], textposition="top center", name="B1",
                  marker=dict(size=15, color="#898781", symbol="x"))
    f.update_layout(title=f"12 · Pareto frontier ({a.algo.upper()}, regime {a.regime}): delivery vs shield hours — up-and-left is better",
                    xaxis_title="Shield hours per year (fewer →)", yaxis_title="Delivered to 45 C (kWh/yr)", xaxis_autorange="reversed", height=520)
    figs.append(f)

    f = make_subplots(rows=1, cols=2, subplot_titles=("% of B0 delivery", "Shield cut vs B0 (%)"))
    for i, (col, k) in enumerate((("pct_of_B0_delivery", 1), ("shield_cut_vs_B0_pct", 2))):
        f.add_box(x=r.shield_weight.astype(str), y=r[col], boxpoints="all", marker_color=COL[0], showlegend=False, row=1, col=k)
    f.add_hline(y=95, line_dash="dot", line_color="#e34948", row=1, col=1)
    f.update_layout(title="13 · Seed robustness across shield weights (3 seeds each); 95 % line = plan 18.1 delivery rule", height=440)
    figs.append(f)

    c = s[(s.regime_id == a.regime) & (s.shield_weight == s[s.regime_id == a.regime].shield_weight.iloc[0]) & False]
    both = s[(s.regime_id == a.regime) & (s.seed == 42) & (s.shield_weight == 1.7) & s.control_step_min.isin([60, 15])]
    if len(both) == 2:
        f = make_subplots(rows=1, cols=2, subplot_titles=("Delivered (kWh/yr)", "Shield hours/yr"))
        f.add_bar(x=both.control_step_min.astype(str) + " min", y=both.E45_kWh, marker_color=COL[0], showlegend=False, row=1, col=1)
        f.add_bar(x=both.control_step_min.astype(str) + " min", y=both.shield_h, marker_color=COL[1], showlegend=False, row=1, col=2)
        f.update_layout(title="14 · Control-step ablation (same weight, same seed)", height=420)
        figs.append(f)

    reg = s[(s.shield_weight == s.regime_id.map({0: 3.9, 1: 1.7, 2: 1.7})) & (s.control_step_min == 60) & (s.seed == 42)]
    if len(reg) > 1:
        f = make_subplots(rows=1, cols=2, subplot_titles=("% of B0 delivery", "Shield cut vs B0 (%)"))
        f.add_bar(x=reg.regime_id.astype(str), y=reg.pct_of_B0_delivery, marker_color=COL[2], showlegend=False, row=1, col=1)
        f.add_bar(x=reg.regime_id.astype(str), y=reg.shield_cut_vs_B0_pct, marker_color=COL[3], showlegend=False, row=1, col=2)
        f.update_layout(title="15 · Regime comparison (default shield weight, seed 42)", height=420)
        figs.append(f)

    # 13b seed robustness, SAC vs DQN, at the regime's default shield weight (plan 19.5)
    from o3.environment.pcm_env import default_w_shield
    w0, parts = default_w_shield(a.regime), []
    for algo in ("dqn", "sac"):
        fp = g / f"gen2_summary_{algo}.csv"
        if fp.exists():
            t = pd.read_csv(fp)
            t = t[(t.regime_id == a.regime) & (t.control_step_min == 60) & (t.shield_weight == w0)].assign(algo=algo.upper())
            parts.append(t)
    if len(parts) == 2 and all(len(t) for t in parts):
        t = pd.concat(parts)
        f = make_subplots(rows=1, cols=2, subplot_titles=("% of B0 delivery (annual)", "Shield hours per year"))
        for i, (algo, d) in enumerate(t.groupby("algo")):
            f.add_box(y=d.pct_of_B0_delivery, name=algo, boxpoints="all", marker_color=COL[i], row=1, col=1, showlegend=False)
            f.add_box(y=d.shield_h, name=algo, boxpoints="all", marker_color=COL[i], row=1, col=2, showlegend=False)
        f.add_hline(y=95, line_dash="dot", line_color="#e34948", row=1, col=1)
        f.add_hline(y=b0.B0_shield_h, line_dash="dot", line_color="#52514e", row=1, col=2)
        f.update_layout(title=f"13b · Seed robustness at the default weight w = {w0:g}, SAC vs DQN (3 seeds each; dotted: 95 % rule / B0 shield hours)", height=440)
        figs.append(f)

    # ablation: what forecast and training-time randomisation are worth (one seed per variant)
    fa = g / f"gen2_ablation_dqn_r{a.regime}.csv"
    if fa.exists():
        ab = pd.read_csv(fa)
        f = make_subplots(rows=1, cols=2, subplot_titles=("% of B0 delivery", "Shield cut vs B0 (%)"))
        f.add_box(x=ab.variant, y=ab.pct_of_B0_delivery, boxpoints="all", marker_color=COL[0], showlegend=False, row=1, col=1)
        f.add_box(x=ab.variant, y=ab.shield_cut_vs_B0_pct, boxpoints="all", marker_color=COL[3], showlegend=False, row=1, col=2)
        f.add_hline(y=95, line_dash="dot", line_color="#e34948", row=1, col=1)
        f.update_layout(title="14b · Ingredient ablation (DQN, per-seed points): full vs no forecast vs no training-time randomisation", height=420)
        figs.append(f)
    fr = g / f"gen2_ablation_robustness_dqn_r{a.regime}.csv"
    if fr.exists():
        rb = pd.read_csv(fr)
        m = rb.groupby(["scenario", "policy"]).total_delivered_kWh.mean().unstack()
        order = list(dict.fromkeys(rb.scenario))
        f = go.Figure()
        for i, v in enumerate([c for c in ("no_dr", "full") if c in m.columns]):
            f.add_bar(x=order, y=[m.loc[sc, v] for sc in order], name={"no_dr": "trained WITHOUT randomisation", "full": "trained WITH randomisation"}[v],
                      marker_color=COL[i])
        f.update_layout(title="16 · Domain-randomisation robustness overlay (phase-5 scenarios, validation years, DQN seed 42)", barmode="group",
                        yaxis_title="Delivered to 45 C (kWh/yr)", height=520)
        figs.append(f)

    out = config_O2.RESULTS_DIR / "plots" / "gen2"
    out.mkdir(parents=True, exist_ok=True)
    html = ['<!doctype html><meta charset="utf-8"><title>O3 Gen-2</title><div style="background:#fff4d6;padding:8px">'
            'Software-in-the-loop simulation. Not field validation.</div>']
    for i, fg in enumerate(figs):
        html.append(pio.to_html(fg, include_plotlyjs="cdn" if i == 0 else False, full_html=False))
    (out / f"gen2_{a.algo}.html").write_text("\n".join(html), encoding="utf-8")
    print(f"{len(figs)} charts -> {out / f'gen2_{a.algo}.html'}")


if __name__ == "__main__":
    main()
