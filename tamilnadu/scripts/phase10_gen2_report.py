"""
Phase O3-10 (report) — consolidate the Generation-2 research-extension study
(forecast observation V2, training-time domain randomisation, reward-weight
Pareto sweep, sub-hourly control ablation, regime 1/2 coverage) into a
summary table and five plots.

Run from the objective3 root:
    python scripts/phase10_gen2_report.py
"""

import json
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT   = pathlib.Path(__file__).parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# validated palette
C_SAC, C_DQN = "#1baf7a", "#2a78d6"
C_B0, C_B1 = "#8f8e88", "#b5b4ae"
C_WEIGHT = {0: "#2a78d6", 10: "#1baf7a", 20: "#eb6834"}


def load_phase10(tag: str) -> dict:
    p = RESULTS_DIR / f"phase10_{tag}.csv"
    df = pd.read_csv(p)
    return {
        "delivered_kWh": df["total_delivered_kWh"].mean(),
        "shields": (df["total_shield_water"] + df["total_shield_pcm"]).mean(),
        "reward": df["total_reward"].mean(),
        "weight": df["shield_weight"].iloc[0],
        "seed": df["seed"].iloc[0],
        "regime": df["regime"].iloc[0],
        "algo": df["algo"].iloc[0],
        "ctrl": df["control_step_min"].iloc[0],
    }


def main():
    print("=" * 70)
    print("Phase O3-10 report — Generation-2 research extensions")
    print("=" * 70)

    # ---- 1. Pareto sweep (SAC, regime 0, 3 weights x 3 seeds) --------------
    pareto_tags = ["pareto_w0_s42", "pareto_w5_s42", "pareto_w10_s42", "pareto_w15_s42", "pareto_w20_s42",
                   "pareto_w0_s0", "pareto_w5_s0", "pareto_w10_s0", "pareto_w15_s0", "pareto_w20_s0",
                   "pareto_w0_s123", "pareto_w5_s123", "pareto_w10_s123", "pareto_w15_s123", "pareto_w20_s123"]
    pareto_rows = []
    for t in pareto_tags:
        if (RESULTS_DIR / f"phase10_{t}.csv").exists():
            pareto_rows.append(load_phase10(t))
    pdf = pd.DataFrame(pareto_rows).sort_values(["weight", "seed"]).reset_index(drop=True)
    print("\n--- Pareto sweep (SAC, regime 0) ---")
    print(pdf.to_string(index=False))

    f1 = go.Figure()
    for w in sorted(pdf["weight"].unique()):
        sub = pdf[pdf["weight"] == w]
        f1.add_scatter(
            x=sub["shields"], y=sub["delivered_kWh"], mode="markers+text",
            name=f"weight={w:.0f}",
            text=[f"s{s:.0f}" for s in sub["seed"]], textposition="top center",
            marker=dict(size=14, color=C_WEIGHT.get(int(w), "#8f8e88")),
        )
    # mean-per-weight trend line
    means = pdf.groupby("weight")[["shields", "delivered_kWh"]].mean().sort_index()
    f1.add_scatter(x=means["shields"], y=means["delivered_kWh"], mode="lines",
                   name="mean per weight", line=dict(color="#0b0b0b", width=1, dash="dot"))
    f1.update_layout(
        title="12 · Reward-weight Pareto frontier — SAC, regime 0 "
              "(3 weights x 2 seeds; reduced-budget ablation, 50k steps/run)",
        xaxis_title="shield trips/yr (lower is better)",
        yaxis_title="delivered kWh/yr (higher is better)",
        template="plotly_white", height=420)

    # ---- 2. Sub-hourly ablation ---------------------------------------------
    pareto_rows_dict = {r["weight"]: r for r in pareto_rows if r["seed"] == 42}
    sub60 = pareto_rows_dict.get(10)
    sub15_tag = "subhourly15_full_w10_s42" if (RESULTS_DIR / "phase10_subhourly15_full_w10_s42.csv").exists() \
        else "subhourly15_w10_s42"
    sub15 = load_phase10(sub15_tag) if (RESULTS_DIR / f"phase10_{sub15_tag}.csv").exists() else None
    scaled = (sub15_tag == "subhourly15_full_w10_s42")
    print(f"\n--- Sub-hourly control ablation (SAC, weight=10, seed=42"
          f"{', calendar-scaled budget: 50k@60min vs 160k@15min' if scaled else ', EQUAL step budget (confounded)'}) ---")
    if sub60 and sub15:
        print(f"  60-min control: delivered={sub60['delivered_kWh']:.1f} kWh  shields={sub60['shields']:.0f}")
        print(f"  15-min control: delivered={sub15['delivered_kWh']:.1f} kWh  shields={sub15['shields']:.0f}")
        shield_delta = 100 * (sub60["shields"] - sub15["shields"]) / sub60["shields"]
        deliv_delta = 100 * (sub15["delivered_kWh"] - sub60["delivered_kWh"]) / sub60["delivered_kWh"]
        print(f"  shield change at 15-min cadence: {shield_delta:+.1f}%  "
              f"(delivery change: {deliv_delta:+.1f}%)")

    f2 = make_subplots(rows=1, cols=2, subplot_titles=(
        "Delivered energy (kWh/yr)", "Shield trips/yr"))
    if sub60 and sub15:
        labels = ["60-min control<br>(50k steps, 5.7yr)", "15-min control<br>(160k ticks, ~4.6yr)"] \
            if scaled else ["60-min control", "15-min control"]
        f2.add_bar(x=labels, y=[sub60["delivered_kWh"], sub15["delivered_kWh"]],
                  marker_color=[C_DQN, C_SAC], row=1, col=1, showlegend=False)
        f2.add_bar(x=labels, y=[sub60["shields"], sub15["shields"]],
                  marker_color=[C_DQN, C_SAC], row=1, col=2, showlegend=False)
    subtitle = ("calendar-scaled step budgets, matched coverage" if scaled
               else "reduced-budget: 50k steps both — confounded, see caveat")
    f2.update_layout(title=f"13 · Control-step ablation — does decision cadence cut shield trips? "
                          f"(SAC, weight=10, seed=42, {subtitle})",
                     template="plotly_white", height=380)

    # ---- 3. Regime comparison -----------------------------------------------
    regime_rows = []
    for algo, tag in [("SAC", "pareto_w10_s42"), ("DQN", None)]:
        pass
    r0_sac = sub60
    r0_dqn = None
    dqn_r0_path = RESULTS_DIR / "phase4_comparison_r0.csv"
    if dqn_r0_path.exists():
        d = pd.read_csv(dqn_r0_path)
        dd = d[d["policy"] == "DQN"]
        if len(dd):
            r0_dqn = {"delivered_kWh": dd["total_delivered_kWh"].mean(),
                      "shields": (dd["total_shield_water"] + dd["total_shield_pcm"]).mean()}
    r1_sac = load_phase10("gen2_sac_r1_w10_s42") if (RESULTS_DIR / "phase10_gen2_sac_r1_w10_s42.csv").exists() else None
    r2_sac = load_phase10("gen2_sac_r2_w10_s42") if (RESULTS_DIR / "phase10_gen2_sac_r2_w10_s42.csv").exists() else None
    r1_dqn = load_phase10("gen2_dqn_r1_w10_s42") if (RESULTS_DIR / "phase10_gen2_dqn_r1_w10_s42.csv").exists() else None
    # prefer the full-budget (300k) regime-2 DQN rerun over the earlier
    # 100k-step run, which produced a degenerate (near-always-bypass) policy
    r2_dqn_tag = "gen2_dqn_r2_full_w10_s42" if (RESULTS_DIR / "phase10_gen2_dqn_r2_full_w10_s42.csv").exists() \
        else "gen2_dqn_r2_w10_s42"
    r2_dqn = load_phase10(r2_dqn_tag) if (RESULTS_DIR / f"phase10_{r2_dqn_tag}.csv").exists() else None

    print("\n--- Regime comparison (GEN2, weight=10, seed=42) ---")
    regimes_tbl = []
    for name, r in [("R0 SAC", r0_sac), ("R0 DQN", r0_dqn), ("R1 SAC", r1_sac),
                    ("R1 DQN", r1_dqn), ("R2 SAC", r2_sac), ("R2 DQN", r2_dqn)]:
        if r:
            print(f"  {name:8s} delivered={r['delivered_kWh']:.1f} kWh  shields={r['shields']:.0f}")
            regimes_tbl.append((name, r["delivered_kWh"], r["shields"]))

    f3 = make_subplots(rows=1, cols=2, subplot_titles=("Delivered (kWh/yr)", "Shields/yr"))
    names = [t[0] for t in regimes_tbl]
    f3.add_bar(x=names, y=[t[1] for t in regimes_tbl],
              marker_color=[C_SAC if "SAC" in n else C_DQN for n in names], row=1, col=1, showlegend=False)
    f3.add_bar(x=names, y=[t[2] for t in regimes_tbl],
              marker_color=[C_SAC if "SAC" in n else C_DQN for n in names], row=1, col=2, showlegend=False)
    f3.update_layout(title="14 · Climate-adaptive coverage — regimes 0/1/2, SAC & DQN "
                          "(GEN2 env: forecast obs + domain randomisation, weight=10, seed=42)",
                     template="plotly_white", height=380)

    # ---- write combined HTML -------------------------------------------------
    out = RESULTS_DIR / "plots" / "R2" / "gen2_extensions.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>O3 Generation-2 Research Extensions</title>',
             '<style>body{background:#f9f9f7;margin:0;padding:16px;'
             'font-family:system-ui,sans-serif;color:#0b0b0b}h1{font-size:20px}'
             '.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);'
             'border-radius:10px;margin:14px 0;padding:8px}'
             '.note{color:#5c5b57;font-size:13px}</style></head><body>',
             '<h1>Objective 3 — Generation-2 research extensions</h1>',
             '<p class="note">Forecast observation (V2) + training-time domain '
             'randomisation active in every run below. Reduced-budget ablation '
             '(50k/100k steps, not the production 300k) — trend evidence, not '
             'final convergence.</p>']
    for i, f in enumerate([f1, f2, f3]):
        inc = "cdn" if i == 0 else False
        parts.append('<div class="card">' + f.to_html(full_html=False, include_plotlyjs=inc) + '</div>')
    parts.append('</body></html>')
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"\n[INFO] plots -> {out}")

    # ---- save summary JSON ----------------------------------------------------
    summary = {
        "pareto_sweep": pdf.to_dict(orient="records"),
        "subhourly_ablation": {"ctrl60": sub60, "ctrl15": sub15},
        "regime_comparison": regimes_tbl,
    }
    jp = RESULTS_DIR / "phase10_gen2_summary.json"
    jp.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"[INFO] summary -> {jp}")
    print("\n[OK] Phase O3-10 report complete.")


if __name__ == "__main__":
    main()
