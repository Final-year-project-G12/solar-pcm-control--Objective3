"""
src/plots/make_plots_results.py
===================================
Figures for the current Objective 2 chain (G7-G11). Every figure only READS saved files under results/; nothing is
re-simulated and no number is typed by hand. Output: results/plots/<name>.png (matplotlib, 150 dpi).
The held-out file is read as a result already produced by the single permitted run; this module never calls the held-out loader.

Run: python -m src.plots.make_plots_results        (or: python pipeline.py --state uttarakhand --stage plots)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR
from src.weather.inventory import REGIMES

RD = RESULTS_DIR
OUT = RD / "plots"
COL = {"explore": "#7f8c8d", "select": "#2980b9", "held-out": "#c0392b", "band": "#27ae60"}


def _csv(name):
    return pd.read_csv(RD / name)


def _band(regime, obj="g_window"):
    b = _csv("wp6_band.csv")
    return float(b[(b.scope == f"r{regime}") & (b.objective == obj)].band.iloc[0])


def _chosen(regime):
    return json.load(open(RD / f"g10_selection_r{regime}.json", encoding="utf-8"))["chosen"]


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("  wrote", OUT / f"{name}.png")


def fig_front_vs_grid():
    """Dense grid (rank-1 PCM) with the optimiser's simulated points and the selected design."""
    fig, ax = plt.subplots(1, len(REGIMES), figsize=(4.7 * len(REGIMES), 4.3), sharey=False)
    for r in REGIMES:
        c = _chosen(r)
        rk = int(c["pcm_rank"])
        pts = sorted(RD.glob(f"g9_points_r{r}_p{rk}*.csv"))[-1]
        p = pd.read_csv(pts)
        p = p[p.feasible.astype(bool)]
        if rk == 1:        # the dense grid exists for the rank-1 PCM only
            g = _csv(f"g9_s6_grid_r{r}_p1.csv")
            g = g[g.status.eq("ok")] if "status" in g else g
            ax[r].scatter(g.d_overheat, g.g_window, s=8, c="#bdc3c7", label=f"dense grid ({len(g)})")
        ax[r].scatter(p.d_overheat, p.g_window, s=10, c="#2980b9", alpha=.6, label=f"optimiser points, O1 rank {rk}")
        ax[r].scatter([c["d_overheat"]], [c["g_window"]], marker="*", s=220, c="#c0392b", zorder=5, label="selected (select years)")
        ax[r].set(title=f"Regime {r}: {c['pcm'][:22]}" + ("" if rk == 1 else " (no dense grid)"), xlabel="extra overheating hours vs water-only (h/yr)", ylabel="g_window gain (kWh/yr)")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Optimiser points against the dense grid (explore years 2016-2020; star = select years 2021-2023)")
    _save(fig, "fig1_front_vs_grid")


def fig_g9_checks():
    """Additive-epsilon gap between optimiser front and grid front, as a share of the repeatability band."""
    g = _csv("g9_grid_check.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(g))
    w = .27
    for i, o in enumerate(("g_window", "g_annual", "d_overheat")):
        ax.bar(x + (i - 1) * w, g[f"eps_{o}"] / g[f"band_{o}"], w, label=o)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(len(g) - .5, 1.03, "gap = band", ha="right", fontsize=8)
    ax.set_xticks(x, g.pair)
    ax.set(ylabel="epsilon gap / band", title="G9 grid check (all values below 1 pass)")
    ax.legend(fontsize=8)
    _save(fig, "fig2_g9_grid_check")


def fig_gain_by_stage():
    """Chosen design: g_window on explore, select and held-out years, next to the band."""
    h = _csv("g10_heldout_results.csv")
    fig, ax = plt.subplots(figsize=(2.7 * len(REGIMES), 4.3))
    for r in REGIMES:
        c = _chosen(r)
        ho = h[(h.regime == r) & (h.role == "chosen")]
        vals = {"explore": c["explore_g_window"], "select": c["g_window"], "held-out": ho.g_window.mean()}
        for i, (k, v) in enumerate(vals.items()):
            ax.bar(r * 4 + i, v, color=COL[k], label=k if r == 0 else None)
        ax.scatter([r * 4 + 2] * len(ho), ho.g_window, c="k", s=14, zorder=4, label="held-out year" if r == 0 else None)
        ax.hlines(_band(r), r * 4 - .4, r * 4 + 2.4, color=COL["band"], ls="--", label="repeatability band" if r == 0 else None)
    ax.set_xticks([r * 4 + 1 for r in REGIMES], [f"Regime {r}\n{_chosen(r)['pcm']}" for r in REGIMES])
    ax.set(ylabel="g_window gain over water-only (kWh/yr)", title="Chosen design: gain by year role")
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.25)          # keep negative gains (regime 3) visible; head room for the legend
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    _save(fig, "fig3_gain_by_year_role")


def fig_pcm_comparison():
    """Held-out g_window of the chosen design and the best design of every other O1 PCM, both test years."""
    h = _csv("g10_heldout_results.csv")
    fig, ax = plt.subplots(1, len(REGIMES), figsize=(4.4 * len(REGIMES), 4))
    for r in REGIMES:
        d = h[h.regime == r].groupby(["role", "pcm"]).g_window.agg(["mean", "min", "max"]).reset_index()
        d = d.sort_values("mean", ascending=False)
        ax[r].barh(d.pcm.str.replace("�", "").str[:28], d["mean"], xerr=[d["mean"] - d["min"], d["max"] - d["mean"]],
                   color=[COL["held-out"] if x == "chosen" else "#95a5a6" for x in d.role], capsize=3)
        ax[r].invert_yaxis()
        ax[r].axvline(_band(r), color=COL["band"], ls="--", lw=1)
        ax[r].set(title=f"Regime {r} (red = selected)", xlabel="held-out g_window (kWh/yr)")
    _save(fig, "fig4_heldout_pcm_comparison")


def fig_null_results():
    """Held-out nulls and costs of the chosen design: night-serve gain, extra shield hours, PCM peak."""
    h = _csv("g10_heldout_results.csv")
    h = h[h.role == "chosen"].groupby("regime")[["gain_tau_serve_mean_h", "d_overheat", "t_pcm_max"]].mean()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (col, t, ref) in zip(ax, (("gain_tau_serve_mean_h", "night-serve duration gain (h)", 0),
                                      ("d_overheat", "extra shield hours vs water-only (h/yr)", 0),
                                      ("t_pcm_max", "peak PCM temperature (C)", 65))):
        a.bar([f"R{r}" for r in h.index], h[col], color=COL["held-out"])
        a.axhline(ref, color="k", lw=1, ls="--" if ref else "-")
        a.set_title(t, fontsize=10)
    fig.suptitle("Held-out 2024-2025, chosen design: what PCM does not improve (dashed = 65 C limit)")
    _save(fig, "fig5_heldout_nulls_and_costs")


def fig_ablation():
    """One-variable ablations around the chosen design, change in g_window against the band."""
    fig, ax = plt.subplots(1, len(REGIMES), figsize=(4.7 * len(REGIMES), 5), sharex=True)
    for r in REGIMES:
        a = _csv(f"g11_ablation_r{r}.csv")
        base = a[a.block == "base"].g_window.iloc[0]
        d = a[a.block.isin(["flow", "count", "arrangement_anchor"]) & a.g_window.notna()].copy()
        d["label"] = np.where(d.block == "count", "fraction " + d.level.astype(str).str.lstrip("f"),
                     np.where(d.block == "flow", "flow " + d.level.astype(str).str.lstrip("q") + " kg/s",
                              d.arrangement.astype(str) + ", stack " + d.anchor.fillna("").astype(str)))
        d["delta"] = d.g_window - base
        d = d.iloc[::-1]
        ax[r].barh(d.label, d.delta, color=["#c0392b" if abs(v) > _band(r) else "#95a5a6" for v in d.delta])
        ax[r].axvspan(-_band(r), _band(r), color=COL["band"], alpha=.2)
        ax[r].axvline(0, color="k", lw=.8)
        ax[r].tick_params(axis="y", labelsize=7)
        ax[r].set(title=f"Regime {r}", xlabel="change in g_window (kWh/yr); green = band")
    fig.suptitle("G11 ablations (explore years): red bars exceed the band")
    _save(fig, "fig6_ablations")


def fig_surrogates():
    """Surrogate families compared by rank agreement (the quantity used to choose), 5-fold over designs and leave-one-year-out."""
    c = _csv("g8_s3_comparison.csv")
    t = c.groupby(["validation", "family"]).spearman.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    t.T.plot.bar(ax=ax, rot=0)
    ax.set(ylabel="mean Spearman rho", ylim=(t.min().min() - .02, 1), title="G8 surrogate comparison (mean over pairs and targets)")
    _save(fig, "fig7_surrogate_comparison")


def fig_fraction_response():
    """G7: window gain against PCM fraction (R_in = 20 mm, staggered, 0.030 kg/s), multi-node tank, mean of the explore years."""
    g = _csv("g7_sensitivity_paired.csv")
    g = g[(g.block == "B_count") & (g.model == "multinode") & g.valid.astype(bool)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ((r, pcm), d) in enumerate(g.groupby(["regime", "pcm"])):
        m = d.groupby("frac_outer").d_E_del45_window_kWh.mean()
        ax.plot(m.index * 100, m.values, marker="o", ms=3, color=plt.cm.tab20(i % 20), ls=("-", "--", "-.", ":")[int(r) % 4],
                label=f"R{r} {pcm[:20].replace(chr(0xfffd), '')}")
    ax.set(xlabel="PCM fraction of tank volume (%)", ylabel="window gain (kWh/yr)", title="G7: window gain against PCM fraction (R_in 20 mm, staggered, 0.030 kg/s)")
    ax.legend(fontsize=6)
    _save(fig, "fig8_fraction_response")


def fig_boundary():
    """Share of the final optimiser front sitting at a bound of the design box (boundary-artefact detector)."""
    b = _csv("g9_boundary_check.csv")
    b = b[b.share_of_front >= .2]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lab = b.pair + "  " + b.bound.str[:34]
    ax.barh(lab, b.share_of_front, color="#8e44ad")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=6)
    ax.set(xlabel="share of final front", title="G9 boundary check (only bounds holding at least 20 % of a front)")
    _save(fig, "fig9_boundary_share")


def main(state="uttarakhand"):
    if state != "uttarakhand":
        raise SystemExit("rework plots exist for uttarakhand only")
    print("Generating rework figures from saved results ->", OUT)
    for f in (fig_front_vs_grid, fig_g9_checks, fig_gain_by_stage, fig_pcm_comparison, fig_null_results,
              fig_ablation, fig_surrogates, fig_fraction_response, fig_boundary):
        f()


if __name__ == "__main__":
    main()
