# solar-pcm-tamilnadu

**Climate-adaptive deep-reinforcement-learning control of PCM solar water
heating — Tamil Nadu (Objective 3 of the FYP, Group 12, Amrita School of
Engineering).**

A safety-shielded DRL controller that chooses the hourly operating action
**{CHARGE, IDLE, BYPASS}** for a frozen PCM solar-water-heating design
(PureTemp 58 / RT57HC, 100 L tank, 2.5 m² collector) across three Tamil Nadu
climate regimes. The safety shield runs *inside* the environment and cannot be
bypassed by the policy, so zero hard-limit violations hold by construction.

## Headline result — held-out test years 2011–2015 (corrected 2026-11-13)

> **Correction notice:** results below were retrained after fixing a critical
> bug (`_load_weather()` silently loaded a single fixed 2016 file regardless of
> the requested year — see `docs/CORRECTIONS_OBJECTIVE3.md`). All numbers on
> this page are from the corrected multi-year run (train 2016–2020 + 2024–2025,
> validate 2021–2023, locked test 2011–2015, genuinely distinct weather per
> year). Earlier numbers quoting "2144.7 kWh / 36,964 shields" etc. were
> computed on identical 2016 data disguised as five different years and are
> retired.

| Regime | Policy | Delivered kWh/yr | Shield activations/yr | Safety violations |
|---|---|---|---|---|
| 0 | B0 always-charge | 2096.9 | 46,969 | 0 |
| 0 | DQN | 2018.0 (96.2 %) | 17,838 (−62 %) | 0 |
| 0 | **SAC** | 2044.6 (97.5 %) | **21,406 (−54 %)** | **0** |
| 1 | B0 always-charge | 2094.8 | 24,815 | 0 |
| 1 | DQN | 2094.7 (~100 %) | 24,298 (−2 %, essentially B0) | 0 |
| 1 | **SAC** | 2067.5 (98.7 %) | **15,569 (−37 %)** | **0** |
| 2 | B0 always-charge | 2068.8 | 34,796 | 0 |
| 2 | DQN | 2068.8 (100 %, identical to B0) | 34,796 (0 %) | 0 |
| 2 | **SAC** | 2012.6 (97.3 %) | **23,138 (−33.5 %)** | **0** |

**Key finding: DQN only learns genuinely anticipatory control on regime 0.**
On regimes 1 and 2 it collapses to B0-identical (regime 2) or near-identical
(regime 1) behaviour. **SAC is the controller that consistently beats
always-on across all three regimes.** SAC does carry one deployment caveat:
under a simulated pyranometer failure on regime 1, its delivered energy
collapses to ~9.6 kWh/yr (from a 1990 kWh/yr nominal) — a severe
sensor-dependency issue specific to that regime (regimes 0 and 2 degrade only
mildly, 0–15 %, under the same fault). See
`docs/METHODOLOGY_AND_FINDINGS.md` §5b for the full sensor-fault table.

**Does it survive a realistic weather forecast?** The table above uses
`obs_version="V1"` — **no forecast fields in the observation at all**.
Retrained with a realistic 1–3h GHI forecast (smart persistence on the
clear-sky index, not a perfect look-ahead): SAC's shield-reduction advantage
holds up well (54%→51.8% regime 0, 37%→26.2% regime 1, 33.5%→35.9% regime 2,
delivery ≥95% of B0 on all three, zero violations) — the strongest evidence
that SAC's result is real and not an artifact of unrealistic information.
DQN's regime-0 result appeared to drop sharply (62%→18.3%), but a controlled
same-seed check (identical observation shape, `forecast_mode="perfect"`
instead of `"smart"`) showed that same seed only reaches 28.3% even with a
*perfect* forecast — so most of the apparent 62%→18.3% drop is ordinary DQN
retrain variance, not a forecast-realism effect; the genuine isolated effect
is 28.3%→18.3%. A third condition, `forecast_mode="none"` (zero forecast,
same observation shape), shows **SAC's result is forecast-independent**
(26–54% shield cut across all three forecast conditions, every regime) while
**DQN collapses completely to B0 (0% shield cut) with zero forecast info**.
Full locked-test + robustness + sensor-fault + deployment-metrics suite now
covers every trained variant — see `docs/METHODOLOGY_AND_FINDINGS.md` §5c–5d
and `docs/CORRECTIONS_OBJECTIVE3.md` §O3-5–O3-6.

## What's here

```
o3/            environment (Gym wrapper over the frozen O2 physics), baselines
               B0/B1/B2/B4, six agent trainers, robustness suite, live demo adapter
scripts/       phase0–phase8 pipeline, make_plots.py (Plotly/Folium dashboards),
               make_live_dashboard.py (animated live/replay controller dashboard),
               make_forecast_dashboard.py (no-forecast vs. realistic-forecast
               comparison -> results/plots/forecast_comparison/dashboard.html)
contracts/     frozen Tamil Nadu environment contract (v3, SHA-verified)
results/       trained models, comparison CSVs, locked-test results, dashboards
docs/          methodology, algorithms, runbook, literature review + references.bib
tests/         pytest suite
```

## Requires

- Python 3.11+, `stable-baselines3`, `sb3-contrib`, `gymnasium`, `pandas`,
  `numpy`, `plotly`, `folium`.
- **The Objective 2 physics codebase** as a sibling directory
  `../objective2_design_optimization` (not in this repo — 1.3 GB; it provides
  the validated multi-node tank simulator and the NASA POWER weather data).

## Quick start

```bash
# validate the frozen contract
python scripts/phase0_audit.py

# re-score all saved policies on the validation years
python scripts/phase4_reeval.py --regime 0 --algo all

# animated controller dashboard — historical replay
python scripts/make_live_dashboard.py --algo sac --year 2021 --days 10
# ...or on LIVE present-day Open-Meteo weather (software-in-the-loop)
python scripts/make_live_dashboard.py --algo sac --live
```

Data protocol: train 2016–2020, validate 2021–2023, locked test 2011–2015
(year roles enforced in code). All results are simulation; field validation is
Objective 4 (`results/obj4_handoff/`).
