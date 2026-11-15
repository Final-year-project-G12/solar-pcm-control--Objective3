# Objective 3 — How to Run

A practical runbook: setup, every command, expected outputs, and the few
Windows-specific gotchas. Run everything from the `objective3/` root.

---

## 1. Environment

- **Python:** the system interpreter at
  `C:\Users\jaswi\AppData\Local\Programs\Python\Python314` (it has numpy,
  gymnasium, stable-baselines3, sb3-contrib, plotly, folium). Do **not** use the
  `.venv` at the project root — it lacks the ML packages and is the reason the
  editor sometimes shows "Cannot find module numpy" squiggles. Those are
  type-checker import warnings only; the code runs fine on Python 3.14.
- **Objective 2 on the path:** scripts add
  `../objective2_design_optimization` to `sys.path` at runtime so the frozen
  physics (`from src.simulation... import`) resolves. Nothing to configure.
- **Windows console encoding:** prefix long training commands with
  `PYTHONIOENCODING=utf-8`. Without it, a non-ASCII character in a progress line
  can raise `UnicodeEncodeError` under the default cp1252 code page and abort the
  run. (The source has been de-unicoded, but the prefix is belt-and-suspenders.)

## 2. Verify the contract and run tests

```bash
python scripts/phase0_audit.py           # validate + hash the frozen contract
python -m pytest -q                       # full suite
python -m pytest tests/test_baselines.py -v
```

## 3. Baselines

```bash
python scripts/phase2_baselines.py        # B0, B1, B2, B4 on validation years
```
Expect zero safety violations for all, B0 highest delivery / most shields,
B2 fewest shields / lowest delivery.

## 4. Train the DRL algorithms (Phase O3-4)

```bash
# all six algorithms, regime 0, 300k steps each (long — see timings below)
PYTHONIOENCODING=utf-8 python -u scripts/phase4_train.py \
    --regime 0 --steps 300000 --algo all --seed 42

# or a subset
PYTHONIOENCODING=utf-8 python -u scripts/phase4_train.py --algo dqn ppo
```
Outputs: `results/<algo>/r0/` (checkpoints + `best_model/`),
`results/phase4_comparison_r0.csv`.

**Approximate single-algorithm training time (CPU, 300k steps, varies with load):**
PPO 35–290 min · DQN 45–90 min · RecurrentPPO 95–220 min · QRDQN 60–105 min ·
MaskablePPO 20–35 min · SAC ~6 h (per-step gradient; the long pole).

### Re-score saved models without retraining
After any eval-harness change, or to compare `best_model` checkpoints:
```bash
PYTHONIOENCODING=utf-8 python scripts/phase4_reeval.py --regime 0 --algo all
```

### Multi-seed robustness (thesis statistics)
```bash
PYTHONIOENCODING=utf-8 python scripts/phase4_multiseed.py \
    --regime 0 --algo dqn --seeds 42 0 123 --steps 300000
```
Outputs mean ± std per metric in `results/phase4_stats_r0.json`.

## 5. Robustness suite (Phase O3-5)

```bash
python scripts/phase5_robustness.py       # GHI noise, T_amb bias, init-state, combined
```

## 6. Locked final evaluation (Phase O3-6) — run ONCE

```bash
python scripts/phase6_locked_eval.py --algo dqn --regime 0
```
Evaluates the selected policy on the held-out test years **2011–2015**, which
must never have influenced training, tuning, or selection. Outputs
`results/phase6_final_test_dqn.csv`, a summary JSON, and the model SHA for
reproducibility. Checks the primary success criteria (zero violations, delivery
≥ 95 % of B0, shields ≤ B0).

## 7. Software-in-the-loop demo (Phase O3-7)

```bash
python scripts/phase7_demo.py --mode stub   --algo dqn --steps 48   # no network
python scripts/phase7_demo.py --mode replay --algo dqn --year 2021  # historical CSV
python scripts/phase7_demo.py --mode live   --algo dqn --steps 24   # Open-Meteo (Madurai)
```
Prints a per-hour action/state table and logs to `results/demo_logs/`.
**Simulation only** — thermal state is from the simulator, not sensors.

## 8. Objective 4 hardware handoff (Phase O3-8)

```bash
python scripts/phase8_o4_handoff.py --algo dqn --regime 0
```
Writes the sensor/actuator/fail-safe spec, policy export + SHA, and a deployment
test plan to `results/obj4_handoff/`.

## 9. Interactive visualisations

All commands are run from the `Objective3` folder. Needs `plotly`, `folium`,
`streamlit` (`pip install plotly folium streamlit`). Open the HTML files by
double-clicking them (internet is needed once for plotly.js).

**1. One interactive page with the key Tamil Nadu plots (start here):**
```bash
python scripts/make_interactive_o3.py
```
Writes `results/plots/O3_interactive_tamilnadu.html`: trade-off scatter and
per-year bars with regime dropdowns, locked-test regimes, sensor faults,
robustness scenarios and the forecast ablation.

**2. Per-regime dashboards (6 Plotly charts + Folium map):**
```bash
python scripts/make_plots.py --regime 0 --tag R0 --selected sac --label "SAC (realistic-forecast model)" ^
  --test-csv results/phase6_final_test_sac_smartfc_r0.csv ^
  --robust-csv results/phase5_robustness_r0_sac_smartfc.csv
```
(On Linux or macOS replace `^` with a backslash.) Repeat with `--regime 1 --tag R1` and `--regime 2 --tag R2` (and the matching `r1` / `r2`
CSV names). The tag is only the output folder name (`results/plots/R0` = regime 0).
`--test-csv` / `--robust-csv` are needed because `phase6_final_test_sac.csv` holds only the
last regime that was run.

**3. Animated controller replay (press Play in the browser):**
```bash
python scripts/make_live_dashboard.py --algo sac --regime 0 --year 2021
python scripts/make_live_dashboard.py --algo sac --regime 0 --live     # real weather from Open-Meteo, needs internet
```
Output: `results/plots/live_demo/`.

**4. Forecast comparison dashboard:** `python scripts/make_forecast_dashboard.py`.

**5. Streamlit portal (live controller + deployment metrics):**
```bash
streamlit run scripts/streamlit_portal.py
```
Opens at http://localhost:8501. Its Generation-2 panel shows a warning that those results pre-date
the weather-year fix; do not quote them.

Pre-fix dashboards and the multi-seed files are kept in `results/archive_prefix_bugfix/`.

## 10. Deployment metrics (Phase O3-9)

```bash
PYTHONIOENCODING=utf-8 python scripts/phase9_deployment_metrics.py --regime 0
```
Computes thermal efficiency, energy savings vs B0/B1, inference latency and
memory footprint, and the sensor-fault robustness suite (observation corrupted,
physics untouched). Writes `results/deployment_metrics_r0.json`,
`results/phase9_sensor_faults_r0.csv`, and plots 8–11 to
`results/plots/R2/deployment_metrics.html`. `--skip-faults` runs only the fast
metrics.

## 11. Live controller dashboard (animated)

```bash
# historical replay (reproducible)
PYTHONIOENCODING=utf-8 python scripts/make_live_dashboard.py --algo sac --year 2021 --start-day 160 --days 10
# LIVE present-day weather from Open-Meteo (needs network; software-in-the-loop)
PYTHONIOENCODING=utf-8 python scripts/make_live_dashboard.py --algo sac --live
```
Writes a self-contained HTML in `results/plots/live_demo/` — open it and press
Play to watch the hourly CHARGE / IDLE / BYPASS decisions, shield overrides,
temperatures against the guard bands, and KPI tiles.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Editor: "Cannot find module numpy/gymnasium" | Wrong interpreter. Ctrl+Shift+P → Python: Select Interpreter → Python 3.14 (not `.venv`). Harmless to runtime. |
| `UnicodeEncodeError` mid-run, job aborts | cp1252 console. Re-run with `PYTHONIOENCODING=utf-8`. |
| "No trained model found for \<algo\>" | Train it first (Phase O3-4), or the dir name differs — recurrent→`recurrent_ppo`, maskable→`maskable_ppo` (handled by the loader). |
| `ImportError: tensorboard not installed` | Trainers must not pass `tensorboard_log` (already removed). |
| All DRL policies == B0 in results | Expected under the R1 (delivery-only) reward; use R2 (shield penalty, the default). |
| DRL delivery far below B0 | Penalty too strong (e.g. w_shield=20 destabilised DQN). Default w_shield=10. |
