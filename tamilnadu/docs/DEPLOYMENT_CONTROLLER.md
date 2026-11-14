# Deployment controller (frozen reference for Objective 4)

**Controller:** SAC, regime 0, trained with the realistic forecast
(`forecast_mode="smart"`, smart persistence on the clear-sky index).
**Path:** `results/sac/r0_smartfc/best_model/best_model.zip`
(the file the locked evaluation loads: best_model preferred over final)

| file | SHA-256 |
|---|---|
| `best_model/best_model.zip` | `fb13d8415f3338f5f1d6caafbf0e79d4c64bb365c8f4daa59d0d854be63aa1b2` |
| `sac_final.zip` (not deployed) | `796153b470dd67cb0ed64b4fc872c4a18d589127f219ce9f06bd1b19ffa0cc88` |

Trained 2026-11-14 03:16-04:00 on years 2016-2020 + 2024-2025, seed 42,
300,000 steps. Locked test 2011-2015, regime 0: 2063.7 kWh/yr delivered
(98.4% of B0), 22,632 shields/yr (-51.8% vs B0), 0 safety violations.

Why this one: it is the only SAC trained and scored with a forecast a real
installation could produce, rather than perfect look-ahead. The canonical
`results/sac/r0` model (commit 11d9ee0) uses the no-forecast V1 observation and
is retained as the reference result, not the deployment controller.

**Freeze rule:** no retraining after the commit that adds this file. Any change
to `o3/environment/*`, the trainers, or this model requires a new record here.

## Verified for this controller (regime 0, `r0_smartfc`)
- Weather robustness (10 scenarios x 3 years) and sensor faults (7 scenarios):
  zero hard-safety violations, no NaN in core metrics
  (`phase5_robustness_r0_sac_smartfc.csv`, `phase9_sensor_faults_r0_smartfc.csv`).
- Nominal 2021: 1971.4 kWh, 18,604 shields (-53.1% vs B0), thermal efficiency 40.9%.
- Dead or stuck pyranometer: delivery falls to 1696.1 kWh (-14%) while shields
  drop to 5,621 (it bypasses more) - degraded but safe; the shield acts on true state.
- Inference: ~5 ms/decision, 1.4 MB (see `deployment_metrics_r0_smartfc.json`).

Note: `scripts/phase9_deployment_metrics.py` previously copied the canonical
models' efficiency table into tagged JSONs. Fixed (`apply_nominal_override`); the
six tagged JSONs were rebuilt from their own nominal rollouts. DQN rows are
absent for regimes 1-2 tagged runs because no tagged DQN model exists.
