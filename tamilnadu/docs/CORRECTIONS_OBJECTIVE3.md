# CORRECTIONS — Objective 3 (Tamil Nadu)

**Applied 2026-11-13.** Companion to `../../objective2_design_optimization/docs/CORRECTIONS.md`
(Objective 2). Kept per-tree so each objective stays self-contained.

---

## O3-1 — The weather-year bug (critical; everything before this date is retired)

**The defect.** `o3/environment/pcm_env.py`'s `_load_weather()` called
`load_hourly_weather(state, cluster_id)` from Objective 2's `src/io_utils.py`,
which takes **no year parameter** and always returns one fixed file
(`weather_regime_tamilnadu_cluster{id}_hourly.csv`, 2016 data only). The old
code then filtered that single file by `self.year`; when the requested year had
no matching rows (every year except 2016), it **silently fell back to using the
whole unfiltered file** instead of raising an error.

**Consequence.** Every "different year" ever used anywhere in Objective 3 —
training years 2016–2020, validation years 2021–2023, the "locked" test years
2011–2015 — was actually identical 2016 data relabelled. This invalidated:

- Every SAC/DQN algorithm-selection result (the comparison table in the old
  `docs/ALGORITHMS.md`)
- The "locked test passed" claim (it was never run on genuinely held-out data —
  2011–2015 as loaded was 2016)
- Every robustness-suite result that varied `years=[...]`
- The Generation-2 extensions (Pareto sweep, regime-1/2 coverage, domain
  randomisation) that depended on real multi-year diversity

**The fix.** `_load_weather()` now constructs the year-specific file path
directly (`weather_regime_tamilnadu_cluster{id}_hourly_{year}.csv`) and
**raises `FileNotFoundError` loudly** if that file is missing, instead of
silently substituting a different year. Per-year files for 2011–2025 were
generated and verified (diurnal GHI profile peaks at a physically sensible
UTC hour, consistent with the LST→UTC correction documented on the Objective 2
side).

**Scope of what changed.** Nothing about the reward design, the safety shield,
the physics, or the algorithm trainers changed. Only the weather-loading path
changed. Every result quoting per-year numbers from before 2026-11-13 is
retired; the corrected re-run is the complete record below.

## O3-2 — TRAIN_YEARS extended to include 2024–2025

The original contract's `forbidden` list reserved 2024–2025 for Objective 2's
own one-shot held-out test (`role: "used_test_v1"`). On explicit instruction,
this project uses 2024–2025 as **additional training-year diversity only** —
`TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020, 2024, 2025]`. This does not reopen
Objective 2's test (its 2011–2015 held-out run is untouched) and does not touch
Objective 3's own 2011–2015 locked test, which remains genuinely held out and
was used exactly once per algorithm per regime, after training was frozen.

## O3-3 — Full corrected retrain: all three regimes

Baselines (B0, B1), DQN, and SAC were fully retrained from scratch (checkpoint
weights discarded, replay buffers discarded) on the corrected weather loader,
for all three Tamil Nadu regimes. Old (invalid) checkpoints were preserved,
not deleted, at `results/archive_2016only_buggy/`.

**PPO was evaluated as a candidate for this pass and dropped** — consistent
with every prior round in this project, it reproduced B0-identical behaviour
and added a third parallel training process without improving coverage, so it
was not carried into the corrected run under the project's time constraints.

Full corrected results (locked test, robustness, deployment metrics, per
regime) are in `METHODOLOGY_AND_FINDINGS.md` and `ALGORITHMS.md`. Headline:

| Regime | DQN differentiates from B0? | SAC differentiates from B0? |
|---|---|---|
| 0 | **Yes** — 62% fewer shields, 96.2% delivery | **Yes** — 54% fewer shields, 97.5% delivery |
| 1 | Barely — 2% fewer shields (≈ B0) | **Yes** — 37% fewer shields, 98.7% delivery |
| 2 | No — numerically identical to B0 | **Yes** — 33.5% fewer shields, 97.3% delivery |

**SAC is the controller that reliably beats always-on across all three
regimes; DQN only does so on regime 0.** This reverses the earlier (buggy-data)
conclusion that DQN was "the max-delivery alternative" worth carrying forward
alongside SAC — on corrected data, DQN is not a reliable second choice outside
regime 0.

## O3-4 — Sensor-fault finding: SAC's regime-1 pyranometer dependency

Deployment-metrics sensor-fault testing (phase 9) found that SAC's regime-1
policy is unusually dependent on the raw GHI observation: under a simulated
dead or stuck pyranometer, delivered energy collapses from ~1990 kWh/yr
(nominal) to **~9.6 kWh/yr** — the policy essentially stops charging for the
entire year. There are zero hard-safety violations (the shield still holds by
construction; the failure mode is "delivers almost no hot water," not
"overheats"), but it is a severe availability failure if a pyranometer fails
in the field.

This is **not** a universal SAC defect: the same fault on regime 0 causes no
degradation (delivery actually rises slightly) and on regime 2 causes only a
mild ~15% drop. It appears to be specific to how SAC's continuous policy
happened to converge on regime 1's reward landscape. Recorded here as an
honest deployment caveat, not fixed — a production deployment on regime 1
would need either a sensor-health fallback (e.g. revert to B1's rule-based
logic if GHI reads implausibly) or retraining with sensor-dropout augmentation
during training (future work; see the forecast-realism work in O3-5).

## O3-5 — Forecast observation was a perfect look-ahead; fixed with smart persistence (2026-11-14)

The V2 observation's 1–3h GHI forecast tail read the stored year's **true
future** GHI value directly, with zero error — a perfect forecast no real
pyranometer/NWP system could provide (the same defect independently flagged
in the parallel Rajasthan work, whose reviewed implementation plan this fix
follows). An earlier pass at this fix (since replaced) used an invented
Gaussian-noise model; that was dropped in favour of the method below, which
needs no assumed noise parameter.

**Terminology note.** The pre-existing "CLEAN" DQN/SAC retrain reported in
O3-3 (62%/54%/37%/33.5% shield cuts) used `obs_version="V1"` — an observation
with **no forecast fields at all**, not a perfect-forecast variant. Earlier
drafts of this section mislabelled that baseline "perfect forecast"; it is
corrected below to "no forecast." The only genuinely controlled, same-seed,
same-obs-shape ablation (`obs_version="V3"`, `forecast_mode="perfect"` vs.
`forecast_mode="smart"`, otherwise identical hyperparameters) was run
afterward for DQN regime 0 specifically because that comparison was missing
— see the table and finding below.

**The fix — smart persistence on the clear-sky index.** `forecast_ghi(k)`
(new method on `PCMEnv`) now supports `forecast_mode`:
- `"perfect"` (default) — unchanged old behaviour, bit-identical to pre-fix,
  kept only as an ablation/upper-bound reference point.
- `"smart"` — `GHI_hat(t+k) = Kc(t) * GHI_clear(t+k)`, where `Kc(t) =
  GHI(t)/GHI_clear(t)` is clipped to [0, 1.2] and held at its last daytime
  value overnight. `GHI_clear` is the `GHI_clearsky_Wm2` column already
  present in every per-year weather file — no new data pipeline needed.
- `"none"` — zero, the no-forecast ablation floor.

Five tests (`tests/test_environment.py::test_forecast_*`) verify default
equivalence, causality (perfect reads exactly t+k; smart never reads the
true future GHI, only the clear-sky envelope + today's Kc), reproducibility,
and boundedness (finite/non-negative at night, sunrise, year-end) — all pass.
Next-hour demand forecast stays perfect by design (O2's fixed schedule is
knowable in advance; a modelling limitation, not a bug).

**The retrain — DQN + SAC, all three regimes, `forecast_mode="smart"`.**
Locked-test result (2011–2015), compared against the earlier no-forecast
(`obs_version="V1"`) numbers from O3-3:

| Regime | Policy | No forecast (V1): delivery / shield cut | Realistic 3h forecast (V3/smart): delivery / shield cut |
|---|---|---|---|
| 0 | DQN | 96.2% / **62%** | 99.6% / **18.3%** |
| 0 | SAC | 97.5% / 54% | 98.4% / **51.8%** |
| 1 | SAC | 98.7% / 37% | 99.7% / **26.2%** |
| 2 | SAC | 97.3% / 33.5% | 98.8% / **35.9%** |

(DQN was only retrained with `forecast_mode="smart"` on regime 0, as a cheap
secondary check — see O3-3 for why DQN is not carried as a cross-regime
alternative.)

**Important caveat on the DQN row, and the controlled check that resolves
it.** The no-forecast-vs-smart-forecast comparison above is **not** a
controlled, same-seed ablation that isolates the forecast's effect from
ordinary retrain-to-retrain variance. This matters most for DQN: it is known
(see O3-3's cross-regime table and ALGORITHMS.md) to have substantially
higher run-to-run training variance in this project than SAC. A controlled
check — identical `obs_version="V3"` shape, identical seed (42) and
hyperparameters, `forecast_mode="perfect"` instead of `"smart"` — was run on
DQN regime 0 (2026-11-14, 300k steps, 61.8 min) to isolate the forecast
effect from everything else:

| Regime | Policy | V1, no forecast (O3-3) | V3 + perfect forecast (controlled) | V3 + smart forecast | Isolated forecast-realism effect |
|---|---|---|---|---|---|
| 0 | DQN | 96.2% / 62% | **97.7% / 28.3%** | 99.6% / 18.3% | **28.3%→18.3%** (10 pts) |

**This is the key result of the controlled check.** Going from the old
`V1` (no forecast) run to the new, same-seed `V3`+`forecast_mode="perfect"`
run, DQN's shield cut falls from 62% to 28.3% — *before the forecast is even
degraded to "smart."* Since `forecast_mode="perfect"` objectively gives the
agent strictly *more* information than `V1`'s no-forecast observation, this
drop cannot be a forecast-realism effect at all — it is almost entirely
retrain-to-retrain / observation-dimensionality variance. The genuine,
isolated forecast-realism effect (perfect → smart, everything else held
fixed) is only **28.3% → 18.3%**, a real but much smaller degradation than
the naive 62%→18.3% comparison implied. **The caveat raised before running
this check was correct: most of the originally-reported 62%→18.3% "forecast
effect" for DQN was actually ordinary DQN retrain variance, not a forecast
effect.**

**Finding (SAC, not confounded by this issue — all four SAC comparisons use
the same V1→V3 no-forecast→realistic-forecast framing, and SAC's variance is
known to be low in this project):** SAC's shield reduction barely moves
(54%→51.8% on regime 0, 33.5%→35.9% on regime 2) and degrades only
moderately on regime 1 (37%→26.2%), with delivery at ≥95% of B0 and zero
safety violations on every regime under the realistic forecast. This is the
strongest evidence in the project that SAC's "beats always-on" result is real
and not dependent on information a deployed controller could not have.

**Finding (DQN, final):** DQN is a real, if modest, improvement over B0 on
regime 0 (28.3% shield cut under a true perfect forecast, 18.3% under a
realistic one) — genuinely degraded by forecast realism, but by far less
than first appeared, and starting from a smaller base than the originally
reported 62%. DQN was never carried as a cross-regime alternative to SAC
(O3-3), so this correction does not change the project's SAC recommendation
— if anything it strengthens it, since DQN's regime-0 result is now shown to
be smaller and noisier than previously stated.

## O3-6 — The missing fourth case: `forecast_mode="none"`, and the full eval suite on every forecast variant (2026-11-14)

The original plan called for a 4-case table (perfect / smart / none /
"optimism check") and 3 seeds; given the 10-hour budget, this project runs
**1 seed, 3 of the 4 cases** (perfect, smart, none — the "optimism check,"
a sensitivity sweep over the Kc clip bound, is out of scope for this pass)
and completes the **full** locked-test + robustness + sensor-fault +
deployment-metrics suite on every trained variant, not just the headline
smart-forecast numbers.

**`forecast_mode="none"` (zero-forecast ablation floor)** was trained for
SAC on all three regimes and DQN on regime 0 (same seed 42, same steps as
every other variant) and run through the full locked test:

| Regime | Policy | V1, no forecast field at all (original) | V3 + `forecast_mode="none"` (same obs shape, forecast zeroed) | V3 + smart forecast |
|---|---|---|---|---|
| 0 | SAC | 97.5% / 54% | 99.1% / **49.8%** | 98.4% / 51.8% |
| 1 | SAC | 98.7% / 37% | 99.3% / **29.0%** | 99.7% / 26.2% |
| 2 | SAC | 97.3% / 33.5% | 98.9% / **41.0%** | 98.8% / 35.9% |
| 0 | DQN | 96.2% / 62% | 100.0% / **0.0%** | 99.6% / 18.3% |

**SAC's result is essentially forecast-independent.** Across none/smart (and
the original V1), SAC's shield cut stays in a 26–54% band on every regime
with no consistent direction — it is not being driven by the forecast at
all; whatever benefit SAC has over B0 comes from its exploration/entropy
mechanism, not from reading a weather forecast. This is further, stronger
evidence for the O3-5 finding that SAC's result is real and not an artifact
of information a deployed controller wouldn't have.

**DQN completely collapses to B0 with zero forecast signal.** `V3` +
`forecast_mode="none"` gives DQN regime 0 a shield cut of exactly **0%** —
delivery and shield count are bit-identical to B0 (100.0% / 46,969 vs
46,969). This is a clean, striking result: DQN *needs* some forecast
field (even a noisy one — see the 28.3%/18.3% perfect/smart numbers in
O3-5) to differentiate from always-charge at all; with literally zero
weather information it finds no incentive to deviate from B0. This
reinforces O3-3/O3-5's conclusion that DQN is not a robust cross-regime
or cross-condition alternative to SAC.

**Robustness suite: PASSED, zero safety violations**, on every one of
these new variants (SAC r0/r1/r2 for both `smart` and `none`; DQN r0 for
both) — 10-scenario suite × 3 validation years each. No regressions found.

**Deployment metrics / sensor-fault re-evaluation — the regime-1 severe
vulnerability from O3-4 does not reproduce under either V3 variant.**
Re-running the sensor-fault suite (GHI dropout/stuck pyranometer) on the
`none` and `smart` V3 models gives a materially different picture from the
original O3-4 finding (which used the `V1` no-forecast model):

| Regime | Variant | Nominal delivery | Under GHI dropout/stuck | Drop |
|---|---|---|---|---|
| 0 | V1 (original, O3-4) | ~1990 kWh | unaffected / slightly higher | ~0% |
| 0 | V3 + none | 1988.0 kWh | 1806.4 kWh | **9.1%** |
| 0 | V3 + smart | 1971.4 kWh | 1696.1 kWh | **14.0%** |
| 1 | V1 (original, O3-4) | ~1990 kWh | **~9.6 kWh** | **~99.5%** |
| 1 | V3 + none | 2000.0 kWh | 1735.8 kWh | 13.2% |
| 1 | V3 + smart | 2008.5 kWh | 1983.6 kWh | **1.2%** |
| 2 | V1 (original, O3-4) | ~1990 kWh | ~15% lower | ~15% |
| 2 | V3 + none | 1970.5 kWh | 1928.2 kWh | 2.2% |
| 2 | V3 + smart | 1971.3 kWh | 1923.2 kWh | 2.4% |

Zero hard-safety violations in every cell of this table (the shield acts on
true state regardless of what the policy observes, so safety always holds
by construction — only delivery/availability degrades).

**Reading this carefully: regime-1's catastrophic ~99.5% collapse appears
to have been specific to that one `V1` training run, not a property of
"SAC on regime 1."** Both V3 retrains (none and smart) on regime 1 show mild
degradation (1.2–13.2%) — in the same range as regimes 0 and 2 — not the
near-total collapse reported in O3-4. Regime 0, conversely, *gained* a
moderate new vulnerability (9–14%) it didn't show under the original V1
model. Plausible explanation: a different random seed/architecture
coincidence in which observation dimension the policy happened to lean on
for each regime, not a deterministic law about regime 1 specifically. This
does **not** retract O3-4 — that finding was real for the checkpoint it was
measured on, and a production deployment should still treat "does this
checkpoint have a sensor single-point-of-failure" as something to test
per-checkpoint — but it does mean the vulnerability is checkpoint-specific
rather than an inherent regime-1 property, which is a more encouraging
picture than O3-4 alone suggested.

**All raw outputs:** `results/phase6_final_test_{sac,dqn}_{smartfc,none}_r{0,1,2}.csv`,
`results/phase5_robustness_r{0,1,2}_{sac,dqn}_{smartfc,none}.csv`,
`results/phase9_sensor_faults_r{0,1,2}_{smartfc,none}.csv`,
`results/deployment_metrics_r{0,1,2}_{smartfc,none}.json`.

---

## Still open

1. **SAC's regime-1 sensor-fault vulnerability** — turns out to be
   checkpoint-specific (see O3-6), not a reproducible regime-1 property, but
   per-checkpoint sensor-fault testing before any production deployment is
   still unmitigated process, not a guarantee.
2. **Regimes 1 and 2's Generation-2 extensions** (Pareto sweep, sub-hourly
   ablation, domain randomization) — were run on pre-fix data and are invalid;
   not yet rerun on corrected data.
2b. **Multi-seed (3 seeds) and the "optimism check" case** from the original
   4-case plan were explicitly descoped to fit the time budget (O3-6) — only
   1 seed and 3/4 cases (perfect/smart/none) were run. A 3-seed study would
   give real confidence intervals instead of point estimates and is the
   natural next step if more time becomes available.
3. **Streamlit portal**: Historical Replay mode now offers the realistic
   smart-forecast model where trained (SAC r0/r1/r2, DQN r0) alongside the
   original no-forecast model. **LIVE (Open-Meteo) mode still cannot use the
   smart-forecast model** — Open-Meteo has no clear-sky-irradiance column,
   so `run_rollout_live` silently falls back to zero forecast fields
   (equivalent to `forecast_mode="none"`) if a smart-forecast model is
   selected there, with a warning shown in the UI. Porting an analytic
   clear-sky model from lat/lon via pvlib (since no stored column exists
   live) is unstarted follow-on work.
4. PPO/RecurrentPPO/QRDQN/MaskablePPO were not retrained on corrected data in
   this pass (all previously collapsed to B0-identical behaviour and were
   deprioritized under time pressure); their old numbers are retired along
   with everything else pre-fix, but no corrected rerun exists for them.
