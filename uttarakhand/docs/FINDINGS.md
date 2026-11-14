# Objective 3 - Uttarakhand: findings

Hourly enable/bypass controller for O2's frozen Uttarakhand PCM water heater (50 L tank, 1.5 m2 collector, 50 C
delivery, 300 L/day demand). Software-in-the-loop simulation only - **not field validation**. All numbers come from
files in `results/`; every result file carries the O3 config SHA-256.

## 1. Headline

**On O2's own primary metric (energy delivered up to 50 C in the 18:00-09:00 window), no hourly on/off controller
beats always-on operation (B0).** PPO learned a different trade-off: it removes almost all shield activity (from
about 600-700 h/yr to 1-4 h/yr, zero hard-limit hours) and gives up 8-11 % of window delivery for it. It still delivers
more than the water-only tank in every PCM regime, but keeps only about 20 % of B0's PCM gain. Under the plan's
lexicographic rule (delivery before shield hours) **B0 remains the best controller**, and the plan (Risk 1) names this as
a legitimate, reportable outcome.

## 2. Headroom gate G2 (validation 2021-2023) - `results/g2_summary_r{k}.json`

| | r0 | r1 | r2 |
|---|---|---|---|
| B1 grid points (35) beating B0 | 0 | 0 | 0 |
| Random wider-family policies beating B0 | 1 / 60 | 2 / 60 | 2 / 60 |
| Best delivery gain over B0 (any probe) | 0.0 kWh | 0.0 kWh | 0.0 kWh |
| Largest single-hour bypass gain in window delivery | 0.0008 kWh | 0.004 kWh | 0.005 kWh |
| Mean shield-hour change per single bypassed hour | -0.60 h | -0.44 h | -0.45 h |
| G2 | pass (tie rule only) | pass (tie rule only) | pass (tie rule only) |

G2 passed only through the tie criterion (delivery within 1 kWh/yr of B0 and fewer shield hours). **Bypassing an hour
never increases delivery**: the kernel already stops the loop when it would lose heat and at the 72 C shield, and the
collector model does not depend on flow, so the only lever is "collect less". The headroom that exists is
*fewer shield hours at roughly equal delivery*.

## 3. Locked test, 2006-2010 (M6, run once: `results/o3_test.lock`, run `o3test_e6ea0abf636a`)

Fully solid start, deterministic demand, 5 years x 4 medoids; PPO = mean of 3 frozen seeds (+- seed SD).

| Regime | Controller | Window delivery kWh/yr | Gain vs water-only twin | Shield h/yr | Hours over limit | Bypass switches |
|---|---|---|---|---|---|---|
| r0 | **B0** | **636.9** | **+84.3** | 717.7 | 0 | 0 |
| | B1 | 613.3 | +60.7 | 305.9 | 0 | 264 |
| | PPO | 568.2 +- 3.6 | +15.6 | 3.0 +- 1.2 | 0 | 306 |
| r1 | **B0** | **631.6** | **+76.1** | 681.4 | 0 | 0 |
| | B1 | 623.5 | +67.9 | 445.0 | 0 | 153 |
| | PPO | 569.5 +- 16.4 | +13.9 | 4.2 +- 5.9 | 0 | 426 |
| r2 | **B0** | **653.6** | **+69.8** | 573.5 | 0 | 0 |
| | B1 | 641.2 | +57.4 | 295.6 | 0 | 167 |
| | PPO | 601.6 +- 0.6 | +17.7 | 1.3 +- 1.3 | 0 | 367 |
| r3 | B0 | 626.3 | -5.6 | 180.4 | 0 | 0 |
| | B1 | 623.3 | -8.6 | 100.8 | 0 | 68 |

The highest temperatures in any test year for any controller are 72.6 C water / 63.6 C PCM (limits 75 / 65). Episode energy
residual is 0.0 % (O2 Gate 1 threshold 0.1 %). Test results agree with validation in ranking and size (section 4),
so selection on 2021-2023 did not overfit.

**Regime 3** (cold high-altitude, UKP_0041) has no PPO: O2 chose 4 capsules (2 % PCM) and flagged the PCM as not
meaningful, and the PCM tank delivers *less* than the water-only tank (-5.6 kWh/yr). No PCM controller is needed there.

## 4. Robustness (M5, validation years) - `results/m5_robustness_r{k}.csv`

Window delivery, mean over years / seeds:

| Regime | Demand | B0 | B1 | PPO | PPO shield h |
|---|---|---|---|---|---|
| r0 | deterministic | 616.2 | 593.0 | 551.9 | 3.8 |
| | stochastic (assumed) | 572.6 | 549.8 | 509.5 | 4.9 |
| r1 | deterministic | 616.1 | 608.9 | 557.5 | 4.8 |
| | stochastic (assumed) | 572.8 | 565.7 | 520.9 | 13.2 |
| r2 | deterministic | 634.9 | 623.6 | 590.2 | 1.9 |
| | stochastic (assumed) | 602.3 | 592.8 | 561.2 | 4.5 |

The reset state (fully solid / partially charged / fully liquid PCM) changes window delivery by less than 1 kWh/yr for
every controller. The ranking B0 > B1 > PPO is the same in all 36 regime x demand x reset cells, with zero hard-limit
hours everywhere. Stochastic-demand ranges (daily volume N(1, 0.15) clipped to 0.6-1.4, shift of -1/0/+1 h) are
**assumptions** (plan Q3 still open).

## 4b. Night serve metrics with O2's own definition (tau_serve, serve_frac)

O2 defines tau_serve as a restart scenario: from each day's own 18:00 state, 15 h with no sun and the normal demand;
hours until the top-node outlet first drops below 50 C. `o3/night.py` computes it for any controller and
`tests/test_night.py` shows it equals O2's own `run_case` value to 1e-9 (PCM tank and water-only twin). Mean over days,
years and PPO seeds (PPO seed SD <= 0.02 h):

| Regime | Years | B0 | B1 | PPO | Water-only twin | B0 gain vs twin |
|---|---|---|---|---|---|---|
| r0 | validate 2021-23 | 0.248 h | 0.212 h | 0.179 h | 0.355 h | -0.107 h |
| | test 2006-10 * | 0.256 h | 0.221 h | 0.183 h | 0.367 h | -0.111 h |
| r1 | validate | 0.185 h | 0.167 h | 0.108 h | 0.291 h | -0.107 h |
| | test * | 0.193 h | 0.173 h | 0.111 h | 0.299 h | -0.105 h |
| r2 | validate | 0.103 h | 0.075 h | 0.038 h | 0.264 h | -0.160 h |
| | test * | 0.113 h | 0.081 h | 0.038 h | 0.276 h | -0.163 h |
| r3 | validate | 0.082 h | 0.075 h | - | 0.085 h | -0.003 h |
| | test * | 0.091 h | 0.084 h | - | 0.095 h | -0.004 h |

\* computed after the lock (`scripts/m6b_postlock_checks.py`, same frozen policies, config and years) and labelled
EXPLORATORY; M6 itself reported only the in-run variant `tau_serve_insitu_h`. Files: `results/m5b_night_metrics.csv`,
`results/m6b_night_metrics_EXPLORATORY.csv`.

Every controller holds 50 C for only minutes after 18:00, and the PCM tank does **not** extend evening service at 50 C
compared with the same tank without PCM. This is the same thing O2's hand-off says (plan section 0: "it does not extend
the volume or time at 45 C in the evening versus the same tank"). Controllers that keep the tank cooler (B1, PPO) shorten
it further. The PCM's gain shows up in window energy delivered *below* 50 C (section 3), not in time held at 50 C.

## 5. Why PPO lands where it does

* The contract reward's safety term (w3, set by the dominance rule to 2x the largest possible one-hour normalised
  delivery) fires whenever the top node reaches 72 C or the PCM 62 C, i.e. whenever the shield *would* act. PPO
  learns to keep the tank just below that band. It does exactly what the agreed reward asks; the reward and the
  plan's selection rule disagree about what matters most.
* Across all 12 weight sets, w2 (overheat) and w4 (switching) changed delivery by at most about 10 kWh, much less than
  the 60-65 kWh gap to B0, so the result does not depend on the sweep choice. All 36 runs are in `results/m4_runs.csv`
  and all 12 sets in `results/m4_weight_sets.csv` (Q2: report all).
* An honest reading for the paper: *with O2's in-kernel shield already preventing every limit breach, shield hours have
  no physical cost in this simulator; a controller that avoids them buys a safety margin that the metrics do not price,
  and pays for it in delivered energy.* If shield operation does have a real cost (pump cycling, wear, thermal
  stress), that cost has to be modelled before an RL controller can be shown to add value.

## 6. Design limit inherited from O2 (not an O3 defect)

Only 1-4 % of the hot water drawn in the window leaves the tank at >= 50 C (`serve_frac_window`; B0 r1 2.6 %), and
the top node falls to about 18 C by 23:00 on a winter day. A 50 L tank cannot carry a 300 L/day demand into the evening
whatever the controller does. Window "delivery" is therefore mostly partial-temperature energy up to 50 C. This is
an O2 sizing question and should be raised there (plan section 10).

## 7. Comparison with the Tamil Nadu Objective 3 (`../tamilnadu/`)

| Aspect | Uttarakhand (this folder) | Tamil Nadu |
|---|---|---|
| Physics | O2's compiled multinode kernel called **unchanged**, stepped hourly; test T1 shows 8760 steps = one O2 full-year call to rtol 1e-9 | Single-node per-hour loop **re-implemented** ("inlined from run_year") in `o3/environment/pcm_env.py`; no equivalence test against O2 |
| O2 inputs | Byte-for-byte snapshot `o2_frozen/` (96 files re-hashed by a test) + contract v2 hash check | Reads `../objective2_design_optimization`, which **does not exist on this machine**, so TN cannot be re-run here; contract "v3" (plan section 2 said TN needs a v2 contract first) |
| Weather by year | Separate O2 point-year file per year | `_load_weather` falls back to the whole frame when there is no `year` column: **2021, 2022 and 2023 give bit-identical results** for every controller in `phase4_comparison_r0.csv`, so the "3 validation years" are one weather sequence |
| Test years | O3-only 2006-2010, downloaded and quality-checked, behind an O3 lock | 2011-2015, which are **O2's already-used held-out years** (plan 8.2: not usable as untouched data); no locked-test results file exists |
| Action space | 2 actions (enable / bypass) - in O2's kernel CHARGE and IDLE are the same thing | 3 actions CHARGE / IDLE / BYPASS (IDLE and BYPASS both switch the pump off) |
| Shield metric | Shield-active **hours** from the kernel (O2's definition) | Shield **events** per solver sub-step (about 50,000 / yr for B0) |
| Regimes | r0-r2 trained, r3 baselines only; all 4 in the locked test | Regime 0 only |
| Algorithms | PPO, 4 weight sets x 3 seeds per regime (36 runs); post-lock: DQN 3 regimes x 3 seeds, PPO with forecast obs + domain randomisation, w2 / w3 frontier (section 9) | PPO, DQN, RecurrentPPO, QRDQN, MaskablePPO, SAC. The final plan PDF (sections 16.9, 18) reports DQN on 3 seeds and selects SAC; only regime-0 comparison files are in this repo (`results/phase4_comparison_r0.csv`, one seed) |
| Delivery result | B0 best; PPO and DQN -8 to -11 % delivery, about -99 % shield hours | B0 2146 kWh; DQN 2144 kWh (-0.1 %) with 26 % fewer shield events; SAC 2097 kWh (-2.3 %) with 54 % fewer (PDF section 18.1); PPO / RecurrentPPO / QRDQN identical to B0 |
| Rule baseline | B1 -1 to -4 % delivery, -35 to -60 % shield hours | B1 -6.1 % delivery, -55 % shield events |

**What agrees across the two states:** with O2's shield inside the plant, no controller beats always-on on delivered
energy; every rule or learned policy that reduces shield activity pays in delivery; and PPO either copies B0 (TN) or
trades heavily toward safety margin (UK), depending on how the reward prices the shield. The TN DQN result (same
delivery, 26 % fewer shield events) is the one claim that goes further. The final plan PDF reports it on 3 seeds and
behind a locked test, but that test uses 2011-2015, which O2 had already used, the model is a re-implemented
single-node loop, and the TN validation file in this repo gives bit-identical numbers for 2021, 2022 and 2023 (one
weather sequence). In Uttarakhand, with the contract reward, DQN behaves like PPO, not like B0 (section 9). The TN
claim **should be re-run on the TN multinode kernel with per-year weather and O3-only test years before it is cited**. The Uttarakhand protocol (unchanged-kernel
stepper + equivalence tests, O3-only test years, seeds, lock) is the one to port to Tamil Nadu.

## 8. Limits

* Simulation only; PCM temperature, melt fraction and stored energy are simulator states, not sensor values.
* Train and validate years were used by O2 to choose the design, so gains against B0 there are optimistic; 2006-2010
  is a hold-out by **year** at the same four medoids, not by location.
* Current-hour GHI in the observation is the hour mean (mild look-ahead); the live demo uses Open-Meteo, whose
  radiation is a model product on a ~11 km grid (r1 cell elevation 1,743 m vs medoid 1,348 m).
* No MPC baseline (B3) was built; the literature suggests MPC is the strongest baseline, and its absence must be stated.
  The forecast rule B2 was built after the lock (section 9) and adds nothing over B1.
* Pumping energy is not modelled (O2 reports it as 0).

## 9. Additions from the final plan's as-built sections 16-19 (built after the lock)

The final plan PDF adds Tamil Nadu's as-built extensions (sections 16-19). Their Uttarakhand equivalents were built
**after** the one-shot test, so they are judged on the validation years 2021-2023 only; nothing here changes the
frozen policies or the M6 numbers.

### 9.1 B2 forecast-aware rule (PDF 16.2) - `results/m9_b2_*`, `b2_params_r{k}.json`

B1 plus a perfect 2-4 h GHI look-ahead (pre-emptive bypass when the PCM is nearly full, the tank is hot and strong sun
is coming), tuned on the validation years by the same rule as B1. **It lands on B1** in every regime (window delivery
equal to 0.1 kWh/yr, at most 2 fewer shield hours), and 10 % / 20 % forecast error changes nothing. A forecast cannot
create delivery headroom that G2 showed does not exist.

### 9.2 Deployment metrics (PDF 18.2) - `results/m10_deployment_r{k}.json`, `m10_sensor_faults_r{k}.csv`

| Regime | Controller | Thermal efficiency | Collector efficiency | Demand coverage | Annual delivery vs B1 |
|---|---|---|---|---|---|
| r0 | B0 / B1 / PPO | 43.0 / 42.2 / 40.7 % | 52.2 / 50.6 / 48.4 % | 32.1 / 31.5 / 30.4 % | +23.2 / 0 / -39.0 kWh |
| r1 | B0 / B1 / PPO | 45.4 / 45.1 / 43.6 % | 53.9 / 53.1 / 50.4 % | 29.8 / 29.7 / 28.6 % | +7.1 / 0 / -41.6 kWh |
| r2 | B0 / B1 / PPO | 46.2 / 45.7 / 44.5 % | 53.6 / 52.6 / 50.5 % | 29.5 / 29.2 / 28.4 % | +11.3 / 0 / -33.2 kWh |
| r3 | B0 / B1 | 52.5 / 52.3 % | 58.6 / 58.2 % | 24.0 / 23.9 % | +2.8 / 0 kWh |

Thermal efficiency = annual energy delivered to 50 C / solar incident on the 1.5 m2 collector; demand coverage =
delivered / energy needed to heat every drawn kg from mains to 50 C (the 50 L tank meets under a third of the
300 L/day demand, section 6). Backup-heater electricity = the delivery difference with an assumed resistive element
(efficiency 1.0, O3 assumption): PPO would need 33-42 kWh/yr MORE backup than B1. **PPO inference:** 0.24 ms median
per hourly decision (1 thread), 10,563 parameters, 42 kB fp32 weights, 157 kB checkpoint.

**Sensor faults** (observation corrupted, physics untouched; 6 scenarios x 3 years x PPO / B1 / B2): **zero hard-limit
hours in every case**, because the shield acts on the true state. PPO depends on the pyranometer: with it dead or
stuck, PPO stops bypassing, falls back to B0-like behaviour (+3 to +6 % delivery, shield hours back to B0 level).
A +2 C water-temperature bias makes B1 / B2 bypass earlier (-3 to -6 % delivery); PPO is insensitive to it.

### 9.3 O4 hand-off package (PDF 16.4) - `results/o4_handoff/r{k}/`

Per regime: hardware interface spec (13 contract signals, 4 of which need an estimator; one actuator, the collector
pump on/off at the fixed design flow; O2's 72 C / 62 C shield restated as an independent hardware interlock, read
from O2's own config; 75 / 65 C hard limits; fail-safe to B0), the frozen primary policy with its SHA-256 and
normalisation file, and a 6-step test plan ending at hardware-in-the-loop. Simulation-derived; nothing is validated
on hardware. Regime 3: no policy, always enable under the interlock.

### 9.4 Extension study (PDF 16.8 / 19.1) - `results/ext/`, `results/m12_extensions*.csv`

Validation window delivery, mean +- SD over 3 seeds (B0: r0 615.8, r1 615.7, r2 634.6 kWh/yr):

| Study | r0 | r1 | r2 | Shield h/yr |
|---|---|---|---|---|
| Frozen PPO (M4) | 551.6 +- 2.6 | 557.1 +- 15.8 | 589.9 +- 0.7 | 1.9-4.8 |
| **DQN**, same reward and selection rule | 545.9 +- 3.6 | 561.5 +- 8.0 | 588.3 +- 6.1 | 1.2-3.5 |
| **PPO, forecast obs (1-3 h GHI, next-hour draw) + training-time domain randomisation** | - | 564.4 +- 2.4 | - | 1.4 |

**DQN behaves like PPO**, not like B0 as in Tamil Nadu: with the contract reward, the exploration method does not
change the answer. Forecast inputs plus domain randomisation gain about 7 kWh/yr and make the seeds much more
consistent (SD 2.4 vs 15.8), but stay about 51 kWh/yr below B0.

**Delivery vs shield frontier, regime 1** (PDF 19.1 item 3):

| Reward setting | Window delivery | Shield h/yr |
|---|---|---|
| w2 (overheat) = 0, 1, 3, 10, 30, w3 by the dominance rule (29.7) | 553.7-558.0 | 1.0-4.8 |
| w3 = 10 (DIAGNOSTIC) | 565.7 +- 2.6 | 1.1 |
| w3 = 3 (DIAGNOSTIC) | 574.4 +- 4.3 | 8.5 |
| w3 = 0 (DIAGNOSTIC: no guard penalty) | 614.3 +- 2.3 | 607 |
| B0 always-on | 615.7 | 607 |

Sweeping the overheat weight w2 is flat, because the safety-guard weight w3 already removes shield time. Lowering w3
(diagnostic: Q2 fixes w3 by the dominance rule, so these runs are never selectable) shows a **cliff, not a curve**:
with no guard penalty PPO learns exactly B0 (the same thing Tamil Nadu found for its delivery-only reward, PDF 16.1),
and even a tenth of the agreed penalty removes about 99 % of shield time. There is no setting that keeps B0's
delivery with materially fewer shield hours; B1 is the only point in between (608.5 kWh, 406 h).

### 9.5 Not built, and why

* **Sub-hourly control (PDF 19.1 item 1):** the unchanged O2 kernel consumes whole-hour weather rows (plan 3.0), so a
  15-min decision step would need an edit to O2's physics, which the plan forbids. Tamil Nadu could do it only
  because its environment re-implements the physics.
* **SAC (PDF 16.3):** added afterwards for the algorithm comparison in section 9.6 (continuous action binned to
  bypass / enable); it is the weakest of the three agents here.
* **Regimes 1 and 2 (PDF 19.1 item 4):** already covered: Uttarakhand trained and tested regimes 0, 1 and 2 from the
  start, and regime 3 with baselines.

### 9.6 Realistic forecast and the choice of agent (post-lock, 2026-11-14) - `results/m13_*`, `results/ext/fcalgo/`

**Question.** Every V2 result in 9.4 gave the agent the stored future GHI (a perfect forecast), which no real controller
has. With a forecast a real controller can make, which DRL agent should Uttarakhand use: PPO (as frozen), DQN
(Rajasthan's choice) or SAC (Tamil Nadu's)?

**Forecast.** Smart persistence on the clear-sky index (Inman et al. 2013): GHI_hat(t+h) = Kc(t) x GHI_cs(t+h),
Kc(t) = GHI(t) / GHI_cs(t) clipped to [0, 1.2], held through night hours; GHI_cs is the weather file's NASA POWER
clear-sky column (`o3/env_ext.py`, `forecast_mode="smart"`; `"perfect"` is the old behaviour, bit-identical,
`"none"` gives zeros). Measured on the validation years 2021-2023, regimes 0-2, daytime hours: rRMSE 8.4 / 12.7 /
16.6 % at +1 / +2 / +3 h (MAE 19 / 31 / 41 W/m2), against 31.7 / 59.0 / 81.4 % for plain persistence. Tests:
`tests/test_forecast.py` (default unchanged over a full year, causal, bounded, reproducible, physics unaffected,
SAC action binning).

**Study** (`scripts/m13_forecast_algo_study.py`, `scripts/m13_summary.py`): regime 1, frozen M4 reward weights
(w2 = 0, w4 = 0.3, w3 by the dominance rule), V2 observation, training-time domain randomisation, 600,000 steps,
3 seeds, best validation checkpoint by the M4 rule. Validation 2021-2023, window delivery at 50 C:

| Agent | Forecast | Window kWh/yr | % of B0 | Overheat h/yr | Switches/yr | Hours over limit |
|---|---|---|---|---|---|---|
| B0 always enable | - | 615.7 | 100 | 607 | 0 | 0 |
| B1 reactive rule | - | 608.5 | 98.8 | 406 | 134 | 0 |
| **PPO** | **smart** | **567.5 +- 5.0** | **92.2** | **1.4** | **348** | **0** |
| PPO | perfect | 564.4 +- 2.4 | 91.7 | 1.4 | 371 | 0 |
| PPO trained perfect, scored smart | (no retraining) | 561.0 +- 2.8 | 91.1 | 1.1 | 372 | 0 |
| PPO | none (V1, M4) | 557.1 +- 15.8 | 90.5 | 4.8 | 416 | 0 |
| DQN | smart | 558.8 +- 15.3 | 90.8 | 4.9 | 411 | 0 |
| DQN | none (V1) | 561.5 +- 8.0 | 91.2 | 1.6 | 587 | 0 |
| SAC | smart | 521.9 +- 21.7 | 84.8 | 0.0 | 925 | 0 |

**Choice: PPO with the smart-persistence forecast.** It has the highest delivery, the smallest seed spread among the
learned agents (all three seeds 562-571 kWh), few overheat hours and the fewest switches. DQN reaches similar
delivery on its best seed but varies three times as much between seeds and overheats more; SAC removes the last overheat
hours by bypassing much more (about 925 switches a year) and loses another 7-8 % of delivery. Unlike Tamil Nadu and
Rajasthan, off-policy exploration buys nothing here, because the contract reward already pushes every agent to the same
'almost never touch the guard band' corner (section 9.4); the on-policy agent reaches that corner most consistently.

**What the forecast is worth.** Little. Swapping the perfect look-ahead for smart persistence at evaluation costs 3.4
kWh/yr (optimism check); training with smart persistence recovers it (567.5 vs 564.4), and removing the forecast
altogether costs about 10 kWh/yr, inside the V1 seed spread. In Uttarakhand the result does not rest on perfect
foresight, unlike Tamil Nadu's 'no forecast -> back to always-charge'.

**All three regimes** (PPO, smart forecast, each regime's frozen weights, 3 seeds; validation 2021-2023):
r0 546.8 +- 4.3 kWh (88.8 % of B0, overheat 0.4 h), r1 567.5 +- 5.0 (92.2 %, 1.4 h), r2 586.8 +- 1.6 (92.5 %, 0.7 h);
zero hours over a hard limit in every run.

**Test years 2006-2010, EXPLORATORY** (`scripts/m13b_postlock_test.py`; attaches read-only to the spent lock; the
agent was chosen after the test years had been seen once, so these numbers do not replace section 3):

| Regime | B0 (M6) | Frozen PPO (M6) | PPO + smart forecast (exploratory) | Overheat h/yr frozen -> smart | Switches frozen -> smart |
|---|---|---|---|---|---|
| r0 | 636.9 | 568.2 +- 3.6 (89.2 %) | 562.7 +- 3.7 (88.3 %) | 3.0 -> 0.9 | 306 -> 274 |
| r1 | 631.6 | 569.5 +- 16.4 (90.2 %) | 580.8 +- 4.8 (91.9 %) | 4.2 -> 2.3 | 426 -> 350 |
| r2 | 653.6 | 601.6 +- 0.6 (92.0 %) | 600.2 +- 1.8 (91.8 %) | 1.3 -> 0.4 | 367 -> 246 |

Zero hours over a hard limit for every seed and year. The recommended agent matches the frozen PPO on delivery (better
in r1 by 11 kWh with a third of the seed spread, within about 5 kWh elsewhere) with fewer overheat hours and fewer pump
switches, and it uses only inputs a real installation has. **B0 still delivers the most energy (8-12 % more); the
headline of section 1 does not change.** The frozen M4 policies and the M6 lock are untouched; the O4 hand-off and the
MATLAB twin still carry the frozen V1 PPO and would need a re-export (15 + 4 inputs, with the clear-sky envelope
computed on the device) to switch to this agent.
