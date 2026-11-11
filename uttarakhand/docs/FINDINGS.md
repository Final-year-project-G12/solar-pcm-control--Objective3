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
| Algorithms | PPO, 4 weight sets x 3 seeds per regime (36 runs) | PPO, DQN, RecurrentPPO, QRDQN, MaskablePPO, SAC; one seed shown |
| Delivery result | B0 best; PPO -8 to -11 % delivery, about -99 % shield hours | B0 2146 kWh; DQN 2144 kWh (-0.1 %) with 26 % fewer shield events; PPO / RecurrentPPO / QRDQN identical to B0 |
| Rule baseline | B1 -1 to -4 % delivery, -35 to -60 % shield hours | B1 -6.1 % delivery, -55 % shield events |

**What agrees across the two states:** with O2's shield inside the plant, no controller beats always-on on delivered
energy; every rule or learned policy that reduces shield activity pays in delivery; and PPO either copies B0 (TN) or
trades heavily toward safety margin (UK), depending on how the reward prices the shield. The TN DQN result (same
delivery, 26 % fewer shield events) is the one claim that goes further, but it rests on a re-implemented
single-node model, one weather sequence repeated three times, one seed and no locked test, so it **should be re-run
on the TN multinode kernel with per-year weather before it is cited**. The Uttarakhand protocol (unchanged-kernel
stepper + equivalence tests, O3-only test years, seeds, lock) is the one to port to Tamil Nadu.

## 8. Limits

* Simulation only; PCM temperature, melt fraction and stored energy are simulator states, not sensor values.
* Train and validate years were used by O2 to choose the design, so gains against B0 there are optimistic; 2006-2010
  is a hold-out by **year** at the same four medoids, not by location.
* Current-hour GHI in the observation is the hour mean (mild look-ahead); the live demo uses Open-Meteo, whose
  radiation is a model product on a ~11 km grid (r1 cell elevation 1,743 m vs medoid 1,348 m).
* No MPC or forecast baseline (B2, B3) was built; the literature suggests MPC is the strongest baseline, and its absence
  must be stated.
* Pumping energy is not modelled (O2 reports it as 0).
