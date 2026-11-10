# The four objectives, and the algorithms each one actually runs

**Project:** *Climate-Adaptive Intelligent Control and Optimization of PCM
Thermal Storage for Solar Water Heating* (final-year B.Tech CSE).

Status verified 2026-11-09. "Actually runs" means the algorithm is in code that
produced the results in `obj1_results/` and `obj2_results/` — not something
planned, approved or described in a proposal.

---

## The four objectives

| | objective | scope | status |
|---|---|---|---|
| **O1** | Climate-region-aware PCM **selection** | Which PCM suits each discovered climate regime | **Complete** |
| **O2** | PCM-storage **design optimisation** | Capsule geometry, arrangement, flow — the physical hardware and its validated operating envelope | **Complete** |
| **O3** | Real-time **adaptive control** | Which action — charge, discharge or bypass — at each timestep | **Not started** (blocked, see below) |
| **O4** | **Embedded hardware** deployment & validation | Validate the integrated system on hardware under representative Indian climates | **Not started** (depends on O3) |

**The boundary, which no objective may cross:** O1 selects the PCM shortlist.
O2 selects the physical design and envelope. O3 selects the real-time action
and *nothing else* — it must not change PCM identity, capsule geometry, tank
volume, safety limits or the released physics model. O4 validates the whole
stack on hardware.

### What blocks O3

The O3 environment contract does not exist for the current O2 results. The file
`objective2_design_optimization/objective3/obj3_environment_contract_tamilnadu.json`
(2026-10-27) describes a **superseded** run, and the generator
`src/handoff/build_obj3_contract_v2.py` is hardcoded for **Rajasthan**. Full
account in `README.md` → "Known gaps". The data the generator needs
(`wp6_band.csv`, `g10_heldout_results.csv`, `g10_selection_r{0,1,2}.json`) is
current and present — the work is porting the generator, not re-running O2.

---

## Objective 1 — algorithms that run

Source: `tamilnadu_pipeline/`. Full phase map in
`obj1_key_docs/00_MASTER_OVERVIEW.md`.

| phase | algorithm / method | where |
|---|---|---|
| 1 — data collection | Population-weighted spatial sampling: WorldPop 2020 density × GADM v4.1 boundary over a 0.25° grid → 133 points covering ~87.5 % of state population. Per-point, per-date solar geometry (**pvlib**) for sunrise / solar noon / sunset, so sampling follows the sun rather than a fixed clock | `00a`, `00b`, `00c`, `01`, `01b` |
| 2 — preprocessing | **Deaccumulation** of ERA5 accumulated fields to fluxes (`accum_to_flux`); 13-step physical QC; **per-season quantile mapping** to reconcile ERA5 against NASA POWER (two independent sources, cross-validated per row, not averaged blindly) | `02`, `02b`, `03b`, `04` |
| 3 — climate signature | Two-tier feature construction: Tier 1 instantaneous sun-event statistics, Tier 2 true daily integrals. Derives PCM-facing targets `Tm_target`, `L_required` (`SHARE_PCM = 0.5`) and the literature-anchored `kt_worst_month` charging ceiling | `04b` |
| 4 — climate regimes | **Gaussian Mixture Model**, `covariance_type="diag"` (chosen to stop overfitting), *k* auto-selected by `cluster_lib.suggest_k`'s 3-tier cascade → **k = 3** (not hand-picked). **Level B**: per-point, per-season re-clustering to detect regime shift, visualised as a Sankey | `05`, `05a` |
| 5 — feasibility | 8-constraint physical/corrosion/safety screening filter (Table 12 order, C6 = `Tm ≤ Tm_target_capped_C`), plus a **κ-calibration** companion pass. Full per-candidate audit retained — 9–15 of 62 survive per cluster | `06`, `07` |
| 6 — MCDM ranking | Four independent rankers — **TOPSIS**, **GRA**, **PROMETHEE II**, **VIKOR** — fused by **Borda count**, over 9 criteria (8 Table-13 + `thermal_margin`). **Entropy weighting** with the supercooling weight capped at 2× its prior. **Monte Carlo uncertainty propagation, N = 5000**. Agreement measured by **Kendall's W** + pairwise method agreement | `08` |
| 7 — physics validation | Grey-box **lumped-enthalpy** tank model, **backward-Euler**, `UA_TANK_W_K = 2.0`, driven by each regime medoid's real 10-year daily weather. Falsifiable check: **Spearman ρ** between MCDM rank and simulated solar fraction | `10` |
| 8 — recommendation cards | Per-regime markdown cards aligned to a 300 L/day domestic draw profile | `09` |
| post-6 | Seasonal PCM re-rank sensitivity (does the winner flip by season?) | `11` |

**Result:** 3 regimes; consensus Top-1 PCM `RT57HC` in all three.

**Disclosed open finding:** the `thermal_margin` criterion fixed MCDM-vs-physics
rank agreement in Cluster 0 (ρ −0.595 → +0.381) but **not** in Clusters 1 and 2.
Proven, not assumed — the identical PCM `CrodaTherm 60` scores 38 % / 79 % / 66 %
simulated performance across the three regimes on unchanged properties, so the
driver is a dynamic climate–PCM interaction no static criterion can capture.
Kept as a documented limitation rather than chased with more criteria tuning,
which would overfit the ranking to one physics model. See
`tamilnadu_pipeline/docs/era5_tamilnadu/09_PHASE_7_AUDIT.md`.

---

## Objective 2 — algorithms that run

Source: `objective2_design_optimization/`. 30 stages, `g2`…`g24`, run by
`scripts/tn_run_all.py`. **89/89 tests pass** (verified 2026-11-09).

> ⚠ **Do not use `docs/RESULTS.md`'s own "Algorithms used" table.** It sits
> below that file's 2026-10-28 superseded banner and describes the *retired*
> architecture — "a single surrogate-scoring pass, not an optimization loop",
> "no active-learning, no NSGA-II", `src/doe/`, `src/surrogate/`,
> `src/robustness/`. None of those modules exist any more. The table below is
> the current pipeline, read from the code.

### Design space searched

| variable | range |
|---|---|
| capsule inner radius | 10–40 mm |
| PCM volume fraction of tank | 2–30 % |
| flow rate | 0.010–0.050 kg/s |
| capsule arrangement | single-layer, staggered, radial |
| PCM identity | O1's Top-3 shortlist for that regime (inherited, never re-ranked) |

Fixed system: 100 L tank, 2.5 m² collector.

### The chain

| stage | algorithm actually running | code |
|---|---|---|
| **Simulator** (the optimiser's evaluator) | **Enthalpy-method** capsule model; **multi-node stratified tank, N = 10**, dt = 150 s; **backward-Euler lagged-PCM** timestepping with **adaptive sub-stepping** near the melt band; **Ergun** pressure drop across the capsule bed; collector model; local-time demand profile; hard temperature-trip **safety shield** (72 °C water / 62 °C PCM, a 3 °C guard band below the 75/65 °C hard limits) | `src/simulation/` |
| **g2 — convergence** | Timestep halving (300→150→75 s, and 150→75→37.5 s): accept only if annual useful energy moves **< 0.05 %** per halving and the Gate-1 energy-balance residual stays ≤ 0.1 % | `g2_timestep_convergence.py` |
| **g4 — discretisation** | Node-count study, N = 10 vs 20 | `g4_multinode_report.py` |
| **g7 — sensitivity** | One-factor-at-a-time **paired** sweeps against a water-only twin at identical flow; max-packing-fraction check | `g7_*.py` |
| **g8/S1 — screening** | **Taguchi L9 orthogonal array** (4 variables × 3 levels) + **ANOVA**, replicated over 5 explore years | `g8_s1_screening.py` |
| **g8/S2 — DOE** | **Scrambled Sobol** quasi-random sequence, `qmc.Sobol(d=4, scramble=True, seed=SEED).random_base2(7)` = **128 points**, identical point set per (regime, PCM) pair so pairs stay comparable | `g8_s2_doe.py` |
| **g8/S3 — surrogate bake-off** | **Five families compared on the same folds**: `nn_ensemble` (5× `MLPRegressor` (32,32) tanh, bootstrap resamples — spread across members = uncertainty); `xgboost` (`XGBRegressor`, 300 trees, depth 3); `rsm_quadratic` (full 2nd-order polynomial + **Ridge**); `gp_ard` (**Gaussian process, anisotropic ARD Matérn-5/2** + fitted white noise — closed-form posterior σ for active learning, and a learned length scale per input that cross-checks the Taguchi ANOVA); `stacked` (**out-of-fold stacking** with a **non-negative ridge** meta-learner over GP/XGB/RSM/NN). **The bake-off decides, not reputation** — XGBoost won 21 of 27 groups on the real DOE; the GP came third despite beating it 11.2× on a synthetic benchmark | `src/optimize/surrogates.py`, `surrogates_advanced.py`, `g8_s3_compare.py` |
| **g9/S4 — multi-objective search** | **NSGA-II**: population 100, 100 generations, 3 seeds. **Simulated-binary crossover** (η = 15, p_c = 0.9), **polynomial mutation** (η = 20, p_m = 1/n), **binary tournament on (rank, crowding distance)**, **fast non-dominated sort**, **constrained domination**. Minimises f₁ = −g_window, f₂ = −g_annual, f₃ = d_overheat under exact geometry feasibility and a predicted PCM-peak + 1 °C guard ≤ 65 °C | `src/optimize/nsga2.py`, `g9_s4_s5.py` |
| **g9/S5 — active learning** | Up to 5 rounds: from the union of 3 seeds' Pareto sets, take the 10 points of **largest committee disagreement** (NN + XGB + RSM), simulate them for real, add, refit, re-run NSGA-II. Stop when the simulator-evaluated front's **hypervolume** moves < 1 % for two rounds. Plus a **24-corner box-vertex** step | `g9_s4_s5.py` |
| **g9/S6 — cross-check** | **Dense regular grid**: 6 radii × 8 fractions × 5 flows × 3 arrangements = **720 designs**, to confirm the search missed no region. Compared with **exact hypervolume** and the **additive ε-indicator** | `g9_s6_grid.py` |
| **g9 — repeatability band** | The noise floor, **measured not assumed**: perturb dt (150→75 s), node count (10→20), PCM latent heat (±10 %), and year-to-year weather scatter. Any gain smaller than this band is noise | `g9_band.py` → `wp6_band.csv` |
| **g10 — selection** | Documented **S0/S1/S2 funnel** with pre-declared tie-breaks: reject infeasible → designs indistinguishable within the band collapse to one set → tie-break on overheating, then PCM mass, capsule count, temperature margin | `g10_select.py` |
| **g10 — held-out test** | **5 years (2011–2015) reserved from the start, evaluated once, under a lock that cannot be released** (`held_out_test.lock`). Covers the chosen design **plus the best design of every other shortlisted PCM** — 9 designs, not 3, so it cannot be a cherry-pick | `g10_test.py` |
| **g20 — independent cross-check** | **Gaussian-process Bayesian optimisation** on g_window: scrambled-Sobol start, then **expected-improvement** acquisition on a Matérn-5/2 GP, best of a 4000-point random pool per round, infeasible designs penalised so the GP learns to avoid them. A comparison track, not a replacement — it changes no design | `g20_gp_bo_comparison.py` |
| **CFD — physics check** | **ANSYS Fluent**, **enthalpy-porosity** solidification/melting model, first-order upwind. One capsule, six hours, ~30 min per PCM. **Chooses no design** — it tests the assumptions the fast model buys its speed with | `ansys/`, `src/cfd/` |
| **g11–g24 — diagnostics** | Ablation, shield split, control comparisons, anchor check, serve-metric variants, widened-bound boundary-artefact test, sizing study, extra sensitivity, Chen-2025 benchmark, flow/F_R study | `g11`…`g24` |

**Why Fluent and not OpenFOAM** (which the source Rajasthan study used):
OpenFOAM's standard solvers have **no native solidification/melting model** —
they conduct heat into a solid capsule with sensible heating only, no latent
heat. For a study whose premise is that PCM stores heat *in its phase change*,
that solver can only show the uninteresting half.

### Results

| regime | PCM | capsule Ø | PCM fraction | flow | capsules | PCM mass |
|---|---|---|---|---|---|---|
| 0 | PureTemp 58 | 81.6 mm | 29.6 % | 0.0388 kg/s | 104 | 24.8 kg |
| 1 | RT57HC | 81.6 mm | 27.9 % | 0.0295 kg/s | 98 | 23.6 kg |
| 2 | RT57HC | 81.6 mm | 29.9 % | 0.0267 kg/s | 105 | 25.3 kg |

All staggered. Held-out gates, on years never used to choose anything:
**9/9 beat the plain-water twin**, 9/9 PCM under 65 °C (63.4–64.0 °C), 9/9 water
under 75 °C (72.2–72.9 °C). Margins 178–269 kWh/yr against a repeatability band
of 26–30 — **6–9× the noise floor**.

---

## Objective 3 — algorithms planned (none implemented)

Nothing here runs yet. Recorded so the plan is not mistaken for progress.

| element | intended approach |
|---|---|
| Environment | Refactor `tank_model.run_year()`'s inner loop into a stateful `step(action) → (obs, reward, done, info)`. **Reuse the physics unchanged** — only the control surface changes, `flow_rate_kg_s` becoming an action instead of a constant |
| State | 13 fields incl. stratified `T_water_top_C` / `T_water_bottom_C`, `T_pcm_C`, melt fraction, GHI, ambient, hour, draw mass, mains temp, and `recoverable_energy_above_45C_kWh` (an **estimator**, not a sensor) |
| Action space | Discrete charge / discharge / bypass, with a recommended continuous-flow hybrid. The controller may **never** command outside the flow envelope or temperature limits |
| Reward | Built from the O2 window metrics: delivered E_del,45 minus overheat time, safety violation and bypass-switching penalties. **Weights deliberately unset** — O3 must choose and justify them; the old `w1..w5` defaults came from retired designs and are not carried over |
| Baseline first | A **trivial rule-based controller** must pass the contract's acceptance test *before* any learning code is written. The target is to beat O2's blunt reactive on/off shield on finer metrics (shield-activation frequency, discharge timing) — **not** to fix a broken baseline, since the inherited environment is already safe and delivery-reliable |
| Candidate learners | DQN / PPO / SAC-family, benchmarked against each other on one environment and reward before committing — precedent [Sivaraj2023], [Emami2026]; ANN-MPC as a non-DRL contrast [Terfai2025] |
| Weather split | train 2016–2020, validate 2021–2023, **never tune on** 2011–2015 or 2024–2025 (already spent by O2's held-out tests) |

---

## Objective 4 — algorithms planned (none implemented)

Embedded deployment and validation of the integrated framework on hardware
across representative Indian climate regions, measuring thermal efficiency,
energy savings and operational reliability. Depends entirely on a trained O3
controller. No code, no hardware work started.

---

## Honest limits carried across O1 and O2

Stated here rather than left for a reader to find:

- **CFD melt fractions are conservative.** First-order upwind is the only
  discretisation under which all four PCMs converge, and it is more diffusive
  than second order. The **ordering** of PCMs is the result; absolute values
  are low.
- **The CFD mesh is a symmetry half-slice**, so buoyancy-driven convection in
  the melt is excluded. That enhancement is carried empirically by the tank
  model — a division of labour, not an oversight.
- **Wind and humidity reach no equation.** They are in the weather files, but
  collector and tank loss coefficients are fixed. For coastal regimes that is a
  real simplification.
- **Irradiance is used as plane-of-array directly**, no tilt model. It applies
  equally to the PCM tank and its plain twin, so the reported *difference* is
  largely insulated; the absolute energies are not.
- **Collector F_R is flow-independent**, so any flow optimum sitting at
  0.010 kg/s is a model artefact.
- **The 30 % PCM-fraction cap is binding.** At 40 % the gain is higher by more
  than the band in every regime with both peaks still inside limits. The cap is
  administrative; the only literature in the files read (Chen 2025) covers
  10–20 %.
- **O1's `monsoon_index` is a proxy** — NASA POWER precipitation was never
  downloaded. External cluster validation against Köppen-Geiger / NBC-ECBC
  zones is not implemented.

### A defect worth knowing about

Partway through O2, **every result had to be thrown away and re-run.** The
per-point-year weather cache carried its solar columns **shifted +5 hours**
against their own timestamps — IST truncated to whole hours, written onto
local-clock stamps but labelled UTC. That moved collector gain from ~11:30 to
~16:30 local, far closer to the 19:00 demand peak than reality, which
systematically **flattered PCM** on the study's primary objective.

It was caught by checking irradiance against computed sunrise and sunset, and
confirmed three independent ways. The cause was a second raw dataset whose
header declares **Local Solar Time**, read as though its keys were UTC. All 45
point-years were rebuilt from raw behind a **mandatory ephemeris gate**: the
rebuild refuses to write unless clearsky daylight brackets true sunrise and
sunset on every day. The inventory now records *how* each year's timestamps
were derived rather than asserting a time standard, because:

> Provenance metadata is not verification.
