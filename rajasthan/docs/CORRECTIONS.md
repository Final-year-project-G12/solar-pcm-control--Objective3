# CORRECTIONS — Objective 3 (Rajasthan)

Every place where this implementation differs from the Tamil Nadu (TN) reference in
`solar-pcm-control--Objective3/tamilnadu/`, from the plan (`.claude/Objective_3_Implementation_Plan_FINAL.docx`),
or from what an earlier file claimed — and why. Date: 2026-11-11. Nothing under `tamilnadu/` was modified.
Nothing under `OBJECTIVE2/` was changed except ONE new file (D1) and the O2 numba cache files Python writes
next to the kernel when it runs (the same thing every O2 run does).

Sources used for the decisions below (all read in this session):
`.claude/Objective_3_Implementation_Plan_FINAL.docx` (sections cited as "plan §x"),
`OBJECTIVE2/Objective---2/objective2-rajasthan/src/simulation/tank_model_multinode.py` and `run_case.py`
(the frozen kernel), `.../results/obj3_environment_contract_rajasthan_v2.json` and `g10_*`, `wp6_band.csv`,
`held_out_test.lock`, `.../configs/*.yaml`, `.claude/Sources/Emami2026DRL_Solar_ORC_TES_summary.md` (DRL-for-solar-TES
precedent, hourly steps over 8760 h, safety filter) and `.claude/Sources/IS12976_2023.pdf` (collector/tank ratio,
cited via the O2 YAML comments, not re-read).

---

## Verification first (what was actually run, 2026-11-11)

| What | Result |
|---|---|
| Contract (phase 0) | PASSED: v3, 50 L / 1.5 m², 3 regimes, flow 0.010–0.050, provenance hashes of 8 O2 files |
| Test suite (`python -m pytest tests/`) | 117 passed |
| T1 stepped year vs one O2 `run_case(multinode)` year | identical: max \|ΔT_top\| = 0, \|ΔT_pcm\| ≤ 2e-14, window energy equal to 1e-9 relative |
| Serial vs parallel baselines / dataset / robustness | identical metrics; dataset files byte-identical |
| `SubprocVecEnv` vs `DummyVecEnv` PPO, same seed | identical network weights after 256 steps |
| Smoke training, all six agents, 3,000 steps | runs end to end through phase 4 (numbers are NOT results) |
| Phase 6 dry run on a throw-away lock | works once, second run refused; the real lock was never created |

**Disclosure about the test years.** To prove the lock works, phase 6 was exercised ONCE as a dry run with a throw-away
3,000-step smoke policy, regime 1, test year 2011 only, against a lock file in the scratch folder (not `results/o3_test.lock`).
Its output (B0 / B1 / twin / smoke-policy numbers for 2011) was not used for any decision and is not in the repository.
`test_T7_lock_is_exclusive` also loads a 2011 weather file to check its shape. 2012–2015 were never loaded outside the tests'
refusal checks. The real one-shot test was later spent on the selected Gen-2 DQN run (2026-11-12; see the Gen-2 section).

---

## D1 — The contract mismatch (decided with the student, option (a))

TN's `contract_loader` requires `contract_version = obj3_contract_v3.x`, `static_design_per_regime`,
`fixed_system` = 100 L / 2.5 m², `delivery_target_temp_C`, `flow_min/max_kg_s`. The Rajasthan file
(`results/obj3_environment_contract_rajasthan_v2.json`) has no version field, `regimes[]`,
`delivery_temp_target_C`, `pump_flow_*`, and the system is 50 L / 1.5 m².

* NEW file in the O2 tree: `src/handoff/build_obj3_contract_rajasthan_v3.py` → `results/obj3_environment_contract_rajasthan_v3.json`
  (`contract_version = obj3_contract_v3.0_rajasthan_2026-11-11`). It re-expresses existing results in the v3 schema.
  No O2 value is changed; `verify_against_sources()` re-reads the sources and checks 26 copied fields (design, limits,
  held-out rows, 50 L / 1.5 m²); `tests/test_contract_builder.py` repeats it.
* `o3/utils/contract_loader.py` keeps the TN structure and function names but validates Rajasthan's own values
  (state = rajasthan, 50 L / 1.5 m², flow envelope 0.010–0.050, three regimes, flow of each design inside the envelope)
  and re-hashes the O2 files named in `provenance` (a stale contract is refused).
* The held-out limit check looks only at `role == "chosen"` rows. The O2 held-out table also holds `best_of_pcm`
  alternatives, which are not released designs.
* Computed in the contract, not asserted: `held_out_test.gates_chosen_designs` — 15/15 chosen rows keep PCM ≤ 65 C and
  water ≤ 75 C and have positive `g_window`, but only **3 of 15** rows have a positive `gain_serve_frac_45_mean`.
  PCM does not extend the time at 45 C in the evening in every regime (plan §0). This is O2's result, reported as it is.
* v2 recommended the continuous-flow hybrid action; v3 recommends the discrete version because F_R is flow-independent
  (`collector.flow_dependent_fr.enabled = false`; plan §0, §3.3).

## D2 — The environment steps the real multi-node kernel (TN did not)

TN's `pcm_env.py` hand-inlined a single-node tank (an older copy of `run_year`) although its contract says
"multi-node". Rajasthan's `o3/environment/o2_adapter.py` calls the O2 numba `_kernel` directly, one hour at a time
(plan §3.1 fallback: "call the same compiled kernel function directly"), carrying node temperatures and enthalpies.
The only code copied from O2 is the Python set-up that `run_year_multinode` does before calling `_kernel`
(node masses, loss coefficients, parameter vector). `tests/test_environment.py` T1–T3 prove it equals O2.

* "Bypass" = the hour's collector arrays zeroed (what GHI = 0 does in `_collector_arrays`).
* `T_w` in the state dict and info = the TOP node (the delivery node the shield reads).
* End-of-hour state is read from the kernel's own hourly output (profiling: recomputing it in Python was a third of the
  step time). `test_end_of_hour_state_equals_adapter_derived` keeps an independent formula honest.

## D3 — Observation: the 13 contract fields (TN: 11 dims)

Plan §3.2 / contract `dynamic_state_schema`: T_top, T_bottom, T_mean, T_pcm, f_melt, GHI, T_amb, hour, draw now, T_mains,
stored energy above mains, recoverable energy above 45 C, minutes since last draw (`obs_version="V2"`, default). TN's
11-dim layout stays as `"V1"`. Scaling constants are FIXED and declared in `o3/environment/obs.py` (plan §3.2 said
"normalise with training-years statistics"; fixed constants carry no information from any weather year at all, so the
leakage question cannot arise — same choice as TN). Baselines, masks and the demo read observations through
`obs.decode()`, which accepts both layouts.
`GHI` is the hour's mean value, a mild look-ahead (plan §3.2); live mode uses a nowcast.

## D4 — Reward: delivered-to-45 C, and a re-derived shield weight

* Energy term = `E_del45` (energy delivered up to 45 C, contract / plan §6). TN used `E_load` (all heat to the load).
* **The TN shield weight (10) cannot be used for Rajasthan.** TN tuned it against TN's counters. The multi-node kernel
  counts a PCM shield per NODE per sub-step, so Rajasthan's B0 collects 1.3e5–3.3e5 activations a year (TN: ~5e4). With
  weight 10 the shield term is ~5× the delivery term: B0, the best policy on the plan's own metric (window gain +136…+152
  kWh/yr versus the water-only twin, regime 1), scored −5.4 M while the twin scored −2.7 M.
* Rajasthan value = TN's METHOD on Rajasthan data: 1.19 × the break-even weight w* at which B0 and B1 tie, measured on the
  TRAINING years 2016–2020 only (TN: weight 10 vs break-even 8.4). Measured w*: regime 0 = 3.26, regime 1 = 1.43,
  regime 2 = 1.45 → `W_SHIELD_BY_REGIME = {0: 3.9, 1: 1.7, 2: 1.7}` in `pcm_env.py`. Reproduce:
  `python scripts/phase2_baselines.py --regime 1 --calibrate-shield`.
* **Provisional.** Plan §6/Q2 says the shield weight is swept on validation years on a log grid including 0, ≥ 3 seeds each.
  That sweep is a full training job and has NOT been run (`--w-shield` on phases 4 exposes it). The safety-violation weight
  (50), unmet weight (2) and switching weight (0.1) are TN values, unswept.
* Plan Q2 asks for E_del45 to be divided by the water-only twin's value over 2016–2020; not done (TN did not either).
  Rewards are therefore in Wh-equivalents, not normalised.

## D5 — Three actions, but IDLE ≡ BYPASS in the kernel

The plan's prototype is `Discrete(2)` (loop on / off); TN's as-built (plan §16) has CHARGE / IDLE / BYPASS. Rajasthan
keeps three actions for TN parity (`test_idle_and_bypass_are_physically_identical` proves they are the same physics).
They differ only in the switching penalty and in what the policy reports. If a cleaner action space is wanted, the
change is `N_ACTIONS = 2` plus the mask; nothing else depends on it.

## D6 — Test years: fresh 2006–2010 (updated 2026-11-11; first version used O2's 2011–2015)

Plan §8.2/Q1: the O3-only test set is 2006–2010 at RJP_0132, RJP_0192, RJP_0083. With the student's go-ahead,
`scripts/phase0_fetch_test_weather.py` downloaded the 15 point-years from NASA POWER (same request as O2's g19), applied
O2's inventory acceptance rules (≥ 98 % valid hours, no gap > 3 h, unbroken hourly UTC sequence) without printing or plotting
any weather value, and wrote `data/weather_test_2006_2010/` (raw JSON, cleaned CSVs, `inventory.csv` with SHA-256). **All 15 were
accepted**, so the test set is complete (no shrinkage to report).
`o3/utils/weather_roles.py`: test = 2006–2010 behind `results/o3_test.lock` (exclusive-create, plan §15.8); 2011–2015 (O2's
held-out) and 2024–2025 are `used_o2` and always refused. O2's `roles.load_point_year` is not used. It is still a temporal
hold-out at the same three sites. Earlier disclosure in this file about a 2011 dry run is therefore moot for the real test
set: the dry run touched O2-used data, not the fresh years; the fresh years have been loaded only by the lock-guard tests
(shape check) and never evaluated.

## D7 — Action mask threshold (a TN discrepancy)

TN's `gym_wrapper` documents "CHARGE masked when T_water > 68 C" but compares the normalised value against 1.2, which is
78 C, above the 72 C shield — the mask never fired. Rajasthan decodes physical values: mask CHARGE at top node ≥ 68 C or
GHI ≤ 10 W/m² (the collector cutoff). B4 uses the same function. `test_action_mask_physical_thresholds`.

## D8 — Robustness suite (TN bugs fixed)

TN's `tamb_noise_C` was applied only when > 0 (its "tamb_minus2C" scenario did nothing), scaled by /50 instead of the
observation's /25; `demand_scale` was accepted and ignored; the noise RNG was unseeded. Rajasthan: signed bias, correct
scale, demand scenarios change the physics (`PCMEnv(demand_volume_multiplier=…)`), noise seeded from (seed, year) so the
CSV is identical for any `--n-jobs`. Three scenarios added (demand ×0.7, ×1.3, stochastic daily demand; plan M5): 13 total.
Observation noise is a TN carry-over; the plan defers sensor noise/forecast error (§8.1).

## D9 — Live demo (TN bug fixed) and Streamlit app

`app/streamlit_app.py` (plan §15.9): persistent banner on every view; replay (regime, allowed year, date range, B0/B1/B2/twin and any trained agent side by side, temperatures, melt fraction, draw, action, shield status, cumulative energy, over-limit time, CSV download) and live mode (one hour per click, provider/latency/quality logged, visible fallback). Run `streamlit run app/streamlit_app.py`. Tested with Streamlit's AppTest and a real server start (`tests/test_streamlit_app.py`).


TN's `LiveDemoRunner.step_once` read weather but never gave it to the environment (it reset the env every step when the
env had no `_state_to_obs`). Rajasthan injects each frame through `PCMEnv.set_weather_override()` and runs the same
`env.step()`, so replaying a stored validation year reproduces the evaluation numbers (`tests/test_demo.py`).
Live mode: Open-Meteo at one of the three O2 medoids (0 Udaipur, 1 Nagaur, 2 Jaipur — plan Q6 option); failures fall back
visibly (persistence, then stub) and are logged. Provider choice (plan Q5) is still the student's.
Every row carries "Software-in-the-loop simulation. Not field validation."

## D10 — Metrics

One evaluation loop (`o3/evaluation/rollout.py`) replaces TN's 14 copies. Per episode: E_del45 annual and in the O2 window
18:00–09:00 (same definition as `tes_metrics.WINDOW_HOURS`), gain versus the water-only twin (`TWIN_water_only`, same tank,
flow, shield, weather, demand, zero capsules), shield hours (water, PCM), hours over limit, peak top/PCM temperature,
bypass switches, energy residual, in-situ serve fraction. **Not computed:** O2's `tau_hold`/`tau_serve` night
scenarios (they restart the solver from each day's 18:00 state); plan §1 lists "serve fraction / tau_serve" — only the
in-situ serve fraction is reported.

## D11 — Training

* On-policy agents take `--n-envs` worker environments; `n_steps` is the TOTAL samples per update (8760), split across
  envs, so the batch size matches TN. Off-policy agents use one env (reason in `o3/agents/common.py`).
* EvalCallback evaluates all three validation years (TN evaluated 2021 three times) — `cycle_years=True`.
* Training episodes draw the reset scenario from the three contract scenarios (plan §8.1); TN always used "solid".
* Torch/BLAS threads pinned to 1 everywhere.
* Hyper-parameters are TN's (PPO lr 3e-4, DQN buffer 50k, etc.); no search was run (plan §15.7).
* Results are reproducible for a fixed (seed, n_envs); changing n_envs changes the experience stream.

## D12 — Multi-seed, selection, freezing

Seed directories are `results/<algo_dir>/r<regime>_seed<seed>` (TN's trainers wrote `results/<algo>/…` but its loader
looked in `<algo_dir>/…`, so recurrent/maskable multi-seed models were never found by phase 6). Phase 4 multi-seed picks
the best seed per algorithm by plan §8.3's lexicographic rule on VALIDATION years and records seed + SHA-256 in
`results/phase4_selected_r<regime>.json`. Phase 6 fingerprints (models + contract + algo) into the lock.

## D13 — Regime 0 (O1 candidate pool undersized, n = 4)

Recorded in the contract (`small_sample_policy`, `o1_candidate_pool_status`) and printed by phase 0. O3 does not re-rank
PCMs: it controls the design O2 released (savE OM55, 52 capsules, 0.0155 kg/s). Policy decision from 2026-09-17: accept the
calibrated-κ ranking and report the caveat. Regime 1 (Nagaur, healthy pool, O1 #1 = O2 choice) is the plan's first
prototype regime and the default for tests and demos.

## D14 — 50 L / 1.5 m² (33.3 L/m²) limitation carried

Below the 40–100 L/m² band of IS 12976:2023 (O2 YAML comment, memory 2026-09-13). O3 may not change it. Absolute kWh
figures are not comparable with TN's 100 L / 2.5 m².

## D15 — Desert weather: what the numbers say, untuned

Regime 1 (Nagaur) B0 on training years 2016–2018: water-shield time 504–556 h/yr and PCM-shield time 544–607 h/yr
(≈ 1,050–1,160 h/yr together, counted separately); top node peaks 72.2 C, PCM 62.3 C, zero hours over the hard limits.
Untuned B1 and B2 (TN's rules) have a NEGATIVE window gain versus the water-only twin (B1 −12 to −21 kWh, B2 −36 to −48 kWh;
B0 +134 to +152). The rules switch the loop off in the 17–21 h evening window and below 200 W/m² of sun, which in a desert
costs useful heat (the collector still yields at 100 W/m² when T_amb ≈ 30 C). So the plan's headroom gate G2 ("does ANY
on/off policy beat B0?") is open: B1's grid (`phase2 --tune`, validation years) and the random-search half have not been run.
The honest possible outcome stays "no improvement over B0" (plan risk 1).

## D16 — Stochastic demand and the other assumed numbers

Stochastic daily demand (plan §3.4): volume factor LogNormal(mean 1, sd 0.20) clipped to [0.5, 1.5], timing shift
N(0, 1 h) per day. **Assumed ranges** — no measured Indian draw-off data was found (plan Q3). Used only in robustness
scenario `demand_stochastic`; training uses O2's deterministic curve.

## D17 — Paths and environment

TN code expected `objective2_design_optimization/` next to `objective3/`; that folder does not exist in this tree. Here
`config_O2.py` locates the O2 Rajasthan tree (override: `O2_RAJASTHAN_ROOT`) and `o3/__init__.py` puts it on `sys.path`
in every worker process. `O3_RESULTS_DIR` redirects all outputs (used for smoke tests so `results/` stays clean).
Python 3.10; gymnasium 1.4.0, stable-baselines3 2.9.2, sb3-contrib 2.9.2, torch 2.10 (CPU) — installed in a throw-away
virtual environment outside the repo (`requirements.txt` lists them). `folium` is not installed, so the regime map is
skipped (the dashboard needs only plotly).

## D18 — Additions that TN does not have

`scripts/phase1_acceptance.py` (the contract's six acceptance checks → pytest nodes), `scripts/benchmark_speedup.py`,
`o3/utils/{parallel,cli,models,weather_roles}.py`, `o3/environment/{o2_adapter,obs}.py`, `o3/evaluation/rollout.py`,
`--n-jobs` / `--n-envs` on every phase script, tests T1–T8 / parallel / demo / contract-builder, `--tune` for B1 (grid on
validation years), `--calibrate-shield`.

---

## Results of the plan's milestones M3, G2, M4, M5 (run 2026-11-11, regime numbers from `results/`)

**M3 — baselines on 2016–2023, all regimes** (`results/phase2_baseline_results_r*.csv`; zero hard-limit violations, residual 0.00 %):

| Regime | B0 window gain vs twin (kWh/yr) | B0 shield h (water + PCM) | B1 untuned gain | B2 gain |
|---|---|---|---|---|
| 0 Udaipur | +131.9 | 507 + 356 | +12.7 | −19.1 |
| 1 Nagaur | +138.4 | 507 + 554 | −16.2 | −41.6 |
| 2 Jaipur | +144.2 | 430 + 710 | +34.7 | +3.9 |

B1 tuned on validation years (24-point grid, `phase2_b1_tuning_r*.csv`): best window energy 503 / 449 / 525 kWh versus B0's
≈ 623 / 603 / 637 — tuning does not close the gap.

**G2 — headroom gate** (`scripts/phase2_headroom_gate.py`, `results/phase2_headroom_gate_r*`): 92 rule policies per regime
(24 grid + 8 near-B0 corners + 60 seeded random draws of the rule family) on validation years. Declared rule: headroom exists
iff the best policy's window gain exceeds B0's by more than O2's repeatability band for `g_window` (21.1 / 14.4 / 21.9 kWh).
**Verdict in all three regimes: NO policy beats B0 on window delivery** (best = +0.0 kWh over B0, reached only by never switching
the loop off). Secondary: within the band of B0, shield time can be cut in regime 2 (1,112 h → 764 h at −6.3 kWh, ~1,000 switches/yr);
regime 0 837 → 834 h; regime 1 none.

**M4 — PPO, 3 seeds × 300k steps, regime 1, run as a CONFIRMATION of G2** (the plan says stop training when G2 fails; a learned
policy searches a richer class than the rules, so it was run once to check the negative result, not to get round it):

| Reward | Seeds 42 / 0 / 123 on validation years |
|---|---|
| R2 with shield weight 0 (delivery only), `results/sweep_wshield0/` | all three = B0 exactly: 925.7 kWh, window gain +137.5, 0 switches |
| R2 with shield weight 1.7 (default), `results/ppo/`, `phase4_stats_r1.json` | all three = B0 exactly, although this reward scores B1 above B0 |

So PPO finds nothing better than always-on, and with the shield penalty it stays in the always-on optimum instead of finding the
(reward-better, delivery-worse) low-shield rules — the same collapse TN reported for PPO. Frozen: seed 42, SHA-256 in
`results/phase4_selected_r1.json`. Not run: the plan-Q2 weight grid (w2, w4) — the reward form is implemented
(`--reward-mode Q2`, twin normalisation in `contracts/reward_normalisation_rajasthan.json`, w3 by the dominance rule ≈ 11.3–11.5),
but after G2 a sweep cannot change the delivery ceiling; other algorithms and regimes 0, 2 likewise not trained.

**M5 — reset scenarios and stochastic demand, validation years** (`results/phase5_robustness_r1_{ppo,b0}.csv`, 13 scenarios × 3 years):
zero hard-limit violations and zero hours over limit everywhere; PPO ≡ B0 in every scenario; liquid / partial starts change the
year by < 1 kWh; demand ×0.7 / ×1.3 gives 854 / 979 kWh; assumed stochastic demand 933 kWh (window 466 kWh: the assumed timing
shifts move draws out of the 18–09 window).

**M6 — locked test: NOT run.** The fresh 2006–2010 set and the lock are ready; the frozen policy is behaviourally B0, so the test
would report B0 against the twin on unseen years. The one-shot is the student's to spend.

**Honest conclusion (plan risk 1).** With loop on/off as the only action and O2's flow-independent collector, hourly control has
no headroom on delivered energy in Rajasthan; the released passive design with its in-kernel shield is already the delivery
optimum. This is a reportable result. The plan's next step is to revisit the action space — enabling F_R(flow) is a declared O2
physics change and was not made.

Also added in this pass (plan 15.4, 15.8): `config_sha` (reward weights, demand parameters, reset mix, year roles, contract hash) on
every result row; git commit in the phase-6 fingerprint.

---

## Plan sections 16.9–19 (added to the plan 2026-11-11/12): what was done for Rajasthan, and what it shows

The updated plan adds Tamil Nadu's as-built selection (DQN, then SAC), deployment metrics (§18.2), and five advisor-feedback
items (§19.1) with a Generation-2 training study (§19.3). Those TN numbers (2146 kWh, 50,307 shields, 128.8 kWh savings …) are
TN's and are NOT used here. Rajasthan equivalents:

**Code (D19–D22)**
* D19 sub-hourly control: `PCMEnv(control_step_min=…)`, default 60 (unchanged). The O2 kernel runs one hour per call; for
  sub-hour periods `o2_adapter.period_kernel()` builds a copy from the O2 SOURCE TEXT with exactly two lines replaced
  (`nsteps = int(round(3600/dt))` → `int(P[35])`, `mdot_d = hour_mass/3600` → `hour_mass/(nsteps*dt)`); it raises if O2's source
  changes. The 150 s sub-stepping and the shield are untouched. `tests/test_gen2.py`: four 15-minute steps with one action equal one
  hourly step to 1e-9.
* D20 observation V3 (17 dims) = the 13 contract fields + GHI at +1/+2/+3 h + next-hour draw. (The plan's "V2 15-dim" is TN's
  layout over TN's 11-field V1; Rajasthan's V2 is already the 13-field contract vector, so the forecast version is named V3.)
  The forecast is the stored year's actual irradiance — a PERFECT forecast, like B2's.
* D21 training-time domain randomisation (`domain_randomize=True`): per episode GHI ×U[0.90,1.10], T_amb +U[−2,2] C, demand
  ×U[0.85,1.15], applied to the physics inputs; seeded; validation/test/baselines never randomise.
* D22 `scripts/train_one.py`, `scripts/gen2_study.py`, `scripts/phase9_deployment_metrics.py`, `scripts/make_gen2_plots.py`,
  `o3/agents/gen2.py`; phase 6 accepts Gen-2 runs (`--run`) and now checks the plan-18.1 rule (delivery ≥ 95 % of B0, annual and
  window) instead of "window gain ≥ B0".
* The shield-weight sweep uses multiples {0, 0.5, 1, 1.5, 2} × Rajasthan's per-regime default (D4), not TN's {0, 5, 10, 15, 20}.

**DQN Generation-2 study, regime 1 first (18 runs × 300k steps, validation years 2021–2023, `results/gen2/`)**

| Shield weight | seeds meeting "delivery ≥ 95 % of B0" | mean % of B0 delivery | mean shield cut |
|---|---|---|---|
| 0 | 3 / 3 | 100.0 | 0 % (collapsed to B0) |
| 0.85 | 3 / 3 | 100.0 | 0 % (collapsed to B0) |
| **1.7 (default)** | **2 / 3** | 96.8 | 32.8 % |
| 2.55 | 1 / 3 | 84.0 | 62.5 % |
| 3.4 | 0 / 3 | 74.4 | 97.5 % |

* **First Rajasthan policy that is not B0.** At the default weight, seed 42 delivers 98.2 % of B0's energy (97.3 % in the night
  window) with **57 % fewer shield hours** (447 vs 1043 h) and zero hard-limit hours; it is the selected run under the plan-18.1
  rule (`results/gen2/gen2_selected_dqn.json`, SHA-256 inside). Frontier: more shield penalty buys fewer shield hours at a steep
  delivery cost; the 95 % delivery constraint is met at weight ≤ 1.7 for 2 of 3 seeds.
* **Seed spread is large**: at w = 1.7 seed 123 collapses to B0 (100 % / 0 % cut), seed 0 gives 92 % delivery / 41 % cut, seed 42
  98 % / 57 %. The selection is the best of three on validation years — the claim is "an anticipatory policy exists and can be
  found", not "DQN reliably finds it".
* **This does not contradict G2.** G2 asked for headroom on DELIVERY and found none; the DQN agent trades ~2 % of delivery
  (−16 kWh in the night window, just outside O2's 14.4 kWh band) for the shield-time cut. What changed from the failed PPO run
  is the information (forecast), the training distribution (randomisation) and the learner (off-policy ε-greedy), not the
  action space. Which of the three matters has NOT been separated (no ablation without forecast / without randomisation);
  the plan's item 16 (randomisation overlay) is therefore not produced.
* **Sub-hourly control (feedback item 1) did not help here**: 15-minute control, same weight and seed: 98.8 % of B0 delivery but
  only a 0.2 % shield cut (it returned to near-B0 behaviour). One run, one seed: not evidence either way.
* **Regimes 0 and 2** (one seed, default weight): regime 2 collapsed to B0; regime 0 delivers 93.4 % of B0 with a 58 % shield
  cut, so it does NOT meet the 95 % rule. "Climate-adaptive" coverage is therefore not established beyond regime 1.
  **Follow-up, 3 seeds × shield weights 0.25×–4× the default (DQN, validation years):** regime 2 stays at exactly B0 in 14 of 15
  runs; only the two heaviest-weight seeds (w 2.55 and 3.4, seed 42) leave B0, at 84 % of B0 delivery (shield cut 74 % / 94 %), which fails the
  95 % rule. Regime 0 is B0 in most runs; the runs that leave B0 do so by giving up delivery (w 3.9 seed 42: 93.4 %; w 5.85 and 7.8:
  70–87 %). No weight in either regime satisfies the plan-18.1 rule, so the regime-0 and regime-2 "selected" entries in
  `gen2_selected_dqn.json` are B0-equivalent fall-backs (100 % delivery, 0 % shield cut), not learned policies. This is recorded as a
  limit: DQN works in regime 1 only.
* Run times: 8.6–10.9 min per DQN run, 6 in parallel, 34.5 min wall for 18 runs.

**Deployment metrics (phase 9, regime 1, selected DQN, validation years, `results/deployment_metrics_r1.json`)**
Thermal efficiency (E45 / incident solar on 1.5 m² = 2,955 kWh/yr): DQN 30.8 %, B0 31.3 %, twin 26.4 %, B1 26.2 %. Demand
coverage: 37.4 / 38.1 / 32.1 / 31.8 %. Extra heat versus B1: DQN +136 kWh/yr (+17.6 %), B0 +153 (+19.8 %) — B1 is the
weakest controller here, unlike TN. Inference: 0.21 ms/decision, 5,507 acting-network parameters, 22 kB fp32 weights
(single thread, this CPU; against a 1-hour control period). Sensor faults (7 scenarios, observation only): zero hard-limit
violations everywhere (the shield acts on the true state); DQN loses ≤ 0.2 % delivery in the worst scenario, B1 loses 97.5 %
with a dead pyranometer (it never charges when it reads no sun).
"Backup-heater electricity avoided" (TN §18.2): no project file gives a heater efficiency, so the figure is reported as an
**assumption**, not a result. At an ASSUMED resistive efficiency of 1.0 the selected DQN avoids +136.7 kWh/yr of backup energy vs B1
(+17.6 %); B0 avoids +153.2 kWh/yr but with 1043 shield-hours against DQN's 447. The JSON also carries the sensitivity at efficiencies
0.85 / 1.0 / 3.0 (`backup_energy_avoided_vs_B1_kWh_yr_at_efficiency`). Replace the assumed value before quoting it in the paper.

**SAC Generation-2 (regime 1, w = 1.7, 3 seeds, 300k steps, 98 min wall at 3 parallel; `results/gen2/gen2_summary_sac.csv`):**
delivery 94.5 / 94.7 / 95.0 % of B0 (window 91.7–92.2 %), shield hours 136–140 vs 1043 (cut 87 %), 0 violations, but ~3,100–3,700
on/off switches per year. No run meets the plan-18.1 rule (delivery >= 95 % of B0, annual AND window): SAC lands just under the
line and is very consistent across seeds (unlike DQN), trading ~5 % delivery for an 87 % shield cut. DQN seed 42 remains the
selected run. No SAC weight sweep was run (a lower weight would likely move SAC above 95 %); not claimed.

**LOCKED TEST SPENT (2026-11-12, user-approved), regime 1, selected run `dqn_r1_w1.7_s42_V3_60min_dr`, test years 2006-2010
(fresh weather, 3 initial PCM states x 5 years, `results/phase6_final_test_dqn.csv`, `phase6_summary_dqn.json`, lock `o3_test.lock`).**
Mean delivered to 45 C: DQN 941.0 kWh, B0 960.6, B1 797.9, water-only twin 805.5. DQN = 98.0 % of B0 annual, 96.9 % in the night
window (plan-18.1 rule PASSED on unseen years); shield hours 464 vs B0 1180 (-61 %); 0 hard-limit violations; energy residual
2.6e-12 %. Window gain vs twin 128.2 kWh (B0 147.8): the agent is NOT better than B0 on delivery; it matches ~98 % of it with
far less shield action. Validation numbers (98.2 % / 57 % cut) held up out of sample. Only regime 1 was tested; the test is
now spent and nothing may be tuned on it. Cosmetic: the run banner prints "Algorithm: ppo | Regimes: [0]" (argparse defaults
not overridden by --run); the evaluated rows are regime 1 / DQN as the CSV shows.

**Ingredient ablation (2026-11-12; DQN, regime 1, w = 1.7, 300k steps, validation years, 3 seeds per variant;
`scripts/gen2_ablation.py`, `results/gen2/gen2_ablation_dqn_r1.csv`).** Seeds 42 / 0 / 123:

| Variant | delivery % of B0 | shield cut vs B0 | meets the 95 % rule |
|---|---|---|---|
| full (V3 forecast + training randomisation) | 98.2 / 92.3 / 100.0 | 57 / 41 / 0 % | 1 of 3 (seed 123 is B0) |
| no forecast (V2) | 100.0 / 100.0 / 100.0 | 0 / 0 / 0 % | 3 of 3, but all three ARE B0 |
| no randomisation (V3) | 81.1 / 88.1 / 97.9 | 99 / 57 / 64 % | 1 of 3 (seed 123: 97.9 %, window 96.8 %, cut 64 %) |

* **The forecast is what lets DQN leave B0**: without it all three seeds sit exactly on B0. This holds across seeds.
* **Training-time randomisation is NOT shown to matter.** With one seed (42) it looked as if randomisation moved the policy from a
  near-always-bypass one to the 98 % one; the other seeds contradict that: the no-randomisation seed 123 meets the 95 % rule with
  a larger shield cut (64 %) than the selected randomised run (57 %). Both variants find a good policy in about 1 seed of 3.
  Randomisation is kept as the default because the plan asks for it and robustness did not suffer, not because it was shown to help.
* The selected run used for the locked test was chosen by the plan-18.1 rule among the main-study runs before this ablation; the
  ablation runs are not eligible for selection and the locked test is not repeated.
* Robustness (phase-5, 13 scenarios x 2 validation years, seed-42 runs, `gen2_ablation_robustness_dqn_r1.csv`): zero violations for
  every variant in every scenario. The selected (full) run's delivery moves by 128 kWh across scenarios, almost all of it the
  demand scenarios (x0.7 / x1.3); 20 % GHI noise changes it by < 1 kWh. Plot 16 (randomisation overlay) is drawn from these files.
* Code for this round: phase 5 and the Streamlit replay accept Gen-2 runs (`phase5_robustness.py --run`, `o3/demo/replay.py`),
  `scripts/gen2_ablation.py`, `make_gen2_plots.py` plots 13b / 14b / 16, and `gen2_study.py summarise` counts only V3 + randomised
  runs so ablation runs cannot change the selection.

**Live mode for Gen-2 agents (2026-11-12).** `LiveDemoRunner(..., obs_version="V3")`, `phase7_demo.py --run results/gen2/<tag>` and the
Streamlit live tab now accept Gen-2 agents. A V3 agent's forecast fields come from the DEMO's weather source (Open-Meteo's own next
three hours; the next three recorded hours in replay; the synthetic day in stub mode; persistence of the current GHI if a fetch
fails), never from the stored year: `PCMEnv.set_weather_override(ghi, tamb, hour, forecast_ghi)`. Replay of a stored year through
the demo reproduces the evaluation numbers for a V3 agent (`tests/test_demo.py`). Live mode is a nowcast + provider forecast, so a
live session is not exactly the perfect-forecast setting the agent was trained and validated in.

**Still not done:** no SAC Pareto sweep; the locked 2006–2010
test is spent (see above) (it should be run on the selected DQN run: `phase6_locked_eval.py --run results/gen2/<tag>`); no run trained
without forecast / randomisation for an ablation; Streamlit portal items of §19.4 are already covered by `app/streamlit_app.py`
(replay/live, banner) and do not yet list Gen-2 agents.

---

## Still open (not fixed here)

1. B1 tuning: done. Shield weight: 0 and 1.7 trained (both collapse to B0); the plan-Q2 (w2, w4) grid is implemented but not run.
2. Headroom gate G2: run, FAILED on delivery in all regimes (no on/off RULE beats B0); a Gen-2 DQN trades ~2 % delivery for a 57 % shield cut in regime 1 (see above). F_R(flow) still not enabled.
3. (done 2026-11-11) 2006–2010 fresh test weather downloaded and accepted (D6).
4. Live-weather provider and location mapping (plan Q5/Q6).
5. O2's own caveats stand: 30 % PCM cap and 40 mm radius bind, `cfd_closure_v1` unvalidated, F_R(flow) not enabled,
   simulation only (no hardware).
