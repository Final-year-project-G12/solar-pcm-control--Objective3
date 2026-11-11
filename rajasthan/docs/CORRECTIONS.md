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
refusal checks. The real one-shot test is still unspent.

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

## Still open (not fixed here)

1. B1 tuning: done. Shield weight: 0 and 1.7 trained (both collapse to B0); the plan-Q2 (w2, w4) grid is implemented but not run.
2. Headroom gate G2: run, FAILED in all regimes (no on/off policy beats B0). Next step per plan: revisit the action space (F_R(flow)) — an O2 change, not made.
3. (done 2026-11-11) 2006–2010 fresh test weather downloaded and accepted (D6).
4. Live-weather provider and location mapping (plan Q5/Q6).
5. O2's own caveats stand: 30 % PCM cap and 40 mm radius bind, `cfd_closure_v1` unvalidated, F_R(flow) not enabled,
   simulation only (no hardware).
