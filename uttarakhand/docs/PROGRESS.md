# Objective 3 - Uttarakhand: progress log

Plan: "Objective 3 - Implementation plan (final, consolidated)", section 15.10 milestones, applied to Uttarakhand.
Results and interpretation: `FINDINGS.md`. Commands: `HOW_TO_RUN.md`.

## Milestones - all complete (2026-11-11)

| # | Exit gate (plan wording) | Status | Evidence |
|---|---|---|---|
| M0 | contract hash passes; Q11 answered; test point-year files accepted or rejected with reasons, SHA-256 recorded | done | `contracts/*.sha256`; `data/weather_test_2006_2010/weather_inventory_test.csv`: 20/20 accepted; Q11 below |
| M1 | speed recorded; training budget fixed from it | done | one 8760-h episode about 1.2 s (env) / 2 s (evaluation); 600k steps/run in `configs/o3_config.yaml` |
| M2 | tests T1-T8 all pass | done | `python -m pytest -q`: **21 passed** (T1-T8, O2 snapshot, O2 night metric, demo, lock guards) |
| M3 | metric table per year; **B1 passes the contract acceptance list** | done | `results/m3_*_r{0..3}.csv`; `results/m3_acceptance.json`: **all 6 contract items pass** for B0, B1 and every frozen PPO seed, all 4 regimes |
| G2 | headroom gate, reported honestly | done - pass on the tie rule only, no delivery headroom | `results/g2_*_r{0,1,2}.*` |
| M4 | frozen policy hash | done | 36/36 runs; `results/m4_runs.csv`, `m4_weight_sets.csv`, `results/frozen/r{0,1,2}/frozen.json` |
| M5 | results per scenario | done | `results/m5_robustness_r{0,1,2}.csv`; O2-definition night metrics `results/m5b_night_metrics.csv` |
| M6 | lock file + results file with hashes | done | `results/o3_test.lock` (run `o3test_e6ea0abf636a`, fingerprint `fd91543a...826b`, 2026-11-11T17:12:03Z); `results/m6_test_results.csv`, `m6_test_summary.json` |
| M7 | banner present; **replay reproduces M6 numbers** | done | banner on every view; `results/m6b_summary.json`: the demo code path reproduces **all 85 M6 rows exactly**; replay of 2021 r1 also equals M5; one real Open-Meteo fetch OK (66 h, 1.05 s, 0 missing) |

Frozen choices (validation seed means): r0 `w2=3, w4=0.3` (primary seed 0), r1 `w2=0, w4=0.3` (seed 1),
r2 `w2=0, w4=0.3` (seed 0). Regime 3 is water-only by O2's verdict: baselines only.

## Contract acceptance list (`acceptance_test_before_drl_training`) - `scripts/m3_acceptance.py`

Each controller x regime x validation year (51 episodes):

| Contract item | How it is checked | Result |
|---|---|---|
| actions within flow envelope / temperature limits | design flow in 0.010-0.050 kg/s; 0 s over 75 / 65 C; peaks <= limits | pass |
| transitions physically valid | actions in {0,1}; bypassed hour collects exactly 0 Wh; collector heat >= 0 | pass |
| energy conserved (< 0.1 %) | episode residual | pass (max 3e-12 %) |
| shield cannot be bypassed | counterfactual: every ENABLE hour in which the water shield engaged is re-run from the same state with only the water shield removed; shielded hour never collects more or ends hotter. **8,141 such hours, 8,129 changed by the shield** (not vacuous); plus T6 | pass |
| same seed, same trajectory | episode run twice, every hourly column identical | pass |
| matches O2 for identical actions | each controller's own action sequence replayed through O2 `run_case` (GHI = 0 in bypassed hours); hourly T_top, T_pcm, f_melt, E_del, collector heat to rtol 1e-9 | pass |

## Plan questions

* **Q1, Q2, Q4, Q10** - decided in the plan; implemented as decided (Q10 applies to Rajasthan; Uttarakhand primary
  regime 1 by population).
* **Q3** stochastic demand - still **open**: no Indian measured draw-off data was found. Ranges are labelled assumptions
  in `configs/o3_config.yaml`; M5 shows the ranking B0 > B1 > PPO is unchanged under them (36/36 cells).
* **Q5** live provider - **closed**: Open-Meteo forecast API (the same provider as the Tamil Nadu demo). Terms checked
  2026-11-11 (open-meteo.com/en/terms, /en/licence): free tier allows non-commercial use and public research, 10,000
  calls/day (the demo makes about one per hour); data CC BY 4.0, credit and licence link shown next to the data in the
  live tab, with the changes made (hour-end -> hour-start, clip, persistence fill).
* **Q6** location-to-regime mapping - **closed**: the user picks one of the four medoids (no automatic mapping).
* **Q7** compute - **closed**: CPU only; about 13 min per PPO run, 2 runs in parallel in 16 GB RAM.
* **Q8** O1 K=5 vs O2 k=4 - O3 follows O2's four regimes (the design O3 must not change). The O1 discrepancy is an O1/O2
  task (plan section 10).
* **Q9** - Rajasthan/plan-document question, not needed for Uttarakhand.
* **Q11** unread O2 files (energy_balance.py, capsule_*.py, heat_transfer.py, contract builder) - **closed, not needed**:
  O3 never re-implements them. `o3/o2_bridge.py` calls O2's own `run_case`, which runs them, and captures the compiled
  kernel's arguments; test T1 (8760 hourly steps = one O2 full-year call, rtol 1e-9) and the acceptance item "matches O2"
  cover them, and `test_o2_snapshot_unchanged` re-hashes all 96 snapshot files, including those modules.

## Log

* **2026-11-11** - M0-M3 built. First G2 attempt crashed with out-of-memory while running beside three PPO trainers
  (each trainer commits about 1.6 GB). The first reward grid (w2 in {0, 0.3, 3}, 54 runs) was cut to off/on per weight
  (36 runs) under the Q2 compute clause before any run finished.
* **2026-11-11** - Added `o3/live.py`, `app/streamlit_app.py`, `tests/test_live.py`, `docs/`. Fixed a demo crash:
  pyarrow 22 needs NumPy 2 -> pinned `pyarrow==17.0.0` in `requirements.txt`.
* **2026-11-11** - G2 re-run (r2 again alone after a second out-of-memory error); M4 grid with 2 parallel jobs, about
  13 min per run; M4 select; M5; **M6 run once at 17:12 UTC**; M7 checked headless (Streamlit AppTest) and with one
  live fetch.
* **2026-11-11 (after M6)** - Closed the remaining gaps:
  `scripts/m3_acceptance.py` (explicit contract acceptance list; a first shield test was vacuous and a second version
  wrongly disabled both shields via P[28]; the final version removes only the water shield via P[29]);
  `o3/night.py` + `tests/test_night.py` (O2's own tau_serve / serve_frac, equal to O2 to 1e-9);
  `scripts/m5b_night_metrics.py` (validation years);
  `roles_o3.attach_existing_lock` + `scripts/m6b_postlock_checks.py` (read-only attach to the existing lock, refuses
  any other policy fingerprint, never creates a lock): demo reproduces M6 exactly, and O2-definition night metrics on the
  test years, **labelled EXPLORATORY** (plan 8.3 step 8). The lock was not deleted or re-created.
  Open-Meteo attribution added to the demo.
* **2026-11-12** - Re-audit against the plan text closed four more gaps:
  (a) **config hash in every result file (15.4)**: 17 files lacked it. M3 outputs carry an EARLIER config hash
  (`444ca3c...`) than M4 onward (`c1e2162...`): the config was edited between M3 and M4 and the old version was never
  committed. Recomputing normalisation and the B1 choice under today's config gave bit-identical results in all 4
  regimes, so the edit did not affect M3. The 4 M3 grid files were stamped with the hash they were made under, G2 / M4 /
  post-lock files with the current one; every script now writes it. `m6_test_summary.json` (locked output) was left
  untouched; its companion `m6_test_results.csv` carries the hash.
  (b) **demo panels (15.9)**: bottom temperature, melt fraction, draw and over-limit time added (8 panels).
  (c) **nowcast in live mode (3.2)**: the policy sees the previous completed hour's GHI / T_amb; the kernel runs the
  actual hour. Replay keeps the training convention, so replay still reproduces M6 exactly.
  (d) **raw provider data stored on every fetch (9)**, not only on "save".
  Plus `scripts/m6c_reset_scenarios_postlock.py` (results per reset scenario on the test years, EXPLORATORY: < 1 kWh/yr
  effect, ranking unchanged).
* **2026-11-12** - Items from the final plan PDF's as-built sections 16-19 (Tamil Nadu additions), built for
  Uttarakhand after the lock, validation years only unless labelled EXPLORATORY: B2 forecast rule (`m9`), deployment
  metrics and sensor faults (`m10`), O4 hand-off package (`m11`), results dashboard with regime map (`make_plots`),
  extension study (`m12`: DQN 9 runs, w2 sweep 9 runs, forecast obs + domain randomisation 3 runs, diagnostic w3
  frontier 9 runs). The O1 point-to-regime file for the map was copied from the O1 tree after checking its SHA-256
  against the O1 manifest (`data/o1/SOURCE.md`). Sub-hourly control (19.1 item 1) was not built: the unchanged O2 kernel
  consumes whole-hour weather rows, so it would need an edit to O2's physics, which the plan forbids. SAC (16.3) was not
  run: with two actions it adds nothing DQN does not test.

## Caveats to keep in view

* **Git:** the policies were frozen before `uttarakhand/` was first committed, so every `frozen.json` records commit
  `7a6875b` (the Tamil Nadu commit) with `uncommitted_changes: true`. `frozen.json` is not rewritten (that would break
  the lock fingerprint chain). Every policy, normalisation, config and contract file is pinned by SHA-256 in the
  fingerprint, and the commit that first contains these exact files is recorded below.
* **Test lock:** `results/o3_test.lock` exists. Any further run on 2006-2010 is exploratory and must be labelled so.
  Deleting the lock must be logged here with the reason.
* **Still not built (plan 11.1 deferrals or impossible under the plan):** B3 MPC (its absence is stated in FINDINGS),
  SAC, recurrent policies, flow control, sub-hourly control, member-point / location hold-out.

## Commit record

* Branch `uttarakhand-objective3`, merged into `main` (merge `be989c9`) and pushed 2026-11-12.
* `bc2452e` - first commit of `uttarakhand/` (git line-ending normalisation altered some CRLF files: contract JSON,
  weather CSVs; do not check out this commit alone for hash checks).
* `caa7e8d` - `.gitattributes` `* -text`: files stored byte-exact. Verified 2026-11-11 with a fresh clone made with
  `core.autocrlf=true`: all 642 files byte-identical to the working folder; contract check, all frozen-policy hashes,
  the lock fingerprint and the 96-file O2 snapshot test pass inside the clone. **`caa7e8d` is the first commit that
  contains exactly the files the frozen policies and the M6 lock refer to.**
