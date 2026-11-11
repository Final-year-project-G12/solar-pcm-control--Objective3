# Objective 3 — Methodology and Findings (Rajasthan)

The scientific narrative: the question made measurable, the method, what has been found so far, and what has **not** yet been
established. Plan reference: `.claude/Objective_3_Implementation_Plan_FINAL.docx`. Everything below marked **(measured)** was
run on 2026-11-11 and can be reproduced with the commands in `HOW_TO_RUN.md`; everything marked **(not run)** needs the full
training jobs.

## 1. The research question, made measurable

Plan §1: with O2's shield left intact, can a policy that chooses *loop enabled / loop off* each hour beat (i) always-on
operation (B0 = O2's released passive PCM design), (ii) a tuned reactive rule (B1) and (iii) the water-only twin, **on years
it has never trained on**? Success is O2's own metrics: zero hard-limit hours (water 75 C, PCM 65 C), energy residual below 0.1 %,
delivered energy to 45 C in the 18:00–09:00 window versus the twin (`g_window`), serve fraction, shield hours, bypass switches —
per year, per reset scenario, mean ± spread over seeds.

**Why a controller at all** (plan §0, decided Q4): O2's hand-off says the case is hour-by-hour operation, not "the best PCM
changes with the weather". The O1 ranking does not change with season within a cluster (0 seasonal re-rankings in 11 cells);
the annual-vs-seasonal #1 mismatch in clusters 0 and 2 (O1's "8 of 11") is unexplained and is not evidence of seasonal change.
The PCM tank spends about 400–1,200 h/yr throttled by its own shield, and (O2's held-out table) does not extend the time at 45 C
in the evening in 12 of 15 chosen rows. Whether timing the loop can recover some of that is the open question.

## 2. The method

* **Plant:** O2's frozen multi-node (N = 10) stratified tank kernel, Δt = 150 s, wrapped so that one call = one control hour
  (`o3/environment/o2_adapter.py`). Same physics, same shield, same demand curve (300 L/day, local time) as O2. The stepped
  environment equals O2's `run_case` to 1e-9 (tests T1–T3).
* **Regimes:** three O2 regimes, each its own design — regime 0 savE OM55 (Udaipur, flow 0.0155 kg/s), regime 1 PureTemp 60
  (Nagaur, 0.0493), regime 2 PureTemp 58 (Jaipur, 0.0236); all 52 capsules, 50 L tank, 1.5 m² collector. A policy is trained per regime.
* **Action:** loop enabled / IDLE / BYPASS (IDLE ≡ BYPASS physically; D5). Flow is a constant of the design: the collector model
  is flow-independent, so letting an agent set flow would reward a model artefact (plan §0).
* **Observation:** the contract's 13 fields, fixed scaling (D3). T_pcm, f_melt and the two energy fields are simulator
  internals; a real tank would need an estimator (the demo labels them "simulator state").
* **Reward R2:** `1·E_del45 − 2·E_unmet − 50·hard_violations − 0.1·switch − w_shield·shield_activations`, with `w_shield`
  per regime (3.9 / 1.7 / 1.7) re-derived from Rajasthan training-year baselines (D4). Provisional until the plan's Q2 sweep.
* **Split:** train 2016–2020, validate 2021–2023 (reward weights, B1 thresholds, checkpoint choice), test 2006–2010 (fresh) once behind
  `results/o3_test.lock` (D6). 2024–2025 never used.
* **Procedure** (plan §8.3): verify contract → tests T1–T8 → baselines on train + validation → headroom gate G2 → train ≥ 3 seeds
  per weight set → choose by validation (lexicographic) → freeze and hash → one locked test run → anything later is exploratory.

## 3. What has been established (measured)

1. **The environment is the O2 simulator.** Stepped year = O2 year: max |ΔT_top| = 0, |ΔT_pcm| ≤ 2e-14, window energy and shield hours
   equal to 1e-9 relative; a split run, a random bypass schedule (= GHI zeroed in those hours), determinism, the shield
   overriding CHARGE with a 80 C tank, and the < 0.1 % energy balance all pass (110 tests).
2. **O2's design beats its water-only twin in the night window.** Measured here for regime 1 on all six train/validation years
   run: +134…+152 kWh/yr (regimes 0 and 2 were run for B0 only on 2016 and for calibration; O2's own held-out table has a positive
   `g_window` for all 15 chosen design-years). But the shield costs it: ~500 h/yr of water-side and ~550 h/yr of PCM-side shield time in Nagaur, peaks 72.2 C / 62.3 C,
   zero hours over the hard limits.
3. **TN's rules do not transfer.** B1 (−12…−21 kWh) and B2 (−36…−48 kWh) are *below* the twin in the window, because switching the
   loop off in the evening and below 200 W/m² discards heat that a hot-dry climate still delivers. They do cut shield time by ~90 %.
   Whether a *tuned* rule or a learned policy can keep B0's delivery and lose some of the shield time is the headroom question (G2): **open**.
4. **The TN shield weight is a trap here.** With weight 10 the reward ranks B0 last by a factor of two although B0 is best on the
   plan's metric (D4). The break-even weight is 1.4–3.3 per regime, not 8.4. This is the same lesson TN learned in reverse
   (R1 → R2): a reward is only as good as its scale against the kernel's counters.
5. **Parallelism does not change a number.** Baselines, dataset (byte-identical) and robustness give identical results serially
   and in a pool; `SubprocVecEnv` and `DummyVecEnv` train identical weights from the same seed.
6. **Compute.** Env ≈ 3 ms/hour-step is the O2 kernel (≈ 55 % of the step; the numba kernel cannot be vectorised without editing O2);
   Python overhead around it was cut from ≈ 45 % to ≈ 10–15 % by reading the end-of-hour state from the kernel output.

## 3b. Milestone results (M3, G2, M4, M5) — added 2026-11-11

Full tables are in `CORRECTIONS.md` ("Results of the plan's milestones"). In one paragraph: B0 beats the water-only twin by
+132 / +138 / +144 kWh/yr in the night window (regimes 0 / 1 / 2, 2016–2023). The headroom gate G2 tried 92 rule policies per
regime on validation years and none beats B0 on window delivery; a 3-seed, 300k-step PPO in regime 1 converged to exactly the
B0 policy under both a delivery-only reward and the default shield-penalised reward; robustness (13 scenarios × 3 years) shows
zero hard-limit hours. Conclusion: with loop on/off as the only action, hourly control has no headroom on delivered energy here.
Items below marked "not run" that are now done: G2, B1 tuning, a PPO multi-seed run, M5 for the frozen policy.

## 4. What has NOT been established (not run)

* Any trained agent's performance. Smoke training (3,000 steps) only proves the plumbing; the smoke PPO collapsed to a near-
  random 'idle' policy (window gain −466 kWh on the throw-away dry run) — exactly what 3,000 steps should do.
* Headroom gate G2, the shield-weight sweep, B1 tuning (plan §15.6, Q2).
* Robustness of a trained policy (phase 5 ran for a smoke policy and B1; no violations in 13 scenarios).
* The locked test: not run, lock not created.
* Stochastic-demand training (only an evaluation scenario), forecast inputs, flow control (needs F_R(flow)), hardware.

## 5. Threats to validity (carry into the paper)

* Train and validation years were used by O2 to *choose the design*, so gains versus the passive design are optimistic (plan §8.2).
* The test years are a fresh set (D6), but at the same three sites only — a temporal hold-out, not a spatial one.
* O2 is a simulation: CFD runs are model checks (`cfd_closure_v1` unvalidated), the PCM resistance is the lumped r/(5k) closure,
  F_R is flow-independent, 50 L / 1.5 m² sits below the IS 12976 ratio band, regime 0's O1 pool has n = 4.
* Stochastic-demand ranges are assumptions (Q3). Live mode is software-in-the-loop only: "Not field validation".
* The reward weights other than `w_shield` are TN values and unswept.

## 6. What this does and does not establish

When the runs in `HOW_TO_RUN.md` are done it can support: "a contract-constrained policy around O2's unchanged simulator and
shield, compared with always-on operation, a tuned rule and the water-only twin on years it never trained on" (plan §12).
It cannot support: hardware validation, other locations, or a claim that the O1 ranking changes with the weather.
