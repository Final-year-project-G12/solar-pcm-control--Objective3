# CORRECTIONS — Objective 2 (Tamil Nadu)

**Applied 2026-11-09.** Companion to
`../../tamilnadu_pipeline/docs/era5_tamilnadu/CORRECTIONS.md` (Objective 1).
Kept per-tree so each objective stays self-contained.

**One code change, no re-runs.** The only non-documentation edit is a corrected
module docstring in `src/optimize/surrogates_advanced.py` (O2-1). No pipeline
stage was re-run and **no Objective 2 result changed**. 89/89 tests pass.

> **What this pass can and cannot do.** Item O2-1 found a genuinely wrong claim
> in the repository. Fixing it does not change which surrogate the search used —
> it changes what the repository *says* it used, bringing the text in line with
> the results file that was authoritative all along. The designs, the held-out
> numbers and the gates are untouched and remain correct.

---

## Verification first

Each of the eight reviewed items was checked against the results files before
any edit. **Three of the eight were already correct in the repository**, and two
of the review's "blocking" claims were false.

| item | verdict after checking | action |
|---|---|---|
| **1** Conflicting surrogate winner | **Real defect — but the opposite of what the review said.** `stacked won all 9 pairs` is **correct and authoritative**. The *other* claim — "XGBoost won 21 of 27 groups, GP came third" — is the wrong one, and it is not in a report but in a **code docstring**. No current results file supports it. | Corrected the docstring. **O2-1** |
| **2** Held-out test status unclear | **False — the test ran.** `results/held_out_test.lock` records `run_id heldout_2da11230bfb5` at `2026-10-30T09:56:22Z` with a selection fingerprint; `g10_heldout_results.csv` holds the per-year rows for 9 designs × 5 years. **Not a blocker.** | Recorded the evidence. **O2-2** |
| **3** Tank/collector specs wrong | **False for the pipeline.** `configs/system_config_shared.yaml` already reads `volume_L: 100.0` and `area_m2: 2.5`. The only 50 L / 1.5 m² design figures sit in the already-superseded `objective3/` handoff; other mentions are deliberate comparisons to the Rajasthan source study. | Confirmed; see **O2-3** |
| **4** "Global optimum" overclaimed | **True.** The 30 % cap is demonstrably binding and the flow optimum may be a model artefact. | New section. **O2-4** |
| **5** CFD presented as validation | **Partly true.** `docs/CFD_VERIFICATION.md` was already honest ("a sanity check, not a replacement", "not a validated, production-grade CFD study", and it reports 0 % liquid fraction openly). But `OVERVIEW.md` called it an "independent physics check". | Reclassified. **O2-5** |
| **6** Pareto front vs selection not separated | **True** — the data existed but was never presented as front-then-funnel. | New section. **O2-6** |
| **7** "Surrogate-assisted search" may outrun the evidence | **Largely unfounded.** Every stage `g2`…`g24` has result files on disk. Two real scope limits found instead. | Evidence table. **O2-7** |
| **8** "Validation" used for three different things | **True.** | Terminology standard. **O2-8** |

---

## O2-1 — The surrogate winner, settled

**Authoritative file: `results/g8_s3_selection.csv`.** It picks `stacked` in
**all 9 (regime, PCM) pairs**. That is the surrogate the NSGA-II search used, so
*"stacked won all 9 pairs" was right.*

The claim that conflicts with reality is the other one. `surrogates_advanced.py`
asserted in its module docstring: *"on the real DOE, XGBoost won 21 of 27 groups
and the GP came third."* Measured from `results/g8_s3_comparison.csv` (360 rows
= 9 pairs × 4 targets × 5 families × 2 validation schemes), winner = lowest RMSE
per group:

| grouping | result |
|---|---|
| per (pair, target), `designs_5fold`, 36 groups | `stacked` 14, `gp_ard` 13, `rsm_quadratic` 5, **`xgboost` 4** |
| per (pair, target), `leave_one_year_out`, 36 groups | `stacked` 14, `gp_ard` 14, `rsm_quadratic` 4, **`xgboost` 4** |
| base families only (stacked removed), 72 groups pooled | **`gp_ard` 42**, `rsm_quadratic` 18, `xgboost` 12 |

So **XGBoost wins 4 of 36, not 21 of 27**, and the **GP is first among the base
families, not third.** The "27 groups" arithmetic (9 pairs × 3 targets) belongs
to an older 3-target DOE and never applied to this 4-target one.

The docstring now carries the measured table and an explicit instruction not to
quote the old figures. **Nothing downstream changes:** the search consumed
`g8_s3_selection.csv`, not the docstring.

**This error had propagated.** `documentation/OBJECTIVES_AND_ALGORITHMS.md`
repeated the 21-of-27 figure from the docstring; it has been corrected too.

## O2-2 — Held-out test: run, locked, recorded

Not an open question.

| evidence | value |
|---|---|
| `results/held_out_test.lock` | `run_id heldout_2da11230bfb5`, `utc 2026-10-30T09:56:22.980791+00:00`, selection fingerprint `2da11230…1f6e298` |
| `results/g10_heldout_results.csv` | 9 designs × 5 years (2011–2015), per-year paired rows |
| coverage | the chosen design **plus the best design of every other shortlisted PCM** — 9, not 3, so it cannot be a cherry-pick |
| outcome | 9/9 beat the water-only twin; PCM peak 63.4–64.0 °C (limit 65); water peak 72.2–72.9 °C (limit 75) |

The "not yet performed" wording the review cites is from a progress document
written *before* 2026-10-30. It was true when written and is now stale.

## O2-3 — Tank and collector specs: already correct

`configs/system_config_shared.yaml` → `tank.volume_L: 100.0`,
`collector.area_m2: 2.5`. Every result in `results/` was produced against this.

The 50 L / 1.5 m² figures appear in three distinct ways, and only one is wrong:

1. **Wrong, already flagged:** `objective3/obj3_environment_contract_tamilnadu.json`
   and `objective3/OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md` (both 2026-10-27)
   describe the superseded 50 L design. These must be regenerated before
   Objective 3 starts — see `../../documentation/README.md` → "Known gaps".
2. **Legitimate:** `system_config_shared.yaml`'s own comment
   *"TAMIL NADU: project instruction (hers is 50 L)"* — a deliberate note that
   the Rajasthan source study used 50 L.
3. **Legitimate:** historical entries in `RESULTS.md` and
   `RAJASTHAN_METHOD_PORT_PROGRESS.md` below their superseded banners, which
   record the 2026-10-27 resize as history.

## O2-4 — Not a global optimum

Added to `OVERVIEW.md`. The selected designs are **the best found within the
declared search bounds**, and two bounds matter:

- **The 30 % PCM-fraction cap is binding.** The `g17` widened-bound diagnostic
  shows 40 % raises window gain by more than the repeatability band in **every**
  regime, with both peaks still inside 65 °C and 75 °C. The cap is
  administrative — the packing engine accepts 28–48 % at R 40 mm, and the only
  literature in the files read (Chen 2025) covers 10–20 %. A 40 % run is
  planned, not done.
- **The flow optimum may be a model artefact**, because the collector's F_R is
  flow-independent in this model. `g17` does establish that the *upper* flow
  bound is not limiting (0.080 kg/s moves `g_window` by less than the band).

Required wording: **"best design within the declared 2–30 % PCM-fraction and
0.010–0.050 kg/s search bounds"**, never "optimal design".

**Consequence for Objective 3:** it inherits a design that is *good and
verified*, not one proven optimal. It must not treat the inherited flow value as
a physical optimum to defend.

## O2-5 — CFD reclassified

`OVERVIEW.md`'s two-simulator table called Fluent an "independent physics
check"; it now reads **"model consistency cross-check (NOT experimental
validation)"**.

The substance was already documented honestly in `docs/CFD_VERIFICATION.md` —
only the framing needed fixing. The limits, restated: one capsule, six hours;
three v3 cases returned **0 % liquid fraction**, which that document records as
the honest result rather than suppressing; melt fractions elsewhere are
conservative because first-order upwind is the only scheme under which all four
PCMs converge; the mesh is a symmetry half-slice, excluding buoyancy-driven melt
convection.

Required wording: **"CFD cross-check of selected modelling assumptions"**, never
"CFD-validated design".

## O2-6 — Pareto front and selected design, reported separately

Added to `OVERVIEW.md`. NSGA-II returns a *set*; the front and the pick are now
separate tables.

Front quality (`g9_grid_check.csv`, rank-1 PCM per regime): final fronts hold
65 / 50 / 53 designs against the 720-point dense grid's 11 / 10 / 11, with
hypervolume ratios 1.103 / 1.095 / 1.016 — the search front dominates the grid
everywhere — and ε-indicator values 0.62 / 3.06 / 11.97 all inside the 28.0
band.

Funnel (`g10_selection_r{0,1,2}.json`): 90 candidates per regime → S0 (within
band on `g_window`) 14 / 10 / 12 → S1 5 / 3 / 5 → S2 3 / 3 / 2 → chosen.

**The tie-breaks were declared before results were seen** (plan WP6, decisions
D9/D10). Regime 0 shows the rule doing real work: RT57HC scores *higher* on
window energy than the chosen PureTemp 58 (297.1 vs 283.4 kWh/yr), but the gap
is inside the band, so S0 treats them as tied and overheating decides — 856 vs
1059 shield-hours, favouring PureTemp 58. The higher-energy design was **not**
chosen, and the documented rule is why.

## O2-7 — Stage evidence audit

Every stage has results on disk, so "surrogate-assisted constrained
multi-objective search" is a supportable description of what ran:

| stage | evidence |
|---|---|
| `g2` timestep convergence | 2 CSVs (300/150/75 and 150/75/37.5 ladders) |
| `g3` metrics, `g4` multinode | `g3_metrics_report.csv`, night-daily CSVs, `g4_multinode_report.csv` |
| `g6` O1 PCM ingest | `results/phase4_verification_o1_pcms.txt` — Go/No-Go: **GO** |
| `g7` sensitivity, packing, baseline | 6 CSVs |
| `g8_s1` Taguchi L9 + ANOVA | 18 CSVs (9 pairs × design + ANOVA) |
| `g8_s2` Sobol DOE | 9 CSVs (one per pair, 128 points each) |
| `g8_s3` surrogate bake-off | `g8_s3_comparison.csv` (360 rows), `g8_s3_selection.csv` |
| `g9_s4/s5` NSGA-II + active learning | 10 seed-front CSVs |
| `g9_s6` dense grid | 3 CSVs |
| `g9_band` repeatability | 3 CSVs + `wp6_band.csv` |
| `g10` selection + held-out | 6 selection files, `g10_heldout_results.csv`, lock |
| `g11`, `g13`–`g18`, `g20`–`g24` | 3 CSVs each (per regime), `g18` 1 |

**Two real scope limits, stated rather than implied:**

1. **The dense grid covers rank-1 PCM pairs only** — 3 of 9 pairs, by design
   (`g9_s6_grid_r{0,1,2}_p1.csv`). Ranks 2–3 have no dense-grid cross-check, so
   the "search missed no region" claim is established for the rank-1 pairs only.
2. **`g18`'s with/without comparison is incomplete.** Its CSV holds only the
   `base` state; `novert` and `vert` never ran. The generated results document
   now prints an explicit *"This comparison is INCOMPLETE"* note instead of the
   previously hardcoded reading, which quoted numbers absent from the data.

## O2-8 — Three kinds of checking, three names

Added to `OVERVIEW.md`:

| what it is | call it | status |
|---|---|---|
| energy-balance and unit tests in code (89 tests, Gate-1 residual) | **simulator verification** | done |
| ANSYS Fluent against the fast model's assumptions | **model consistency cross-check** | done, with O2-5 caveats |
| physical measurement on hardware | **experimental validation** | **NOT DONE** — Objective 4 |

---

## O2-9 — The Objective 3 contract, built (2026-11-09, second pass)

**This was the single blocker on starting Objective 3, and it is now closed.**

The problem: no Objective 3 contract existed for the current results. The only
Tamil Nadu contract (2026-10-27) described a superseded run, and the newer
generator `build_obj3_contract_v2.py` is **hardcoded for Rajasthan** — it reads
`obj3_environment_contract_rajasthan.json`, which is absent from this tree, and
writes `..._rajasthan_v2.json`.

**What was built:**

| artefact | what |
|---|---|
| `src/handoff/build_obj3_contract_tamilnadu.py` | new generator. Reuses the unchanged generic blocks (action space, reset scenarios, acceptance tests, state schema) from the 2026-10-27 contract; rebuilds everything design- and result-specific from live files |
| `scripts/g25_obj3_contract.py` | pipeline-convention wrapper, with an input-presence check that fails with a readable message rather than a traceback |
| `tn_run_all.py` stage 31 | `g25_obj3_contract`, registered after `g10_test` since it reads the held-out outputs |
| `results/obj3_environment_contract_tamilnadu_v3.json` | the contract, 35.8 kB, also copied to `objective3/` |

**Everything in it is read from a file** — `g10_selection_r{0,1,2}.json`,
`g10_heldout_results.csv`, `held_out_test.lock`, `wp6_band.csv`,
`system_config_shared.yaml`, `states/tamilnadu.yaml` — and each is SHA-256
recorded in the contract's `provenance` block.

**The corrections from this pass are carried into the contract itself**, so
Objective 3 cannot inherit a claim this audit has already retired:

- `design_space_caveats` — `NOT_A_GLOBAL_OPTIMUM: true`, the binding 30 % cap,
  and an explicit instruction not to treat the nominal flow as a physical
  optimum (O2-4).
- `temperature_specification` — the master table, and the finding that
  67 °C vs 65 °C is **not** a safety problem, plus the open 60 °C/45 °C
  service mismatch.
- `verification_status` — simulator verification / model consistency
  cross-check / experimental validation kept separate, with Objective 1's weak
  non-significant physics agreement stated plainly (O2-5, O2-8).
- `held_out_test` — status, lock id, timestamp and fingerprint, closing the
  "was it run?" question inside the machine-readable artefact.

**A chain-integrity assertion is built in**: the generator asserts every
selected PCM is in that regime's Objective 1 shortlist, so a substituted
material cannot reach Objective 3 silently.

**Unit confusion removed.** The docs quote 81.6 mm; the selection files store
`R_in_m = 0.04`, i.e. 80.0 mm inner. These differ by the 0.8 mm aluminium wall
on each side. The contract now emits **both** `capsule_inner_diameter_mm` and
`capsule_outer_diameter_mm`.

**Retired:** the 2026-10-27 contract and `OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md`
moved to `archive/superseded_obj3_contract_2026-10-27/`. The handoff document
was rewritten against real output; its old headline claims — "PCM loses to a
plain tank in every regime (−12.5 % to −14.8 %)" and "P(meets demand) = 0–4 %" —
are **false for the current designs**, which beat the water-only twin 9 of 9.

---

## O2-10 — The generated results document (applied earlier the same day)

**Recorded here 2026-11-09 after an audit found it missing from this file.** It
was applied earlier in the day and written up only in
`../../documentation/README.md`, which is not where an Objective 2 correction
belongs.

`scripts/g12_results_doc.py` **crashed every time it ran**:
`IndexError: single positional indexer is out-of-bounds` at line 270, on
`w1[(w1.pair == tag) & (w1.state == st)].iloc[0]` — an empty selection indexed
without a guard. Cause: `g18_issue1_with_without.csv` holds only the `base`
state, but the generator assumed `base`, `novert` and `vert` all existed.

Three things were wrong, all fixed:

1. **Its output had never existed.** It writes `docs/04_RESULTS.md` — *not*
   `docs/RESULTS.md`, which is a separate hand-maintained file. Because the
   script had never once succeeded, the auto-generated results document — the
   one in which no number is typed by hand — was simply **absent from the
   repository**. It now exists and regenerates cleanly.
2. **The narrative asserted numbers its own table could not show.** The
   hardcoded reading quoted `novert`/`vert` figures ("23.5 kWh against 13.8",
   "3.1", "r1 5.0 without, 1.8 with") that are not in the results at all. It now
   prints an explicit *"This comparison is INCOMPLETE"* note naming the missing
   states, and the full reading only when all three are present.
3. **The document was titled "Objective 2, Rajasthan"** inside the Tamil Nadu
   tree — a leftover from the method port. Now "Tamil Nadu".

## O2-11 — The stale-path-reference count, and a correction to my own figure

**Also applied earlier the same day and missing from this file until now.**

`OVERVIEW.md` said: *"`g12_verify_docs` reports ~68 stale path references,
almost all inside the historical progress log, which legitimately describes
paths as they were."*

**The count was right; my first correction of it was wrong.** I initially
reported the real figure as **236** and called the claim understated. That 236
came from the stale snapshot `results/_log_g12_verify_docs_all.log`
(2026-10-31 16:46) — and the docs were repaired three minutes later at 16:49. A
live run on 2026-11-09 gives **68**. The original "~68" was correct all along,
and the 236 figure has been reverted everywhere it appeared.

**The second half of the original claim is still wrong**, and that part stands
corrected. Only 15 of the 68 are in the progress log:

| doc | missing refs |
|---|---|
| `docs/DATA_REFERENCE.md` | 31 |
| `docs/RAJASTHAN_METHOD_PORT_PROGRESS.md` | 15 |
| `docs/HOW_TO_RUN.md` | 15 |
| `README.md` | 3 |
| `docs/CFD_VERIFICATION.md` | 3 |
| `docs/RESULTS.md` | 1 |

So "almost all in the progress log" covers 22 %, not almost all. `OVERVIEW.md`
now carries 68 with this breakdown. **The 68 references themselves are not
repaired** — only the claim about them is honest.

## O2-12 — Optimisation method references added

**This was review item 2.8 and had not been done.** An audit on 2026-11-09
found `docs/REFERENCES.md` contained **no citation for NSGA-II at all**, though
it is the core search algorithm of this objective. Six method sources added:

| key | covers |
|---|---|
| `Deb2002` | NSGA-II itself — non-dominated sorting, crowding distance, elitism, constrained domination, SBX and polynomial mutation (`src/optimize/nsga2.py`) |
| `Bhattacharjee2016` | multi-surrogate competition rather than one chosen model (`g8_s3_compare.py`) |
| `Bradford2018` | GP surrogates for expensive black-box MOO (`gp_ard`, `g20_gp_bo_comparison.py`) |
| `Kumar2024` | the Taguchi → surrogate → NSGA-II *combination* |
| `Gadhave2020` | MCDM-based PCM selection for domestic water heating (the O1 application) |
| `Sobol1967` / `Owen1998` | the scrambled-Sobol DOE in `g8_s2_doe.py` |

Each entry states what it does **and does not** support — they are method
precedents, not evidence that this project's bounds, DOE size or results are
correct.

---

## Still open in Objective 2 (not fixed here)

1. **The 40 % PCM-fraction run** that `g17` indicates would do better (O2-4).
2. **`g18` `novert`/`vert` runs** (O2-7).
4. **Dense-grid cross-check for rank-2/3 PCM pairs** (O2-7).
5. **68 stale path references** reported by `g12_verify_docs` — 31 in
   `DATA_REFERENCE.md`, 15 in `HOW_TO_RUN.md` (which still documents a
   `pipeline.py` that no longer exists), 15 in the progress log, 3 + 3 + 1
   elsewhere. The count claim in `OVERVIEW.md` is now accurate; the references
   themselves are unrepaired.
6. **Physics limits unchanged** — flow-independent collector F_R, symmetry-slice
   CFD mesh, wind and humidity reaching no equation, plane-of-array irradiance
   with no tilt model. All listed under `OVERVIEW.md` → "Honest limits".
7. **No experimental validation of any kind** (O2-8). Objective 4 scope.
