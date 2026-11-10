# Documentation

Everything in `new_obj/` that is **not** Objective 1 or Objective 2 source code,
plus copies of the four most important documents from those two objectives.

The two objective trees stay where they are:

| Objective | Folder |
|---|---|
| Objective 1 — climate regimes → PCM shortlist | `../tamilnadu_pipeline/` |
| Objective 2 — PCM capsule/flow design optimisation | `../objective2_design_optimization/` |

---

## What is in here

| path | what |
|---|---|
| `OBJECTIVE_3_README.md` | **Objective 3 — the DRL controller: overview, phases, results, plots** |
| `METHODOLOGY_AND_FINDINGS.md` | the scientific narrative — the R1→R2 reward-design journey and why DQN wins |
| `ALGORITHMS.md` | every baseline and algorithm: why included, config, and how each performed |
| `HOW_TO_RUN.md` | practical runbook — setup, every command, timings, troubleshooting |
| `objective3_plans/` | the O3 implementation plans (FINAL.docx + complete plan .md/.docx) |
| `from_before/` | O1/O2 reference docs carried into this tree |
| `OBJECTIVES_AND_ALGORITHMS.md` | **all four objectives, and the algorithms each one actually runs** |
| `obj1_key_docs/` | copies of Objective 1's plan + implementation docs |
| `obj2_key_docs/` | copies of Objective 2's plan + results docs |
| `obj1_results/` | copies of Objective 1's headline result files |
| `obj2_results/` | copies of Objective 2's headline result files |
| `conference_paper/` | LLNCS paper source (`main.tex`, `references.bib`, figures) |
| `Presentation/` | review slides, speech notes, per-objective explanations, course reports |
| `G12-Tea/` | group submission material (course summaries, paper PDF, submission proof) |
| `G12-Tea.zip` | zipped copy of the above |

### The four key docs

| objective | role | file | original |
|---|---|---|---|
| 1 | **plan** | `obj1_key_docs/Objective1_PCM_Climate_Framework_Plan_v3.md` | `tamilnadu_pipeline/docs/era5_tamilnadu/` |
| 1 | **implementation** | `obj1_key_docs/00_MASTER_OVERVIEW.md` | `tamilnadu_pipeline/docs/era5_tamilnadu/` |
| 2 | **plan** | `obj2_key_docs/Objective2_PCM_Design_Optimization_Workflow.md` | `objective2_design_optimization/docs/reference/` |
| 2 | **implementation / results** | `obj2_key_docs/RESULTS.md` | `objective2_design_optimization/docs/` |

**These are copies, not the source of truth.** The originals are still in each
objective's own `docs/` tree and are what the pipelines regenerate. If a
pipeline is re-run, re-copy rather than editing a copy here.

### The result copies

| folder | source | contents |
|---|---|---|
| `obj1_results/` | `tamilnadu_pipeline/data/processed/` | cluster assignments (Level A + Level B), κ-calibrated feasibility survivors, full MCDM rankings, Top-k by cluster, method agreement, recommendation cards |
| `obj2_results/` | `objective2_design_optimization/results/` + `docs/` | per-regime selection (`g10_selection_r{0,1,2}.json`), held-out test results, repeatability band, metrics report, dense-grid check, seed stability, held-out lock, results registry, generated `04_RESULTS.md` |

The live `results/` directory was **not** renamed. `config.py`'s `RESULTS_DIR`
and all 30 `g*` stage scripts resolve against that exact name, so renaming it
would break the pipeline. The per-objective naming lives here instead.

---

## Status as of 2026-11-09

### Objective 1 — complete

All 8 phases implemented and run end-to-end against elevation-corrected data
(last full run 2026-09-16). Headline outputs:

- 133 population-weighted points, ~87.5 % of state population, 2016–2025
- k=3 climate regimes (GMM, `covariance_type="diag"`, k auto-selected by
  `cluster_lib.suggest_k`'s 3-tier cascade — not hand-picked)
- 62-PCM database → 8-constraint feasibility filter → 9-criterion MCDM
  (TOPSIS / GRA / PROMETHEE II / VIKOR + Borda, Monte Carlo N=5000)
- consensus Top-1 PCM is `RT57HC` in all three regimes

**One documented open finding, deliberately not chased:** MCDM-vs-physics rank
agreement responded to the added `thermal_margin` criterion in Cluster 0
(ρ −0.595 → +0.381) but **not** in Clusters 1 and 2. This is kept as a
disclosed limitation rather than fixed with more criteria tuning, which would
risk overfitting the ranking to one physics model's assumptions. See
`tamilnadu_pipeline/docs/era5_tamilnadu/09_PHASE_7_AUDIT.md`.

Note: `12_FINAL_READINESS_REPORT.md` carries an "UPDATE 2026-09-16" banner —
its body text below that banner is **historical**, not current state. Read
`00_MASTER_OVERVIEW.md`'s status table for current numbers.

### Objective 2 — complete

30 pipeline stages (`g2`…`g24`), **89/89 tests pass** (verified 2026-11-09).
Final designs, in a 100 L tank with a 2.5 m² collector, all staggered:

| regime | PCM | capsule Ø | PCM fraction | flow | capsules | PCM mass |
|---|---|---|---|---|---|---|
| 0 | PureTemp 58 | 81.6 mm | 29.6 % | 0.0388 kg/s | 104 | 24.8 kg |
| 1 | RT57HC | 81.6 mm | 27.9 % | 0.0295 kg/s | 98 | 23.6 kg |
| 2 | RT57HC | 81.6 mm | 29.9 % | 0.0267 kg/s | 105 | 25.3 kg |

Held-out test (5 years reserved from the start, evaluated once under lock),
covering the chosen design **plus the best design of every other shortlisted
PCM** — 9 designs, not 3, so the result cannot be a cherry-pick:

| gate | result |
|---|---|
| beats the plain-water twin | 9 / 9 |
| PCM stays under 65 °C | 9 / 9 (63.4–64.0 °C) |
| water stays under 75 °C | 9 / 9 (72.2–72.9 °C) |

Margins are 178–269 kWh/yr against a repeatability band of 26–30 — roughly
6–9× the noise floor.

---

### Objective 3 — in progress (Tamil Nadu)

The contract blocker below is **resolved**: `obj3_environment_contract_tamilnadu_v3.json`
was generated from the current O2 results (PureTemp 58 / RT57HC / RT57HC,
staggered, 98–105 capsules) and is the frozen input O3 trains against. The full
O3 pipeline (phases O3-0 through O3-8) is implemented — step-based environment,
B0/B1/B2/B4 baselines, six DRL algorithms (PPO, DQN, RecurrentPPO, QRDQN,
MaskablePPO, SAC), robustness suite, locked-eval harness, software-in-the-loop
demo, and the Objective 4 hardware handoff spec.

**Key finding so far:** under a delivery-only reward (R1) every DRL algorithm
collapsed to the B0 "always-charge" policy because the safety shield made shield
activations cost-free in the reward. A shield-aware reward (R2, penalty per
guard-band trip) was added so DRL has a reason to learn anticipatory control.
Training under R2 is the current comparison of record.

See **`OBJECTIVE_3_README.md`** for the full methodology, run commands, and the
interactive Plotly/Folium visualisations in `results/plots/`.

---

## Known gaps

### (Resolved) was: blocking Objective 3 — contract did not exist

**Resolved 2026-11-09.** The text below is the original gap description, kept for
history; the v3 contract now exists and O3 is underway (see the O3 status section
above).

**The Objective 3 environment contract does not exist for the current
Objective 2 results.** This is the one thing that must be built before
Objective 3 work can start.

- `objective2_design_optimization/objective3/obj3_environment_contract_tamilnadu.json`
  is dated **2026-10-27** and describes a **superseded** run:
  n-Hexacosane (C26) / n-Pentacosane (C25), *radial* and *single-layer*
  arrangements, 32–35 capsules, 72–80 mm, 4.9–7.4 kg PCM. The current designs
  (table above) are PureTemp 58 / RT57HC, *staggered*, 98–105 capsules,
  81.6 mm, 23.6–25.3 kg. **Nothing in that file matches the released designs.**
- `objective3/README.md` says so itself: *"The environment contract here must
  be regenerated after the Objective-2 rebuild completes: it was written from
  the pre-shield-fix designs."*
- `objective3/OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md` (also 2026-10-27) is built
  on that stale contract. Its §1/§2 tables, its "PCM loses to a plain tank in
  every regime (−12.5 % to −14.8 %)" claim, and its "P(meets demand) = 0–4 %"
  claim all describe the superseded run. The current held-out result is the
  opposite: PCM beats the plain twin 9/9. **Do not plan Objective 3 from this
  document's numbers.**
- The generator `src/handoff/build_obj3_contract_v2.py` (2026-10-29) is newer,
  but it is **hardcoded for Rajasthan**, not Tamil Nadu: it reads
  `results/obj3_environment_contract_rajasthan.json` (which does not exist in
  this tree) and writes `..._rajasthan_v2.json`. Its two real data
  dependencies — `results/wp6_band.csv` and `results/g10_heldout_results.csv`
  — **do** exist and are current.
- `docs/HOW_TO_RUN.md:274` still documents the handoff as
  `python pipeline.py --state tamilnadu --stage handoff`, but `pipeline.py`
  no longer exists; the pipeline is now `scripts/tn_run_all.py` with the
  `g2`…`g24` stages, and no stage runs the handoff generator.

**So the work to start Objective 3 is:** port
`build_obj3_contract_v2.py` to Tamil Nadu (parameterise the state, drop the
dependency on the missing old Rajasthan contract so it builds the generic
blocks from scratch, point it at `g10_selection_r{0,1,2}.json`, `wp6_band.csv`
and `g10_heldout_results.csv`), wire it into `tn_run_all.py` as a stage, run
it, then rewrite `OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md` against the real output.

### Fixed on 2026-11-09

- **`g12_results_doc` crashed; now fixed.** `scripts/g12_results_doc.py:270`
  raised `IndexError: single positional indexer is out-of-bounds` on
  `w1[(w1.pair == tag) & (w1.state == st)].iloc[0]` — an empty selection
  indexed without a guard. Cause: `g18_issue1_with_without.csv` holds only the
  `base` state; the `novert` and `vert` runs were never completed, but the
  generator assumed all three.

  Two things were wrong beyond the crash, both corrected:

  1. **The generator's output never existed.** It writes `docs/04_RESULTS.md`,
     not `docs/RESULTS.md`. Since it had never once succeeded, the
     auto-generated results document — the one where no number is typed by
     hand — was simply **absent** from the repo. `docs/RESULTS.md` is a
     separate, hand-maintained, layered file. It now exists.
  2. **The narrative asserted numbers its own table could not show.** The
     hardcoded reading quoted `novert`/`vert` figures ("23.5 kWh against 13.8",
     "3.1", "r1 5.0 without, 1.8 with") that are not in the current results at
     all. It now prints an explicit *"This comparison is INCOMPLETE"* note
     naming the missing states instead, and the full reading only when all
     three states are present.

  Also fixed: the generated doc was titled *"Objective 2, **Rajasthan**"* in
  the Tamil Nadu tree — a leftover from the method port.

  Still outstanding (not a bug): `scripts/g18_issue1_compare.py` needs
  re-running for `novert` and `vert` before section 5f answers anything.

- **`OVERVIEW.md`'s stale-reference claim corrected.** It said *"~68 stale path
  references, almost all inside the historical progress log, which legitimately
  describes paths as they were."* Both halves were wrong. The real count is
  **236**, and the progress log holds **65** of them — 28 %, not "almost all":

  | doc | missing refs |
  |---|---|
  | `docs/DATA_REFERENCE.md` | 88 |
  | `docs/RAJASTHAN_METHOD_PORT_PROGRESS.md` | 65 |
  | `docs/HOW_TO_RUN.md` | 39 |
  | `docs/RESULTS.md` | 15 |
  | `README.md` | 15 |
  | `docs/CFD_VERIFICATION.md` | 14 |

  `OVERVIEW.md` now carries the real figure and this breakdown. The 39 in the
  run guide matter most, since people follow it: it still documents
  `python pipeline.py --stage ...` and `pipeline.py` no longer exists. The
  remaining 171 stale references are **not yet repaired** — only the claim
  about them is now honest.

### Non-blocking, still open

- **`docs/RESULTS.md`'s "Algorithms used" table is stale.** It sits below that
  file's own 2026-10-28 superseded banner and describes the retired
  architecture — "a single surrogate-scoring pass, not an optimization loop",
  "no active-learning, no NSGA-II", and modules (`src/doe/`, `src/surrogate/`,
  `src/robustness/`) that no longer exist. The current pipeline does run
  NSGA-II, active learning, a Sobol DOE and a five-family surrogate bake-off.
  `OBJECTIVES_AND_ALGORITHMS.md` in this folder is the accurate version, read
  from the code.
- Objective 2's own disclosed physics limits (conservative CFD melt fractions,
  symmetry-half-slice mesh excluding melt convection, wind/humidity reaching no
  equation, irradiance used as plane-of-array with no tilt model) are listed in
  `objective2_design_optimization/OVERVIEW.md` under "Honest limits" and are
  unchanged.

---

## The Objective 1 / 2 / 3 boundary

Carried forward from `OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md` §"The formal
boundary", which still holds even though that document's numbers do not:

- **Objective 1** selected the PCM shortlist.
- **Objective 2** selected the physical hardware design — PCM identity, capsule
  geometry, arrangement, flow — and the validated operating envelope.
- **Objective 3** selects the real-time action, and nothing else. It must not
  change PCM identity, capsule geometry, tank volume, safety limits, or the
  released physics model.
