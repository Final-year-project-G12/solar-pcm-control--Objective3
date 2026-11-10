# Objective 2 — what this folder is doing

**One sentence:** for each of Tamil Nadu's three climate regimes, find the PCM
capsule design and flow rate that make a 100 L solar water tank deliver the
most useful hot water in the evening — and prove it beats the same tank with
no PCM in it.

---

## The question, precisely

A solar water heater stores heat as hot water. Add sealed capsules of
**phase-change material (PCM)** and some of that heat is stored in the
material's melting instead, which can hold the tank warmer into the evening
when people actually want hot water.

Whether that helps depends on choices nobody can make by intuition:

| choice | range searched |
|---|---|
| capsule inner radius | 10–40 mm |
| PCM volume fraction of the tank | 2–30 % |
| flow rate | 0.010–0.050 kg/s |
| capsule arrangement | single-layer, staggered, radial |
| which PCM | Objective 1's shortlist for that regime |

Every candidate is judged against an **identical plain-water tank at the same
flow, weather and demand**. The number that matters is the *difference*, so a
design is never credited for something the plain tank would have done anyway.

---

## How the answer is reached

Thirty pipeline stages, run in order. The chain that actually decides a design:

1. **Convergence** — prove the solver's timestep is fine enough that halving it
   changes annual energy by < 0.05 %.
2. **Sensitivity** — one-factor sweeps to find which parameters matter at all.
3. **Screening** — a Taguchi L9 array with ANOVA.
4. **DOE** — a 128-point scrambled-Sobol design of experiments per regime and PCM.
5. **Surrogate bake-off** — five model families compared; the winner is used to
   guide the search.
6. **NSGA-II + active learning** — multi-objective search that simulates where
   the surrogate is least certain, then refits.
7. **Dense grid cross-check** — 720 designs, to confirm the search did not miss
   a region.
8. **Repeatability band** — how much a result moves under year-to-year scatter,
   timestep, node count and material-property perturbation. Anything smaller
   than this band is noise.
9. **Selection** — a documented S0/S1/S2 funnel with explicit tie-breaks.
10. **Held-out test** — five years reserved from the start, evaluated **once**,
    under a lock that cannot be released.

A separate **Bayesian optimisation** track runs the same problem independently
as a cross-check, and **ANSYS Fluent** checks the physics the fast model assumes.

---

## What it found

| regime | PCM | capsule Ø | PCM fraction | flow | capsules | PCM mass |
|---|---|---|---|---|---|---|
| 0 | PureTemp 58 | 81.6 mm | 29.6 % | 0.0388 kg/s | 104 | 24.8 kg |
| 1 | RT57HC | 81.6 mm | 27.9 % | 0.0295 kg/s | 98 | 23.6 kg |
| 2 | RT57HC | 81.6 mm | 29.9 % | 0.0267 kg/s | 105 | 25.3 kg |

All designs are staggered, in a 100 L tank with a 2.5 m² collector.

**Every gate passes, on years never used to choose anything:**

| gate | held-out result |
|---|---|
| beats the plain tank | **9 / 9** designs |
| PCM stays under 65 °C | 9 / 9 (63.4–64.0 °C) |
| water stays under 75 °C | 9 / 9 (72.2–72.9 °C) |

Nine, not three, because the test covered the chosen design **plus the best
design of every other shortlisted PCM** — so it could not be a cherry-pick.

The margins are 178–269 kWh/yr over a repeatability band of 26–30. That is
roughly 6–9× the noise floor, so the PCM advantage here is decisive rather
than marginal.

---

## Two simulators, and why

| | tank model | ANSYS Fluent |
|---|---|---|
| scope | whole tank, whole year | one capsule, six hours |
| runtime | seconds per simulated year | ~30 min per PCM |
| role | **the optimiser's evaluator** | independent physics check |
| chooses the design | yes | **no** |

The search ran over 1,100 design evaluations; at CFD speed that is years of
compute. But the fast model buys its speed with assumptions, and Fluent is how
those get tested.

**Why Fluent and not OpenFOAM** (which the source Rajasthan study used):
OpenFOAM's standard solvers have **no native solidification/melting model**.
They resolve heat conduction into a solid capsule but with sensible heating
only — no latent heat. For a study whose premise is that a PCM stores heat *in
its phase change*, that solver can only show the uninteresting half. Fluent has
the enthalpy-porosity model built in.

---

## Where things are

| path | what |
|---|---|
| `src/` | the simulator, design space, surrogates, optimiser, CFD helpers |
| `scripts/` | the 30 pipeline stages (`g2`…`g24`) plus `tn_*` helpers |
| `configs/` | tank, collector, solver and per-state settings |
| `data/` | frozen inputs: Objective 1 outputs, weather, demand profile |
| `results/` | every stage's output, plus `plots/` |
| `ansys/` | Fluent cases, meshes, the live demo, `figures/` |
| `docs/` | results, CFD history, how to run, data provenance |
| `tests/` | 89 tests |
| `archive/` | superseded work — nothing deleted, only retired |

### Start here

| question | file |
|---|---|
| What were the results? | `docs/RESULTS.md` (hand-written, layered: read only above the 2026-10-28 superseded banner) |
| The same results, generated from `results/` with no hand-typed numbers | `docs/04_RESULTS.md` (`scripts/g12_results_doc.py`) |
| Which algorithms actually run? | `../documentation/OBJECTIVES_AND_ALGORITHMS.md` — `docs/RESULTS.md`'s own "Algorithms used" table sits **below** the superseded banner and describes the retired architecture |
| Why two simulators? | `docs/SIMULATOR_AND_CFD.md` |
| What does the CFD show, and what went wrong? | `docs/CFD_VERIFICATION.md` |
| How do I run it? | `README.md`, `ansys/HOW_TO_RUN_ANSYS.md` |
| How does weather enter the model? | `O2_Climate_Parameters_and_Simulators.docx` |
| The full write-up | `O2_TamilNadu_Pipeline_Report.docx` |

```bash
python scripts/tn_run_all.py --all --keep-going --workers 6   # the pipeline
python scripts/tn_final_summary.py                            # designs + gates
python -m src.plots.make_plots_results                        # figures
python -m pytest tests/ -q                                    # 89 tests
cd ansys && python fluent_live_demo.py                        # live CFD demo
```

---

## Honest limits

These are stated here rather than left for a reader to find:

- **CFD melt fractions are conservative.** First-order upwind is the only
  discretisation under which all four PCMs converge, and it is more diffusive
  than second order. The *ordering* of the PCMs is the result; the absolute
  values are low.
- **The CFD mesh is a symmetry half-slice**, so buoyancy-driven convection in
  the melt is excluded. That enhancement is carried empirically by the tank
  model — a division of labour, not an oversight.
- **Wind and humidity do not affect any result.** They are in the weather files
  but reach no equation; the collector and tank loss coefficients are fixed. For
  coastal regimes that is a real simplification.
- **Irradiance is used as plane-of-array directly**, with no tilt model. It
  applies equally to the PCM tank and its plain twin, so the reported
  *difference* is largely insulated — the absolute energies are not.
- **`g12_verify_docs` reports 236 stale path references** (verified 2026-11-09).
  An earlier version of this file said "~68, almost all inside the historical
  progress log" — that is wrong on both counts. The progress log holds 65 of
  them, 28 %, and it is the only file where describing paths as they once were
  is legitimate. The other 171 are genuinely stale:

  | doc | missing refs |
  |---|---|
  | `docs/DATA_REFERENCE.md` | 88 |
  | `docs/RAJASTHAN_METHOD_PORT_PROGRESS.md` | 65 |
  | `docs/HOW_TO_RUN.md` | 39 |
  | `docs/RESULTS.md` | 15 |
  | `README.md` | 15 |
  | `docs/CFD_VERIFICATION.md` | 14 |

  The 39 in the run guide matter most, since people follow it: it still
  documents `python pipeline.py --stage ...`, and `pipeline.py` no longer
  exists (the pipeline is `scripts/tn_run_all.py`, stages `g2`…`g24`).

---

## A defect worth knowing about

Partway through, every Objective-2 result had to be thrown away and re-run.

The per-point-year weather cache carried its solar columns **shifted +5 hours**
against their own timestamps — IST truncated to whole hours, written onto
local-clock stamps but labelled UTC. That moved collector gain from about 11:30
to about 16:30 local, far closer to the 19:00 demand peak than reality, which
systematically flattered PCM on the study's primary objective.

It was found by checking irradiance against computed sunrise and sunset, and
confirmed three independent ways. The cause turned out to be a second raw data
set whose header declares **Local Solar Time**, not UTC, read as though its keys
were UTC.

All 45 point-years were rebuilt from raw with a **mandatory ephemeris gate**:
the rebuild refuses to write unless clearsky daylight brackets true sunrise and
sunset on every day. The lesson is in `docs/CFD_VERIFICATION.md` and the
progress log, and it is the reason the inventory now records *how* each year's
timestamps were derived rather than just asserting a time standard:

> Provenance metadata is not verification.
