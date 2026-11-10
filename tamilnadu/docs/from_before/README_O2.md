# Objective 2 — PCM Capsule / Tank Design Optimization (Tamil Nadu)

Determines the optimal PCM thickness, capsule arrangement, number of
capsules, and flow rate for a solar-water-heater tank, per climate regime,
using **Objective 1's own shortlisted PCMs** (material choice is never
re-derived here — see `archive/old_implementation/_archive/docs/` for the decision that settled it).

The pipeline is state-agnostic: nothing in `src/` knows which state it is
running for, only `configs/states/<state>.yaml` does.

---

## Status — 2026-10-31

**Objective 2 is complete — all 30 stages, the held-out test included.** The
methodology follows the Rajasthan study's pipeline with **ANSYS Fluent as the
CFD tool** in place of her OpenFOAM, a 100 L tank in place of her 50 L, and a
stacked ensemble in place of XGBoost.

| | |
|---|---|
| Optimal designs | `results/tn_optimal_designs.csv` |
| Acceptance gates | `results/tn_threshold_report.csv` — **9/9 pairs pass** |
| Figures | `results/plots/` — **all 9** |
| Report | `O2_TamilNadu_Pipeline_Report.docx` (11 tables) |
| Why two simulators | `docs/SIMULATOR_AND_CFD.md` |
| CFD | `ansys/` — see `ansys/HOW_TO_RUN_ANSYS.md` |

Headline: in all three regimes the selected design beats an identical
plain-water tank at the same flow by **+178 to +269 kWh/yr over a
repeatability band of 26–30 kWh/yr**, and every shortlisted PCM clears every
gate — not only the winners. Full numbers in `docs/RESULTS.md`.

**Nothing is outstanding.** All 30 pipeline stages have run, including the
one-shot held-out test on 2011–2015 — years reserved from the start and
evaluated exactly once, under a lock that cannot be released. Every
difference between the select years and the held-out years is inside the
repeatability band, so the designs are not overfitted to the years they were
chosen on.

| gate, on the held-out years | result |
|---|---|
| beats the plain tank | **9 / 9** |
| PCM peak ≤ 65 °C | 9 / 9 (63.4–64.0 °C) |
| water peak ≤ 75 °C | 9 / 9 (72.2–72.9 °C) |

Known limits, all stated rather than hidden: CFD melt fractions are
conservative (first-order upwind is the only setting under which all four
PCMs solve), the CFD mesh is a symmetry half-slice so buoyancy in the melt is
excluded, and `g12_verify_docs` still reports 68 stale path references, almost
all of them inside the historical progress log. See
`docs/RESULTS.md` and `docs/CFD_VERIFICATION.md`.

### Quick start

Run everything from the project root.

```bash
python scripts/tn_run_all.py --all --keep-going --workers 6   # full pipeline
python scripts/tn_final_summary.py                            # designs + gates
python -m src.plots.make_plots_results                        # figures
python -m pytest tests/ -q                                    # 88 tests

cd ansys && python fluent_live_demo.py                        # live melting demo
```

### Layout

```
.
├── config.py        paths and STATE; every path derives from this file
├── configs/         per-state YAML (regimes, PCM shortlists, mains temps)
├── data/            frozen inputs: objective1/, weather/, demand/
├── scripts/         pipeline stages (g2 ... g24) + tn_* helpers
├── src/             simulator, design space, surrogates, optimiser, cfd/
├── tests/           88 tests
├── results/         stage outputs + plots/
├── figures/         phase2-8 figures from the earlier implementation
├── docs/            RESULTS.md, CFD_VERIFICATION.md, HOW_TO_RUN.md, ...
├── ansys/           ANSYS Fluent cases, meshes, live demo
├── tools/           report builder (node) and maintenance scripts
└── archive/         superseded work — nothing here is deleted
```

> **A note on the history.** Until 2026-10-30 this pipeline lived in a
> subfolder named `.` (project root), which named the *method's* origin — Manvitha's
> Rajasthan build — and made the project look like it was about Rajasthan. It
> is not: every input, design and result is Tamil Nadu. That directory was the
> whole working pipeline, so it was promoted to the root and the older
> implementation moved to `archive/old_implementation/`.

| | |
|---|---|
| Tank / collector | 100 L / 2.5 m² (40 L/m², low end of IS 12976:2023) |
| Timestep | **150 s to search, 37.5 s to report** — the PCM cases converge at order p≈0.8, so her <0.05 %/halving gate is met only at 37.5 s. Her 50 L tank passed at 150 s; our 100 L one does not. |
| Tank model | 10-node stratified, numba-compiled, **844× faster** than the reference |
| Search space | `pcm_volume_fraction` 0.02–0.30 (searched), capsule count derived |
| Energy-balance residual | ≤ 0.055 % (gate: < 0.5 %) |
| Safety | PCM ≤ 65 °C, water ≤ 75 °C, both honoured |

**Progress and every finding behind these numbers:**
[`docs/RAJASTHAN_METHOD_PORT_PROGRESS.md`](docs/RAJASTHAN_METHOD_PORT_PROGRESS.md)

### What is done
- 10-node stratified tank + PAV inversion removal, numba kernel, **equivalence-validated to machine precision (24/24 checks)** against the pure-Python reference
- **Two-part safety shield** (water-side bypass + PCM-side charge block) — the fix that turned `g_window` from negative in every regime to **+107 … +148 kWh**
- Model repairs **G1** (local-hour floor), **G2** (inner-sphere PCM mass + shell heat capacity), **D3** (inner-radius conduction)
- dt convergence study with Richardson extrapolation over a 4-point ladder
  (300/150/75/37.5 s) → `results/g2_richardson_summary.csv`
- Parallel execution layer (process pool, worker-side input caching)
- NSGA-II (SBX, polynomial mutation, constrained domination, crowding, hypervolume) — validated against the ZDT1 benchmark
- Surrogate bake-off: **Gaussian-process/Kriging**, quadratic RSM, NN ensemble, XGBoost-as-baseline
- 15 physics-invariant tests, all passing

### What is pending
Sobol DOE execution → surrogate fit on real data → NSGA-II search →
active learning → confirmation on select years → repeatability band →
S0/S1/S2 funnel → held-out test → diagnostics → new .docx report.
`results/` is regenerated by these stages; until they run, see the
registry's `NOT RUN` markers rather than assuming a file is current.

---

## Two trees, and which one to use

**`.` (project root) is the current pipeline.** It is Manvitha's *actual*
Rajasthan implementation (from `Github/Objective---2/objective2-rajasthan`
— 4,247 lines of `src/`, 3,056 of `scripts/`, gates g2–g24), copied in and
pointed at Tamil Nadu's frozen inputs with the 100 L tank.

It replaced a hand-port written from her `.docx` report's prose. Comparing
the hand-port against the real source found **ten material physics
divergences**, the decisive one being that her collector is a *mass-flow
loop* — flow leaves node 0 and re-enters at a stratified return node, so
flow rate sets the loop ΔT and acts on stratification — while the
hand-port had modelled it as a pure heat source. Flow rate is one of the
four things Objective 2 exists to determine, so that difference mattered
more than the convenience of keeping the port. Adopting her code removed
all ten divergences at once and makes any remaining difference in results
a difference in *inputs*, not in method.

```bash
cd rajmethod
python scripts/tn_run_all.py --list                  # the 31-stage sequence
python scripts/tn_run_all.py --all --keep-going      # run it
python scripts/tn_run_all.py --only g8_s2_doe        # one stage
```

Her scripts are used **unmodified**; `tn_run_all.py` only decides the
order and what runs concurrently, so every result is reproducible by
running her command lines by hand. Three adaptation scripts are ours:
`tn0_add_local_hour.py`, `tn1_build_point_years.py`, and the launchers.

Deliberate differences from her tree, and only these:

| | hers | Tamil Nadu |
|---|---|---|
| tank | 50 L | **100 L** (instruction) |
| collector | 1.5 m² (33.3 L/m²) | 2.5 m² — holds 40 L/m², the low end of IS 12976:2023 §4.2; her 33.3 L/m² is *below* that band, which her own report flags |
| state | rajasthan | tamilnadu, her config format filled from TN's Objective 1 output |

Her absolute kWh figures are therefore **not** comparable to ours; the
algorithms, gates and pass/fail criteria are.

**`src/` is the legacy tree.** It holds the original Phase 1–8 pipeline
plus the numba kernel and parallel layer. Its numba multinode kernel is
still the fastest simulator here (1.0–1.9 s/simulated-year against her
implementation's 4.3 s at the same dt=150 s / N=10) and is
equivalence-validated to machine precision, but its *physics* is the
superseded hand-port. Use it for the kernel, the parallel layer and the
test suite; do not use it to produce design results.

## Quick start

```bash
# 1. the expensive DOE (explore years only)
python -m src.doe.run_sobol_doe --points 128 --years 5 --dt 150

# 2. everything downstream, in order
python run_rajasthan_pipeline.py --stage all

# or one stage at a time
python run_rajasthan_pipeline.py --stage surrogate
python run_rajasthan_pipeline.py --stage search
```

**Windows note:** any script that fans work out to the process pool must
keep its body behind `if __name__ == "__main__":`. Workers are *spawned*
and re-import `__main__`, so an unguarded script makes every worker build
its own pool. `run_tasks_parallel()` detects this and says so.

### Verification scripts (these are the evidence, and are meant to be re-run)

```bash
python validate_multinode_fast.py predictive   # kernel == reference, 24 checks
python validate_multinode_fast.py lagged       # same, other shield mode
python validate_nsga2.py                       # NSGA-II vs the ZDT1 front
python validate_surrogates.py                  # GP vs XGBoost, sd calibration
python -m src.verify.dt_convergence            # timestep + Richardson
python -m pytest tests/ -q                     # 15 physics invariants
python benchmark_parallel.py 240               # throughput
```

---

## Year roles — why the splits are kept apart

| years | role | used for |
|---|---|---|
| 2016–2020 | explore | the DOE, surrogate fitting, NSGA-II, active learning |
| 2021–2023 | select | hard safety constraints + the selection funnel |
| 2011–2015 | **test** | **one-shot** held-out test, behind a SHA-256 lock |
| 2024–2025 | extra | spare |

The DOE refuses to build a task for a select or test year, and the
held-out test is guarded by an atomic single-use lock file. A held-out
test you can re-run is a validation set you tuned against.

---

## Layout

```
configs/                 frozen inputs (system, design bounds v3.0, per-state)
data/                    weather (15 years), PCM database, demand profile
src/
  design/                geometry, constraints, volume-fraction <-> count
  simulation/            tank models + the numba kernel
    tank_model.py             single-node (regression reference)
    tank_model_multinode.py   10-node reference, pure Python
    tank_model_multinode_fast.py  the numba kernel (production)
    capsule_mass.py           G2/D3 inner-sphere mass + shell capacity
  doe/                   Sobol, Taguchi L9 + ANOVA, LHS (legacy)
  surrogate/             model zoo + per-group bake-off
  optimize/              NSGA-II, surrogate search, active learning, funnel
  verify/                dt convergence, repeatability, held-out lock, diagnostics
  parallel/              process pool + worker-side caching
  metrics/               g_window, g_annual, d_overheat, tau_hold/tau_serve
results/<state>/         current outputs + results_registry.csv
tests/                   physics invariants
objective3/              Objective-3 hand-off (kept out of results/)
_archive/                superseded material — nothing here is current
```

`archive/old_implementation/_archive/results_pre_shield_fix/` holds outputs computed **before** the
2026-10-28 shield fix and the v3.0 design space. Those numbers are wrong,
not merely old.

---

## Reading the results honestly

- `results/<state>/results_registry.csv` is the only list of citable
  numbers. Every row carries its source file and a content hash. A
  quantity with no row was not computed.
- A difference smaller than the **repeatability band** is not a result.
  The band is measured (year + timestep + node-count + PCM-property
  effects, combined in quadrature), and candidates the band cannot
  separate are reported as indistinguishable rather than as a winner.
- Feasibility is never surrogated — `check_design()` is exact and runs on
  every candidate.

## Docs

| file | what it holds |
|---|---|
| `docs/RAJASTHAN_METHOD_PORT_PROGRESS.md` | the port: every gate, every finding, every measurement |
| `docs/HOW_TO_RUN.md` | stage-by-stage run instructions |
| `docs/RESULTS.md` | current results narrative |
| `docs/DATA_REFERENCE.md` | where every input comes from |
| `docs/CFD_VERIFICATION.md` | ANSYS Fluent cross-check |
| `docs/REFERENCES.md` | literature and standards |
