# Objective 3 — The Full Project Journey (Start to Now)

A single chronological account of Objective 3, phase by phase, decision by
decision. For reference material (algorithm detail, how to run each script,
literature) see `OBJECTIVE_3_README.md`, `ALGORITHMS.md`, `HOW_TO_RUN.md`,
`LITERATURE_REVIEW.md`. This file is the story; those are the manuals.

---

## 1. The starting point

Objective 2 had already shipped a frozen, validated physical design: a 100 L
tank, 2.5 m² collector, and a PCM selected per climate (PureTemp 58 /
RT57HC) across three Tamil Nadu regimes. It also shipped a blunt reactive
**safety shield** — trip the collector off at 72 °C water / 62 °C PCM, 3 °C
below the 75/65 °C hard limits — that fires roughly 50,000 times a year under
the simplest possible operating rule (always charge).

The question Objective 3 set out to answer:

> Can a *learned* controller operate the same frozen hardware so it delivers
> the same hot water while firing that shield far less often — anticipatory
> control instead of reactive — without ever breaching a hard limit?

## 2. Building the environment (Phase O3-0 to O3-1)

The O2 physics (hour-by-hour backward-Euler tank + PCM simulation) was
wrapped, unedited, in a step-based Gymnasium environment
(`o3/environment/pcm_env.py`). The agent only ever chooses one of three
actions each hour — **CHARGE**, **IDLE**, **BYPASS** — and the shield runs
*inside* the environment, authoritative over the policy. The contract
(`contracts/obj3_environment_contract_tamilnadu_v3.json`) was hash-verified
before any training began, and the year-locked data split was fixed and
enforced in code: train 2016–2020, validate 2021–2023, test 2011–2015
(touched once, at the very end).

## 3. Baselines before any learning (Phase O3-2)

Before training a single network, three non-learning controllers were built
to set the comparison floor: **B0** (always charge — the O2 operating
point), **B1** (reactive rules), **B2** (forecast-aware rules, a 4-hour GHI
look-ahead). B1 already cut shields ~56% versus B0, at a ~7% delivery cost —
establishing that *some* anticipation is cheap, and giving DRL something
concrete to beat.

## 4. The first training run — and a result about reward, not RL (Phase O3-3/O3-4, round 1)

Six algorithms were trained under a reward that only priced delivered energy
against unmet demand and hard-limit violations (**R1**). Because the shield
makes hard violations ≈0 by construction, nothing in R1 penalised the
50,000 shield firings — so *every* DRL algorithm correctly learned that
"always charge" was optimal. It matched B0 exactly. This was the first real
finding: **a safety shield that makes violations free also makes the thing
you actually want to minimise invisible to the learner**, unless the reward
says so explicitly.

## 5. The fix: R2, and finding the right penalty (round 2)

A shield-activation penalty was added: `reward -= W_SHIELD × (shield
events)`. At `W_SHIELD = 10`, the break-even point, the ranking becomes
B1 > B0 > B2 on reward — shield reduction now pays. Under R2, **DQN** learned
the anticipatory policy for the first time: B0-level delivery, 26% fewer
shields. A stronger penalty (`W_SHIELD = 20`) was tried to push *more*
algorithms off the always-charge trap — it backfired, destabilising DQN's
value learning until it collapsed back to always-charge even though the
better policy still scored higher. **Second finding: penalty magnitude past
the break-even point is not the lever; it just destabilises training.**
Reverted to 10.

## 6. Why DQN (and later SAC) escape, and the others don't

Across all six algorithms, the separator wasn't architecture — it was
**exploration mechanism**. DQN's ε-greedy forces random actions regardless
of policy confidence; the on-policy family (PPO, RecurrentPPO, MaskablePPO)
stops exploring once their policy sharpens, so none of them ever stumble
into the better region. QRDQN, also value-based, stayed stuck at this
seed/weight — so it's not purely "value-based wins," it's "sustained,
forced exploration wins."

## 7. Selection, robustness, and the first locked test (Phase O3-4 → O3-6)

DQN's escape was verified across seeds 42/0/123 — not a lucky seed, a
reproducible method. A lexicographic selection rule was fixed *before*
looking at the test set: zero hard violations → delivery ≥95% of B0 →
maximum shield reduction, never reward alone. Seed 42 was selected (the
only one keeping delivery ≥95%). It was run **once** on the held-out
2011–2015 test years: 2144.7 kWh delivered (99.9% of B0), 36,964 shields
(−26%), zero violations. Robustness (O3-5) confirmed stability under GHI
noise, ambient bias, and varied initial PCM state.

## 8. SAC closes the loop — and actually wins (later round)

SAC had been scoped in the plan but not finished in the first pass. Once
run to completion (on a continuous-action wrapper, `PCMContinuousGymEnv`,
binned back to the 3 discrete actions), it also escaped always-charge — via
entropy regularisation + replay, a *different* exploration mechanism than
DQN's. It passed the same locked test: 2097.4 kWh (97.7% of B0), **23,117
shields (−54%)**, zero violations, and the **highest reward of any
controller**. Under the pre-fixed selection rule, SAC wins on shield
reduction — it became the selected policy, with DQN retained as the
validated max-delivery alternative. Two independent algorithms reaching the
same qualitative result (anticipatory control, zero violations) is stronger
evidence than either alone.

## 9. Hardening the result: deployment metrics and the live demo (Phase O3-7 to O3-9)

With a policy selected, the questions shifted from "does it work" to "is it
deployable": thermal efficiency, energy savings versus both baselines,
inference latency and memory footprint on the embedded target class,
robustness under *sensor* faults (dead/stuck pyranometer, biased
water-temperature sensor — distinct from the earlier *weather* robustness
suite). Zero hard-safety violations held even on corrupted sensor input,
because the shield acts on true simulated state, not on what the policy was
told. A software-in-the-loop demo (`scripts/make_live_dashboard.py`,
`scripts/streamlit_portal.py`) was built so the controller's hour-by-hour
decisions — including when the shield overrides the policy — can actually
be watched, on historical replay or on present-day live weather.

## 10. Advisor feedback, round one: the Generation-2 study

A review surfaced five concrete gaps. Each was closed in the environment
itself, not bolted on:

- **Control step** — the shield fires at ~150 s sub-steps inside the hour;
  an hourly *action* can't react that fast. `control_step_min` was added,
  backward-compatible (60 = the original hourly environment, bit-for-bit).
- **Forecast** — the observation only had the current hour's GHI. A V2
  observation (15-dim) appends the next 1h/2h/3h GHI and the next hour's
  demand — the same look-ahead B2's rule already used.
- **Pareto curve, not two points** — a reward-weight sweep, so the
  delivery/shield trade-off is shown as a frontier, not asserted.
- **Regime 1 & 2** — the "climate-adaptive" claim had only ever been tested
  on regime 0.
- **Domain randomisation** — GHI, ambient temperature, and demand are
  randomised *during training* (not only at evaluation), so the policy has
  seen variability before the robustness suite ever tests it.

A first, reduced-budget pass (50k–100k steps, to fit a compressed timeline)
answered all five directionally. Two results needed a second look: the
sub-hourly ablation came back flat (not the predicted improvement), and
regime-2 DQN looked degenerate (low delivery, high unmet energy, almost no
shields). Both were investigated rather than reported as-is.

## 11. The confound, found and fixed

The sub-hourly ablation's flat result turned out to be an artefact: at
15-minute control, the same step budget covers four times fewer *calendar
years* of weather than at 60-minute control, so the finer-grained policy was
simply less trained, not genuinely unaffected by control granularity.
Re-run with the step budget scaled to match calendar coverage, the result
flipped decisively: **58% fewer shield trips for a 4% delivery cost** —
confirming the original advisor hypothesis once the confound was removed.
The regime-2 DQN anomaly was the same story at a different scale: a rerun
at the full 300,000-step budget gave a sane result (2086.9 kWh, 31,619
shields), confirming the earlier run was under-trained, not a real property
of that regime. Both corrections, and the reasoning behind them, are
recorded in the docs rather than quietly overwritten.

## 12. Where this stands now

Objective 3's **core deliverable is complete, validated, and has been for
some time**: SAC selected (DQN as a validated alternative), both locked-
tested on held-out years, both robustness-tested, deployment metrics
computed, a working interactive demo. None of what follows changes that
result — it is additional rigor on top of it.

What remains **in progress**: finishing the Generation-2 Pareto sweep at the
full 300,000-step budget across all five reward weights (0, 5, 10, 15, 20),
and the full-budget regime-1/2 SAC upgrades. This has been interrupted
twice by the training session itself restarting mid-run (background
processes die when that happens); each time, the partially-overwritten
files were identified and reverted to the last known-good, already-
documented state before relaunching, so no corrupted or unvalidated result
has ever been reported as final. Current status and exact numbers are kept
in `OBJECTIVE_3_README.md` §5d and the `_v1` plan document's §19, both
updated each time a batch completes.

## 13. What this does, and does not, establish

It establishes, **in simulation**, that a learned controller — two
independent ones, in fact — can hold delivery within a few percent of the
maximum-charge baseline while cutting protective shield trips by roughly a
quarter to a half, with zero safety violations, reproducibly, across
climate regimes, and robust to both weather and sensor-level perturbation.

It does **not** establish field performance. Every result here is
simulation; the live demo is software-in-the-loop, not hardware-in-the-
loop. Experimental validation on physical hardware is Objective 4's scope,
for which this objective's final phase (O3-8) produced the sensor list,
actuator interface, and failsafe specification as a handoff.

## 14. The weather-year bug, found and fixed (2026-11-13)

Everything in §§1–13 above was written before this date and describes the
project as it was understood at the time — including results that turned out
to rest on a critical defect. A bug report (independently surfaced while
fixing the same class of issue in the parallel Rajasthan work) pointed at
`_load_weather()`: it called an Objective-2 helper that returns one fixed
2016-only weather file with no year parameter, and the old filtering logic
silently fell back to that file whenever a requested year had no matching
rows — which was every year except 2016. **Every "different year" ever used
in this project — the five training years, the three validation years, the
five "locked test" years — was identical 2016 data relabelled.**

This was not a cosmetic issue. It meant the seed-robustness study, the
algorithm selection (DQN vs SAC vs the others), the "locked test passed"
claim, and the Generation-2 extension work were never actually tested against
year-to-year weather variation at all. The fix itself was small — load the
correct per-year file directly and raise loudly instead of substituting —
but its blast radius covered essentially every quantitative claim in this
document.

**What happened next, under real time pressure** (the user had explicitly
budgeted hours, not days, for the recovery): baselines (B0/B1), DQN, and SAC
were retrained from scratch on the corrected weather loader for all three
regimes, genuinely distinct years per split. PPO was considered and dropped
again — still B0-identical, still not worth a third parallel process. The
retrain itself was repeatedly interrupted by further session restarts (the
same infrastructure instability noted in §12), which motivated building
checkpoint-resume support into both trainers mid-stream so an interrupted
300k-step run could pick up from its last saved checkpoint instead of
restarting from zero — this saved roughly 2–3 hours of wall-clock time across
the recovery.

**The corrected result changed the project's conclusion, not just its
numbers.** Pre-fix, DQN and SAC appeared to be two validated alternatives,
with DQN offering higher delivery and SAC offering deeper shield reduction —
a genuine Pareto choice. Post-fix, that picture holds only on regime 0. On
regimes 1 and 2, DQN does not reliably escape the always-charge policy (on
regime 2 its locked-test result is numerically identical to B0 in every
column). **SAC is the only controller that consistently beats always-on
across all three regimes**, and is now the project's primary result. SAC also
turned up a new finding under deployment-metrics sensor-fault testing: on
regime 1 specifically, a simulated dead pyranometer collapses its delivered
energy by over 99% (to ~9.6 kWh/yr from a ~1990 kWh/yr nominal) — a severe,
regime-specific availability risk not present on regimes 0 or 2. Full detail
in `CORRECTIONS_OBJECTIVE3.md`, which is now the authoritative account of
what changed and why; `METHODOLOGY_AND_FINDINGS.md` and `ALGORITHMS.md` carry
the corrected numbers inline with notices pointing back here.

A second, related correctness gap was found and fixed in the same pass: the
V2 forecast observation (1–3h-ahead GHI used by the Generation-2 "forecast
observation" extension) read the stored year's **true future** value
directly — a perfect forecast no real sensor or weather model could provide.
Replaced with V3: a 5-hour forecast with multiplicative error growing from
~17% at 1 hour to ~37% at 5 hours, deterministic per (year, regime) for
locked-test reproducibility but genuinely wrong at every step, the way a real
forecast is. Retraining against V3 is scoped as follow-on work, starting with
regime 0 given the time remaining.
