# Objective 3 — Climate-Adaptive DRL Controller for PCM Solar Water Heating

**Tamil Nadu implementation.** A safe, shielded deep-reinforcement-learning
controller for the *frozen* Objective 2 hardware design. Objective 3 chooses the
real-time operating action (charge / idle / bypass); it never changes the PCM,
capsule geometry, tank size, safety limits, or released physics.

---

## 1. What this objective does

Objective 2 selected the physical design (PCM, 100 L tank, capsule count, flow
rate) for three Tamil Nadu climate regimes. It also ships a blunt reactive safety
shield that trips the collector off at 72 °C water / 62 °C PCM (3 °C below the
75/65 °C hard limits). Objective 3 asks:

> Can a learned controller operate this fixed hardware *better* than the reactive
> shield — delivering as much hot water while triggering the shield far less
> often — without ever violating a safety limit?

The controller wraps O2's hour-by-hour physics in a Gymnasium environment and
learns a policy over the discrete action set **{CHARGE, IDLE, BYPASS}**. The
in-environment safety shield is always active and authoritative — the policy can
never disable it. O3's contribution is *anticipatory* control: acting before the
guard band is hit, not reacting at it.

## 2. The three Tamil Nadu regimes (frozen from O2)

| Regime | Character | Selected PCM | O1 rank | Capsules | Flow (g/s) |
|---|---|---|---|---|---|
| 0 | coastal/interior lowland (medoid 156 m) | PureTemp 58 | #3 | 104 | 38.8 |
| 1 | elevated / Nilgiris-adjacent (627 m) | RT57HC | #1 | 98 | 29.5 |
| 2 | coastal metro lowland (44 m) | RT57HC | #1 | 105 | 26.7 |

All three PCMs are drawn from Objective 1's own shortlists. Regime 0 uses the #3
consensus PCM because, in full-system simulation, PureTemp 58 beat the
higher-ranked RT57HC on overheating behaviour at comparable window energy — a
legitimate O2 selection, recorded in the contract with its funnel rationale.

See the interactive **[regime map](../results/plots/R2/tamilnadu_regimes_map.html)**
for all 133 climate points and the per-regime designs.

## 3. Environment, observation, action, reward

- **Environment** (`o3/environment/pcm_env.py`): reuses O2's backward-Euler tank
  physics unchanged; exposes `reset()` / `step(action)`. A `PCMGymEnv` wrapper
  (`o3/environment/gym_wrapper.py`) adds the Gymnasium interface and an action
  mask; `PCMContinuousGymEnv` exposes a continuous action for SAC.
- **Observation V1** (11-dim, normalised): GHI, ambient T, water T, PCM T, melt
  fraction, demand, mains T, hour, day-of-year, previous action, water safety
  margin.
- **Action A1** (discrete): `0 = CHARGE` (pump on), `1 = IDLE` (pump off), `2 =
  BYPASS` (isolate collector).
- **Reward** — see §5. The reward version is the crux of this objective.

## 4. Baselines and algorithms

**Non-learning baselines** (the comparison ladder):
- **B0** — fixed charge (always on): the exact O2 operating point.
- **B1** — reactive rule controller (current-state rules).
- **B2** — forecast-aware rules (4 h irradiance look-ahead; anticipatory bypass).
- **B4** — random safe policy (diagnostic floor).

**Deep RL algorithms** (all trained on the same environment):
- **PPO** — on-policy policy gradient (stable baseline).
- **DQN** — value-based (double DQN).
- **RecurrentPPO** — LSTM policy, for temporal/monsoon patterns.
- **QRDQN** — distributional, risk-aware value learning.
- **MaskablePPO** — PPO with safety action masking.
- **SAC** — off-policy, entropy-regularised (continuous-action view).

## 5. The reward, and the key finding (R1 → R2)

**R1 (delivery-only)** rewarded delivered energy minus unmet-demand and
hard-limit penalties. Because the safety shield prevents *all* hard-limit
breaches, the safety term was always ≈ 0 — so **nothing penalised the ~50 000
shield activations per year.** Under R1, every DRL algorithm correctly learned
that "always charge" (identical to B0) is optimal: maximum delivery, zero penalty
for hammering the shield. The RL matched B0 exactly and the rule-based B1/B2 were
the *only* policies that reduced shield firing (≈ 56 %) — at a ≈ 7 % delivery
cost. **This is a reward-design finding, not an algorithm failure**, and it is
exactly what the implementation plan warned about ("do not define success only as
higher reward").

**R2 (shield-aware)** adds a penalty of `W_SHIELD = 10` per guard-band
activation. The break-even is ≈ 8.4/shield (B0 leads B1 by ~235 k on the
delivery/unmet terms while firing ~27.9 k more shields), so at 10 the reward
ranks **B1 > B0 > B2** — shield reduction now pays, and there is an interior
optimum the DRL agent can beat by keeping delivery high *while* cutting shield
events. R2 is the default; pass `reward_weights={"shield": 0.0}` to recover R1.

The R1 results are archived under `results/archive_R1_reward/` because the
collapse is itself a reportable result about reward shaping.

## 5b. Final result — selected policy: SAC; DQN is the regime-0-only alternative

*(Corrected 2026-11-13 after the weather-year bug fix; see
`CORRECTIONS_OBJECTIVE3.md`. Earlier numbers in this section, e.g. 2097.4 kWh /
23,117 shields, were computed on one repeated weather year and are retired.)*

Held-out test years 2011–2015 (never seen in training, tuning or selection),
mean per year:

| Regime | Policy | Delivered kWh/yr | Shield/yr | Safety violations |
|---|---|---|---|---|
| 0 | B0 always-charge | 2096.9 | 46,969 | 0 |
| 0 | DQN | 2018.0 (96.2 %) | 17,838 (-62 %) | 0 |
| 0 | **SAC** | 2044.6 (97.5 %) | 21,406 (-54 %) | 0 |
| 1 | B0 always-charge | 2094.8 | 24,815 | 0 |
| 1 | DQN | 2094.7 (~100 %) | 24,298 (-2 %) | 0 |
| 1 | **SAC** | 2067.5 (98.7 %) | 15,569 (-37 %) | 0 |
| 2 | B0 always-charge | 2068.8 | 34,796 | 0 |
| 2 | DQN | 2068.8 (identical to B0) | 34,796 (0 %) | 0 |
| 2 | **SAC** | 2012.6 (97.3 %) | 23,138 (-33.5 %) | 0 |

**Why SAC is the final pick:** it is the only learned controller that cuts
shield activations (33-54 %) while keeping >= 97 % of B0 delivery on *all three*
regimes, with zero safety violations. DQN is slightly better on regime 0
(-62 % shields) but collapses to B0 on regimes 1-2, so it adds nothing there.
On validation SAC also has the highest reward in regimes 0 and 1. Honest
caveats: in regime 2 SAC's reward is ~2.5 % *below* B0 (it trades ~3 % delivery
for 34 % fewer shields); SAC is ~5x slower at inference and ~34x larger than
DQN (still negligible against a 1-hour control period); and on regime 1 a dead
pyranometer collapses SAC's delivery (see METHODOLOGY_AND_FINDINGS.md 5b).

Data note: `results/phase6_final_test_{dqn,sac}.csv` is overwritten by each
regime run and currently holds regime 2 only; the regime 0/1 locked-test
numbers above come from the run logs.

## 5c. Deployment metrics (Phase O3-9, regime 0, validation years)

- **Thermal efficiency** (delivered / incident solar on 2.5 m^2): B0 41.6 %,
  DQN 40.1 %, SAC 40.6 %, B1 39.3 %, B2 38.7 %; demand coverage
  80.4 / 78.9 / 79.5 / 78.0 / 77.0 %.
- **Vs B1:** DQN +2.2 % delivery (15 kWh/yr backup avoided), SAC +3.5 %
  (28 kWh/yr avoided). Vs B0 both deliver ~2-3 % less but cut shields 54-59 %.
- **Inference cost:** DQN ~1 ms/decision, 40 kB weights; SAC ~5 ms, 1.4 MB.
- **Sensor faults:** zero hard-safety violations in every scenario (the shield
  acts on true state). Source: `results/deployment_metrics_r{0,1,2}.json`.

## 5d. Generation-2 research extensions (advisor feedback round, 2026-11-11/12)

Five gaps raised in review, each addressed in the environment
(`o3/environment/pcm_env.py`): a 15-dim **V2 observation** (+3h GHI forecast,
+1h demand), a configurable **control-step** (`control_step_min`), and
**training-time domain randomisation** (GHI/ambient/demand sampled per
episode, applied to the physics). A reduced-budget study (50k steps/SAC run,
100k/DQN run — a trend ablation, not full convergence) ran all three regimes:

**Pareto sweep** (SAC, regime 0, weights {0,10,20} × seeds {42,0,123} —
50k-step reduced budget; weights 5/15 and the full 300k-step budget deferred
for time, noted as future work):

| weight | delivered kWh (mean of 3 seeds) | shields (mean of 3 seeds) |
|---|---|---|
| 0 | 2146.1 | 50,296 |
| 10 | 2098.2 | 26,426 |
| 20 | 2021.2 | 15,114 |

Clean, monotonic frontier across all three seeds (full 3-seed coverage now
matches DQN's existing seed-robustness study): higher shield penalty → fewer
shields, less delivery, consistently. Confirms the Pareto trade-off is real,
controllable via the reward weight, and robust to seed.

**Sub-hourly control ablation** (weight=10, seed=42) — run twice. First at an
equal 50k-step budget for both control cadences: 60-min gave 28,588 shields,
15-min gave 28,762 — flat, confounded by the 15-min run covering only ~1.4
training-years vs ~5.7 at 60-min for the same step count. **Rerun at a
calendar-scaled budget** (60-min: 50k steps/5.7yr; 15-min: 160k ticks/~4.6yr,
closely matched coverage): **60-min → 28,588 shields; 15-min → 11,882
shields — a 58% reduction, for a 4% delivery cost (2092.6 → 2008.5 kWh).**
This **confirms the feedback's central claim**: control-step granularity is
a major lever for cutting guard-band activations, once the training-budget
confound is removed.

**Regime 1 & 2 coverage** (weight=10, seed=42) — SAC generalises cleanly to
both: R1 2118.3 kWh/22,180 shields; R2 2054.9 kWh/28,332 shields. DQN R1 is
also healthy (2129.8 kWh/25,691 shields). **DQN R2 was rerun at the full
300k-step budget** after the initial 100k-step run produced a degenerate,
near-always-bypass result (1647.7 kWh, 744 kWh unmet, 2 shields); the
full-budget rerun gives a sane **2086.9 kWh delivered, 31,619 shields**,
confirming the earlier anomaly was a training-budget artefact, now resolved.

Plots: `results/plots/R2/gen2_extensions.html` (Pareto frontier, sub-hourly
comparison, regime comparison). Raw data: `results/phase10_*.csv`,
`results/phase10_gen2_summary.json`.

## 6. Phases and how to run them

Run everything from the `objective3/` root. On Windows, prefix long-running
training with `PYTHONIOENCODING=utf-8`.

| Phase | Script | What it does |
|---|---|---|
| O3-0 | `scripts/phase0_audit.py` | validate + hash the frozen contract |
| O3-2 | `scripts/phase2_baselines.py` | run B0/B1/B2/B4 baselines |
| O3-3 | `scripts/phase3_generate_dataset.py` | safe-trajectory dataset |
| O3-4 | `scripts/phase4_train.py --regime 0 --steps 300000 --algo all` | train all algorithms |
| O3-4 | `scripts/phase4_multiseed.py` | multi-seed stats (mean ± std) |
| O3-4 | `scripts/phase4_reeval.py --regime 0` | re-score saved models without retraining |
| O3-5 | `scripts/phase5_robustness.py` | noise / bias / init-state robustness suite |
| O3-6 | `scripts/phase6_locked_eval.py --algo <best>` | **locked** test on 2011–2015 (run once) |
| O3-7 | `scripts/phase7_demo.py --mode stub\|replay\|live` | software-in-the-loop demo |
| O3-8 | `scripts/phase8_o4_handoff.py --algo <best>` | Objective 4 hardware handoff spec |
| O3-9 | `scripts/phase9_deployment_metrics.py` | efficiency, savings, latency/memory, sensor-fault suite |
| viz | `scripts/make_plots.py --tag R2 --selected sac` | interactive Plotly dashboard + Folium map |
| viz | `scripts/make_live_dashboard.py --algo sac [--live]` | animated live/replay controller dashboard |

**Train / validation / test split (year-locked):** train 2016–2020, validate
2021–2023, final test 2011–2015 (never touched until O3-6).

## 7. Visualisations

`scripts/make_plots.py` produces, in `results/plots/<tag>/`:

1. **Policy comparison** — delivered energy, shield activations, reward per policy.
2. **Delivery-vs-shield tradeoff** — the headline plot; up-and-left is better.
3. **Locked test** — the final verified result on 2011–2015.
4. **Robustness** — delivery across perturbation scenarios.
5. **Multi-seed** — seed-to-seed spread (the "precision" of each method).
6. **Reward decomposition** — delivery reward vs unmet/shield penalties (R2).
7. **Per-year delivery** — consistency across validation years.
8–11. **Deployment metrics** (`deployment_metrics.html`) — thermal efficiency &
   demand coverage, energy savings vs baselines, inference latency & memory,
   sensor-fault robustness.
- **Folium regime map** — 133 climate points, 3 regimes, medoid designs.
- **Live controller dashboards** (`results/plots/live_demo/`) — animated
  hour-by-hour CHARGE/IDLE/BYPASS playback, in historical-replay and LIVE
  (present-day Open-Meteo weather) modes, for SAC and DQN.

Open `results/plots/R2/dashboard.html` (and the linked pages) in a browser.

## 8. Safety status and scope

- Zero hard-limit violations is enforced structurally by the shield, which runs
  inside the physics and cannot be bypassed by the policy.
- **All results are simulation.** The live demo (O3-7) is software-in-the-loop
  only — thermal state comes from the frozen simulator, not sensors. Experimental
  validation is Objective 4's scope (O3-8 produces the handoff spec).
- Scope: Tamil Nadu, three regimes. Rajasthan / Assam / Uttarakhand reuse the
  same environment API with their own contracts (future work).

## 9. Directory map

```
objective3/
  o3/
    environment/   pcm_env.py, gym_wrapper.py (+ continuous wrapper)
    baselines/     b0_fixed, b1_reactive, b2_forecast, b4_random
    agents/        ppo, dqn, recurrent_ppo, qrdqn, maskable_ppo, sac trainers
    evaluation/    robustness.py
    demo/          live_adapter.py
    utils/         contract_loader.py
  scripts/         phase0..phase8, phase4_reeval, make_plots
  contracts/       obj3_environment_contract_tamilnadu_v3.json (frozen)
  results/         model checkpoints, comparison CSVs, plots/, obj4_handoff/
  tests/           pytest suite
  docs/            this file, METHODOLOGY_AND_FINDINGS.md, ALGORITHMS.md,
                   HOW_TO_RUN.md, LITERATURE_REVIEW.md, references.bib,
                   objective3_plans/ (FINAL_v1 plan)
```
