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
| viz | `scripts/make_plots.py --tag R2` | interactive Plotly dashboard + Folium map |

**Train / validation / test split (year-locked):** train 2016–2020, validate
2021–2023, final test 2011–2015 (never touched until O3-6).

## 7. Visualisations

`scripts/make_plots.py` produces, in `results/plots/<tag>/`:

1. **Policy comparison** — delivered energy, shield activations, reward per policy.
2. **Delivery-vs-shield tradeoff** — the headline plot; up-and-left is better.
3. **Learning curves** — validation reward vs training steps per algorithm.
4. **Reward decomposition** — delivery reward vs unmet/shield penalties (R2).
5. **Per-year delivery** — consistency across validation years.
6. **Folium regime map** — 133 climate points, 3 regimes, medoid designs.

Open `results/plots/R2/dashboard.html` (and the linked map) in a browser.

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
  docs/            this file, objective3_plans/, from_before/
```
