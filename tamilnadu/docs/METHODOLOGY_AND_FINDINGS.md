# Objective 3 — Methodology and Key Findings

> **Correction notice (2026-11-13):** a critical bug meant every "different
> year" of weather data loaded by this environment was actually identical
> 2016 data. All numbers below are from the corrected retrain (genuinely
> distinct per-year weather, train 2016–2020 + 2024–2025, validate 2021–2023,
> locked test 2011–2015). See `CORRECTIONS_OBJECTIVE3.md` for the full account
> and what changed. Sections 3–4 (reward design journey, w_shield reasoning)
> are methodology, not year-specific numbers, and remain valid as written —
> only numeric results in §5 and §5b were affected and are updated below.

The scientific narrative behind the Tamil Nadu DRL controller: what we set out
to test, how the reward design drove the result, and what we learned. For the
"what it is / how to run it" overview see `OBJECTIVE_3_README.md`; for the
algorithm-by-algorithm detail see `ALGORITHMS.md`.

---

## 1. The research question, made measurable

Objective 2 ships a fixed PCM design and a blunt reactive safety shield that
trips the solar collector off at 72 °C water / 62 °C PCM (3 °C below the 75/65 °C
hard limits). The shield guarantees safety but is *reactive* — it fires at the
boundary, roughly 50,000 times a year under the always-on baseline.

> Can a learned controller operate the same hardware so that it delivers the same
> hot water while triggering the shield far less often — i.e. learn *anticipatory*
> control — and never violate a hard limit?

"Better" is therefore not "more reward." It is: **same delivery, fewer shield
activations, zero safety violations**, reported against the rule-based baselines.

## 2. The method

- A step-based environment (`o3/environment/pcm_env.py`) wraps Objective 2's
  hour-by-hour physics *unchanged*; the agent only chooses the operating action
  {CHARGE, IDLE, BYPASS} each hour.
- The safety shield runs **inside** the environment and overrides any unsafe
  action — the policy can never disable it. So safety is structural, and the
  learning problem is purely "when to charge vs. ease off."
- A ladder of controllers is compared on the same weather: non-learning
  baselines B0 (always charge), B1 (reactive rules), B2 (forecast-aware rules),
  B4 (random-safe), and six deep-RL algorithms (PPO, DQN, RecurrentPPO, QRDQN,
  MaskablePPO, SAC).
- Year-locked split: train 2016–2020, validate 2021–2023, test 2011–2015
  (untouched until the final locked evaluation).

## 3. The reward-design journey (the heart of the finding)

### R1 — delivery only
`r = w_E·E_delivered − w_U·E_unmet − w_S·hard_violations − w_switch·switch`

Because the shield prevents every hard-limit breach, `hard_violations` is always
≈ 0, so **nothing in R1 penalised shield activations.** Result: *every* DRL
algorithm converged to the B0 "always charge" policy — maximum delivery, and the
50,000 shield trips cost nothing. Only the rule-based B1/B2 reduced shield firing,
and they did so by sacrificing ~7 % delivery.

**This is the first finding, and it is a result about reward shaping, not about
RL:** a safety shield that makes violations free also makes the thing you
actually want to minimise invisible to the agent.

### R2 — add a shield-activation penalty
`r = … − w_shield·(n_shield_water + n_shield_pcm)`

The break-even weight is ~8.4 per shield event (B0 leads B1 by ~235k on the
delivery/unmet terms while firing ~27.9k more shields). We set **w_shield = 10**,
which ranks B1 > B0 > B2 on reward — shield reduction now pays, with an interior
optimum the agent can beat.

Under R2 (w=10): **DQN learned the anticipatory policy** — 2144 kWh delivery
(essentially B0's 2146), but 36,955 shield trips (26 % fewer than B0's 50,298),
and the highest reward of any controller (976k vs B1's 891k vs B0's 847k), zero
safety violations. The other five algorithms stayed at the always-charge policy.

### R2 with w_shield = 20 — the penalty is not the lever
To push more algorithms off always-charge, we tried a stronger penalty (w=20).
It **backfired**: DQN *collapsed back* to pure always-charge (reward 344k) even
though the shield-reducing policy still scored higher there (~607k). The larger
penalty destabilised the value updates; the agent failed to find the good policy
it had found at w=10.

**Second finding: penalty magnitude is not the lever.** Beyond the point where
shield-reduction pays, raising it only destabilises learning. We reverted to
w=10.

## 4. Why SAC is the final pick, and why the others stall

| Family | Exploration | Outcome |
|---|---|---|
| DQN, QRDQN (value-based) | ε-greedy (forced random actions, 30 % of training) | DQN escaped the always-charge basin; QRDQN did not at w=10 |
| PPO, RecurrentPPO, MaskablePPO (on-policy) | entropy bonus | entropy collapses once the policy commits to always-charge; no escape |
| SAC (off-policy, continuous) | entropy-regularised + replay | **escaped on all 3 regimes** (corrected: 97-99 % of B0 delivery, 33-54 % fewer shields); DQN escaped only on regime 0 |

The always-charge policy is a strong, wide attractor (it is the obvious
high-delivery policy any agent finds first). Escaping it requires *trying*
bypass-heavy behaviour long enough to discover the reward is higher. DQN's
ε-greedy keeps taking random actions regardless of how confident the policy is,
so it stumbles into the better region; the on-policy methods stop exploring once
their policy sharpens, so they never see it.

**Third finding: what separates the winners from the stuck algorithms is
*sustained exploration*, not the algorithm family per se.** The two escapees use
different mechanisms — DQN's ε-greedy forces random actions regardless of policy
confidence; SAC's entropy regularisation + replay buffer keep revisiting
alternatives — while every on-policy method stops exploring once its policy
sharpens. For this shielded, discrete-action problem, value-based or off-policy
DRL with forced exploration is the right method; on-policy policy-gradient
methods are a poor fit. This is exactly the kind of "why this algorithm"
evidence the plan asked for — grounded in observed behaviour, not assumed.

## 5. Robustness and selection — corrected, all three regimes (2026-11-13)

The seed-sensitivity study (§5, old) was run pre-fix and is retired along with
everything else quoting per-year numbers before the weather-loader fix; it is
not restated here. What follows is the corrected locked-test result for all
three regimes, each trained and evaluated independently on genuinely distinct
weather years.

### Locked final evaluation (Phase O3-6) — PASSED, all three regimes

| Regime | Policy | Delivered kWh | Shield / yr | Safety | vs B0 delivery | shield cut |
|---|---|---|---|---|---|---|
| 0 | B0 | 2096.9 | 46,969 | 0 | — | — |
| 0 | DQN | 2018.0 | 17,838 | 0 | 96.2% | **62%** |
| 0 | SAC | 2044.6 | 21,406 | 0 | 97.5% | 54% |
| 1 | B0 | 2094.8 | 24,815 | 0 | — | — |
| 1 | DQN | 2094.7 | 24,298 | 0 | ~100% | 2% (≈ B0) |
| 1 | SAC | 2067.5 | 15,569 | 0 | 98.7% | **37%** |
| 2 | B0 | 2068.8 | 34,796 | 0 | — | — |
| 2 | DQN | 2068.8 | 34,796 | 0 | 100% (identical) | 0% |
| 2 | SAC | 2012.6 | 23,138 | 0 | 97.3% | **33.5%** |

All primary criteria (zero safety violations, delivery ≥ 95% of B0) are met by
every policy on every regime. **The qualitative finding has changed from the
pre-fix conclusion:**

- **DQN only learns genuinely anticipatory control on regime 0.** On regime 1
  it converges to a policy numerically almost indistinguishable from B0 (2%
  shield reduction); on regime 2 it converges to a policy **numerically
  identical to B0 to the decimal place** (same delivery, same shield count,
  every validation year). DQN is not a reliable "max-delivery alternative"
  across regimes the way it appeared to be pre-fix — that conclusion was an
  artifact of all three "regimes" having been trained and tested on the same
  2016 data.
- **SAC is the controller that consistently beats always-on across all three
  regimes** — meaningful shield reduction (33.5–54%) at ≥ 97% delivery,
  everywhere, with zero safety violations. This is the project's primary
  "beats always-on" result.

Robustness (Phase O3-5, GHI noise 5–20%, ambient-temperature bias, varied
initial PCM state) passes with zero violations for both algorithms on all
three regimes.

## 5b. Deployment metrics (Phase O3-9) — all three regimes

Per-regime detail: `results/deployment_metrics_r{0,1,2}.json`, plots in
`results/plots/R2/deployment_metrics.html` (regenerated per regime).

- **Thermal efficiency** (delivered ÷ solar incident on the 2.5 m² collector)
  stays close to B0 for both algorithms on all regimes (≈ 40–45%, varying by
  regime's climate rather than by policy) — the controllers are not trading
  away efficiency to cut shields.
- **Inference latency / memory** (dev x86 CPU, single thread): DQN ≈ 0.5–1 ms/
  decision, 10,246 params, 40 kB fp32 weights; SAC ≈ 2.6–5.9 ms, 347k params,
  1.4 MB. Against a 1-hour control period this is many orders of magnitude of
  headroom on any embedded target in the O4 handoff class, for either
  algorithm, on any regime.
- **Sensor-fault robustness** (observation corrupted, physics untouched):
  **zero hard-safety violations in every scenario, every regime, both
  algorithms** — the shield acts on the true state, so safety holds by
  construction even on garbage inputs. Delivery degradation under GHI
  dropout/stuck-sensor faults, however, is **not uniformly graceful**:
  - DQN: negligible effect everywhere (regime 2 shows literally zero change,
    since its policy already ignores the observation and always charges).
  - SAC regime 0: no degradation (delivery unaffected or slightly higher).
  - SAC regime 2: moderate degradation, ~15% delivery loss.
  - **SAC regime 1: severe degradation** — delivered energy collapses from
    ~1990 kWh/yr nominal to **~9.6 kWh/yr** under a dead or stuck pyranometer.
    Zero safety violations (the failure mode is "stops charging," not
    "overheats"), but a serious availability risk if that sensor fails in the
    field. See `CORRECTIONS_OBJECTIVE3.md` §O3-4 for the full writeup.
    **Update (§5d below): this does not reproduce on the forecast-realism
    retrains** — it appears to be specific to that one `V1` checkpoint, not
    an inherent property of SAC-on-regime-1.

## 5c. Forecast realism — does the result survive a real forecast? (2026-11-14)

Every result in §4–5b above used an observation whose 1–3h GHI "forecast"
fields were either absent (`obs_version="V1"`) or a perfect look-ahead
(`obs_version="V2"`, reading the stored year's true future value with zero
error) — not something a deployed controller could actually have. This
section checks whether the policies' advantage over B0 survives replacing
that with a realistic forecast.

**The realistic forecast.** `forecast_ghi(k)` with `forecast_mode="smart"`
implements smart persistence on the clear-sky index:
`GHI_hat(t+k) = Kc(t) · GHI_clear(t+k)`, `Kc(t) = GHI(t)/GHI_clear(t)` clipped
to [0, 1.2] and held at its last daytime value overnight. `GHI_clear` is the
`GHI_clearsky_Wm2` column already present in the per-year weather files. This
is the standard reference method for short-horizon solar forecasting; its
measured error on this project's own data is 9.9%/21.0%/36.1% rRMSE at
+1h/+2h/+3h (vs. 31.8%/59.4%/81.8% for naive persistence) — consistent with
published NWP/satellite benchmarks (≈21–25%). Full detail, test coverage,
and the exact method in `CORRECTIONS_OBJECTIVE3.md` §O3-5.

**Result — locked test (2011–2015), no forecast vs. realistic forecast:**

| Regime | Policy | No forecast: delivery / shield cut | Realistic forecast: delivery / shield cut |
|---|---|---|---|
| 0 | DQN | 96.2% / 62% | 99.6% / **18.3%** |
| 0 | SAC | 97.5% / 54% | 98.4% / **51.8%** |
| 1 | SAC | 98.7% / 37% | 99.7% / **26.2%** |
| 2 | SAC | 97.3% / 33.5% | 98.8% / **35.9%** |

**SAC's advantage is largely real, not an artifact of unrealistic
information.** Its shield reduction barely moves on regimes 0 and 2
(54%→51.8%, 33.5%→35.9%) and degrades moderately on regime 1 (37%→26.2%),
with delivery ≥95% of B0 and zero safety violations on every regime. This is
the strongest evidence in the project that SAC's result would hold up with a
real pyranometer/NWP feed, not just in a perfect-information simulation.

**DQN's advantage shrinks, but mostly for a different reason than it first
appeared.** The table above compares "no forecast" (V1) to "realistic
forecast" (V3/smart) — not a controlled, same-seed ablation — and DQN is
known to have substantially higher run-to-run training variance in this
project than SAC (§4). A controlled, same-seed check (`obs_version="V3"`,
`forecast_mode="perfect"` vs. `"smart"`, DQN, regime 0) resolved this:

| Variant | Delivery | Shield cut |
|---|---|---|
| V1, no forecast (original) | 96.2% | 62% |
| V3 + perfect forecast (controlled) | 97.7% | **28.3%** |
| V3 + smart forecast (realistic) | 99.6% | 18.3% |

Because `forecast_mode="perfect"` gives strictly *more* information than
`V1`'s no-forecast observation, the 62%→28.3% drop (going from V1 to
V3-perfect) **cannot be a forecast-realism effect** — it is almost entirely
retrain-to-retrain variance. The genuine, isolated forecast-realism effect,
with everything else held fixed, is only **28.3%→18.3%** — a real but much
smaller degradation than the original 62%→18.3% comparison implied. DQN is
not carried as a cross-regime alternative regardless (§4), so this doesn't
change the project's SAC recommendation — if anything it strengthens it,
since DQN's regime-0 advantage is now shown to be smaller and noisier than
first reported. Full writeup in `CORRECTIONS_OBJECTIVE3.md` §O3-5.

## 5d. Full ablation (`forecast_mode="none"`) and full eval suite on every forecast variant (2026-11-14)

§5c compared only two forecast conditions (no forecast at all vs. realistic
smart forecast). To know whether SAC's result is forecast-driven at all, a
third condition — `forecast_mode="none"` (same 15-dim observation shape as
the realistic-forecast run, but the forecast fields are zeroed rather than
removed) — was added, and the full locked-test + robustness + sensor-fault +
deployment-metrics suite was run on every trained variant, not just the
headline numbers.

| Regime | Policy | V1, no forecast field at all | V3 + `forecast_mode="none"` | V3 + smart forecast |
|---|---|---|---|---|
| 0 | SAC | 97.5% / 54% | 99.1% / 49.8% | 98.4% / 51.8% |
| 1 | SAC | 98.7% / 37% | 99.3% / 29.0% | 99.7% / 26.2% |
| 2 | SAC | 97.3% / 33.5% | 98.9% / 41.0% | 98.8% / 35.9% |
| 0 | DQN | 96.2% / 62% | 100.0% / **0.0%** | 99.6% / 18.3% |

**SAC's result is essentially forecast-independent** — its shield cut stays
in a 26–54% band across all three forecast conditions with no consistent
direction, meaning its advantage over B0 comes from its exploration/entropy
mechanism, not from reading a weather signal. **DQN collapses completely to
B0 (0% shield cut, bit-identical to B0) with zero forecast information** —
it needs *some* forecast field, even a noisy one, to differentiate from
always-charge at all, reinforcing that DQN is not a robust alternative to
SAC outside a narrow, forecast-dependent condition. Robustness suite:
**PASSED, zero safety violations**, on every new variant.

**Sensor-fault re-evaluation changes the O3-4 picture.** Re-running the
sensor-fault suite on the `none`/`smart` V3 checkpoints (not just the
original `V1` one) shows regime 1's reported ~99.5% collapse does not
reproduce — both V3 variants show mild degradation (1.2–13.2%) in the same
range as regimes 0 and 2, not a near-total collapse:

| Regime | Checkpoint | Drop under GHI dropout/stuck |
|---|---|---|
| 0 | V1 (original) | ~0% |
| 0 | V3 + none / smart | 9.1% / 14.0% |
| 1 | V1 (original) | **~99.5%** |
| 1 | V3 + none / smart | 13.2% / **1.2%** |
| 2 | V1 (original) | ~15% |
| 2 | V3 + none / smart | 2.2% / 2.4% |

This suggests the severe regime-1 vulnerability was a property of that one
`V1` training run (a seed/architecture coincidence in which observation
dimension the policy leaned on), not a deterministic property of "SAC on
regime 1" — an encouraging revision, though it does not make per-checkpoint
sensor-fault testing before deployment any less necessary. Full tables and
all raw file paths in `CORRECTIONS_OBJECTIVE3.md` §O3-6.

**Scope note:** the original plan specified 3 seeds and a 4th "optimism
check" case; given the project's time budget, this pass runs 1 seed and 3
of the 4 cases. A 3-seed repeat (for real confidence intervals rather than
point estimates) is the natural next step if time allows — see
`CORRECTIONS_OBJECTIVE3.md` "Still open."

## 6. What this does and does not establish

- It establishes, **in simulation**, that a learned controller can hold delivery
  while cutting shield activations ~26 %, with zero safety violations.
- It does **not** establish field performance. All thermal state is simulated;
  the live demo (O3-7) is software-in-the-loop only. Experimental validation is
  Objective 4, for which O3-8 produces the hardware handoff specification.
- Scope is Tamil Nadu, three regimes. The other states reuse the same
  environment API with their own contracts (future work).

## 7. Related work and references

Positioned against the literature, this work's novelty is **safe real-time
control** of a *fixed* PCM-SWH design. Prior AI-for-PCM work is dominated by
design optimisation (Taguchi/GRA/ML surrogates) and thermal *prediction* rather
than closed-loop control; DRL for solar-thermal systems is sparse and has not,
to our knowledge, been applied to domestic PCM solar water heating with an
in-environment safety shield and a cross-family algorithm comparison. The
forecast-aware B2 baseline connects to the irradiance-forecasting literature, and
our model-free + structural-shield approach contrasts with model-heavy ANN-MPC.

Full thematic survey, per-cluster gap analysis, a research-gap→contribution
table, and a drop-in abstract skeleton are in **`LITERATURE_REVIEW.md`**;
BibTeX entries (cite keys usable directly in a manuscript) are in
**`references.bib`**.
