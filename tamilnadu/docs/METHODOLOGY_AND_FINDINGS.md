# Objective 3 — Methodology and Key Findings

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

## 4. Why DQN wins and the others stall

| Family | Exploration | Outcome |
|---|---|---|
| DQN, QRDQN (value-based) | ε-greedy (forced random actions, 30 % of training) | DQN escaped the always-charge basin; QRDQN did not at w=10 |
| PPO, RecurrentPPO, MaskablePPO (on-policy) | entropy bonus | entropy collapses once the policy commits to always-charge; no escape |
| SAC (off-policy, continuous) | entropy-regularised + replay | under evaluation |

The always-charge policy is a strong, wide attractor (it is the obvious
high-delivery policy any agent finds first). Escaping it requires *trying*
bypass-heavy behaviour long enough to discover the reward is higher. DQN's
ε-greedy keeps taking random actions regardless of how confident the policy is,
so it stumbles into the better region; the on-policy methods stop exploring once
their policy sharpens, so they never see it.

**Third finding: for this shielded, discrete-action problem, value-based DRL with
forced exploration is the right method; on-policy policy-gradient methods are a
poor fit.** This is exactly the kind of "why this algorithm" evidence the plan
asked for — it is grounded in observed behaviour, not assumed.

## 5. Robustness and selection

DQN's escape at w=10 was first seen on one seed. To claim "DQN is the method" we
verify it across seeds (0, 123) — a single lucky seed is not a method. The
selected policy for the locked evaluation is chosen by a lexicographic rule:
first zero hard violations, then delivery reliability, then shield reduction —
never by reward alone.

## 6. What this does and does not establish

- It establishes, **in simulation**, that a learned controller can hold delivery
  while cutting shield activations ~26 %, with zero safety violations.
- It does **not** establish field performance. All thermal state is simulated;
  the live demo (O3-7) is software-in-the-loop only. Experimental validation is
  Objective 4, for which O3-8 produces the hardware handoff specification.
- Scope is Tamil Nadu, three regimes. The other states reuse the same
  environment API with their own contracts (future work).
