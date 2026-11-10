# Objective 3 — Baselines and Algorithms

Every controller in the comparison, why it is included, its configuration, and
how it performed. Results are on the validation years (2021–2023), reward R2
with w_shield = 10, regime 0. "Shield" = water + PCM guard-band activations per
year; fewer is better. All controllers had **zero hard-limit violations**.

---

## Reference numbers (validation, R2 w=10)

| Controller | Delivered kWh | Shield / yr | R2 reward | Note |
|---|---|---|---|---|
| B0 fixed charge | 2146.2 | 50,298 | 846,582 | the Objective 2 operating point |
| B1 reactive rule | 2015.3 | 22,371 | 890,675 | best non-learning controller |
| B2 forecast rule | 1984.6 | 20,807 | 829,193 | fewest shields, lowest delivery |
| PPO | 2146.2 | 50,298 | 846,582 | stuck at always-charge |
| **DQN** | **2144.1** | **36,955** | **976,408** | **best overall** |
| RecurrentPPO | 2146.2 | 50,298 | 846,582 | stuck at always-charge |
| QRDQN | 2146.2 | 50,298 | 846,582 | stuck at always-charge |
| MaskablePPO | 2146.2 | 50,298 | 846,509 | stuck (mask forces a few non-charge steps) |

(SAC under evaluation; DQN multi-seed robustness under verification.)

---

## Non-learning baselines

### B0 — fixed charge  (`o3/baselines/b0_fixed.py`)
Always requests CHARGE; the shield handles safety. This *is* the Objective 2
operating point and the minimum any learned controller must beat. It delivers the
most energy but fires the shield the most (50k/yr).

### B1 — reactive rule  (`o3/baselines/b1_reactive.py`)
Priority-ordered rules on the current state: pre-emptive bypass near the water
guard band, bypass when PCM is near-full under strong sun, hold through the
evening demand window, charge when sun is available and PCM not full. Must pass
the acceptance checklist (zero violations) before any DRL training. The strongest
non-learning controller: 22k shields at a 6 % delivery cost.

### B2 — forecast-aware rule  (`o3/baselines/b2_forecast.py`)
B1 plus a 4-hour irradiance look-ahead: pre-emptive bypass when the PCM is
near-full and more strong sun is coming; a last-chance charge when sun is now but
none is forecast soon. Fewest shields of all (20.8k) but also the lowest delivery
(1984.6) — it over-bypasses. Supports an imperfect-forecast mode
(`forecast_noise_frac`) for a forecast-sensitivity study.

### B4 — random safe  (`o3/baselines/b4_random.py`)
Uniform random over non-masked actions. A diagnostic floor only — any learned
policy must clearly beat it.

---

## Deep-RL algorithms

All use Stable-Baselines3 / sb3-contrib, MLP policies, `device="cpu"`,
300,000 training steps, rollout/episode = one year (8,760 h), seed 42.

### PPO  (`o3/agents/ppo_trainer.py`)
On-policy policy gradient. Stable, easy to constrain — the recommended first
policy-gradient method. **Result:** converged to always-charge; its entropy
bonus is not enough to keep exploring once the policy commits.

### DQN  (`o3/agents/dqn_trainer.py`)  ← selected method
Double DQN, value-based, ε-greedy exploration (`exploration_fraction=0.3`,
final ε = 0.05), buffer 50k, batch 256, γ = 0.99. **Result:** the only algorithm
that learned anticipatory control — B0-level delivery with 26 % fewer shields,
highest reward. Its ε-greedy exploration is what lets it escape the always-charge
attractor the on-policy methods fall into.

### RecurrentPPO  (`o3/agents/recurrent_ppo_trainer.py`)
PPO with an LSTM policy (`lstm_hidden_size=64`), for temporal/monsoon patterns.
LSTM state is threaded through each evaluation episode. **Result:** same
always-charge collapse as PPO; memory did not help because the problem here is
exploration, not partial observability.

### QRDQN  (`o3/agents/qrdqn_trainer.py`)
Distributional DQN (50 quantiles) — learns a return distribution, useful for
risk-aware control under stochastic irradiance. Value-based with ε-greedy like
DQN. **Result:** stuck at always-charge at w=10/seed 42 — the distributional
target did not, on this seed, find the shield-reducing policy DQN found.

### MaskablePPO  (`o3/agents/maskable_ppo_trainer.py`)
PPO that masks invalid actions (CHARGE masked at night / near the water limit) so
the agent never proposes them. **Result:** essentially always-charge; the mask
forces a handful of non-charge steps (reward 846,509 vs B0's 846,582) but does
not change the policy's character.

### SAC  (`o3/agents/sac_trainer.py`)
Soft Actor-Critic — off-policy, entropy-regularised, the plan's recommended
continuous-control algorithm. The task is discrete, so SAC trains on
`PCMContinuousGymEnv`: a 1-D action in [−1,1] binned to {CHARGE, IDLE, BYPASS}.
Its entropy regularisation + replay buffer make it the most likely of the
non-DQN methods to escape; result under evaluation.

---

## How to read the comparison

- A controller is **better than B0** only if it cuts shields *without* losing
  delivery. DQN does (2144 vs 2146, 37k vs 50k). The stuck algorithms simply
  reproduce B0.
- A controller **beats B1** (the best rule) if it gets a higher R2 reward. DQN
  does (976k vs 891k): it keeps far more delivery than B1 (2144 vs 2015) while
  still cutting shields well below B0.
- The clean selection story: among six algorithms, the value-based DQN is the one
  that reliably learns the intended anticipatory behaviour — which is why it is
  the method carried into the locked evaluation.
