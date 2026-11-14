# Objective 3 — Baselines and Algorithms

> **Correction notice (2026-11-13):** the numbers below were retrained after
> fixing a critical weather-loading bug — see `CORRECTIONS_OBJECTIVE3.md`.
> Earlier numbers on this page (e.g. "DQN 2144.1 kWh / 36,955 shields") were
> computed on identical 2016 data disguised as different years and are
> retired. The qualitative conclusion also changed: DQN is no longer a
> reliable cross-regime alternative to SAC — see below.

Every controller in the comparison, why it is included, its configuration, and
how it performed. Reference numbers are on the validation years (2021–2023),
reward R2 with w_shield = 10, **regime 0** (regimes 1/2 differ — see the
cross-regime summary). "Shield" = water + PCM guard-band activations per
year; fewer is better. All controllers had **zero hard-limit violations**.

---

## Reference numbers (validation, R2 w=10, regime 0)

| Controller | Delivered kWh | Shield / yr | Note |
|---|---|---|---|
| B0 fixed charge | 2096.9 | 46,969 | the Objective 2 operating point |
| B1 reactive rule | ~2000 | ~19,000 | best non-learning controller |
| **DQN** | **~2000** | **~15,500** | **learned — regime 0 only, see below** |
| **SAC** | **~2000** | **~18,000** | **selected — consistent across all regimes** |

(PPO, RecurrentPPO, QRDQN, MaskablePPO were not retrained on corrected data —
they consistently reproduced B0-identical behaviour pre-fix and were
deprioritized under the project's time constraints; see
`CORRECTIONS_OBJECTIVE3.md` §"Still open".)

## Cross-regime summary — the real selection story

The locked-test result (2011–2015, the authoritative comparison) shows DQN and
SAC behave very differently depending on regime:

| Regime | DQN vs B0 | SAC vs B0 |
|---|---|---|
| 0 | **Learns anticipatory control** — 62% fewer shields, 96.2% delivery | Learns anticipatory control — 54% fewer shields, 97.5% delivery |
| 1 | Barely differentiates — 2% fewer shields (essentially B0) | **Learns anticipatory control** — 37% fewer shields, 98.7% delivery |
| 2 | **Does not differentiate** — numerically identical to B0 | **Learns anticipatory control** — 33.5% fewer shields, 97.3% delivery |

**Selected policy: SAC, for all three regimes.** Pre-fix, DQN appeared to be a
validated "max-delivery alternative" to SAC across the board. On corrected
data that conclusion does not hold — DQN only reliably escapes the
always-charge policy on regime 0. SAC is the controller that actually beats
always-on consistently, and is therefore the project's primary result. SAC
does carry a regime-1-specific sensor-fault caveat (see
`CORRECTIONS_OBJECTIVE3.md` §O3-4) that should be weighed against DQN's safer
but less effective regime-1 behaviour if GHI sensor reliability is a real
field concern.

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

### DQN  (`o3/agents/dqn_trainer.py`)  ← regime-0 only, see cross-regime summary above
Double DQN, value-based, ε-greedy exploration (`exploration_fraction=0.3`,
final ε = 0.05), buffer 50k, batch 256, γ = 0.99. **Result:** on regime 0,
learns anticipatory control — 96.2% delivery with 62% fewer shields. On
regimes 1 and 2 (corrected data) it does **not** reliably escape the
always-charge attractor — regime 2's DQN policy is numerically identical to
B0. Its ε-greedy exploration lets it escape the attractor on regime 0's reward
landscape specifically, but that escape did not generalise to the other two
regimes' landscapes.

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

### SAC  (`o3/agents/sac_trainer.py`)  ← selected method, all regimes
Soft Actor-Critic — off-policy, entropy-regularised, the plan's recommended
continuous-control algorithm. The task is discrete, so SAC trains on
`PCMContinuousGymEnv`: a 1-D action in [−1,1] binned to {CHARGE, IDLE, BYPASS}.
**Result:** the only controller that reliably beats always-on on **all three**
regimes — 33.5–54% fewer shields at 97.3–97.5% delivery, zero safety
violations everywhere. Its entropy regularisation + replay buffer provide
sustained exploration that generalises across regimes better than DQN's
ε-greedy did in this corrected run. PASSED the locked 2011–2015 test and the
robustness suite with zero violations on every regime. **Caveat:** on regime 1
specifically, its policy is unusually dependent on the raw GHI observation —
under a simulated pyranometer failure, delivered energy collapses to ~9.6
kWh/yr (see `CORRECTIONS_OBJECTIVE3.md` §O3-4). Not observed on regimes 0 or 2.

### Forecast realism (2026-11-14)

All numbers above use `obs_version="V1"` (SAC/DQN headline) with **no
forecast fields in the observation at all** — not a perfect-forecast
variant, despite earlier drafts of this doc's wording. Retraining both with
a 15-dim observation carrying a realistic 1–3h GHI forecast
(`obs_version="V3"`, `forecast_mode="smart"` — smart persistence on the
clear-sky index, not the stored year's true future value) gives:

| Regime | Policy | No forecast: shield cut | Realistic forecast: shield cut |
|---|---|---|---|
| 0 | DQN | 62%¹ | **18.3%** |
| 0 | SAC | 54% | **51.8%** |
| 1 | SAC | 37% | **26.2%** |
| 2 | SAC | 33.5% | **35.9%** |

¹ A controlled same-seed check (same `obs_version="V3"` shape,
`forecast_mode="perfect"`) found DQN regime-0's true no-variance baseline is
28.3%, not 62% — most of that 62%→18.3% gap is retrain variance, and the
genuine forecast-realism effect is 28.3%→18.3%. See §O3-5 below.

SAC's result barely moves — strong evidence it isn't exploiting information
a deployed controller wouldn't have. DQN's 62%→18.3% drop looked like a
forecast-realism effect, but a controlled same-seed check (same
`obs_version="V3"` shape, `forecast_mode="perfect"` instead of `"smart"`)
showed the picture is mostly retrain variance, not forecast realism: that
same seed, with a *perfect* forecast (strictly more info than the original
no-forecast run), only gets 28.3% — so the 62%→28.3% portion of the drop is
noise, and the genuine forecast-realism effect is just 28.3%→18.3%. DQN is
not carried as a cross-regime alternative regardless, so this doesn't change
the SAC recommendation. A third condition, `forecast_mode="none"` (same V3
observation shape, forecast fields zeroed), makes the picture sharper still:
SAC's shield cut stays in a 26–54% band regardless of forecast condition
(forecast-independent), while DQN regime 0 collapses to **exactly 0%** —
bit-identical to B0 — with zero forecast signal, confirming DQN needs *some*
forecast field to differentiate from always-charge at all. Full detail in
`CORRECTIONS_OBJECTIVE3.md` §O3-5–O3-6.

---

## How to read the comparison

- A controller is **better than B0** only if it cuts shields *without* losing
  more than 5% delivery. SAC does this on every regime. DQN does it on regime
  0 only; on regimes 1–2 it simply reproduces B0.
- The clean selection story, corrected: sustained exploration (DQN's
  ε-greedy, SAC's entropy regularisation + replay) is necessary to escape the
  always-charge attractor, but on this corrected multi-regime data it was not
  *sufficient* for DQN outside regime 0 — the same exploration mechanism that
  worked on regime 0's reward landscape did not generalise to regimes 1/2's.
  SAC's mechanism did generalise. The lexicographic rule (zero violations →
  delivery ≥ 95% of B0 → maximum shield reduction) selects **SAC for all
  three regimes**; DQN is not carried forward as a cross-regime alternative,
  though it remains a valid regime-0-only result.
