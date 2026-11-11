# Objective 3 — Baselines and Algorithms (Rajasthan)

What each controller is, why it is in the comparison, how it is configured, and where its code lives.
**No trained-agent result is reported here: full training has not been run** (see `HOW_TO_RUN.md` for what it costs).
The only numbers below are the deterministic baselines, which are cheap and have been run.

Success is judged on the plan's metrics (plan §1), never on reward alone: hard-limit violations (must be 0), delivered
energy to 45 C in the 18:00–09:00 window **versus the water-only twin** (`gain_window_kWh`), shield hours, bypass
switches, energy residual (< 0.1 %). Rewards in these tables are shown only because every script prints them.

Sources: plan `.claude/Objective_3_Implementation_Plan_FINAL.docx` §5 (algorithm roster, precedents) and §16.8 (as-built
roster); DRL-for-solar-thermal-storage precedent `.claude/Sources/Emami2026DRL_Solar_ORC_TES_summary.md` (DDPG supervisor on
8,760 h of hourly solar data, paraffin-PCM storage, safety filter between policy and plant — the same structure as here).

---

## Reference numbers — baselines, regime 1 (Nagaur, PureTemp 60, 52 capsules), validation years 2021–2023

Run with `python scripts/phase2_baselines.py --regime 1 --years 2021 2022 2023` (identical serial and parallel).

| Controller | E to 45 C (kWh/yr) | Window 18–09 (kWh) | **Window gain vs twin (kWh)** | Water-shield h | PCM-shield h | Hard-limit violations |
|---|---|---|---|---|---|---|
| Water-only twin | 781 | 461 | — | 800 | 0 | 0 |
| **B0** always on (O2 as released) | 926 | 598 | **+137** | 494 | 549 | 0 |
| B1 reactive rule (TN thresholds, untuned) | 773 | 446 | −15 | 40 | 0 | 0 |
| B2 forecast rule (perfect 4 h forecast, untuned) | 744 | 420 | −41 | 42 | 0 | 0 |
| B4 random (safe) | ≈ 485 | ≈ 249 | ≈ −219 | 11 | 0 | 0 |

(means over the three years; B4 from training years 2016–2018.) Reading: the released O2 design with its passive
shield beats its water-only twin by ~137 kWh in the night window. The untuned rules cut the shield time by 90 % **but give
up more delivery than they save** — the opposite of Tamil Nadu's finding for the same rules (see `CORRECTIONS.md` D4, D15).
Whether any on/off policy beats B0 on this metric is plan gate **G2, which is still open**.

---

## Non-learning baselines

### Water-only twin  (`o3/evaluation/rollout.py::evaluate_twin`)
Same tank, flow, shield, weather and demand with zero capsules (the O2 baseline convention, `run_case(pcm_name=None)`).
It is the reference every gain is measured against (plan §1, §15.6). Not a controller.

### B0 — always on  (`o3/baselines/b0_fixed.py`)
Loop enabled every hour; the shield in the kernel acts when it must. This **is** the O2 released design;
`test_T1` proves a stepped B0 year equals one O2 `run_case` year. Any learned policy must beat it to claim improvement.

### B1 — reactive rule  (`o3/baselines/b1_reactive.py`)
Priority rules on the current state: top node ≥ 70 C → bypass; PCM ≥ 60 C with > 500 W/m² → bypass; evening 17–21 h with
f_melt ≥ 0.15 → idle (let the PCM discharge); GHI ≥ 200 W/m² and f_melt < 0.9 → charge; default idle. The thresholds are
TN's (2 C under the shield). Plan §15.6: tune on **validation years only** by grid search —
`python scripts/phase2_baselines.py --regime 1 --years 2021 2022 2023 --tune` runs the 24-point grid
(`B1_GRID`) and ranks it lexicographically (zero violations → window delivery → shield hours → switches); every grid point is
written to `results/phase2_b1_tuning_r<regime>.csv`. Not yet run.

### B2 — forecast-aware rule  (`o3/baselines/b2_forecast.py`)
B1 plus a 4-hour irradiance look-ahead read from the environment's own weather (perfect forecast; `forecast_noise_frac` for an
imperfect one). It uses information a real controller would need a forecast feed for (an O4 question). The plan lists B2/B3
as deferred (plan §5); B2 stays because TN built it. B3 (MPC) is absent and its absence must be stated in the paper
(plan §5: probably the strongest baseline in the literature).

### B4 — random safe  (`o3/baselines/b4_random.py`)
Uniform over unmasked actions; diagnostic and the "random search" half of gate G2. Seeded → reproducible (T5).

---

## Deep-RL algorithms (Stable-Baselines3 2.9.2 / sb3-contrib 2.9.2, CPU)

All six train on `PCMGymEnv`: 13-field observation, `Discrete(3)` {CHARGE, IDLE, BYPASS}, one-year episodes drawn from
2016–2020 with the reset scenario drawn from {solid, partial, liquid}; validation on 2021–2023 (`EvalCallback`, all three
years, best checkpoint by validation reward saved). Hyper-parameters are the TN values (no search was run, plan §15.7).
Seeds: ≥ 3 for anything reported (plan §5). Hard-limit protection is the kernel's shield, never the policy.

| Algorithm | File | Family | Parallelism (`--n-envs`) | Why it is here |
|---|---|---|---|---|
| PPO | `o3/agents/ppo_trainer.py` | on-policy actor-critic | SubprocVecEnv, n_envs workers | the plan's prototype algorithm (§5, §15.7) |
| MaskablePPO | `o3/agents/maskable_ppo_trainer.py` | on-policy + action masks | SubprocVecEnv | physically pointless actions (charge at ≥ 68 C or without sun) masked by construction |
| RecurrentPPO | `o3/agents/recurrent_ppo_trainer.py` | on-policy, LSTM(64) | SubprocVecEnv | multi-day cloud/demand patterns; the contract observation has no forecast |
| DQN | `o3/agents/dqn_trainer.py` | off-policy, double DQN | 1 env (DummyVecEnv) | value-learning baseline |
| QRDQN | `o3/agents/qrdqn_trainer.py` | off-policy, distributional (50 quantiles) | 1 env | stands in for Rainbow (not in SB3); risk-sensitive view of cloudy-day returns |
| SAC | `o3/agents/sac_trainer.py` | off-policy, entropy-regularised | 1 env | continuous-control comparison on the binned 1-D action view (`PCMContinuousGymEnv`) |

Parallel training: PPO-family `n_steps` is the TOTAL per update (8,760 = one year) split over the workers
(`rollout_steps_per_env`), so the batch matches the serial run's; off-policy agents keep one environment because extra
environments would change their updates-per-sample ratio. Reproducible for a fixed (seed, n_envs); a SubprocVecEnv and a
DummyVecEnv with the same n_envs and seed learn identical weights (`tests/test_parallel.py`).

PPO/DQN/SAC defaults: PPO `lr 3e-4, batch 256, 10 epochs, γ 0.99, λ 0.95, clip 0.2, ent 0.01`; MaskablePPO `ent 0.005`;
DQN/QRDQN `lr 1e-4, buffer 50k, batch 256, train_freq 4, target update 1000, ε 1→0.05 over 30 %`; SAC `lr 3e-4, buffer 100k,
τ 0.005, auto entropy`.

---

## How to read a comparison

1. Hard-limit violations first (must be 0 for every row — the kernel's shield makes this nearly automatic).
2. `gain_window_kWh` against B0 (+137 kWh in regime 1) and against the twin (0). A policy below B0 on this column has not
   improved the system, whatever its reward says.
3. Shield hours and bypass switches as secondary (shield time is wasted collector potential; switches are actuator wear).
4. Mean ± spread over seeds, per year and per reset scenario, never pooled only (plan §1).
5. The reward is a training signal. R2's shield term makes "do less" attractive; that is why the shield weight was re-derived
   per regime (`CORRECTIONS.md` D4) and why it must be swept before a final run.

What is deliberately missing (plan §11.1, deferred): MPC and MPC-guided RL, constrained/Lagrangian RL, model-based RL,
offline RL, hybrid [mode, flow] actions (needs F_R(flow) enabled as a declared physics change), forecasts as inputs,
pooled/meta policies across states.
