# Objective 3 — Climate-Adaptive DRL Controller for PCM Solar Water Heating (Rajasthan)

Status 2026-11-11: **code, tests and smoke runs done; full training, the shield-weight sweep and the locked test are not run.**
Plan: `docs/objective3_plans/Objective_3_Implementation_Plan_FINAL.docx`. Deviations: `CORRECTIONS.md`.

## 1. What this objective does
Learns (and compares with rules) an hourly controller that decides whether the solar collector loop circulates through a
PCM-in-tank store, around O2's **unchanged** simulator and in-kernel safety shield (water 72 C / PCM 62 C guard bands, limits 75 / 65 C).
It may not change the PCM, capsules, tank, collector, limits or physics.

## 2. The three Rajasthan regimes (frozen from O2; 50 L tank, 1.5 m² collector, 45 C delivery)

| Regime | Medoid | PCM (O2 choice) | Capsules | Flow (kg/s) | Note |
|---|---|---|---|---|---|
| 0 | RJP_0132 Udaipur | savE OM55 | 52 | 0.0155 | O1 candidate pool undersized (n = 4) |
| 1 | RJP_0192 Nagaur | PureTemp 60 | 52 | 0.0493 | first prototype regime (healthy pool; O2 = O1 #1) |
| 2 | RJP_0083 Jaipur | PureTemp 58 | 52 | 0.0236 | |

## 3. Environment, observation, action, reward
* `PCMEnv` (`o3/environment/pcm_env.py`) steps O2's multi-node kernel one hour at a time (`o2_adapter.py`); equal to O2's `run_case` to 1e-9.
* Observation: the 13 contract fields (`obs.py`), fixed scaling. Action: Discrete(3) CHARGE / IDLE / BYPASS (IDLE ≡ BYPASS physically).
* Reward R2: delivered-to-45 C − 2·unmet − 50·hard violations − 0.1·switch − w_shield·shield activations, `w_shield` = 3.9 / 1.7 / 1.7
  (re-derived, provisional). Splits: train 2016–20, validate 2021–23, test 2006–10 (fresh) behind `o3_test.lock`.

## 4. Baselines and algorithms
Twin, B0, B1, B2, B4; PPO, MaskablePPO, RecurrentPPO, DQN, QRDQN, SAC — `ALGORITHMS.md`.

## 5. What is known so far (regime 1, measured)
B0 beats the water-only twin by +134…+152 kWh/yr in the 18–09 window with ~500 h/yr water-side and ~550 h/yr PCM-side shield time;
untuned TN rules lose 12–48 kWh versus the twin; no trained agent has been evaluated. See `METHODOLOGY_AND_FINDINGS.md`.

## 6. Phases and commands
| Phase | Script | Flags |
|---|---|---|
| 0 contract audit, test weather | `phase0_audit.py`, `phase0_fetch_test_weather.py` | — |
| 1 acceptance (extra) | `phase1_acceptance.py` | — |
| 2 baselines, calibration, B1 tuning | `phase2_baselines.py` | `--n-jobs` |
| 3 dataset | `phase3_generate_dataset.py` | `--n-jobs` |
| 4 training / multi-seed / re-score | `phase4_train.py`, `phase4_multiseed.py`, `phase4_reeval.py` | `--n-envs`, `--n-jobs` |
| 5 robustness | `phase5_robustness.py` | `--n-jobs` |
| 6 locked test (once) | `phase6_locked_eval.py` | `--n-jobs` |
| 7 demo | `phase7_demo.py`, `app/streamlit_app.py` | — |
| 8 O4 hand-off spec | `phase8_o4_handoff.py` | — |
| plots, benchmark | `make_plots.py`, `benchmark_speedup.py` | — / `--n-jobs --n-envs` |
Details and timings: `HOW_TO_RUN.md`.

## 7. Safety status and scope
Zero hard-limit hours in every run so far; the shield is inside the kernel and cannot be overridden (test T6). Simulation only —
no hardware, no field validation (that is Objective 4). The demo is software-in-the-loop.

## 8. Directory map
```
rajasthan/
  config_O2.py  conftest.py  requirements.txt
  contracts/obj3_environment_contract_rajasthan_v3.json      (copy of the O2-built file, hashed)
  o3/environment/{pcm_env,o2_adapter,obs,gym_wrapper}.py
  o3/baselines/{b0_fixed,b1_reactive,b2_forecast,b4_random}.py
  o3/agents/{common,ppo,maskable_ppo,recurrent_ppo,dqn,qrdqn,sac}_trainer.py
  o3/data/trajectory_generator.py   o3/demo/live_adapter.py
  o3/evaluation/{rollout,robustness}.py
  o3/utils/{contract_loader,weather_roles,parallel,cli,models}.py
  scripts/phase0..phase8 *.py, make_plots.py, benchmark_speedup.py
  tests/   results/   docs/
```
