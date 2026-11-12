# Objective 3 (Rajasthan) — How to Run

Setup, every command, the two parallelism flags, measured speed-ups, compute estimates for the full runs, troubleshooting.
Run everything from `solar-pcm-control--Objective3/rajasthan/`.

## 1. Environment

* Python 3.10, CPU only. `pip install -r requirements.txt` (numpy, pandas, numba, joblib, psutil, torch, gymnasium,
  stable-baselines3, sb3-contrib, plotly, pytest; optional folium). Versions used: gymnasium 1.4.0, SB3 2.9.2, sb3-contrib 2.9.2.
* The O2 Rajasthan tree is found automatically at `../../OBJECTIVE2/Objective---2/objective2-rajasthan` (override with
  `O2_RAJASTHAN_ROOT`). Nothing in it is edited by O3.
* `O3_RESULTS_DIR=<folder>` redirects all outputs (use it for smoke tests so `results/` stays clean).
* Windows: prefix long runs with `PYTHONIOENCODING=utf-8`.
* TensorFlow/tensorboard is blocked on import (`o3/__init__.py`): SB3 would otherwise load TensorFlow in every worker
  (~1 GB each) and 10 workers exhausted this laptop's memory. Set `O3_ALLOW_TENSORBOARD=1` to undo.

## 2. Parallelism flags (every phase script)

| flag | meaning | default |
|---|---|---|
| `--n-jobs N` | worker **processes** for independent jobs (seeds, regimes, years, scenarios); `1` = serial in-process | auto = physical cores − 1 (11 on this machine) |
| `--n-envs N` | environments per **on-policy** training run (PPO, MaskablePPO, RecurrentPPO; SubprocVecEnv). Off-policy agents use 1. | auto = physical cores − 1 |

A job depends only on its own arguments (own seed, torch/BLAS threads pinned to 1), so deterministic results are identical for any
`--n-jobs`. Training is reproducible for a fixed (seed, n_envs). Scripts that cannot use a flag accept and ignore it (stated in
their docstring). Which phase uses what: 2/3/5/6 and `phase4_reeval` → `--n-jobs`; `phase4_train` → both (jobs for baselines and
scoring, envs for training); `phase4_multiseed` → `--n-jobs` (one process per algorithm×seed, 1 env each); 0, 1, 7, 8, plots → none.

## 3. Verify, then test

```bash
python scripts/phase0_fetch_test_weather.py   # once: 2006-2010 test weather (network); --no-download re-checks
python scripts/phase0_audit.py            # contract: v3, 50 L / 1.5 m2, provenance hashes
python scripts/phase1_acceptance.py       # the contract's six acceptance checks -> pytest nodes (~4 min)
python -m pytest -q                       # full suite, ~8 min
```

## 4. Baselines (Phase O3-2), shield-weight calibration, B1 tuning

```bash
python scripts/phase2_baselines.py --regime 1 --years 2021 2022 2023            # twin, B0, B1, B2, B4
python scripts/phase2_baselines.py --regime 1 --verify-parallel                 # serial vs pool, identical?, speed-up
python scripts/phase2_baselines.py --regime 1 --calibrate-shield                # break-even shield weight, training years
python scripts/phase2_baselines.py --regime 1 --years 2021 2022 2023 --tune     # B1 grid, VALIDATION years only (24 points)
python scripts/phase2_headroom_gate.py --regime 1 --n-random 60 --n-jobs 6      # gate G2: does any on/off rule beat B0?
python scripts/phase2_baselines.py --reward-norm                                # twin normalisation for --reward-mode Q2
```
Expect zero hard-limit violations everywhere, B0 largest window gain, B1/B2 fewest shield hours (`ALGORITHMS.md`).

## 5. Dataset (Phase O3-3) and training (Phase O3-4)

```bash
python scripts/phase3_generate_dataset.py --regime 1 --n-jobs 11
python scripts/phase4_train.py --regime 1 --steps 300000 --algo ppo maskable recurrent --n-envs 10 --n-jobs 11
python scripts/phase4_train.py --regime 1 --steps 300000 --algo dqn qrdqn sac               # off-policy: 1 env each
python scripts/phase4_train.py --regime 1 --steps 3000 --algo all --results-dir C:\tmp\o3smoke --n-envs 2   # smoke
python scripts/phase4_train.py ... --w-shield 0.5          # R2 shield weight; also --w-switching/--w-unmet/--w-safety
python scripts/phase4_multiseed.py ... --reward-mode Q2 --w2 0.1 --w4 0.1   # plan section-6 reward form (twin-normalised)
python scripts/phase4_multiseed.py --regime 1 --steps 300000 --seeds 42 0 123 --algo ppo maskable --n-jobs 6
python scripts/phase4_reeval.py --regime 1 --n-jobs 11     # re-score saved models without retraining
```
`phase4_multiseed.py` freezes the best seed per algorithm (lexicographic rule, validation years) into
`results/phase4_selected_r<regime>.json` with its SHA-256.

## 5a. Generation-2 study (plan §19) and deployment metrics (Phase O3-9)

```bash
python scripts/train_one.py --algo dqn --regime 1 --steps 300000 --seed 42 --shield-weight 1.7 --obs-version V3 --control-step-min 60 --domain-randomize
python scripts/gen2_study.py --algo dqn --study A B C --regime 1 --n-jobs 6     # 18 runs, ~35 min wall (6 parallel, 9-11 min each)
python scripts/gen2_study.py --algo sac --study A --multipliers 1 --n-jobs 3    # SAC: ~30 steps/s, hours
python scripts/gen2_study.py --algo dqn --summarise-only                        # re-score + selection (plan 18.1)
python scripts/phase9_deployment_metrics.py --regime 1 --algo dqn --n-jobs 5
python scripts/train_one.py --algo dqn --regime 1 --seed 42 --shield-weight 1.7 --obs-version V2          # ablation: no forecast
python scripts/train_one.py --algo dqn --regime 1 --seed 42 --shield-weight 1.7 --no-domain-randomize     # ablation: no randomisation
python scripts/gen2_ablation.py --regime 1 --n-jobs 3                                                      # scores them + robustness
python scripts/phase5_robustness.py --run results/gen2/<tag> --years 2021 2022 --n-jobs 6                  # robustness of any Gen-2 run
python scripts/make_gen2_plots.py --algo dqn
python scripts/phase6_locked_eval.py --run results/gen2/<selected tag> --n-jobs 5    # the one-shot locked test, 2006-2010
```

## 5b. Robustness (Phase O3-5)

```bash
python scripts/phase5_robustness.py --regime 1 --algo ppo --years 2021 2022 2023 --n-jobs 11   # 13 scenarios
python scripts/phase5_robustness.py --regime 1 --algo b1
```

## 6. Locked final evaluation (Phase O3-6) — run ONCE

```bash
python scripts/phase6_locked_eval.py --algo ppo --regime 0 1 2 --n-jobs 11
```
Takes `results/o3_test.lock` (exclusive create) before loading a single test hour and refuses to run again. Test years are
2006–2010 (fresh, downloaded by `phase0_fetch_test_weather.py`; `CORRECTIONS.md` D6). Delete the lock only deliberately and log it. Development dry run:
`--lock-path <tmp>/lock --results-dir <tmp> --years 2006`.

## 7. Demo (Phase O3-7), hand-off (Phase O3-8), plots

```bash
streamlit run app/streamlit_app.py     # replay + live, banner on every view
python scripts/phase7_demo.py --mode stub --algo b1 --steps 48
python scripts/phase7_demo.py --mode replay --year 2021 --algo ppo --regime 1 --steps 168
python scripts/phase7_demo.py --mode live --algo ppo --regime 1 --steps 24 --interval 3600   # needs internet
python scripts/phase8_o4_handoff.py --algo ppo --regime 1
python scripts/make_plots.py --regime 1 --tag R2
```
Every demo screen and log row is labelled "Software-in-the-loop simulation. Not field validation."

## 8. Measured speed-ups (this laptop: 12 physical / 16 logical cores, Windows 11)

| Job | Serial | Parallel | Speed-up | Result identical? |
|---|---|---|---|---|
| Phase 2: 5 years × (twin, B0, B1, B2, B4) = 25 episodes, regime 1 | 63.5 s | 17.7 s (6 workers) | ×3.6 | yes (all metrics) |
| Phase 2: 15 episodes | 41.7 s | 13.1 s (6) · 14.3 s (4) · 16.2 s (11) | ×3.2 | yes |
| Phase 3 dataset: 2 years × 3 inits × B0,B1 = 12 episodes | 52.4 s | 17.6 s (6) | ×3.0 | file byte-identical |
| PPO steady-state throughput, 1 vs 10 SubprocVecEnv workers | 652 steps/s | 1,936 steps/s | ×3.0 | same weights as DummyVecEnv, same n_envs |

Why not ×11: the numba kernel is ~55 % of an env step and cannot be changed (O2 frozen); the laptop has 4 performance + 8 efficiency
cores, so more than 6 workers adds little; PPO's gradient updates and evaluation episodes run in the main process (starting 10
workers costs ~20 s). Per-job start-up in a pool ≈ 1.4 s import + 3 s numba cache load. PPO scoring jobs (model load) are
slower than baseline jobs (13 scenarios: B1 51 s vs PPO 212 s at 6 workers, torch import per worker).
Environment profile: one hour-step ≈ 0.3 ms kernel + overhead; a simulated year ≈ 3–4 s serial.

## 9. Compute estimates for the FULL runs (flag: more than a casual laptop session)

Measured single-environment training throughput: PPO 650 steps/s, MaskablePPO 560, DQN 675, RecurrentPPO 185, SAC ≈ 30–45,
QRDQN 12–110 (unstable measurement, re-time before planning). One policy evaluation = 3 validation years ≈ 12–20 s.

| Job (one regime) | Estimate on this laptop |
|---|---|
| PPO / MaskablePPO / DQN, 300k steps, 1 seed | measured: PPO 15 min per seed incl. evaluations (3 seeds in parallel, 16 min wall); estimate ≈ 8–10 min training (+ ~6 min of validation evaluations); PPO-family with 10 envs ≈ 3 min + evals |
| RecurrentPPO 300k | ≈ 27 min (10 envs: ~10 min) |
| SAC 300k | ≈ 2–3 h (one gradient step per sample; the long pole) |
| QRDQN 300k | 45 min – several hours (re-time) |
| 6 algorithms × 3 seeds × 3 regimes | ≈ 1.5–3 days of CPU; ≈ 1 day with `--n-jobs 6` on the PPO-family only |
| Shield-weight sweep (plan Q2): ~5 weights × 3 seeds × 1 algorithm × 300k steps | ≈ 2–3 h with `--n-jobs 6` |
| Phase 6, 3 regimes × 5 years × 3 resets × (twin, B0, B1, model) | ≈ 6 min |
| Emami-scale studies (9,000 episodes ≈ 100 h per run, plan §5) | out of laptop scope — not needed here |

A GPU is not needed or used (small MLP).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `WinError 1450/1114` or "paging file too small" with many workers | TensorFlow was being imported per worker — blocked in `o3/__init__.py`; also lower `--n-envs`/`--n-jobs`; never run two pooled jobs at once |
| `O2 Rajasthan tree not found` | set `O2_RAJASTHAN_ROOT` |
| `Provenance mismatch ...` | an O2 file changed after the contract was built: re-run `python -m src.handoff.build_obj3_contract_rajasthan_v3` in the O2 tree and copy the JSON into `contracts/` |
| `TestLockError` | a test year was requested without the O3 lock — only phase 6 may do that |
| Slow first call | numba loads/compiles the kernel once per machine (cached under the O2 `src/simulation/__pycache__`) |
