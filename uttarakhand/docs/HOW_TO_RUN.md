# Objective 3 - Uttarakhand: how to run

All commands from `uttarakhand/` with the project venv (Python 3.12, pip wheels from `requirements.txt`; do not mix
with a conda MKL numpy/torch, which caused import crashes on this laptop).

```powershell
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
```

## Pipeline (milestones of plan section 15.10)

| Step | Command | Writes |
|---|---|---|
| tests (must pass first) | `python -m pytest -q` | - |
| M0 test weather (no values printed) | `python scripts/m0_fetch_test_weather.py` then `python scripts/m0_build_test_inventory.py` | `data/weather_test_2006_2010/` |
| M3 baselines + normalisation | `python scripts/m3_baselines.py` | `results/normalisation_r*.json`, `b1_params_r*.json`, `m3_*.csv` |
| G2 headroom gate | `python scripts/g2_headroom.py --regime k` (k = 0, 1, 2) | `results/g2_*_r{k}.*` |
| M4 PPO grid | `python scripts/m4_run_grid.py --jobs 2` | `results/ppo/r{k}/<set>/seed{s}/` |
| M4 select + freeze | `python scripts/m4_select.py` | `results/m4_*.csv`, `results/frozen/r{k}/frozen.json` |
| M5 robustness (validation years) | `python scripts/m5_robustness.py` | `results/m5_robustness_*.csv` |
| contract acceptance list (after M4) | `python scripts/m3_acceptance.py` | `results/m3_acceptance.csv/.json` |
| O2-definition night metrics, validation | `python scripts/m5b_night_metrics.py` | `results/m5b_night_metrics.csv` |
| M6 locked test (ONCE) | `python scripts/m6_locked_test.py --confirm-run-once` | `results/o3_test.lock`, `m6_test_*` |
| post-lock checks (EXPLORATORY; demo reproduces M6, night metrics on test years) | `python scripts/m6b_postlock_checks.py` | `results/m6b_*` |
| M7 demo | `streamlit run app/streamlit_app.py` | `results/demo_logs/` (when saved) |

Memory: each PPO training process commits about 1.6 GB. With 16 GB RAM and other programs open, use `--jobs 2`
(three parallel G2 processes beside three trainers ran out of memory on 2026-11-11).

## Rules the code enforces

* `o2_frozen/` is a read-only byte-for-byte snapshot of O2 Uttarakhand (commit in `O2_SNAPSHOT.json`);
  `tests/test_env_guards.py::test_o2_snapshot_unchanged` re-hashes all 96 files. The environment refuses to start if
  the contract or any O2 input hash fails (`o3/o2_bridge.py: verify_contract`).
* Year roles (`o3/roles_o3.py`): train 2016-2020, validate 2021-2023, O2-used 2024-2025 (never loadable), O3 test
  2006-2010 (only after `acquire_o3_test_lock` in the same process).
* `results/o3_test.lock` is created once by M6 with exclusive-create. Any later run on 2006-2010 is exploratory.
  Deleting the lock must be logged in `docs/PROGRESS.md`. This is a guard rail, not security.
* Changing `configs/o3_config.yaml` after M4 invalidates the frozen policies (`o3/frozen.py` checks its hash).

## Demo

`streamlit run app/streamlit_app.py`. Tabs: replay (train / validate years only), live (Open-Meteo at the medoid
point, persistence fallback when missing or > 2 h stale, raw JSON saved unmodified), re-run of a saved live log.
Every view carries the banner "Software-in-the-loop simulation. Not field validation."
