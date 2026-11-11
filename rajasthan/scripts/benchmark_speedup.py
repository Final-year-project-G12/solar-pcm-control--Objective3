"""
Benchmark: speed-up of the parallel paths against serial runs (Step 3 report).

    python scripts/benchmark_speedup.py --what baselines dataset ppo --n-jobs 6 --n-envs 10

  baselines : phase-2 episodes (twin, B0, B1, B2, B4 x years), serial vs n_jobs processes; metrics compared
  dataset   : phase-3 trajectory generation, serial vs n_jobs; file bytes compared
  ppo       : PPO learn() for 4 rollouts (35,040 steps) with NO callbacks, n_envs = 1 vs n_envs workers,
              start-up of the worker processes EXCLUDED (steady-state throughput, steps/s), plus the
              start-up time separately

Prints a table; nothing is written to results/.
"""

import argparse
import pathlib
import sys
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args, resolve  # noqa: E402


def bench_baselines(regime, n_jobs, years):
    from scripts.phase2_baselines import run_all
    t = time.time(); ser = run_all(regime, years, 1); ts = time.time() - t
    t = time.time(); par = run_all(regime, years, n_jobs); tp = time.time() - t
    same = all(a[k] == b[k] for a, b in zip(ser, par) for k in ("total_delivered_kWh", "E_del45_window_kWh", "total_reward"))
    print(f"baselines ({len(ser)} episodes, regime {regime}): serial {ts:6.1f}s | n_jobs={n_jobs} {tp:6.1f}s | x{ts / tp:.2f} | identical={same}")


def bench_dataset(regime, n_jobs, years):
    from o3.data.trajectory_generator import generate_dataset
    d = pathlib.Path(tempfile.mkdtemp())
    t = time.time(); generate_dataset(d / "a.csv", regime, years, ["solid", "liquid"], n_jobs=1); ts = time.time() - t
    t = time.time(); generate_dataset(d / "b.csv", regime, years, ["solid", "liquid"], n_jobs=n_jobs); tp = time.time() - t
    print(f"dataset  ({len(years) * 4} episodes, regime {regime}): serial {ts:6.1f}s | n_jobs={n_jobs} {tp:6.1f}s | x{ts / tp:.2f} | "
          f"identical={(d / 'a.csv').read_bytes() == (d / 'b.csv').read_bytes()}")


def bench_ppo(regime, n_envs):
    from stable_baselines3 import PPO
    from o3.agents import common as C
    C.pin_main_threads(1)
    res = {}
    for ne in (1, n_envs):
        t = time.time()
        env = C.make_train_vec(regime, ne, "auto")
        t_start = time.time() - t
        model = PPO("MlpPolicy", env, n_steps=C.rollout_steps_per_env(8760, ne), batch_size=256, seed=0, device="cpu", verbose=0)
        model.learn(1)                                   # warm-up (first reset / numba cache load in the workers)
        t = time.time(); model.learn(4 * 8760, reset_num_timesteps=False); dt = time.time() - t
        env.close()
        res[ne] = (t_start, 4 * 8760 / dt)
        print(f"ppo n_envs={ne:2d}: start-up {t_start:5.1f}s | steady state {res[ne][1]:7.0f} steps/s")
    print(f"ppo steady-state speed-up x{res[n_envs][1] / res[1][1]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["baselines", "dataset", "ppo"], choices=["baselines", "dataset", "ppo"])
    ap.add_argument("--regime", type=int, default=1)
    add_parallel_args(ap)
    args = ap.parse_args()
    n_jobs, n_envs = resolve(args)
    if "baselines" in args.what:
        bench_baselines(args.regime, n_jobs, [2016, 2017, 2018, 2019, 2020])
    if "dataset" in args.what:
        bench_dataset(args.regime, n_jobs, [2016, 2017])
    if "ppo" in args.what:
        bench_ppo(args.regime, n_envs)


if __name__ == "__main__":
    main()
