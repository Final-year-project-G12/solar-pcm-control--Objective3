"""
Phase O3-4 — Full DRL algorithm benchmark (single seed, development).

Trains the selected algorithms and compares them with B0/B1/B2 and the water-only twin on the
validation years (2021-2023).

Algorithms:
  ppo        — PPO (memoryless MLP policy)                 on-policy  -> parallel envs (--n-envs)
  dqn        — double DQN (value-based)                    off-policy -> 1 env
  recurrent  — RecurrentPPO (LSTM)                         on-policy  -> parallel envs
  qrdqn      — QRDQN (distributional)                      off-policy -> 1 env
  maskable   — MaskablePPO (action masking)                on-policy  -> parallel envs
  sac        — SAC on the continuous-action view           off-policy -> 1 env

Run from the rajasthan O3 root:
    python scripts/phase4_train.py --regime 1 --steps 200000 --seed 42 --algo ppo maskable --n-envs 11
    python scripts/phase4_train.py --regime 1 --steps 3000 --algo all --results-dir <tmp> --n-envs 2   # smoke

Flags: --n-envs (SubprocVecEnv workers for on-policy agents), --n-jobs (processes for the baseline and
evaluation episodes), reward weights --w-shield / --w-switching / --w-unmet / --w-safety (plan Q2: sweep on
validation years only).

Outputs:
    results/<algo_dir>/r{regime}/   — model checkpoints
    results/phase4_comparison_r{regime}.csv
"""

import argparse
import csv
import importlib
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.models import model_dir  # noqa: E402
from o3.evaluation.rollout import (evaluate_baseline, evaluate_twin, evaluate_model, add_twin_gain,  # noqa: E402
                                   ALGO_LABEL, ALGO_DIR)

RESULTS_DIR = config_O2.RESULTS_DIR
ALL_ALGOS = ["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"]
ONPOLICY = {"ppo", "recurrent", "maskable"}
TRAINERS = {
    "ppo": ("o3.agents.ppo_trainer", "train_ppo"),
    "dqn": ("o3.agents.dqn_trainer", "train_dqn"),
    "recurrent": ("o3.agents.recurrent_ppo_trainer", "train_recurrent_ppo"),
    "qrdqn": ("o3.agents.qrdqn_trainer", "train_qrdqn"),
    "maskable": ("o3.agents.maskable_ppo_trainer", "train_maskable_ppo"),
    "sac": ("o3.agents.sac_trainer", "train_sac"),
}


def _baseline_job(kind: str, **kw):
    return evaluate_twin(**kw) if kind == "TWIN" else evaluate_baseline(**kw)


def collect_baselines(regime_id: int, years: list[int], n_jobs: int = 1, reward_weights: dict | None = None) -> list[dict]:
    jobs = []
    for y in years:
        jobs.append(dict(kind="TWIN", regime_id=regime_id, year=y, reward_weights=reward_weights))
        for b in ("b0", "b1", "b2"):
            jobs.append(dict(kind=b, baseline=b, regime_id=regime_id, year=y, reward_weights=reward_weights))
    rows = run_parallel(_baseline_job, jobs, n_jobs=n_jobs)
    twins = [r for r in rows if r["policy"] == "TWIN_water_only"]
    pol = add_twin_gain([r for r in rows if r["policy"] != "TWIN_water_only"], twins)
    for r in pol:
        print(f"  {r['policy']:18s} y={r['year']} del={r['total_delivered_kWh']:.1f} kWh  safety={r['n_safety_violations']}")
    return twins + pol


def train_one(algo: str, regime_id: int, steps: int, seed: int, out_dir: pathlib.Path, n_envs: int,
              reward_weights: dict | None):
    mod, fn = TRAINERS[algo]
    train_fn = getattr(importlib.import_module(mod), fn)
    kw = dict(regime_id=regime_id, total_timesteps=steps, seed=seed, out_dir=out_dir, reward_weights=reward_weights)
    if algo in ONPOLICY:
        kw["n_envs"] = n_envs
    train_fn(**kw)
    return out_dir / f"{ALGO_DIR[algo]}_final.zip"


def eval_trained(algo: str, model_path: pathlib.Path, regime_id: int, years: list[int], n_jobs: int,
                 reward_weights: dict | None) -> list[dict]:
    jobs = [dict(algo=algo, model_path=str(model_path), regime_id=regime_id, year=y, reward_weights=reward_weights)
            for y in years]
    rows = run_parallel(evaluate_model, jobs, n_jobs=n_jobs)
    for r in rows:
        print(f"  {r['policy']:18s} y={r['year']} del={r['total_delivered_kWh']:.1f} kWh  safety={r['n_safety_violations']}")
    return rows


def print_comparison(rows: list[dict]) -> None:
    print("\n" + "=" * 118)
    print(f"{'Policy':20s} {'Year':5s} {'Reg':3s} {'E45 kWh':8s} {'Win kWh':8s} {'GainWin':8s} {'Unmet':8s} "
          f"{'Shield_w h':10s} {'Shield_p h':10s} {'Safety':6s} {'Reward':10s}")
    print("-" * 118)
    for r in rows:
        g = r.get("gain_window_kWh")
        print(f"{r['policy']:20s} {r['year']:5d} {r['regime_id']:3d} {r['total_delivered_kWh']:8.1f} {r['E_del45_window_kWh']:8.1f} "
              f"{('%8.1f' % g) if g is not None else '     n/a'} {r['total_unmet_kWh']:8.1f} {r['shield_water_h']:10.1f} "
              f"{r['shield_pcm_h']:10.1f} {r['n_safety_violations']:6d} {r['total_reward']:10.0f}")
    print("=" * 118)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo", type=str, nargs="+", default=["ppo", "recurrent", "qrdqn", "maskable"],
                        choices=ALL_ALGOS + ["all"], help="Algorithms to train (space-separated, or 'all')")
    parser.add_argument("--results-dir", type=str, default=None, help="where to write models/CSV (default results/)")
    parser.add_argument("--w-shield", type=float, default=None)
    parser.add_argument("--w-switching", type=float, default=None)
    parser.add_argument("--w-unmet", type=float, default=None)
    parser.add_argument("--w-safety", type=float, default=None)
    parser.add_argument("--reward-mode", type=str, default=None, choices=["R2", "Q2"], help="Q2 = plan section-6 form (twin-normalised)")
    parser.add_argument("--w2", type=float, default=None, help="Q2 overheat weight")
    parser.add_argument("--w4", type=float, default=None, help="Q2 bypass-switch weight")
    add_parallel_args(parser)
    args = parser.parse_args()
    n_jobs, n_envs = resolve(args)

    algos = ALL_ALGOS if "all" in args.algo else args.algo
    res_dir = pathlib.Path(args.results_dir) if args.results_dir else RESULTS_DIR
    rw = {k: v for k, v in (("shield", args.w_shield), ("switching", args.w_switching), ("unmet", args.w_unmet),
                            ("safety", args.w_safety)) if v is not None} or None
    for _k, _v in (("mode", args.reward_mode), ("w2", args.w2), ("w4", args.w4)):
        if _v is not None:
            rw = {**(rw or {}), _k: _v}

    print("=" * 70)
    print("Phase O3-4 — DRL Algorithm Benchmark (Rajasthan)")
    print(f"Regime: {args.regime}  |  Steps: {args.steps:,}  |  Seed: {args.seed}  |  Algos: {algos}")
    print(f"n_jobs: {n_jobs}  |  n_envs (on-policy): {n_envs}  |  reward weights: {rw or 'defaults (R2)'}")
    print("=" * 70)

    all_results: list[dict] = []
    t0 = time.time()
    print("\n--- Baselines (twin, B0, B1, B2) ---")
    all_results += collect_baselines(args.regime, VAL_YEARS, n_jobs, rw)

    timings = {}
    for algo in algos:
        print(f"\n--- {ALGO_LABEL[algo]} ---")
        out_dir = model_dir(algo, args.regime, None, res_dir)
        t1 = time.time()
        path = train_one(algo, args.regime, args.steps, args.seed, out_dir, n_envs, rw)
        timings[algo] = time.time() - t1
        twins = [r for r in all_results if r["policy"] == "TWIN_water_only"]
        rows = eval_trained(algo, path, args.regime, VAL_YEARS, n_jobs, rw)
        all_results += add_twin_gain(rows, twins)

    print_comparison(all_results)
    out_path = res_dir / f"phase4_comparison_r{args.regime}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in all_results for k in r))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\n[INFO] Results saved -> {out_path}   (total {time.time() - t0:.0f}s; training s: "
          f"{ {k: round(v) for k, v in timings.items()} })")

    rl = {ALGO_LABEL[a] for a in algos}
    rl_viols = [r for r in all_results if r["policy"] in rl and r["n_safety_violations"] > 0]
    if rl_viols:
        print("\n[WARN] Safety violations found in RL policies on val set:")
        for r in rl_viols:
            print(f"  {r['policy']} y={r['year']}: {r['n_safety_violations']}")
    else:
        print("\n[OK]   Phase O3-4 PASSED.  Ready for Phase O3-5 (Robustness).")


if __name__ == "__main__":
    main()
