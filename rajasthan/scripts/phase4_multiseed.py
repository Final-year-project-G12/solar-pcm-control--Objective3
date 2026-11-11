"""
Phase O3-4 — Multi-seed training for statistical comparison.

Trains each algorithm with N seeds, computes mean +/- std across seeds, picks the best seed per
algorithm with the plan's lexicographic rule (zero hard-limit violations, then window delivery, then
shield hours, then switches; plan 8.3) on the VALIDATION years, and freezes that policy with its SHA-256.

Run from the rajasthan O3 root:
    python scripts/phase4_multiseed.py --regime 1 --steps 300000 --seeds 42 0 123 --algo ppo maskable --n-jobs 3
    python scripts/phase4_multiseed.py --regime 1 --steps 3000 --seeds 1 2 --algo ppo --results-dir <tmp>      # smoke

Parallel: ONE PROCESS PER (algorithm, seed) JOB (--n-jobs). Each job trains with its own seed on a single
in-process environment (n_envs = 1) and torch/BLAS threads pinned to 1, then scores itself on the
validation years. A job depends only on its own arguments, so a serial run (--n-jobs 1) and a parallel run
give the same models for the same seeds. --n-envs is accepted for CLI uniformity but NOT used here:
nesting SubprocVecEnv inside the job pool would oversubscribe the cores.

This is the scientifically rigorous version for the thesis; single-seed runs (phase4_train.py) are for development.
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.models import model_dir, sha256_of_file, lexicographic_key  # noqa: E402
from o3.evaluation.rollout import ALGO_LABEL, ALGO_DIR, evaluate_model, add_twin_gain  # noqa: E402
from scripts.phase4_train import collect_baselines, train_one, ALL_ALGOS  # noqa: E402

RESULTS_DIR = config_O2.RESULTS_DIR
METRICS = ["total_delivered_kWh", "E_del45_window_kWh", "gain_window_kWh", "total_unmet_kWh", "total_shield_water",
           "total_shield_pcm", "n_safety_violations", "total_reward", "action_switches"]


def train_eval_job(algo: str, regime_id: int, seed: int, steps: int, res_dir: str, reward_weights: dict | None,
                   years: list[int]) -> dict:
    """Train one (algo, seed) and score it on `years`. Module-level -> runs one process per job."""
    out_dir = model_dir(algo, regime_id, seed, res_dir)
    path = train_one(algo, regime_id, steps, seed, out_dir, 1, reward_weights)
    rows = [evaluate_model(algo, str(path), regime_id, y, reward_weights=reward_weights) for y in years]
    for r in rows:
        r["algo_key"], r["seed"] = algo, seed
    return {"algo": algo, "seed": seed, "model": str(path), "rows": rows}


def compute_stats(rows_per_seed: list[list[dict]]) -> dict:
    """Given N seed runs, compute mean +/- std of key metrics (years averaged within a seed first)."""
    seed_means = [{k: float(np.mean([r.get(k, np.nan) for r in rows])) for k in METRICS} for rows in rows_per_seed]
    out = {"n_seeds": len(rows_per_seed)}
    for k in METRICS:
        v = [m[k] for m in seed_means]
        out[f"{k}_mean"], out[f"{k}_std"], out[f"{k}_min"], out[f"{k}_max"] = float(np.nanmean(v)), float(np.nanstd(v)), float(np.nanmin(v)), float(np.nanmax(v))
    return out


def print_summary_table(stats_by_algo: dict) -> None:
    print("\n" + "=" * 112)
    print(f"{'Algorithm':20s}  {'E45 kWh (mean±std)':20s}  {'Window gain vs twin':20s}  {'Safety viol':12s}  {'Reward mean':12s}")
    print("-" * 112)
    for a, s in stats_by_algo.items():
        g = s.get("gain_window_kWh_mean", float("nan"))
        gs = s.get("gain_window_kWh_std", float("nan"))
        print(f"{a:20s}  {s['total_delivered_kWh_mean']:8.1f} ± {s['total_delivered_kWh_std']:6.1f}   {g:8.1f} ± {gs:6.1f}   "
              f"{s['n_safety_violations_mean']:>6.1f} ± {s['n_safety_violations_std']:.1f}   {s['total_reward_mean']:10.0f}")
    print("=" * 112)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 123])
    parser.add_argument("--algo", type=str, nargs="+", default=ALL_ALGOS, choices=ALL_ALGOS + ["all"])
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--w-shield", type=float, default=None)
    parser.add_argument("--w-switching", type=float, default=None)
    parser.add_argument("--reward-mode", type=str, default=None, choices=["R2", "Q2"], help="Q2 = plan section-6 form (twin-normalised)")
    parser.add_argument("--w2", type=float, default=None, help="Q2 overheat weight")
    parser.add_argument("--w4", type=float, default=None, help="Q2 bypass-switch weight")
    add_parallel_args(parser)
    args = parser.parse_args()
    n_jobs, _ = resolve(args)
    algos = ALL_ALGOS if "all" in args.algo else args.algo
    res_dir = pathlib.Path(args.results_dir) if args.results_dir else RESULTS_DIR
    rw = {k: v for k, v in (("shield", args.w_shield), ("switching", args.w_switching)) if v is not None} or None
    for _k, _v in (("mode", args.reward_mode), ("w2", args.w2), ("w4", args.w4)):
        if _v is not None:
            rw = {**(rw or {}), _k: _v}

    print("=" * 70)
    print("Phase O3-4 Multi-Seed — Algorithm Comparison with Statistics (Rajasthan)")
    print(f"Regime: {args.regime}  Steps: {args.steps:,}  Seeds: {args.seeds}  Algos: {algos}  n_jobs: {n_jobs}")
    print("=" * 70)
    t0 = time.time()

    print("\n[Baselines]")
    base_rows = collect_baselines(args.regime, VAL_YEARS, n_jobs, rw)
    twins = [r for r in base_rows if r["policy"] == "TWIN_water_only"]
    all_raw: list[dict] = list(base_rows)
    stats_by_algo: dict = {}
    for name in ("B0_fixed_charge", "B1_reactive_rule", "B2_forecast_rule"):
        stats_by_algo[name] = compute_stats([[r for r in base_rows if r["policy"] == name]])

    jobs = [dict(algo=a, regime_id=args.regime, seed=s, steps=args.steps, res_dir=str(res_dir), reward_weights=rw,
                 years=VAL_YEARS) for a in algos for s in args.seeds]
    outs = run_parallel(train_eval_job, jobs, n_jobs=n_jobs)

    selected = {}
    for algo in algos:
        mine = [o for o in outs if o["algo"] == algo]
        per_seed = []
        for o in mine:
            add_twin_gain(o["rows"], twins)
            all_raw += o["rows"]
            per_seed.append(o["rows"])
        stats_by_algo[ALGO_LABEL[algo]] = compute_stats(per_seed)
        # lexicographic choice of the seed (plan 8.3): validation years only
        keyed = []
        for o in mine:
            agg = {"n_safety_violations": sum(r["n_safety_violations"] for r in o["rows"]),
                   "gain_window_kWh": float(np.mean([r.get("gain_window_kWh", 0.0) for r in o["rows"]])),
                   "shield_water_h": float(np.mean([r["shield_water_h"] for r in o["rows"]])),
                   "shield_pcm_h": float(np.mean([r["shield_pcm_h"] for r in o["rows"]])),
                   "action_switches": float(np.mean([r["action_switches"] for r in o["rows"]]))}
            keyed.append((lexicographic_key(agg), o))
        best = sorted(keyed, key=lambda t: t[0])[0][1]
        selected[algo] = {"seed": best["seed"], "model": best["model"], "sha256": sha256_of_file(best["model"]),
                          "rule": "lexicographic (zero hard-limit violations, window gain, shield hours, switches) on validation years"}

    print_summary_table(stats_by_algo)
    res_dir.mkdir(parents=True, exist_ok=True)
    raw_path = res_dir / f"phase4_multiseed_raw_r{args.regime}.csv"
    keys = list(dict.fromkeys(k for r in all_raw for k in r))
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_raw)
    print(f"\n[INFO] Raw results -> {raw_path}")
    (res_dir / f"phase4_stats_r{args.regime}.json").write_text(json.dumps(stats_by_algo, indent=2))
    (res_dir / f"phase4_selected_r{args.regime}.json").write_text(json.dumps(selected, indent=2))
    print(f"[INFO] Stats table -> {res_dir / f'phase4_stats_r{args.regime}.json'}")
    print(f"[INFO] Frozen policies (seed + SHA-256) -> {res_dir / f'phase4_selected_r{args.regime}.json'}   ({time.time() - t0:.0f} s)")

    violated = {k for k, s in stats_by_algo.items() if s["n_safety_violations_mean"] > 0}
    if violated:
        print(f"\n[WARN] Safety violations in: {violated}")
    else:
        print("\n[OK]   All policies safe on the validation set across all seeds.")


if __name__ == "__main__":
    main()
