"""
Phase O3-4 re-evaluation — recompute the comparison table from SAVED models.

Use this after changing the evaluation harness (metrics, info keys) to refresh
results/phase4_comparison_r{regime}.csv WITHOUT retraining. It loads each algorithm's best checkpoint
(else the final model), re-runs it on the validation years, then re-runs the baselines and the twin.

Run from the rajasthan O3 root:
    python scripts/phase4_reeval.py --regime 1 --n-jobs 8
Parallel: one (algorithm, year) episode per job (--n-jobs). --n-envs is not used here (no training).
"""

import argparse
import csv
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.models import find_model  # noqa: E402
from o3.evaluation.rollout import evaluate_model, add_twin_gain, ALGO_LABEL  # noqa: E402
from scripts.phase4_train import collect_baselines, print_comparison, ALL_ALGOS  # noqa: E402

RESULTS_DIR = config_O2.RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--algo", type=str, nargs="+", default=ALL_ALGOS, choices=ALL_ALGOS)
    ap.add_argument("--results-dir", type=str, default=None)
    add_parallel_args(ap, envs=False)
    args = ap.parse_args()
    n_jobs, _ = resolve(args)
    res_dir = pathlib.Path(args.results_dir) if args.results_dir else RESULTS_DIR

    print(f"Phase O3-4 re-evaluation | regime {args.regime} | n_jobs {n_jobs}")
    rows = collect_baselines(args.regime, VAL_YEARS, n_jobs)
    twins = [r for r in rows if r["policy"] == "TWIN_water_only"]
    jobs = []
    for algo in args.algo:
        p = find_model(algo, args.regime, None, res_dir)
        if p is None:
            print(f"  [skip] {ALGO_LABEL[algo]}: no saved model")
            continue
        jobs += [dict(algo=algo, model_path=str(p), regime_id=args.regime, year=y) for y in VAL_YEARS]
    rows += add_twin_gain(run_parallel(evaluate_model, jobs, n_jobs=n_jobs), twins)

    print_comparison(rows)
    out = res_dir / f"phase4_comparison_r{args.regime}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] Re-evaluated table -> {out}")


if __name__ == "__main__":
    main()
