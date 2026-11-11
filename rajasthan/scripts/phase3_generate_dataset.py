"""
Phase O3-3 — Generate the safe demonstration dataset and the split manifest.

Run from the rajasthan O3 root:
    python scripts/phase3_generate_dataset.py [--regime 0] [--years 2016 2017] [--n-jobs 8]

Generates B0 and B1 trajectories (safe demonstrations) across training years and all PCM initial
states. Also writes the train/val/test split manifest. Test years (2006-2010) are NEVER touched here.

Parallel: one (year, initial state, policy) episode per job, one process per job (--n-jobs). The
file written is byte-identical for any --n-jobs. --n-envs is accepted for CLI uniformity and ignored.
"""

import argparse
import csv
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.data.trajectory_generator import generate_dataset, write_split_manifest  # noqa: E402
from o3.environment.gym_wrapper import TRAIN_YEARS  # noqa: E402
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402

RESULTS_DIR = config_O2.RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--years", type=int, nargs="+", default=TRAIN_YEARS)
    parser.add_argument("--inits", type=str, nargs="+", default=["solid", "liquid", "partial"])
    parser.add_argument("--out", type=str, default=None, help="output CSV (default results/phase3_dataset_r{regime}.csv)")
    add_parallel_args(parser)
    args = parser.parse_args()
    n_jobs, _ = resolve(args)

    print("=" * 60)
    print("Phase O3-3 — Dataset Generation (Rajasthan)")
    print(f"Regime: {args.regime}  |  Years: {args.years}  |  n_jobs: {n_jobs}")
    print("=" * 60)

    write_split_manifest(RESULTS_DIR / "split_manifest.json")
    out_path = pathlib.Path(args.out) if args.out else RESULTS_DIR / f"phase3_dataset_r{args.regime}.csv"
    t0 = time.time()
    generate_dataset(out_path=out_path, regime_id=args.regime, years=args.years, pcm_inits=args.inits, n_jobs=n_jobs)
    print(f"[INFO] generated in {time.time() - t0:.1f} s with n_jobs={n_jobs}")

    test_years = {2006, 2007, 2008, 2009, 2010}
    with open(out_path) as f:
        leaked = [int(r["year"]) for r in csv.DictReader(f) if int(r["year"]) in test_years]
    if leaked:
        print(f"\n[FAIL] Data leakage! Test years found: {set(leaked)}")
        sys.exit(1)
    print(f"\n[OK]   Phase O3-3 PASSED.  Dataset: {out_path}")
    print("       No test-year leakage detected.")
    print("       Ready for Phase O3-4 (DRL training).")


if __name__ == "__main__":
    main()
