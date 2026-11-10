"""
Phase O3-3 — Generate safe demonstration dataset and split manifest.

Run from the objective3 root:
    python scripts/phase3_generate_dataset.py [--regime 0] [--years 2016 2017]

Generates B0 and B1 trajectories (safe demonstrations) across training
years and all PCM initial states.  Also writes the train/val/test split
manifest.  Test years (2011-2015) are NEVER touched here.
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.data.trajectory_generator import generate_dataset, write_split_manifest
from o3.environment.gym_wrapper import TRAIN_YEARS

RESULTS_DIR = REPO_ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime",  type=int,       default=0)
    parser.add_argument("--years",   type=int, nargs="+", default=TRAIN_YEARS)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase O3-3 — Dataset Generation")
    print(f"Regime: {args.regime}  |  Years: {args.years}")
    print("=" * 60)

    # write split manifest (once, regime-independent)
    manifest_path = RESULTS_DIR / "split_manifest.json"
    write_split_manifest(manifest_path)

    # generate trajectory dataset
    out_path = RESULTS_DIR / f"phase3_dataset_r{args.regime}.csv"
    generate_dataset(
        out_path  = out_path,
        regime_id = args.regime,
        years     = args.years,
    )

    # quick data-leakage check
    import csv
    test_years = {2011, 2012, 2013, 2014, 2015}
    with open(out_path) as f:
        reader = csv.DictReader(f)
        leaked = [int(r["year"]) for r in reader
                  if int(r["year"]) in test_years]
    if leaked:
        print(f"\n[FAIL] Data leakage! Test years found: {set(leaked)}")
        sys.exit(1)

    print(f"\n[OK]   Phase O3-3 PASSED.  Dataset: {out_path}")
    print("       No test-year leakage detected.")
    print("       Ready for Phase O3-4 (DRL training).")


if __name__ == "__main__":
    main()
