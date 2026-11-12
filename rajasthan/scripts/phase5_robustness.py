"""
Phase O3-5 — Robustness and ablation evaluation.

Run from the rajasthan O3 root:
    python scripts/phase5_robustness.py --regime 1 --algo ppo --years 2021 2022 --n-jobs 8
    python scripts/phase5_robustness.py --regime 1 --algo b1                     # a baseline through the same suite

Loads a trained model (best checkpoint, else final) and runs the 13-scenario robustness suite on
validation years (observation noise/bias, demand scale and stochastic demand, PCM initial state).
`--algo b0|b1|b2` runs a baseline through the same suite as a reference.

Parallel: one (scenario, year) episode per job, one process per job (--n-jobs); each worker builds its
own environment and seeds its noise from (seed, year), so the CSV is identical for any --n-jobs.
--n-envs is accepted for uniformity and unused (no training here).

Outputs:
    results/phase5_robustness_r{regime}_{algo}.csv
"""

import argparse
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.models import find_model  # noqa: E402
from o3.evaluation.robustness import run_robustness_suite, save_robustness_results, print_robustness_summary  # noqa: E402
from o3.evaluation.rollout import ALGO_LABEL  # noqa: E402

RESULTS_DIR = config_O2.RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac", "b0", "b1", "b2"])
    parser.add_argument("--years", type=int, nargs="+", default=VAL_YEARS[:2])
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--run", type=str, default=None,
                        help="Generation-2 run directory (results/gen2/<tag>): scored in its own observation layout; overrides --algo/--regime")
    add_parallel_args(parser, envs=False)
    args = parser.parse_args()
    n_jobs, _ = resolve(args)
    res_dir = pathlib.Path(args.results_dir) if args.results_dir else RESULTS_DIR

    print("=" * 70)
    print(f"Phase O3-5 — Robustness | algo {args.algo} | regime {args.regime} | years {args.years} | n_jobs {n_jobs}")
    print("=" * 70)

    out_name = f"phase5_robustness_r{args.regime}_{args.algo}.csv"
    if args.run:
        import json
        from o3.agents.gen2 import run_model_path
        rd = pathlib.Path(args.run)
        meta = json.loads((rd / "run_meta.json").read_text())
        args.regime, args.algo = meta["regime_id"], meta["algo"]
        env_kw = dict(obs_version=meta["obs_version"], control_step_min=meta["control_step_min"])
        spec, name = ("model", meta["algo"], str(run_model_path(rd)), env_kw), ALGO_LABEL[meta["algo"]] + "[" + meta["tag"] + "]"
        out_name = f"phase5_robustness_{meta['tag']}.csv"
    elif args.algo in ("b0", "b1", "b2"):
        spec, name = ("baseline", args.algo), args.algo.upper()
    else:
        path = find_model(args.algo, args.regime, None, res_dir)
        if path is None:
            print(f"[FAIL] No trained {args.algo} model for regime {args.regime}. Run phase4_train.py first.")
            sys.exit(1)
        spec, name = ("model", args.algo, str(path)), ALGO_LABEL[args.algo]

    t0 = time.time()
    results = run_robustness_suite(policy_name=name, regime_id=args.regime, years=args.years, policy_spec=spec, n_jobs=n_jobs)
    print_robustness_summary(results)
    save_robustness_results(results, res_dir / out_name)
    print(f"[INFO] {len(results)} episodes in {time.time() - t0:.0f} s with n_jobs={n_jobs}")


if __name__ == "__main__":
    main()
