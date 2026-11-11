"""
Phase O3-6 — Locked final evaluation on the fresh O3-only test set (2006-2010).

!! RUN THIS ONLY ONCE, after policy selection is frozen. !!
The test years must NOT have influenced training, reward shaping, normalisation, early stopping or
algorithm selection. Unlike the Tamil Nadu script (a warning in a docstring), this one takes the O3
test lock (results/o3_test.lock, exclusive-create; plan 15.8) before it touches a single test hour and
refuses to run a second time. Test years are loaded only while the lock is held
(o3/utils/weather_roles.py); worker processes are told the lock's run_id and check it on disk.

Test set: 2006-2010 at the three medoid points, downloaded by phase0_fetch_test_weather.py (plan 8.2, Q1); outside every O2 year role.


Usage:
    python scripts/phase6_locked_eval.py --algo ppo --regime 0 1 2 --n-jobs 8
    # dry run for development (does NOT use the real lock or results folder):
    python scripts/phase6_locked_eval.py --algo ppo --regime 1 --lock-path <tmp>/lock --results-dir <tmp> --years 2006

Outputs (in --results-dir, default results/):
    phase6_final_test_{algo}.csv   — per regime / year / reset scenario / policy
    phase6_summary_{algo}.json     — aggregated table + primary success criteria
    phase6_policy_hash.txt         — model SHA-256 and the lock fingerprint
Parallel: one (policy, regime, year, reset scenario) episode per job (--n-jobs).
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

from o3.utils import weather_roles  # noqa: E402
from o3.utils.weather_roles import TEST_YEARS  # noqa: E402
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.models import find_model, sha256_of_file, sha256_of_obj  # noqa: E402
from o3.evaluation.rollout import evaluate_baseline, evaluate_twin, evaluate_model, add_twin_gain, ALGO_LABEL  # noqa: E402

RESULTS_DIR = config_O2.RESULTS_DIR
CONTRACT_PATH = config_O2.CONTRACT_FILE
PCM_INITS = ["solid", "liquid", "partial"]


def _job(kind: str, lock_run_id: str, lock_path: str, **kw):
    """One episode; worker declares the parent's lock before loading any test year."""
    weather_roles.assume_test_lock(lock_run_id, lock_path)
    if kind == "TWIN":
        return evaluate_twin(**kw)
    if kind == "MODEL":
        return evaluate_model(lock_run_id=None, **kw)
    return evaluate_baseline(**kw)


def run_locked(algo: str, regimes: list[int], years: list[int], inits: list[str], model_paths: dict,
               run_id: str, lock_path: str, n_jobs: int) -> list[dict]:
    jobs = []
    for r in regimes:
        for y in years:
            for pi in inits:
                base = dict(lock_run_id=run_id, lock_path=lock_path, regime_id=r, year=y, initial_pcm_state=pi)
                jobs.append(dict(kind="TWIN", **base))
                for b in ("b0", "b1"):
                    jobs.append(dict(kind=b, baseline=b, **base))
                jobs.append(dict(kind="MODEL", algo=algo, model_path=str(model_paths[r]), **base))
    rows = run_parallel(_job, jobs, n_jobs=n_jobs)
    twins = [r for r in rows if r["policy"] == "TWIN_water_only"]
    out = add_twin_gain([r for r in rows if r["policy"] != "TWIN_water_only"], twins)
    for r in out:
        r["model_sha"] = sha256_of_file(model_paths[r["regime_id"]])[:12] if r["policy"] == ALGO_LABEL[algo] else ""
    return twins + out


def success_criteria(rows: list[dict], algo: str) -> bool:
    """Plan section 1 primary criteria, checked on the test set."""
    label = ALGO_LABEL[algo]
    mine = [r for r in rows if r["policy"] == label]
    b0 = [r for r in rows if r["policy"] == "B0_fixed_charge"]
    print(f"\n--- Primary success criteria: {label} ---")
    ok_all = True
    sv = sum(r["n_safety_violations"] for r in mine)
    ok = sv == 0 and sum(r["hours_over_limit"] for r in mine) == 0
    print(f"  {'PASS' if ok else 'FAIL'}  zero hard-limit violations / zero hours over limit: {sv}")
    ok_all &= ok
    g, g0 = np.mean([r["gain_window_kWh"] for r in mine]), np.mean([r["gain_window_kWh"] for r in b0])
    ok = g >= g0
    print(f"  {'PASS' if ok else 'FAIL'}  window gain vs water-only twin >= B0:  {g:.1f} vs {g0:.1f} kWh")
    ok_all &= ok
    res = max(r["residual_pct_of_collector"] for r in mine)
    ok = res < 0.1
    print(f"  {'PASS' if ok else 'FAIL'}  energy residual < 0.1 % of collector energy: max {res:.2e} %")
    ok_all &= ok
    sh = np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in mine])
    sh0 = np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in b0])
    print(f"  INFO  shield hours: {sh:.0f} vs B0 {sh0:.0f}   |   bypass switches: "
          f"{np.mean([r['bypass_switches'] for r in mine]):.0f}")
    return bool(ok_all)


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "no-git"
    except Exception:      # noqa: BLE001
        return "no-git"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"])
    parser.add_argument("--regime", type=int, nargs="+", default=[0])
    parser.add_argument("--years", type=int, nargs="+", default=TEST_YEARS)
    parser.add_argument("--results-dir", type=str, default=None, help="models are read from and results written here")
    parser.add_argument("--lock-path", type=str, default=None, help="override the lock file (development dry runs only)")
    add_parallel_args(parser, envs=False)
    args = parser.parse_args()
    n_jobs, _ = resolve(args)
    res_dir = pathlib.Path(args.results_dir) if args.results_dir else RESULTS_DIR
    lock_path = pathlib.Path(args.lock_path) if args.lock_path else weather_roles.default_lock_path()

    print("=" * 70)
    print(f"Phase O3-6 — LOCKED FINAL EVALUATION  (test years {args.years})")
    print(f"Algorithm: {args.algo}  |  Regimes: {args.regime}  |  lock: {lock_path}  |  n_jobs {n_jobs}")
    print("WARNING: results from this run must NOT be used for further tuning.")
    print("=" * 70)

    model_paths = {}
    for r in args.regime:
        p = find_model(args.algo, r, None, res_dir)
        if p is None:
            print(f"[FAIL] No trained {args.algo} model for regime {r} in {res_dir}. Freeze a policy with phase 4 first.")
            sys.exit(1)
        model_paths[r] = p

    fingerprint = sha256_of_obj({"models": {r: sha256_of_file(p) for r, p in model_paths.items()},
                                 "contract": sha256_of_file(CONTRACT_PATH), "algo": args.algo, "regimes": args.regime,
                                 "git_commit": _git_commit()})
    run_id = f"o3test_{fingerprint[:12]}"
    try:
        weather_roles.acquire_test_lock(run_id, fingerprint, lock_path)       # raises if the test was already run
    except weather_roles.TestLockError as e:
        print(f"[REFUSED] {e}")
        sys.exit(2)
    print(f"[LOCK] acquired {lock_path}  run_id={run_id}")

    t0 = time.time()
    rows = run_locked(args.algo, args.regime, args.years, PCM_INITS, model_paths, run_id, str(lock_path), n_jobs)
    res_dir.mkdir(parents=True, exist_ok=True)
    out_csv = res_dir / f"phase6_final_test_{args.algo}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[INFO] Results -> {out_csv}  ({time.time() - t0:.0f} s)")

    passed = success_criteria(rows, args.algo)

    summary = {"run_id": run_id, "fingerprint": fingerprint, "passed_primary_criteria": passed, "years": args.years}
    for policy in sorted(set(r["policy"] for r in rows)):
        p_rows = [r for r in rows if r["policy"] == policy]
        summary[policy] = {
            "delivered_kWh_mean": float(np.mean([r["total_delivered_kWh"] for r in p_rows])),
            "delivered_kWh_std": float(np.std([r["total_delivered_kWh"] for r in p_rows])),
            "E_del45_window_kWh_mean": float(np.mean([r["E_del45_window_kWh"] for r in p_rows])),
            "gain_window_kWh_mean": float(np.mean([r.get("gain_window_kWh", np.nan) for r in p_rows])),
            "unmet_kWh_mean": float(np.mean([r["total_unmet_kWh"] for r in p_rows])),
            "shield_h_mean": float(np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in p_rows])),
            "safety_violations": int(sum(r["n_safety_violations"] for r in p_rows)),
            "hours_over_limit": float(sum(r["hours_over_limit"] for r in p_rows)),
        }
    (res_dir / f"phase6_summary_{args.algo}.json").write_text(json.dumps(summary, indent=2))
    (res_dir / "phase6_policy_hash.txt").write_text(
        f"algo={args.algo}\nrun_id={run_id}\nfingerprint={fingerprint}\n" +
        "".join(f"regime{r}: {p}  sha256={sha256_of_file(p)}\n" for r, p in model_paths.items()))
    print(f"[INFO] Summary JSON -> {res_dir / f'phase6_summary_{args.algo}.json'}")
    print("\n[OK]   Phase O3-6 complete. The O3 test lock is now held; anything run after this is exploratory.")


if __name__ == "__main__":
    main()
