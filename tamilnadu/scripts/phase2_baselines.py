"""
Phase O3-2 — Run all baselines and print comparison table.

Run from the objective3 root:
    python scripts/phase2_baselines.py [--regime 0] [--years 2016 2017 2018]

Outputs a CSV to results/phase2_baseline_results.csv.
"""

import argparse
import csv
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_all(regime_id: int, years: list[int]) -> list[dict]:
    from o3.environment.pcm_env import PCMEnv
    from o3.baselines.b0_fixed    import FixedChargePolicy, run_episode as run_b0
    from o3.baselines.b1_reactive import ReactiveRulePolicy, run_episode as run_b1

    env = PCMEnv(CONTRACT_PATH, regime_id=regime_id)

    all_results = []
    for year in years:
        print(f"\n--- Regime {regime_id}, Year {year} ---")
        for name, pol, runner in [
            ("B0_fixed",    FixedChargePolicy(),   run_b0),
            ("B1_reactive", ReactiveRulePolicy(),  run_b1),
        ]:
            print(f"  Running {name}...", end=" ", flush=True)
            r = runner(env, policy=pol, year=year)
            all_results.append(r)
            print(
                f"del={r['total_delivered_kWh']:.1f} kWh  "
                f"unmet={r['total_unmet_kWh']:.1f} kWh  "
                f"shield_w={r['total_shield_water']}  "
                f"safety={r['n_safety_violations']}"
            )

    return all_results


def save_csv(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\n[INFO] Results saved to {path}")


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 95)
    print(f"{'Policy':30s} {'Year':6s} {'Regime':7s} {'Del kWh':8s} {'Unmet kWh':10s} {'Shield_w':9s} {'Shield_p':9s} {'Safety':7s}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['policy']:30s} {r['year']:6d} {r['regime_id']:7d} "
            f"{r['total_delivered_kWh']:8.1f} {r['total_unmet_kWh']:10.1f} "
            f"{r['total_shield_water']:9d} {r['total_shield_pcm']:9d} "
            f"{r['n_safety_violations']:7d}"
        )
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0,
                        help="Regime ID to run (0, 1 or 2)")
    parser.add_argument("--years", type=int, nargs="+", default=[2016],
                        help="Weather years to evaluate (space-separated)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase O3-2 — Baseline Evaluation")
    print(f"Regime: {args.regime}  |  Years: {args.years}")
    print("=" * 60)

    results = run_all(args.regime, args.years)
    print_table(results)

    out_path = RESULTS_DIR / f"phase2_baseline_results_r{args.regime}.csv"
    save_csv(results, out_path)

    # Acceptance check: no hard-limit violations in any baseline
    failures = [r for r in results if r["n_safety_violations"] > 0]
    if failures:
        print("\n[FAIL] Safety violations found in baselines:")
        for r in failures:
            print(f"  {r['policy']} year={r['year']}: {r['n_safety_violations']} violations")
        sys.exit(1)

    print("\n[OK]   Phase O3-2 baselines PASSED. Ready for Phase O3-3 (DRL training).")


if __name__ == "__main__":
    main()
