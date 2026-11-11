"""
Phase O3-2 — Run all baselines and print the comparison table.

Run from the rajasthan O3 root:
    python scripts/phase2_baselines.py [--regime 0] [--years 2016 2017 2018] [--n-jobs 8]
    python scripts/phase2_baselines.py --regime 1 --years 2021 2022 2023 --tune      # B1 grid on VALIDATION years
    python scripts/phase2_baselines.py --regime 1 --verify-parallel                  # serial vs parallel + timing

Baselines: B0 (always on), B1 (reactive rule), B2 (forecast-aware), B4 (random, diagnostic) and the
water-only twin. Each (baseline, year) episode is one independent job: --n-jobs runs them in separate
processes (1 = serial). The result of a job depends only on its own arguments, so serial and parallel
runs give identical numbers (--verify-parallel checks this and prints the speed-up).

Outputs: results/phase2_baseline_results_r{regime}.csv (with gain versus the water-only twin),
         results/phase2_b1_tuning_r{regime}.csv when --tune is used.
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

from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.evaluation.rollout import evaluate_baseline, evaluate_twin, add_twin_gain  # noqa: E402
from o3.utils import weather_roles  # noqa: E402

CONTRACT_PATH = config_O2.CONTRACT_FILE
RESULTS_DIR = config_O2.RESULTS_DIR
BASELINES = ["b0", "b1", "b2", "b4"]


def build_jobs(regime_id: int, years: list[int], baselines=BASELINES, inits=("solid",)):
    jobs = []
    for y in years:
        for pi in inits:
            jobs.append(("TWIN", dict(regime_id=regime_id, year=y, initial_pcm_state=pi)))
            for b in baselines:
                jobs.append((b, dict(baseline=b, regime_id=regime_id, year=y, initial_pcm_state=pi)))
    return jobs


def _run_job(kind: str, kw: dict):
    return evaluate_twin(**kw) if kind == "TWIN" else evaluate_baseline(**kw)


def run_all(regime_id: int, years: list[int], n_jobs: int = 1, baselines=BASELINES) -> list[dict]:
    jobs = build_jobs(regime_id, years, baselines)
    rows = run_parallel(_run_job, [dict(kind=k, kw=kw) for k, kw in jobs], n_jobs=n_jobs)
    twins = [r for r in rows if r["policy"] == "TWIN_water_only"]
    pol = add_twin_gain([r for r in rows if r["policy"] != "TWIN_water_only"], twins)
    return twins + pol


def tune_b1(regime_id: int, years: list[int], n_jobs: int) -> list[dict]:
    """Grid search of B1 thresholds on VALIDATION years only (plan 15.6); every grid point is logged."""
    from o3.baselines.b1_reactive import grid_points
    assert all(weather_roles.role_of(y) == "validate" for y in years), "B1 tuning may only use validation years"
    pts = grid_points()
    jobs = [dict(baseline="b1", regime_id=regime_id, year=y, policy_kwargs=p) for p in pts for y in years]
    rows = run_parallel(evaluate_baseline, jobs, n_jobs=n_jobs)
    out = []
    for i, p in enumerate(pts):
        rs = rows[i * len(years):(i + 1) * len(years)]
        out.append({**p, "n_years": len(years), "mean_E_del45_window_kWh": sum(r["E_del45_window_kWh"] for r in rs) / len(rs),
                    "mean_shield_h": sum(r["shield_water_h"] + r["shield_pcm_h"] for r in rs) / len(rs),
                    "total_safety_violations": sum(r["n_safety_violations"] for r in rs),
                    "mean_switches": sum(r["action_switches"] for r in rs) / len(rs)})
    # lexicographic: zero hard-limit violations, then window delivery, then fewer shield hours, then fewer switches
    out.sort(key=lambda r: (r["total_safety_violations"], -r["mean_E_del45_window_kWh"], r["mean_shield_h"], r["mean_switches"]))
    return out


def calibrate_shield(regime_id: int, years: list[int], n_jobs: int) -> dict:
    """Break-even shield weight w* at which B0 and B1 tie under the R2 reward, measured on TRAINING years only.
    (o3/environment/pcm_env.py W_SHIELD_BY_REGIME = 1.19 x w*, the TN ratio 10 / 8.4.)"""
    assert all(weather_roles.role_of(y) == "train" for y in years), "calibrate on training years only"
    jobs = [dict(baseline=b, regime_id=regime_id, year=y, reward_weights={"shield": 0.0}) for b in ("b0", "b1") for y in years]
    rows = run_parallel(evaluate_baseline, jobs, n_jobs=n_jobs)
    b0 = [r for r in rows if r["policy"].startswith("B0")]
    b1 = [r for r in rows if r["policy"].startswith("B1")]
    mean = lambda rs, f: sum(f(r) for r in rs) / len(rs)          # noqa: E731
    R0, R1 = mean(b0, lambda r: r["total_reward"]), mean(b1, lambda r: r["total_reward"])
    cnt = lambda r: r["total_shield_water"] + r["total_shield_pcm"]  # noqa: E731
    n0, n1 = mean(b0, cnt), mean(b1, cnt)
    w = (R0 - R1) / (n0 - n1)
    return {"regime": regime_id, "years": years, "R0_no_shield_term": R0, "R1_no_shield_term": R1, "n_shield_B0": n0,
            "n_shield_B1": n1, "break_even_w_star": w, "suggested_w_shield_1p19x": 1.19 * w}


def reward_normalisation(n_jobs: int) -> dict:
    """Plan Q2: E_norm = water-only twin's mean hourly E_del45 over the TRAINING years; w3 by the dominance rule."""
    import json
    from o3.environment.o2_adapter import build_runtime
    from o3.utils.contract_loader import load_contract
    c = load_contract(CONTRACT_PATH)
    out = {"source": "water-only twin, training years 2016-2020 (plan section 6 / Q2)", "years": weather_roles.TRAIN_YEARS, "regimes": {}}
    for rid in (0, 1, 2):
        rows = run_parallel(evaluate_twin, [dict(regime_id=rid, year=y) for y in weather_roles.TRAIN_YEARS], n_jobs=n_jobs)
        e_norm = sum(r["total_delivered_kWh"] for r in rows) / len(rows) * 1000.0 / 8760.0
        rt = build_runtime(c, rid)
        max_hour_Wh = float(rt.demand24.max()) * rt.cp_w * (rt.target_C - rt.mains_C) / 3600.0
        out["regimes"][str(rid)] = {"E_norm_Wh_per_hour": e_norm, "max_one_hour_delivery_Wh": max_hour_Wh,
                                    "w3_dominance": 1.01 * max_hour_Wh / e_norm}
    path = config_O2.CONTRACTS_DIR / "reward_normalisation_rajasthan.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["regimes"], indent=1))
    print("->", path)
    return out


def save_csv(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    keys = list(dict.fromkeys(k for r in results for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\n[INFO] Results saved to {path}")


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 128)
    print(f"{'Policy':20s} {'Year':5s} {'Reg':3s} {'E45 kWh':8s} {'Win kWh':8s} {'GainWin':8s} {'Unmet':8s} "
          f"{'Shield_w h':10s} {'Shield_p h':10s} {'Switch':7s} {'Safety':6s}")
    print("-" * 128)
    for r in results:
        g = r.get("gain_window_kWh")
        print(f"{r['policy']:20s} {r['year']:5d} {r['regime_id']:3d} {r['total_delivered_kWh']:8.1f} "
              f"{r['E_del45_window_kWh']:8.1f} {('%8.1f' % g) if g is not None else '     n/a'} {r['total_unmet_kWh']:8.1f} "
              f"{r['shield_water_h']:10.1f} {r['shield_pcm_h']:10.1f} {r['action_switches']:7d} {r['n_safety_violations']:6d}")
    print("=" * 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0, help="Regime ID to run (0, 1 or 2)")
    parser.add_argument("--years", type=int, nargs="+", default=[2016], help="Weather years to evaluate (train/validation roles)")
    parser.add_argument("--tune", action="store_true", help="grid-tune B1 on the given VALIDATION years and write the grid")
    parser.add_argument("--verify-parallel", action="store_true", help="also run serially, assert identical metrics, print speed-up")
    parser.add_argument("--calibrate-shield", action="store_true",
                        help="break-even shield weight w* between B0 and B1 on the given TRAINING years (default 2016-2020)")
    parser.add_argument("--reward-norm", action="store_true",
                        help="write contracts/reward_normalisation_rajasthan.json (twin over training years; plan Q2)")
    add_parallel_args(parser, envs=False)
    args = parser.parse_args()
    n_jobs, _ = resolve(args)

    if args.reward_norm:
        reward_normalisation(n_jobs)
        return
    if args.calibrate_shield:
        yrs = args.years if args.years != [2016] else weather_roles.TRAIN_YEARS
        res = calibrate_shield(args.regime, yrs, n_jobs)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
        save_csv([res], RESULTS_DIR / f"phase2_shield_calibration_r{args.regime}.csv")
        return

    print("=" * 60)
    print("Phase O3-2 — Baseline Evaluation (Rajasthan)")
    print(f"Regime: {args.regime}  |  Years: {args.years}  |  n_jobs: {n_jobs}")
    print("=" * 60)

    if args.tune:
        grid = tune_b1(args.regime, args.years, n_jobs)
        save_csv(grid, RESULTS_DIR / f"phase2_b1_tuning_r{args.regime}.csv")
        print("Best B1 thresholds (validation years only):", {k: v for k, v in grid[0].items() if k in
              ("bypass_water_guard_C", "bypass_pcm_guard_C", "ghi_charge_thresh_W", "f_melt_charge_max")})
        return

    t0 = time.time()
    results = run_all(args.regime, args.years, n_jobs)
    t_par = time.time() - t0
    print_table(results)
    save_csv(results, RESULTS_DIR / f"phase2_baseline_results_r{args.regime}.csv")

    if args.verify_parallel:
        t0 = time.time()
        serial = run_all(args.regime, args.years, 1)
        t_ser = time.time() - t0
        keys = ["total_delivered_kWh", "E_del45_window_kWh", "total_shield_water", "total_shield_pcm", "n_safety_violations", "total_reward"]
        same = all(abs(a[k] - b[k]) == 0 for a, b in zip(serial, results) for k in keys)
        print(f"\n[VERIFY] serial {t_ser:.1f}s  |  n_jobs={n_jobs}: {t_par:.1f}s  |  speed-up x{t_ser / max(t_par, 1e-9):.2f}  "
              f"|  metrics identical: {same}")
        if not same:
            sys.exit(1)

    failures = [r for r in results if r["n_safety_violations"] > 0]
    if failures:
        print("\n[FAIL] Safety violations found in baselines:")
        for r in failures:
            print(f"  {r['policy']} year={r['year']}: {r['n_safety_violations']} violations")
        sys.exit(1)
    print("\n[OK]   Phase O3-2 baselines PASSED. Ready for Phase O3-3 / O3-4.")


if __name__ == "__main__":
    main()
