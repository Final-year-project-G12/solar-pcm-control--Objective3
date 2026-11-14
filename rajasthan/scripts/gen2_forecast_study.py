"""
gen2_forecast_study.py — the imperfect-forecast step (plan v2 section 12.2.3): how much of the DQN result survives a forecast
a real installation could produce?

Four cases, DQN, one regime, seeds 42 / 0 / 123, 300k steps, the regime's default shield weight, validation years only:
    perfect      trained and scored with the stored year's own next-3-hour GHI      (the result quoted so far; optimistic)
    imperfect    trained and scored with smart persistence on the clear-sky index   (forecast_mode='smart')
    none         trained and scored with the look-ahead fields zeroed               (forecast_mode='none')
    optimism     the PERFECT-trained policy scored with smart persistence           (no training; measures the optimism directly)
plus B2 (the forecast-rule baseline) under both forecasts, so the comparison is like-for-like.

    python scripts/gen2_forecast_study.py --regime 1 --results-dir results/rerun_2026-11-13 --n-jobs 6
The 'perfect' runs are read from <results-dir>/gen2 (falling back to results/gen2); smart and none runs are trained here.
Output: <results-dir>/gen2/gen2_forecast_dqn_r<regime>.csv (per case and seed). The locked test is not touched.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.agents.gen2 import train_run, evaluate_run, evaluate_baseline_as, gen2_dir, default_tag  # noqa: E402
from o3.environment.pcm_env import default_w_shield  # noqa: E402


def _shield(r):
    return r["shield_water_h"] + r["shield_pcm_h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--algo", default="dqn")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 123])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--results-dir", default=None)
    add_parallel_args(ap, envs=False)
    a = ap.parse_args()
    n_jobs, _ = resolve(a)
    w, g = default_w_shield(a.regime), gen2_dir(a.results_dir)
    g0 = gen2_dir(None)                                                        # fallback location of earlier perfect-forecast runs

    def tag_of(sd, fm):
        return default_tag(a.algo, a.regime, sd, w, "V3", 60, True, fm)

    todo = [dict(algo=a.algo, regime_id=a.regime, steps=a.steps, seed=sd, shield_weight=w, forecast_mode=fm, results_dir=a.results_dir)
            for fm in ("smart", "none") for sd in a.seeds if not (g / tag_of(sd, fm) / "run_meta.json").exists()]
    print(f"forecast study | regime {a.regime} | {len(todo)} runs to train | n_jobs {n_jobs}", flush=True)
    if todo:
        run_parallel(train_run, todo, n_jobs=n_jobs)

    def run_dir(sd, fm):
        for root in (g, g0):
            d = root / tag_of(sd, fm)
            if (d / "run_meta.json").exists():
                return d
        return None

    jobs, meta = [], []
    for sd in a.seeds:
        for case, train_fm, eval_fm in (("perfect", "perfect", None), ("imperfect", "smart", None), ("none", "none", None),
                                        ("optimism", "perfect", "smart")):
            d = run_dir(sd, train_fm)
            if d is None:
                print(f"[skip] {case} seed {sd}: {tag_of(sd, train_fm)} not found")
                continue
            for y in VAL_YEARS:
                jobs.append(dict(run_dir=str(d), year=y, forecast_mode=eval_fm))
                meta.append((case, sd))
    rows = run_parallel(evaluate_run, jobs, n_jobs=n_jobs)
    b0 = run_parallel(evaluate_baseline_as, [dict(baseline="b0", regime_id=a.regime, year=y) for y in VAL_YEARS], n_jobs=n_jobs)
    b2p = run_parallel(evaluate_baseline_as, [dict(baseline="b2", regime_id=a.regime, year=y, forecast_mode="perfect") for y in VAL_YEARS], n_jobs=n_jobs)
    b2s = run_parallel(evaluate_baseline_as, [dict(baseline="b2", regime_id=a.regime, year=y, forecast_mode="smart") for y in VAL_YEARS], n_jobs=n_jobs)
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))                    # noqa: E731
    B0e, B0w, B0s = mean(b0, "total_delivered_kWh"), mean(b0, "E_del45_window_kWh"), float(np.mean([_shield(r) for r in b0]))

    def summarise(case, seed, rs):
        e, win, sh = mean(rs, "total_delivered_kWh"), mean(rs, "E_del45_window_kWh"), float(np.mean([_shield(r) for r in rs]))
        return {"case": case, "seed": seed, "E45_kWh": e, "pct_of_B0_delivery": 100 * e / B0e, "pct_of_B0_window": 100 * win / B0w,
                "shield_h": sh, "shield_cut_vs_B0_pct": 100 * (1 - sh / B0s), "violations": int(sum(r["n_safety_violations"] for r in rs)),
                "pass_annual_rule": bool(100 * e / B0e >= 95 and sum(r["n_safety_violations"] for r in rs) == 0)}

    out = []
    for case in ("perfect", "imperfect", "none", "optimism"):
        for sd in a.seeds:
            rs = [r for r, (c, s) in zip(rows, meta) if c == case and s == sd]
            if rs:
                out.append(summarise(case, sd, rs))
    out += [summarise("B2_perfect_forecast", "-", b2p), summarise("B2_smart_forecast", "-", b2s)]
    p = g / f"gen2_forecast_{a.algo}_r{a.regime}.csv"
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"B0: {B0e:.1f} kWh, shield {B0s:.0f} h (validation years {VAL_YEARS})")
    print(f"{'case':22s} {'seed':>5s} {'%B0':>6s} {'win%':>6s} {'shield h':>8s} {'cut%':>6s} {'viol':>5s} annual-rule")
    for o in out:
        print(f"{o['case']:22s} {str(o['seed']):>5s} {o['pct_of_B0_delivery']:6.1f} {o['pct_of_B0_window']:6.1f} {o['shield_h']:8.0f} "
              f"{o['shield_cut_vs_B0_pct']:6.1f} {o['violations']:5d} {'PASS' if o['pass_annual_rule'] else 'fail'}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
