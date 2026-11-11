"""
Phase O3-2b — Headroom gate G2 (plan 15.10): can ANY hourly on/off policy clearly beat B0?

"Random search and B1 grid over hourly on/off policies. If nothing beats B0 on the section-1 metrics by more than the
spread, stop training and revisit the action space; report this honestly."

What is searched (all on VALIDATION years 2021-2023 only):
  * the 24-point B1 grid (o3/baselines/b1_reactive.B1_GRID);
  * N random draws of the rule family's parameters (seeded): water guard 60-75 C (>= 72 means "never before the shield"),
    PCM guard 55-70 C, sun threshold for charging 0-300 W/m2, strong-sun threshold 300-800, f_melt cap 0.5-1.01 (1.01 = never
    stop charging), evening hold window on/off with random start/length, f_melt needed to hold 0-0.5;
  * near-B0 corner cases (charge whenever there is any sun; bypass only just under the shield).

Decision rule (declared before the run): headroom EXISTS if the best policy's mean window gain versus the water-only twin
exceeds B0's by more than O2's repeatability band for g_window in this regime (contract repeatability_band, the size below
which O2 treats two designs as tied), with zero hard-limit violations. A secondary line reports the best policy that keeps
B0's delivery (within the band) while cutting shield hours — that is the only other way a controller can "win".

    python scripts/phase2_headroom_gate.py --regime 1 --n-random 60 --n-jobs 6
Output: results/phase2_headroom_gate_r<regime>.csv (every policy tried) and ..._summary.json. --n-envs unused.
"""

import argparse
import csv
import json
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
from o3.utils.contract_loader import load_contract, get_regime_design  # noqa: E402
from o3.evaluation.rollout import evaluate_baseline, evaluate_twin  # noqa: E402
from o3.baselines.b1_reactive import grid_points  # noqa: E402


def random_points(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        start = int(rng.integers(15, 22))
        length = int(rng.choice([0, 0, 2, 4, 6]))                      # 0 = no evening hold
        pts.append({"bypass_water_guard_C": float(rng.uniform(60, 75)), "bypass_pcm_guard_C": float(rng.uniform(55, 70)),
                    "ghi_charge_thresh_W": float(rng.choice([0.0, 0.0, rng.uniform(0, 300)])),
                    "ghi_strong_W": float(rng.uniform(300, 800)), "f_melt_charge_max": float(rng.choice([1.01, rng.uniform(0.5, 1.0)])),
                    "evening_start_h": start, "evening_end_h": min(start + length, 24),
                    "f_melt_discharge_min": float(rng.uniform(0, 0.5)), "f_melt_low": 0.05})
    return pts


def corner_points() -> list[dict]:
    base = {"ghi_charge_thresh_W": 0.0, "f_melt_charge_max": 1.01, "evening_start_h": 17, "evening_end_h": 17, "bypass_pcm_guard_C": 70.0}
    return [{**base, "bypass_water_guard_C": g} for g in (75.0, 71.5, 71.0, 70.0, 68.0)] + \
           [{**base, "bypass_water_guard_C": 75.0, "bypass_pcm_guard_C": g, "ghi_strong_W": 0.0} for g in (61.5, 61.0, 60.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    add_parallel_args(ap)
    args = ap.parse_args()
    n_jobs, _ = resolve(args)
    band = get_regime_design(load_contract(config_O2.CONTRACT_FILE), args.regime)["repeatability_band"]["g_window"]

    pts = [("grid", p) for p in grid_points()] + [("corner", p) for p in corner_points()] + \
          [("random", p) for p in random_points(args.n_random, args.seed)]
    print(f"Headroom gate G2 | regime {args.regime} | {len(pts)} rule policies x {len(VAL_YEARS)} validation years | band {band:.1f} kWh | n_jobs {n_jobs}")

    ref = run_parallel(_ref_job, [dict(kind=k, regime_id=args.regime, year=y) for y in VAL_YEARS for k in ("twin", "b0")], n_jobs=n_jobs)
    twin = {r["year"]: r for r in ref if r["policy"].startswith("TWIN")}
    b0 = [r for r in ref if r["policy"].startswith("B0")]
    b0_gain = float(np.mean([r["E_del45_window_kWh"] - twin[r["year"]]["E_del45_window_kWh"] for r in b0]))
    b0_shield = float(np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in b0]))

    jobs = [dict(baseline="b1", regime_id=args.regime, year=y, policy_kwargs=p) for _, p in pts for y in VAL_YEARS]
    rows = run_parallel(evaluate_baseline, jobs, n_jobs=n_jobs)
    out = []
    for i, (src, p) in enumerate(pts):
        rs = rows[i * len(VAL_YEARS):(i + 1) * len(VAL_YEARS)]
        out.append({"source": src, **p,
                    "gain_window_kWh": float(np.mean([r["E_del45_window_kWh"] - twin[r["year"]]["E_del45_window_kWh"] for r in rs])),
                    "E_del45_kWh": float(np.mean([r["total_delivered_kWh"] for r in rs])),
                    "shield_h": float(np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in rs])),
                    "switches": float(np.mean([r["action_switches"] for r in rs])),
                    "violations": int(sum(r["n_safety_violations"] for r in rs))})
    out.sort(key=lambda r: -r["gain_window_kWh"])
    safe = [r for r in out if r["violations"] == 0]
    best = safe[0]
    tied = sorted([r for r in safe if r["gain_window_kWh"] >= b0_gain - band], key=lambda r: r["shield_h"])
    headroom = best["gain_window_kWh"] > b0_gain + band
    summary = {"regime": args.regime, "years": VAL_YEARS, "n_policies": len(out), "band_g_window_kWh": band,
               "B0_gain_window_kWh": b0_gain, "B0_shield_h": b0_shield,
               "best_gain_window_kWh": best["gain_window_kWh"], "best_minus_B0_kWh": best["gain_window_kWh"] - b0_gain,
               "HEADROOM_ON_DELIVERY": bool(headroom),
               "best_shield_cut_within_band": ({"gain_window_kWh": tied[0]["gain_window_kWh"], "shield_h": tied[0]["shield_h"],
                                                "switches": tied[0]["switches"], "params": {k: tied[0][k] for k in pts[0][1] if k in tied[0]}}
                                               if tied else None),
               "rule": "headroom iff best gain > B0 gain + O2 repeatability band (g_window), zero hard-limit violations; validation years only"}
    res = config_O2.RESULTS_DIR
    res.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(res / f"phase2_headroom_gate_r{args.regime}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    (res / f"phase2_headroom_gate_r{args.regime}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"B0: window gain {b0_gain:+.1f} kWh, shield {b0_shield:.0f} h")
    print(f"best rule policy: window gain {best['gain_window_kWh']:+.1f} kWh ({best['gain_window_kWh'] - b0_gain:+.1f} vs B0), shield {best['shield_h']:.0f} h [{best['source']}]")
    if tied:
        t = tied[0]
        print(f"least shield while within the band of B0: gain {t['gain_window_kWh']:+.1f} kWh, shield {t['shield_h']:.0f} h, switches {t['switches']:.0f}/yr")
    print(f"G2 verdict: {'HEADROOM on delivery' if headroom else 'NO policy beats B0 on window delivery by more than the band'}")


def _ref_job(kind: str, regime_id: int, year: int):
    return evaluate_twin(regime_id, year) if kind == "twin" else evaluate_baseline("b0", regime_id, year)


if __name__ == "__main__":
    main()
