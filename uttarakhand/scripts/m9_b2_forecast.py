"""
scripts/m9_b2_forecast.py -- B2 forecast-aware baseline (plan 11.1 deferred item; built for Tamil Nadu in its
section 16.2). Same protocol as B1: threshold grid on the VALIDATION years 2021-2023 only (fully_solid, deterministic
demand, perfect forecast), choice by the plan 8.3 lexicographic rule; then the chosen rule is re-scored with an
imperfect forecast (multiplicative GHI noise 10 % and 20 %, 3 noise seeds) for a forecast-sensitivity check.
Compared against B0, B1 and the water-only twin on the same years. No test year is used.
Writes results/m9_b2_grid_r{k}.csv, results/b2_params_r{k}.json, results/m9_b2_summary_r{k}.csv
Usage: python scripts/m9_b2_forecast.py --regime 1
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                                       # noqa: E402
from o3 import frozen                                              # noqa: E402
from o3.controllers import B0AlwaysOn, B2Forecast                  # noqa: E402
from o3.evaluate import evaluate, lexicographic_best, summarise    # noqa: E402

GRID = {"theta_top": [68, 72], "theta_f": [0.9, 0.99], "theta_hot": [50, 55, 60],
        "ghi_ahead_Wm2": [200, 400, 600], "horizon_h": [2, 4]}
NOISE = [0.1, 0.2]
NOISE_SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, required=True)
    k = ap.parse_args().regime
    cfg = o3c.load()
    yrs, tie, h = cfg["years"]["validate"], cfg["selection"]["delivery_tie_kWh_per_yr"], o3c.config_hash()
    grid = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        grid.append({**p, **summarise(evaluate(B2Forecast(**p), k, yrs, "validate"))})
    g = pd.DataFrame(grid).assign(o3_config_sha256=h)
    g.to_csv(o3c.RESULTS / f"m9_b2_grid_r{k}.csv", index=False)
    best = lexicographic_best(grid, tie)
    params = {x: best[x] for x in GRID}
    (o3c.RESULTS / f"b2_params_r{k}.json").write_text(json.dumps(
        {"regime": k, "params": params, "grid": GRID, "rule": f"plan 8.3 lexicographic, tie {tie} kWh/yr",
         "years": yrs, "forecast": "perfect (episode's own future GHI)", "o3_config_sha256": h}, indent=1))

    rows = []
    ctrls = [("B0", B0AlwaysOn(), None), ("B1", frozen.b1(k), None), ("B2_perfect", B2Forecast(**params), None)]
    for nf in NOISE:
        for s in NOISE_SEEDS:
            ctrls.append((f"B2_noise{int(nf * 100)}", B2Forecast(**params, noise_frac=nf, seed=s), s))
    for name, c, s in ctrls:
        r = evaluate(c, k, yrs, "validate")
        r["controller"], r["noise_seed"] = name, s
        rows.append(r)
    out = pd.concat(rows, ignore_index=True).assign(o3_config_sha256=h)
    out.to_csv(o3c.RESULTS / f"m9_b2_summary_r{k}.csv", index=False)
    cols = ["E_del_window_kWh", "g_window_kWh", "overheat_h", "hours_over_limit", "bypass_switches"]
    print(f"r{k} B2 chosen {params}")
    print(out.groupby("controller")[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
