"""
scripts/m5_robustness.py -- milestone M5: reset scenarios and stochastic demand, VALIDATION years only.

For each PCM regime: B0, B1 and every frozen PPO seed, on 2021-2023 x {fully_solid, partially_charged, fully_liquid}
x {deterministic demand, stochastic demand with 3 demand seeds}. Each row is paired with the water-only twin under
the SAME year, reset scenario and demand draw. Demand ranges are assumptions (Q3).
Writes results/m5_robustness.csv
Usage: python scripts/m5_robustness.py [--regimes 0 1 2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3.controllers import B0AlwaysOn              # noqa: E402
from o3.evaluate import evaluate                   # noqa: E402

DEMAND_SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=int, nargs="+", default=None)
    a = ap.parse_args()
    cfg = o3c.load()
    regs = a.regimes or cfg["regimes"]["pcm"]
    yrs, scen = cfg["years"]["validate"], cfg["reset_scenarios"]
    out = []
    for k in regs:
        ctrls = [B0AlwaysOn(), frozen.b1(k)] + frozen.ppo_policies(k)
        for c in ctrls:
            out.append(evaluate(c, k, yrs, "validate", scen, "deterministic", 0))
            for s in DEMAND_SEEDS:
                out.append(evaluate(c, k, yrs, "validate", scen, "stochastic", s))
            print(f"r{k} {c.name} done", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["o3_config_sha256"] = o3c.config_hash()
    tag = "_".join(f"r{k}" for k in regs)
    df.to_csv(o3c.RESULTS / f"m5_robustness_{tag}.csv", index=False)
    cols = ["E_del_window_kWh", "g_window_kWh", "overheat_h", "hours_over_limit", "bypass_switches"]
    print(df.groupby(["regime", "demand", "scenario", "controller"])[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
