"""
scripts/m5b_night_metrics.py -- O2-definition night serve metrics (tau_serve, serve_frac; o3/night.py) on the
VALIDATION years 2021-2023 for B0, B1 and every frozen PPO seed, each paired with the water-only twin (fully_solid,
deterministic demand). Complements M5, which reported only the in-run tau variant.
Writes results/m5b_night_metrics.csv
Usage: python scripts/m5b_night_metrics.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3.controllers import B0AlwaysOn              # noqa: E402
from o3.evaluate import evaluate                   # noqa: E402


def main():
    cfg = o3c.load()
    out = []
    for k in cfg["regimes"]["pcm"] + cfg["regimes"]["water_only"]:
        ctrls = [B0AlwaysOn(), frozen.b1(k)] + (frozen.ppo_policies(k) if k in cfg["regimes"]["pcm"] else [])
        for c in ctrls:
            out.append(evaluate(c, k, cfg["years"]["validate"], "validate", night=True))
        print(f"r{k} done", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["o3_config_sha256"] = o3c.config_hash()
    df.to_csv(o3c.RESULTS / "m5b_night_metrics.csv", index=False)
    df["group"] = df.controller.str.replace(r"_seed\d+$", "", regex=True)
    cols = ["tau_serve_o2_mean_h", "twin_tau_serve_o2_mean_h", "gain_tau_serve_o2_h", "serve_frac_o2_mean"]
    print(df.groupby(["regime", "group"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
