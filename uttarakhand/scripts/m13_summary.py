"""
scripts/m13_summary.py -- collect the algorithm x forecast study (scripts/m13_forecast_algo_study.py) for one regime,
validation 2021-2023 only, with the earlier runs it is compared against:

  smart    results/ext/fcalgo/r{k}/<algo>_..._V2_fc_smart_DR/seed*   PPO / DQN / SAC, smart-persistence forecast
  perfect  results/ext/fcalgo/... (if run) or results/ext/v2dr/...    look-ahead = stored future GHI
  none     results/ext/fcalgo/... (if run), else the V1 runs (no forecast fields): M4 PPO and ext/dqn DQN
  optimism check (no training): each perfect-forecast policy re-scored with the smart forecast at evaluation

Each run's score is its best validation checkpoint under the M4 lexicographic rule (unchanged). Writes
results/m13_forecast_algo_r{k}.csv (one row per run) and results/m13_forecast_algo_summary_r{k}.csv.
Usage: python scripts/m13_summary.py --regime 1 [--no-optimism]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                                   # noqa: E402
from o3.controllers import SB3Policy                           # noqa: E402
from o3.env import reward_weights                              # noqa: E402
from o3.evaluate import evaluate, summarise                    # noqa: E402

KEYS = ["E_del_window_kWh", "E_del_annual_kWh", "overheat_h", "guard_hours", "hours_over_limit", "bypass_hours",
        "bypass_switches"]
FROZEN_W = {0: (3.0, 0.3), 1: (0.0, 0.3), 2: (0.0, 0.3)}


def _row(b, source, mode):
    return {"source": source, "algo": b.get("algo", "ppo").upper(), "forecast": mode, "seed": b["seed"],
            "best_steps": b["best"]["steps"], **{k: b["best"][k] for k in KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, required=True)
    ap.add_argument("--no-optimism", action="store_true")
    a = ap.parse_args()
    k, R = a.regime, o3c.RESULTS
    w2, w4 = FROZEN_W[k]
    wtag = f"w2_{w2:g}_w4_{w4:g}"
    rows, perfect_runs = [], []
    for p in sorted((R / "ext" / "fcalgo" / f"r{k}").glob("*/seed*/best.json")):
        b = json.loads(p.read_text())
        mode = b.get("forecast_mode", "perfect")
        rows.append(_row(b, "fcalgo", mode))
        if mode == "perfect":
            perfect_runs.append((p.parent, b))
    cnt = pd.Series([(r["algo"], r["forecast"]) for r in rows], dtype=object).value_counts()
    have = {key for key, n in cnt.items() if n >= 3}                     # a full 3-seed arm replaces the older runs
    if ("PPO", "perfect") not in have:                                   # earlier perfect-forecast PPO (same code path)
        for p in sorted((R / "ext" / "v2dr" / f"r{k}").glob(f"ppo_{wtag}_V2_DR/seed*/best.json")):
            b = json.loads(p.read_text())
            rows.append(_row(b, "v2dr", "perfect"))
            perfect_runs.append((p.parent, b))
    if ("PPO", "none") not in have:                                      # no forecast fields at all (V1, no DR)
        for p in sorted((R / "ppo" / f"r{k}" / wtag).glob("seed*/best.json")):
            rows.append(_row({**json.loads(p.read_text()), "algo": "ppo"}, "m4_V1", "none (V1)"))
    if ("DQN", "none") not in have:
        for p in sorted((R / "ext" / "dqn" / f"r{k}").glob(f"dqn_{wtag}_V1/seed*/best.json")):
            rows.append(_row(json.loads(p.read_text()), "ext_dqn_V1", "none (V1)"))

    if not a.no_optimism:                                                # perfect-trained, scored with smart forecast
        cfg = o3c.load()
        weights = reward_weights(k, w2, w4)
        for d, b in perfect_runs:
            pol = SB3Policy(d / "best_model.zip", b.get("algo", "ppo"),
                            env_kwargs={"obs_version": "V2", "forecast_mode": "smart"})
            s = summarise(evaluate(pol, k, cfg["years"]["validate"], "validate", weights=weights))
            rows.append({"source": "optimism_check", "algo": b.get("algo", "ppo").upper(),
                         "forecast": "trained perfect, scored smart", "seed": b["seed"],
                         "best_steps": b["best"]["steps"], **{x: s[x] for x in KEYS}})
            print("optimism", b.get("algo", "ppo"), b["seed"], round(s["E_del_window_kWh"], 1), flush=True)

    m3 = pd.read_csv(R / f"m3_baselines_r{k}.csv")
    m3 = m3[(m3.purpose == "validate") & m3.controller.isin(["B0", "B1"])]
    for c, g in m3.groupby("controller"):
        rows.append({"source": "baseline", "algo": c, "forecast": "-", "seed": 0,
                     **{x: g[x].mean() for x in KEYS if x in g}})
    d = pd.DataFrame(rows).assign(regime=k, o3_config_sha256=o3c.config_hash(),
                                  label="validation 2021-2023; post-lock extension")
    d.to_csv(R / f"m13_forecast_algo_r{k}.csv", index=False)
    s = d.groupby(["algo", "forecast", "source"])[KEYS].agg(["mean", "std"])
    s.columns = [f"{x}_{y}" for x, y in s.columns]
    s = s.reset_index()
    s["n_seeds"] = d.groupby(["algo", "forecast", "source"]).size().to_numpy()
    b0 = d[d.algo == "B0"].iloc[0]
    s["delivery_pct_of_B0"] = 100 * s["E_del_window_kWh_mean"] / b0["E_del_window_kWh"]
    s["overheat_cut_pct_vs_B0"] = 100 * (1 - s["overheat_h_mean"] / b0["overheat_h"])
    s.to_csv(R / f"m13_forecast_algo_summary_r{k}.csv", index=False)
    print(s[["algo", "forecast", "source", "E_del_window_kWh_mean", "E_del_window_kWh_std", "delivery_pct_of_B0",
             "overheat_h_mean", "overheat_cut_pct_vs_B0", "hours_over_limit_mean", "bypass_switches_mean",
             "n_seeds"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
