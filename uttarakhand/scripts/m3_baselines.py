"""
scripts/m3_baselines.py -- milestone M3 (+ the normalisation files the environment needs).

Per regime (0-3):
  1. normalisation: observation mean/std from B0 on the TRAIN years x 3 reset scenarios; E_norm = the water-only
     twin's mean hourly E_del,50 over the train years (fully_solid); the largest possible one-hour normalised
     delivery (for the w3 dominance rule) = max hourly draw x cp x (50 C - mains) / E_norm.
  2. B1 threshold grid on the VALIDATION years only, chosen by the plan 8.3 lexicographic rule.
  3. B0, B1, B4 (3 seeds) and the water-only twin on train and validation years (fully_solid, deterministic demand).
Writes results/normalisation_r{k}.json, results/b1_params_r{k}.json, results/m3_baselines_r{k}.csv, results/m3_b1_grid_r{k}.csv.
Usage: python scripts/m3_baselines.py [--regimes 0 1 2 3]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                                              # noqa: E402
from o3 import o2_bridge as ob                                            # noqa: E402
from o3.controllers import B0AlwaysOn, B1Reactive, B4Random                # noqa: E402
from o3.env import OBS_FIELDS, UKPCMEnv                                    # noqa: E402
from o3.evaluate import METRIC_WEIGHTS, evaluate, lexicographic_best, summarise, twin_metrics   # noqa: E402

B1_GRID = {"theta_top": [56, 60, 64, 68, 72], "theta_f": [0.9, 0.99, 1.01], "theta_hot": [50, 55, 60]}


def build_normalisation(regime: int) -> dict:
    cfg = o3c.load()
    obs = []
    for y in cfg["years"]["train"]:
        for sc in cfg["reset_scenarios"]:
            env = UKPCMEnv(regime, "train", [y], [sc], normalise=False, weights=METRIC_WEIGHTS)
            env.reset(options={"year": y, "scenario": sc})
            done = False
            while not done:
                obs.append(env._raw_obs())
                _, _, done, _, _ = env.step(1)
    obs = np.array(obs)
    mu, sd = obs.mean(axis=0), obs.std(axis=0)
    sd[sd < 1e-6] = 1.0                                     # constant fields (T_mains; PCM fields in a water-only tank)
    e_twin = np.mean([twin_metrics(regime, y, "fully_solid", "train")["E_del_annual_kWh"] for y in cfg["years"]["train"]])
    e_norm_Wh = e_twin * 1000.0 / 8760.0
    case = env.case
    max_hour_Wh = case.demand24.max() * case.P[0] * (case.target_C - case.mains_C) / 3600.0
    out = {"regime": regime, "fields": OBS_FIELDS, "obs_mean": mu.tolist(), "obs_std": sd.tolist(),
           "E_norm_Wh": e_norm_Wh, "twin_E_del_annual_train_kWh": e_twin,
           "max_hour_delivery_Wh": max_hour_Wh, "max_hour_delivery_normalised": max_hour_Wh / e_norm_Wh,
           "source": "B0 on train years 2016-2020 x 3 reset scenarios; twin = water-only, same flow",
           "o3_config_sha256": o3c.config_hash()}
    p = o3c.normalisation_path(regime)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out


def tune_b1(regime: int, tie: float):
    cfg = o3c.load()
    rows = []
    for th in itertools.product(*B1_GRID.values()):
        params = dict(zip(B1_GRID, th))
        if params["theta_f"] > 1.0 and params["theta_hot"] != B1_GRID["theta_hot"][0]:
            continue                                         # f-branch off: theta_hot irrelevant, keep one
        s = summarise(evaluate(B1Reactive(**params), regime, cfg["years"]["validate"], "validate"))
        rows.append({**params, **s})
    grid = pd.DataFrame(rows)
    best = lexicographic_best(grid.to_dict("records"), tie)
    chosen = {k: best[k] for k in B1_GRID}
    return grid, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()
    ob.verify_contract()
    cfg = o3c.load()
    tie = cfg["selection"]["delivery_tie_kWh_per_yr"]
    res, grids = [], []
    for k in a.regimes:
        n = build_normalisation(k)
        print(f"r{k}: E_norm {n['E_norm_Wh']:.1f} Wh/h, max hour normalised {n['max_hour_delivery_normalised']:.1f}", flush=True)
        grid, chosen = tune_b1(k, tie)
        grid.insert(0, "regime", k)
        grids.append(grid)
        (o3c.RESULTS / f"b1_params_r{k}.json").write_text(json.dumps(
            {"regime": k, "params": chosen, "grid": B1_GRID, "tuned_on": cfg["years"]["validate"],
             "rule": "plan 8.3 lexicographic, delivery tie %.1f kWh/yr" % tie, "o3_config_sha256": o3c.config_hash()}, indent=1))
        print(f"r{k}: B1 chosen {chosen}", flush=True)
        for purpose in ("train", "validate"):
            yrs = cfg["years"][purpose]
            ctrls = [B0AlwaysOn(), B1Reactive(**chosen)] + [B4Random(s) for s in range(3)]
            for c in ctrls:
                df = evaluate(c, k, yrs, purpose, seed=getattr(c, "seed", 0))
                df["purpose"] = purpose
                res.append(df)
            tw = pd.DataFrame([{"controller": "water_only_twin", "regime": k, "year": y, "scenario": "fully_solid",
                                "demand": "deterministic", "seed": 0, "purpose": purpose,
                                **twin_metrics(k, y, "fully_solid", purpose)} for y in yrs])
            res.append(tw)
    out = pd.concat(res, ignore_index=True)
    out["o3_config_sha256"] = o3c.config_hash()
    tag = "_".join(f"r{k}" for k in a.regimes)
    out.to_csv(o3c.RESULTS / f"m3_baselines_{tag}.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(o3c.RESULTS / f"m3_b1_grid_{tag}.csv", index=False)
    cols = ["E_del_window_kWh", "g_window_kWh", "overheat_h", "shield_water_h", "shield_pcm_h", "guard_hours",
            "hours_over_limit", "t_pcm_max_C", "bypass_switches", "residual_pct"]
    print(out.groupby(["regime", "purpose", "controller"])[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
