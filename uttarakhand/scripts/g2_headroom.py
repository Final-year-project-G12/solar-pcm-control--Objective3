"""
scripts/g2_headroom.py -- gate G2 (plan 15.10): is there any hourly on/off policy that beats B0?

Validation years only (2021-2023), fully_solid, deterministic demand. Three probes:
  1. the B1 grid (already in results/m3_b1_grid_r{k}.csv, re-read here)
  2. random search: 60 policies from a family wider than B1 -- bypass when the hour of day is in a random set of
     daytime hours AND the top node >= a random threshold, optionally AND f_melt >= a random threshold
  3. single-hour probe: on one validation year, bypass exactly ONE hour (200 random daytime hours with the loop
     available) and record the change in window and annual delivery against B0. If no single bypass ever raises
     delivery, bypassing can only buy fewer shield hours with delivery.
"Beats B0" = the plan 8.3 lexicographic rule: zero hard-limit hours, then window delivery within the declared
tie (1 kWh/yr) or higher, then fewer shield hours.
Writes results/g2_random_r{k}.csv, results/g2_single_hour_r{k}.csv, results/g2_summary_r{k}.json
Usage: python scripts/g2_headroom.py --regime 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                                               # noqa: E402
from o3.controllers import I, B0AlwaysOn                                   # noqa: E402
from o3.evaluate import episode_metrics, evaluate, run_episode, summarise   # noqa: E402


class HourMaskRule:
    name = "G2rand"
    needs_norm = False

    def __init__(self, hours, theta_top, theta_f):
        self.hours, self.theta_top, self.theta_f = set(hours), theta_top, theta_f
        self._h = 0

    def act(self, raw, norm=None):
        hod = int(round(np.arctan2(raw[I["hour_sin"]], raw[I["hour_cos"]]) / (2 * np.pi) * 24)) % 24
        if hod in self.hours and raw[I["T_water_top_C"]] >= self.theta_top and raw[I["f_melt"]] >= self.theta_f:
            return 0
        return 1


class OneHourBypass:
    name = "one_hour"
    needs_norm = False

    def __init__(self, hour_index):
        self.target, self.t = hour_index, 0

    def act(self, raw, norm=None):
        a = 0 if self.t == self.target else 1
        self.t += 1
        return a


def beats(c, b0, tie):
    if c["hours_over_limit"] > 1e-9:
        return False
    if c["E_del_window_kWh"] > b0["E_del_window_kWh"] + tie:
        return True
    return c["E_del_window_kWh"] >= b0["E_del_window_kWh"] - tie and c["overheat_h"] < b0["overheat_h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, required=True)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--n-single", type=int, default=200)
    a = ap.parse_args()
    cfg = o3c.load()
    k, yrs, tie = a.regime, cfg["years"]["validate"], cfg["selection"]["delivery_tie_kWh_per_yr"]
    rng = np.random.default_rng(1000 + k)
    b0 = summarise(evaluate(B0AlwaysOn(), k, yrs, "validate"))

    grid = pd.read_csv(o3c.RESULTS / f"m3_b1_grid_r{k}.csv")
    n_b1_beat = int(sum(beats(r, b0, tie) for r in grid.to_dict("records")))

    rows = []
    for i in range(a.n_random):
        hours = sorted(rng.choice(np.arange(8, 17), size=int(rng.integers(1, 9)), replace=False).tolist())
        th, tf = float(rng.uniform(45, 72)), float(rng.choice([0.0, rng.uniform(0.5, 1.0)]))
        s = summarise(evaluate(HourMaskRule(hours, th, tf), k, yrs, "validate"))
        rows.append({"policy": i, "hours": " ".join(map(str, hours)), "theta_top": th, "theta_f": tf, **s,
                     "beats_B0": beats(s, b0, tie)})
    rnd = pd.DataFrame(rows)
    rnd["o3_config_sha256"] = o3c.config_hash()
    rnd.to_csv(o3c.RESULTS / f"g2_random_r{k}.csv", index=False)

    y = yrs[0]
    df0, _ = run_episode(B0AlwaysOn(), k, y, "fully_solid", "validate")
    m0 = episode_metrics(df0)
    avail = np.where(df0["loop_available"].to_numpy() & (df0["Q_coll_Wh"].to_numpy() > 0))[0]
    probe = []
    for h in rng.choice(avail, size=min(a.n_single, len(avail)), replace=False):
        df, _ = run_episode(OneHourBypass(int(h)), k, y, "fully_solid", "validate")
        m = episode_metrics(df)
        probe.append({"hour_index": int(h), "hour_of_day": int(df0.hour_of_day[h]), "T_top_before": float(df0.T_top[h - 1]) if h else np.nan,
                      "d_E_del_window_kWh": m["E_del_window_kWh"] - m0["E_del_window_kWh"],
                      "d_E_del_annual_kWh": m["E_del_annual_kWh"] - m0["E_del_annual_kWh"],
                      "d_overheat_h": m["overheat_h"] - m0["overheat_h"]})
    pr = pd.DataFrame(probe)
    pr["o3_config_sha256"] = o3c.config_hash()
    pr.to_csv(o3c.RESULTS / f"g2_single_hour_r{k}.csv", index=False)

    summ = {"regime": k, "validation_years": yrs, "tie_kWh": tie,
            "B0": {kk: b0[kk] for kk in ("E_del_window_kWh", "overheat_h", "hours_over_limit")},
            "b1_grid_n": len(grid), "b1_grid_n_beat_B0": n_b1_beat,
            "b1_best_delivery_gap_kWh": float(grid.E_del_window_kWh.max() - b0["E_del_window_kWh"]),
            "random_n": len(rnd), "random_n_beat_B0": int(rnd.beats_B0.sum()),
            "random_best_delivery_gap_kWh": float(rnd.E_del_window_kWh.max() - b0["E_del_window_kWh"]),
            "single_hour_year": y, "single_hour_n": len(pr),
            "single_hour_max_d_window_kWh": float(pr.d_E_del_window_kWh.max()),
            "single_hour_max_d_annual_kWh": float(pr.d_E_del_annual_kWh.max()),
            "single_hour_n_window_gain_gt_1e-6": int((pr.d_E_del_window_kWh > 1e-6).sum()),
            "single_hour_mean_d_overheat_h": float(pr.d_overheat_h.mean())}
    summ["G2_pass"] = bool(n_b1_beat + summ["random_n_beat_B0"] > 0)
    summ["o3_config_sha256"] = o3c.config_hash()
    (o3c.RESULTS / f"g2_summary_r{k}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
