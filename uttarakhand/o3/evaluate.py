"""
o3/evaluate.py
==============
Run a controller through one episode and compute the plan's section 8.4 metrics with O2's definitions:

  E_del_window_kWh   energy delivered up to the 50 C target in the 18:00-09:00 window (O2 E_del45_window, at 50 C)
  g_window_kWh       E_del_window minus the water-only twin's (same year, reset scenario, demand) -- O2's g_window
  g_annual_kWh       same for the whole year
  serve_frac_window  volume drawn at >= 50 C / volume drawn, in the window (O2 compliance_vol_frac_window_insitu)
  tau_serve_insitu_h mean hours from 18:00 until the outlet first drops below 50 C (<= 15), in-run
                     (O2's tau_serve is a no-sun restart scenario; this in-run variant is NOT the same number)
  tau_serve_o2_*     O2's own definition (night=True; o3/night.py, checked against O2 in tests/test_night.py)
  serve_frac_o2_mean O2's serve_frac (volume at >= 50 C / volume drawn, 18:00-09:00 no-sun restart)
  shield_water_h / shield_pcm_h / overheat_h   kernel shield-active time; overheat = water + PCM (O2 d_overheat basis)
  hours_over_limit   kernel time with water > 75 C or PCM > 65 C (hard limits)
  guard_hours        hours whose max reached 72 C (top) or 62 C (PCM): the contract's safety_violation_t
  bypass_hours, bypass_switches, peak temperatures, episode energy residual (% of collector energy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from o3 import night as o3night
from o3.controllers import B0AlwaysOn
from o3.env import UKPCMEnv, WINDOW_HOURS

METRIC_WEIGHTS = {"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0, "E_norm_Wh": 1.0}   # metric-only runs (no reward use)


def run_episode(controller, regime: int, year: int, scenario: str, purpose: str, demand_mode="deterministic",
                seed: int = 0, water_only: bool = False, weights: dict | None = None, normalise: bool | None = None,
                env: UKPCMEnv | None = None, night: bool = False):
    """Returns (hourly DataFrame, reward sum). night=True adds O2's night serve scenario (o3/night.py) in
    df.attrs['night'] (about 60 % more kernel steps; evaluation only, never in training)."""
    normalise = controller.needs_norm if normalise is None else normalise
    env = env or UKPCMEnv(regime=regime, purpose=purpose, years=[year], scenarios=[scenario], demand_mode=demand_mode,
                          water_only=water_only, normalise=normalise, weights=weights or METRIC_WEIGHTS)
    obs, _ = env.reset(seed=seed, options={"year": year, "scenario": scenario})
    rows, total, states = [], 0.0, []
    done = False
    while not done:
        raw = env._raw_obs()
        a = controller.act(raw, obs)
        obs, r, done, _, info = env.step(a)
        total += r
        rows.append(info)
        if night and info["hour_of_day"] == o3night.START_ROW_HOUR:
            states.append((info["hour_index"], info["T_top"], env.stepper.T.copy(), env.stepper.h.copy()))
    df = pd.DataFrame(rows)
    if night:
        df.attrs["night"] = o3night.summarise(o3night.serve_scenarios(
            env.stepper, states, lambda i: env.demand.for_day(int(env._day[i]))))
    df.attrs["residual_pct"] = rows[-1]["episode_residual_pct"]
    df.attrs["demand_factors"] = env.demand.factors
    return df, total


def tau_serve_insitu(df: pd.DataFrame, target_C: float) -> float:
    """Mean over days of the hours from 18:00 until the end-of-hour outlet first drops below target (<= 15)."""
    hod = df["hour_of_day"].to_numpy()
    top = df["T_top"].to_numpy()
    starts = np.where(hod == 18)[0]
    vals = []
    for s in starts:
        seg = top[s:s + 15]
        if len(seg) < 15:
            break
        below = np.where(seg < target_C)[0]
        vals.append(float(below[0]) if len(below) else 15.0)
    return float(np.mean(vals)) if vals else float("nan")


def episode_metrics(df: pd.DataFrame, target_C: float = 50.0) -> dict:
    win = df["hour_of_day"].isin(WINDOW_HOURS).to_numpy()
    draw_win = df.loc[win, "draw_kg"].sum()
    return {
        "E_del_window_kWh": df.loc[win, "Edel_Wh"].sum() / 1000.0,
        "E_del_annual_kWh": df["Edel_Wh"].sum() / 1000.0,
        "serve_frac_window": df.loc[win, "vol_ge_target_kg"].sum() / draw_win if draw_win > 0 else np.nan,
        "tau_serve_insitu_h": tau_serve_insitu(df, target_C),
        "collector_kWh": df["Q_coll_Wh"].sum() / 1000.0,
        "shield_water_h": df["shield_water_s"].sum() / 3600.0,
        "shield_pcm_h": df["shield_pcm_s"].sum() / 3600.0,
        "overheat_h": df["overheat_h"].sum(),
        "hours_over_limit": df["over_limit_s"].sum() / 3600.0,
        "guard_hours": int(df["guard_hit"].sum()),
        "t_water_max_C": df["max_water_C"].max(),
        "t_pcm_max_C": df["max_pcm_C"].max() if df["max_pcm_C"].max() > -1e8 else np.nan,   # kernel sentinel: no PCM
        "bypass_hours": int((df["action"] == 0).sum()),
        "bypass_switches": int(df["bypass_switch"].sum()),
        "residual_pct": df.attrs.get("residual_pct", np.nan),
        "reward": df["reward"].sum(),
        **df.attrs.get("night", {}),
    }


_twin_cache: dict = {}


def twin_metrics(regime, year, scenario, purpose, demand_mode="deterministic", seed=0, night=False) -> dict:
    key = (regime, year, scenario, purpose, demand_mode, seed, night)
    if key not in _twin_cache:
        df, _ = run_episode(B0AlwaysOn(), regime, year, scenario, purpose, demand_mode, seed, water_only=True,
                            night=night)
        _twin_cache[key] = episode_metrics(df)
    return _twin_cache[key]


def evaluate(controller, regime: int, years, purpose: str, scenarios=("fully_solid",), demand_mode="deterministic",
             seed: int = 0, weights: dict | None = None, keep_hourly: bool = False, night: bool = False):
    """One row per (year, scenario): controller metrics + paired gains against the water-only twin."""
    rows, hourly = [], {}
    for y in years:
        for sc in scenarios:
            df, total = run_episode(controller, regime, y, sc, purpose, demand_mode, seed, weights=weights,
                                    night=night)
            m = episode_metrics(df)
            if weights is not None:
                m["reward"] = total
            tw = twin_metrics(regime, y, sc, purpose, demand_mode, seed, night)
            m.update({"g_window_kWh": m["E_del_window_kWh"] - tw["E_del_window_kWh"],
                      "g_annual_kWh": m["E_del_annual_kWh"] - tw["E_del_annual_kWh"],
                      "twin_E_del_window_kWh": tw["E_del_window_kWh"], "twin_shield_water_h": tw["shield_water_h"]})
            if night:
                m.update({"gain_tau_serve_o2_h": m["tau_serve_o2_mean_h"] - tw["tau_serve_o2_mean_h"],
                          "twin_tau_serve_o2_mean_h": tw["tau_serve_o2_mean_h"]})
            rows.append({"controller": controller.name, "regime": regime, "year": y, "scenario": sc,
                         "demand": demand_mode, "seed": seed, **m})
            if keep_hourly:
                hourly[(y, sc)] = df
    out = pd.DataFrame(rows)
    return (out, hourly) if keep_hourly else out


def summarise(df: pd.DataFrame) -> dict:
    """Mean over the rows (years x scenarios) of one controller's evaluation table."""
    num = df.select_dtypes("number").drop(columns=["regime", "year", "seed"], errors="ignore")
    return num.mean().to_dict()


def lexicographic_best(cands: list[dict], tie_kWh: float) -> dict:
    """Plan 8.3: (1) zero hard-limit hours, (2) window delivery -- every candidate within `tie_kWh` of the best
    delivery is tied, (3) fewest shield hours, (4) fewest switches."""
    ok = [c for c in cands if c["hours_over_limit"] <= 1e-9] or cands
    best_del = max(c["E_del_window_kWh"] for c in ok)
    tied = [c for c in ok if c["E_del_window_kWh"] >= best_del - tie_kWh]
    return min(tied, key=lambda c: (c["overheat_h"], c["bypass_switches"]))
