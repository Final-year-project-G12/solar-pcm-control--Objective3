"""
src/optimize/evaluate.py
===========================
O2 G8 - the ONE function that turns a design into paired objectives by running the real simulator
(multi-node tank, N = 10, dt = 150 s) over the weather years of a role. Nothing here is a surrogate.

Per weather year the gains are paired against the water-only tank at the same flow (baseline.plain_at):
  g_window   E_del,45 in 18:00-09:00 (in-situ) minus the plain tank's        [kWh/yr]   f1, primary
  g_annual   annual E_del,45 minus the plain tank's                          [kWh/yr]   f2
  g_useful   annual useful energy minus the plain tank's                     [kWh/yr]   (old metric, for reference)
  d_overheat (shield-water h + shield-PCM h) minus the plain tank's shield-water h   [h/yr]   f3 (lower is better)
  t_pcm_max  peak PCM temperature [C]; the design is FEASIBLE only if it is <= 65 C in every year (user decision)
Night hold/serve scenarios are skipped here (night_day_stride=0); they are computed for finalists.
"""

from functools import lru_cache

import numpy as np

from src.design.constraints import check_design
from src.design.space import geometry_bounds, load_space, to_design
from src.io_utils import load_system_config
from src.optimize.baseline import plain_at
from src.simulation.run_case import run_case
from src.weather import roles
from src.weather.inventory import CURRENT_MEDOIDS

OBJECTIVE_KEYS = ("g_window", "g_annual", "g_useful", "d_overheat")


@lru_cache(maxsize=64)
def _weather(cluster_id: int, year: int, purpose: str):
    return roles.load_point_year("rajasthan", cluster_id, CURRENT_MEDOIDS[cluster_id], year, purpose)


def evaluate_design(cluster_id, pcm, r_in_m, fraction, flow, arrangement, purpose="explore", years=None):
    """Simulate one design for every weather year of `purpose` (or the given subset)."""
    years = list(years) if years is not None else roles.years_for(purpose)
    dv = to_design(r_in_m, fraction, flow, arrangement)
    cfg = load_system_config()
    out = {"cluster": cluster_id, "pcm": pcm, "R_in_m": r_in_m, "fraction_req": fraction, "flow": flow,
           "arrangement": arrangement, "N": dv.n_capsule}
    geom = check_design(dv, cfg, geometry_bounds())
    out["valid"], out["reason"] = bool(geom["valid"]), geom.get("reason") or ""
    if not geom["valid"]:
        return out
    out["fraction"] = geom["pcm_volume_fraction"]
    per_year = []
    sp = load_space()
    for y in years:
        r = run_case("rajasthan", cluster_id, pcm, dv, design_bounds=geometry_bounds(), weather=_weather(cluster_id, y, purpose),
                     tank_model=sp["model"]["tank_model"], n_nodes=sp["model"]["n_nodes"],
                     stack_anchor=sp["model"]["stack_anchor"], compute_tes_metrics=True, night_day_stride=0)
        m, t = r["metrics"], r["tes"]
        p = plain_at(cluster_id, y, flow, purpose)
        per_year.append({
            "year": y,
            "g_window": t["E_del45_window_kWh"] - p["E_del45_window_kWh"],
            "g_annual": t["E_del45_annual_kWh"] - p["E_del45_annual_kWh"],
            "g_useful": m["useful_energy_kWh"] - p["useful_energy_kWh"],
            "d_overheat": (m["shield_water_h"] + m["shield_pcm_h"]) - p["shield_water_h"],
            "t_pcm_max": m["max_pcm_temp_C"], "t_water_max": m["max_water_temp_C"],
            "hours_over_limit": m["hours_over_limit"], "residual_pct": m["residual_pct_of_collector"],
            "pcm_mass_kg": m["pcm_mass_kg"],
        })
    out["per_year"] = per_year
    for k in OBJECTIVE_KEYS:
        out[k] = float(np.mean([d[k] for d in per_year]))
        out[k + "_sd"] = float(np.std([d[k] for d in per_year], ddof=1)) if len(per_year) > 1 else 0.0
    out["t_pcm_max"] = float(max(d["t_pcm_max"] for d in per_year))
    out["t_water_max"] = float(max(d["t_water_max"] for d in per_year))
    out["pcm_mass_kg"] = per_year[0]["pcm_mass_kg"]
    out["residual_pct"] = float(max(d["residual_pct"] for d in per_year))
    out["feasible"] = bool(out["t_pcm_max"] <= sp["constraints"]["pcm_peak_limit_C"]
                           and out["t_water_max"] <= sp["constraints"]["water_peak_limit_C"])
    return out


def flatten(res: dict) -> dict:
    """One CSV-friendly row (drops the per-year list)."""
    return {k: v for k, v in res.items() if k != "per_year"}


def evaluate_night(cluster_id, pcm, r_in_m, fraction, flow, arrangement, purpose, years=None, stride=15):
    """Night hold/serve scenarios (WP1 B) for a design AND its same-flow water-only twin, run exactly (no grid), on the
    given weather role. Returns per-year dict list with the PCM tank's values and the paired gains (PCM minus plain)."""
    years = list(years) if years is not None else roles.years_for(purpose)
    dv = to_design(r_in_m, fraction, flow, arrangement)
    sp = load_space()
    out = []
    for y in years:
        kw = dict(design_bounds=geometry_bounds(), weather=_weather(cluster_id, y, purpose),
                  tank_model=sp["model"]["tank_model"], n_nodes=sp["model"]["n_nodes"],
                  stack_anchor=sp["model"]["stack_anchor"], compute_tes_metrics=True, night_day_stride=stride)
        a = run_case("rajasthan", cluster_id, pcm, dv, **kw)
        b = run_case("rajasthan", cluster_id, None, dv, **kw)
        ta, tb, ma, mb = a["tes"], b["tes"], a["metrics"], b["metrics"]
        row = {"year": y}
        for k in ("tau_hold_mean_h", "tau_serve_mean_h", "serve_frac_45_mean", "serve_T_out_min_mean_C"):
            row[k] = ta.get(k, np.nan)
            row["plain_" + k] = tb.get(k, np.nan)
            row["gain_" + k] = ta.get(k, np.nan) - tb.get(k, np.nan)
        row.update({"g_window": ta["E_del45_window_kWh"] - tb["E_del45_window_kWh"],
                    "g_annual": ta["E_del45_annual_kWh"] - tb["E_del45_annual_kWh"],
                    "g_useful": ma["useful_energy_kWh"] - mb["useful_energy_kWh"],
                    "d_overheat": (ma["shield_water_h"] + ma["shield_pcm_h"]) - mb["shield_water_h"],
                    "shield_water_h": ma["shield_water_h"], "shield_pcm_h": ma["shield_pcm_h"],        # issue 4: like-for-like split, reported beside f3
                    "plain_shield_water_h": mb["shield_water_h"],
                    "t_pcm_max": ma["max_pcm_temp_C"], "t_water_max": ma["max_water_temp_C"],
                    "sf_pcm": ma["solar_fraction"], "sf_plain": mb["solar_fraction"],
                    "residual_pct": ma["residual_pct_of_collector"], "pcm_mass_kg": ma["pcm_mass_kg"], "N": dv.n_capsule})
        out.append(row)
    return out
