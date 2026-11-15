"""
src/metrics/tes_metrics.py
=============================
O2 G3 / plan WP1 (2026-10-26) — threshold-aware thermal-energy-storage
metrics, replacing "annual E_load" (which counts above-target and below-target
heat alike and rewards overheating).

All energies are relative to the mains temperature, so a tank at mains
temperature stores nothing. T_deliv = 45 C = system_config delivery.target_temp_C.
The outlet temperature is the bulk water temperature (single-node tank; the
multi-node tank of G4 will supply its top-node temperature instead).

Definitions (plan WP1):
  A  E_stored(t)  = m_w*cp*(T_w - T_m) + m_pcm*(h(T_pcm) - h(T_m))   [wall inside h]
     E_rec,45(t)  = m_w*cp*max(T_w-45,0) + m_pcm*max(h(T_pcm)-h(45),0)
     (a PCM with Tm < 45 C therefore contributes only its sensible heat above 45 C)
  B  tau_hold,45  hours, from 18:00 local, that the tank alone (no draw, no sun,
                  real ambient) keeps the outlet >= 45 C. Start state = the
                  design's own 18:00 state of each day. Right-censored at hold_cap_h.
     serve        18:00 -> 09:00 local, no sun, the project demand profile:
                  serve_frac_45 = volume drawn at >=45 C / volume drawn;
                  tau_serve_h   = hours from 18:00 until the end-of-hour outlet
                                  first drops below 45 C (<= 15).
  C  peak temperatures: outlet/bulk water, PCM, per day and per year (never mixed).
  D  latent utilisation: daily melt-fraction swing and equivalent full cycles.
  E  supporting: E_del,45 = sum m_dot*cp*(min(T_out,45)-T_m) (annual and in the
     18:00-09:00 window), draw-weighted compliance.

Choices the plan left open (fixed here, change deliberately):
  * retention start = 18:00 local (fixed clock, not "daily maximum");
  * night window = 18:00-09:00 local (plan sec. 6A item 4; WP1 text says 08:00,
    WP2 says 22:00 -> 09:00; 6A is the authoritative decision log);
  * the night scenarios start from each design's OWN 18:00 state on that weather
    day (so plain tank and PCM tank are compared day by day on the same weather).
"""

import copy

import numpy as np
import pandas as pd

from src.simulation.capsule_enthalpy import enthalpy_of_temperature
from src.simulation.tank_model import run_year

J_PER_KWH = 3.6e6
NIGHT_START_HOUR = 18          # local clock, scenario start
NIGHT_HOURS = 15               # 18:00 -> 09:00
WINDOW_HOURS = set(range(18, 24)) | set(range(0, 9))


# ---------------------------------------------------------------- A: stored energy
def _pcm_enthalpy(T_pcm, props) -> np.ndarray:
    return np.array([enthalpy_of_temperature(float(t), props) for t in np.asarray(T_pcm, dtype=float)])


def stored_energy_kWh(T_w, T_pcm, ctx) -> np.ndarray:
    T_w = np.asarray(T_w, dtype=float)
    e = ctx["m_w_kg"] * ctx["cp_w"] * (T_w - ctx["mains_temp_C"])
    if ctx["has_pcm"]:
        p = ctx["pcm_props"]
        e = e + ctx["m_pcm_kg"] * (_pcm_enthalpy(T_pcm, p) - enthalpy_of_temperature(ctx["mains_temp_C"], p))
    return e / J_PER_KWH


def recoverable_energy_kWh(T_w, T_pcm, ctx, threshold_C=None) -> np.ndarray:
    thr = ctx["target_C"] if threshold_C is None else threshold_C
    T_w = np.asarray(T_w, dtype=float)
    e = ctx["m_w_kg"] * ctx["cp_w"] * np.maximum(T_w - thr, 0.0)
    if ctx["has_pcm"]:
        p = ctx["pcm_props"]
        e = e + ctx["m_pcm_kg"] * np.maximum(_pcm_enthalpy(T_pcm, p) - enthalpy_of_temperature(thr, p), 0.0)
    return e / J_PER_KWH


def closure_residual_pct(hourly: pd.DataFrame, summary: dict, ctx) -> float:
    """|dE_stored - (sum Q_collector - sum Q_load - sum Q_loss)| / sum Q_collector, in %,
    from the HOURLY records (independent of the solver's own accumulator)."""
    if "E_stored_kWh" in hourly.columns:          # node-resolved model supplies exact values
        e_end = float(hourly["E_stored_kWh"].iloc[-1])
    else:
        e_end = stored_energy_kWh(hourly["T_w_C"].iloc[[-1]], hourly["T_pcm_C"].iloc[[-1]], ctx)[0]
    d_e = e_end - summary["E_stored_initial_kWh"]
    q_load = hourly["Q_load_net_Wh"] if "Q_load_net_Wh" in hourly.columns else hourly["Q_load_Wh"]
    flows = (hourly["Q_collector_Wh"].sum() - q_load.sum() - hourly["Q_loss_Wh"].sum()) / 1000.0
    denom = max(hourly["Q_collector_Wh"].sum() / 1000.0, 1e-9)
    return abs(d_e - flows) / denom * 100.0


# ---------------------------------------------------------------- helpers
def _day_id(hourly: pd.DataFrame) -> np.ndarray:
    h = hourly["hour_of_day"].to_numpy()
    return np.concatenate([[0], np.cumsum(np.diff(h) < 0)])


def crossing_hours(T_start: float, T_end_of_hour, threshold: float):
    """Hours until the outlet first falls below threshold (linear within the hour).
    Returns (hours, censored) -- censored=True if it never fell below."""
    if T_start < threshold:
        return 0.0, False
    prev = T_start
    for k, t in enumerate(T_end_of_hour):
        if t < threshold:
            frac = (prev - threshold) / (prev - t) if prev > t else 1.0
            return k + frac, False
        prev = t
    return float(len(T_end_of_hour)), True


# ---------------------------------------------------------------- B: night scenarios
def night_scenarios(weather: pd.DataFrame, demand, mains_temp_C: float, runtime, pcm_props_raw,
                    system_config: dict, hourly: pd.DataFrame, ctx: dict,
                    day_stride: int = 1, hold_cap_h: int = 72, runner=run_year) -> pd.DataFrame:
    """Per-day hold and serve scenarios, restarted from the design's own 18:00 state."""
    target = ctx["target_C"]
    zero_demand = copy.copy(demand)
    zero_demand._mass = np.zeros_like(demand._mass)
    starts = np.flatnonzero(hourly["hour_of_day"].to_numpy() == NIGHT_START_HOUR - 1)[::day_stride]
    out = []
    for r in starts:
        i = int(hourly["hour_index"].iloc[r])           # position in `weather`
        if i + 1 + NIGHT_HOURS > len(weather):
            continue
        state = {"T_w_C": float(hourly["T_w_C"].iloc[r]), "T_pcm_C": float(hourly["T_pcm_C"].iloc[r])}
        if "node_T" in hourly.attrs:                       # node-resolved restart (multi-node tank)
            state = {**state, "T_nodes": hourly.attrs["node_T"][r], "h_nodes": hourly.attrs["node_h"][r]}
        dark = weather.iloc[i + 1: i + 1 + max(NIGHT_HOURS, hold_cap_h)].copy()
        dark["GHI_Wm2"] = 0.0

        serve = runner(dark.iloc[:NIGHT_HOURS], demand, mains_temp_C, runtime, pcm_props_raw,
                         system_config, record_hourly=True, initial_state=state)
        sh = serve.hourly
        tau_serve, _ = crossing_hours(state["T_w_C"], sh["T_w_C"].to_numpy(), target)
        draw = float(sh["draw_kg"].sum())
        hold = runner(dark, zero_demand, mains_temp_C, runtime, pcm_props_raw, system_config,
                        record_hourly=True, initial_state=state, stop_when_Tw_below_C=target)
        tau_hold, censored = crossing_hours(state["T_w_C"], hold.hourly["T_w_C"].to_numpy(), target)
        out.append({
            "hour_index_start": i + 1, "T_w_start_C": state["T_w_C"], "T_pcm_start_C": state["T_pcm_C"],
            "tau_hold_h": tau_hold, "hold_censored": censored,
            "tau_serve_h": tau_serve,
            "serve_frac_45": (float(sh["vol_ge_target_kg"].sum()) / draw) if draw > 0 else np.nan,
            "serve_E_del45_kWh": float(sh["Edel_target_Wh"].sum()) / 1000.0,
            "serve_T_out_min_C": float(sh["T_w_min_C"].min()),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- all metrics
def compute_all(hourly: pd.DataFrame, result_summary: dict, ctx: dict) -> dict:
    """In-run metrics from the hourly record (A, C, D, E). Night scenarios are separate."""
    if "E_stored_kWh" in hourly.columns:
        e_st, e_rec = hourly["E_stored_kWh"].to_numpy(), hourly["E_rec45_kWh"].to_numpy()
    else:
        e_st = stored_energy_kWh(hourly["T_w_C"], hourly["T_pcm_C"], ctx)
        e_rec = recoverable_energy_kWh(hourly["T_w_C"], hourly["T_pcm_C"], ctx)
    day = _day_id(hourly)
    n_days = int(day.max()) + 1
    g = pd.DataFrame({"day": day, "e_rec": e_rec, "tw_max": hourly["T_w_max_C"].to_numpy(),
                      "tp_max": hourly["T_pcm_max_C"].to_numpy(), "fmin": hourly["f_melt_min"].to_numpy(),
                      "fmax": hourly["f_melt_max"].to_numpy()}).groupby("day")
    daily = g.agg(e_rec=("e_rec", "max"), tw_max=("tw_max", "max"), tp_max=("tp_max", "max"),
                  fmin=("fmin", "min"), fmax=("fmax", "max"))
    in_win = hourly["hour_of_day"].isin(WINDOW_HOURS).to_numpy()
    draw_win = float(hourly.loc[in_win, "draw_kg"].sum())
    tot_edel = float(hourly["Edel_target_Wh"].sum()) / 1000.0
    win_edel = float(hourly.loc[in_win, "Edel_target_Wh"].sum()) / 1000.0
    s = result_summary
    pcm = ctx["has_pcm"]
    return {
        # A
        "E_stored_mean_kWh": float(e_st.mean()), "E_stored_max_kWh": float(e_st.max()),
        "E_rec45_mean_kWh": float(e_rec.mean()), "E_rec45_max_kWh": float(e_rec.max()),
        "E_rec45_daily_max_mean_kWh": float(daily["e_rec"].mean()),
        "closure_residual_pct": closure_residual_pct(hourly, s, ctx),
        # C  (single node: outlet == bulk water)
        "T_deliv_max_C": float(hourly["T_w_max_C"].max()),
        "T_w_daily_max_mean_C": float(daily["tw_max"].mean()),
        "T_w_daily_max_p95_C": float(daily["tw_max"].quantile(0.95)),
        "T_pcm_max_C": float(hourly["T_pcm_max_C"].max()) if pcm else float("nan"),
        "T_pcm_daily_max_mean_C": float(daily["tp_max"].mean()) if pcm else float("nan"),
        # D
        "LU_daily_swing_mean": float((daily["fmax"] - daily["fmin"]).mean()) if pcm else 0.0,
        "latent_full_cycles_per_year": float(s["sum_abs_df"] / 2.0) if pcm else 0.0,
        "latent_full_cycles_per_day": float(s["sum_abs_df"] / 2.0 / n_days) if pcm else 0.0,
        # E
        "E_del45_annual_kWh": tot_edel, "E_del45_window_kWh": win_edel,
        "E_del45_window_share": win_edel / tot_edel if tot_edel > 0 else float("nan"),
        "compliance_vol_frac_annual": (s["vol_ge_target_kg"] / s["draw_kg_total"]) if s["draw_kg_total"] > 0 else float("nan"),
        "compliance_vol_frac_window_insitu": (float(hourly.loc[in_win, "vol_ge_target_kg"].sum()) / draw_win)
                                              if draw_win > 0 else float("nan"),
        "n_days": n_days,
    }


def summarise_night(night: pd.DataFrame) -> dict:
    if night.empty:
        return {}
    return {
        "night_n_days": int(len(night)),
        "tau_hold_mean_h": float(night["tau_hold_h"].mean()),
        "tau_hold_p10_h": float(night["tau_hold_h"].quantile(0.10)),
        "tau_hold_p50_h": float(night["tau_hold_h"].median()),
        "tau_hold_p90_h": float(night["tau_hold_h"].quantile(0.90)),
        "tau_hold_censored_frac": float(night["hold_censored"].mean()),
        "tau_serve_mean_h": float(night["tau_serve_h"].mean()),
        "tau_serve_p10_h": float(night["tau_serve_h"].quantile(0.10)),
        "serve_frac_45_mean": float(night["serve_frac_45"].mean()),
        "serve_frac_45_p10": float(night["serve_frac_45"].quantile(0.10)),
        "serve_E_del45_mean_kWh": float(night["serve_E_del45_kWh"].mean()),
        "serve_T_out_min_mean_C": float(night["serve_T_out_min_C"].mean()),
    }
