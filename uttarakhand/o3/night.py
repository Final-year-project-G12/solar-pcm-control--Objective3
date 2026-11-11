"""
o3/night.py
===========
O2's night SERVE scenario (src/metrics/tes_metrics.night_scenarios), computed for any controller's episode.

Definition (O2, unchanged): for every day, take the controller's OWN end-of-17:00 state (= the 18:00 state, all node
temperatures and PCM enthalpies), run 15 hours (18:00 -> 09:00 local) with NO SUN (GHI = 0 -> the collector loop is
off, which is exactly the stepper's BYPASS; test T3) and the episode's demand, then
  tau_serve_h   = hours until the end-of-hour top-node outlet first drops below the target (linear within the hour,
                  O2's crossing_hours, <= 15)
  serve_frac    = volume drawn at >= target / volume drawn, over those 15 hours
Each restart is independent of the episode (it does not change what the controller did next).
tests/test_night.py checks this against O2's own run_case(..., night_day_stride=1) to rtol 1e-9.
"""

from __future__ import annotations

import copy

import numpy as np

from o3 import o2_bridge as ob  # noqa: F401  (O2 path)
from o3.stepper import BYPASS, HourlyStepper
from src.metrics.tes_metrics import NIGHT_HOURS, NIGHT_START_HOUR, crossing_hours


def serve_scenarios(stepper: HourlyStepper, states, demand_for_hour) -> list[dict]:
    """stepper: the episode's stepper (copied, never advanced). states: list of (hour_index_of_17h_row, T_top_C,
    T_nodes, h_nodes). demand_for_hour(i) -> the 24-point curve in force at hour i."""
    st = copy.copy(stepper)
    target = st.case.target_C
    out = []
    for i, t_top, T, h in states:
        if i + 1 + NIGHT_HOURS > st.n_hours:
            continue
        st.t, st.T, st.h = i + 1, T.copy(), h.copy()
        tops, draw, vol = [], 0.0, 0.0
        for k in range(NIGHT_HOURS):
            r = st.step(BYPASS, demand_for_hour(i + 1 + k))
            tops.append(r["T_top"])
            draw += r["draw_kg"]
            vol += r["vol_ge_target_kg"]
        tau, _ = crossing_hours(t_top, np.array(tops), target)
        out.append({"hour_index_start": i + 1, "tau_serve_h": tau, "serve_frac": vol / draw if draw > 0 else np.nan})
    return out


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"tau_serve_o2_mean_h": np.nan, "tau_serve_o2_p10_h": np.nan, "serve_frac_o2_mean": np.nan}
    tau = np.array([r["tau_serve_h"] for r in rows])
    sf = np.array([r["serve_frac"] for r in rows])
    return {"tau_serve_o2_mean_h": float(tau.mean()), "tau_serve_o2_p10_h": float(np.quantile(tau, 0.10)),
            "serve_frac_o2_mean": float(np.nanmean(sf)), "night_n_days": len(rows)}


START_ROW_HOUR = NIGHT_START_HOUR - 1      # the end-of-hour record at 17 h is the 18:00 state
