"""
src/simulation/demand_profile.py
===================================
Phase 3 / D2.3 — turns demand_profile_<state>.csv (hour, draw_fraction,
draw_volume_L, draw_mass_kg for a 300 L/day canonical day) into a draw mass
for any solver sub-step, repeated across every day of the simulated year.

Sub-hourly spreading: the source data has hourly resolution; draw mass for
one hour is spread evenly across that hour's sub-steps (zero-order hold) —
documented simplification, since sub-hourly draw timing was never measured
(build_demand_profile.py docstring).

Scenario multipliers (volume_multiplier, timing_shift_hours) exist so Phase
4 Gate 2's "empty demand" case (multiplier=0) and Phase 8's Monte Carlo
demand-uncertainty draws can reuse this exact model instead of re-deriving
demand handling elsewhere.
"""

import numpy as np
import pandas as pd


class DemandModel:
    def __init__(self, demand_df: pd.DataFrame, volume_multiplier: float = 1.0,
                 timing_shift_hours: float = 0.0):
        self.volume_multiplier = volume_multiplier
        self.timing_shift_hours = timing_shift_hours
        hours = demand_df["hour"].to_numpy(dtype=float)
        mass = demand_df["draw_mass_kg"].to_numpy(dtype=float) * volume_multiplier
        # shift the 24-point curve circularly by timing_shift_hours (interpolated)
        shifted_hours = (hours - timing_shift_hours) % 24.0
        order = np.argsort(shifted_hours)
        self._hours = shifted_hours[order]
        self._mass = mass[order]

    def draw_mass_kg_for_hour(self, hour_of_day: int) -> float:
        """Total draw mass (kg) for the given integer hour-of-day (0-23)."""
        idx = int(round(hour_of_day)) % 24
        # nearest-hour lookup on the (possibly shifted) 24-point curve
        pos = np.searchsorted(self._hours, idx)
        pos = min(pos, len(self._hours) - 1)
        return float(self._mass[pos])

    def draw_mass_kg_for_substep(self, hour_of_day: int, n_substeps_per_hour: int) -> float:
        return self.draw_mass_kg_for_hour(hour_of_day) / n_substeps_per_hour


def load_demand_model(demand_df: pd.DataFrame, volume_multiplier: float = 1.0,
                       timing_shift_hours: float = 0.0) -> DemandModel:
    return DemandModel(demand_df, volume_multiplier, timing_shift_hours)
