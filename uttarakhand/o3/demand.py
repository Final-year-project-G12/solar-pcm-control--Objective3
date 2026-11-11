"""
o3/demand.py
============
Hot-water demand for one episode, as 24-point curves per day fed to the kernel.

O2 uses one deterministic 300 L/day curve every day (data/demand/demand_profile_uttarakhand.csv, local time,
morning peak 07:30). 'Dynamic demand' is added by O3: each day draws a volume factor and a whole-hour timing
shift, and the day's curve is built with O2's own DemandModel(volume_multiplier, timing_shift_hours).
The parameter ranges are ASSUMPTIONS (plan Q3, configs/o3_config.yaml demand.stochastic).
"""

from __future__ import annotations

import numpy as np

from o3 import o2_bridge as ob  # noqa: F401  (sets up the O2 path)
from src.io_utils import load_demand_profile
from src.simulation.demand_profile import DemandModel


def curve24(volume_factor: float = 1.0, shift_h: float = 0.0) -> np.ndarray:
    dm = DemandModel(load_demand_profile(ob.STATE), volume_factor, shift_h)
    return np.array([dm.draw_mass_kg_for_hour(h) for h in range(24)], dtype=float)


class EpisodeDemand:
    """Per-day 24-h curves for a year. mode 'deterministic' repeats O2's curve."""

    def __init__(self, n_days: int, mode: str = "deterministic", rng: np.random.Generator | None = None,
                 params: dict | None = None):
        self.mode = mode
        base = curve24()
        if mode == "deterministic":
            self.days = np.tile(base, (n_days, 1))
            self.factors = np.ones(n_days)
            self.shifts = np.zeros(n_days)
            return
        if mode != "stochastic":
            raise ValueError(mode)
        p = params or {}
        rng = rng or np.random.default_rng(0)
        lo, hi = p.get("volume_factor_clip", [0.6, 1.4])
        self.factors = np.clip(rng.normal(1.0, p.get("volume_factor_sd", 0.15), n_days), lo, hi)
        self.shifts = rng.choice(np.asarray(p.get("timing_shift_h", [-1, 0, 1]), dtype=float), n_days)
        cache = {}
        rows = []
        for f, s in zip(self.factors, self.shifts):
            if s not in cache:
                cache[s] = curve24(1.0, s)
            rows.append(cache[s] * f)
        self.days = np.array(rows)

    def for_day(self, day: int) -> np.ndarray:
        return self.days[min(day, len(self.days) - 1)]
