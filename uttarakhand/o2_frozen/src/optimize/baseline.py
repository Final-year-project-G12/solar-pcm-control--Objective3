"""
src/optimize/baseline.py
===========================
O2 G8 - the paired baseline. Decision (user, 2026-10-26): every PCM design is compared with a
WATER-ONLY tank at the SAME flow, weather year, demand, shield and tank model. Flow is continuous, so the
plain tank is simulated on a flow grid per (regime, year) and interpolated (monotone PCHIP); the interpolation
error is measured against exact plain runs in tests/test_design_space.py.

Grids are built per weather-year ROLE: 'explore' and 'select' freely; the held-out 'test' baseline may only be
built after roles.acquire_test_lock() (load_point_year enforces it).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.design.space import geometry_bounds, to_design
from src.simulation.run_case import run_case
from src.weather import roles
from src.weather.inventory import CURRENT_MEDOIDS

OUT_DIR = Path(__file__).resolve().parents[2] / "results"
FLOW_GRID = np.round(np.linspace(0.010, 0.050, 17), 6)
METRICS = ["useful_energy_kWh", "E_del45_window_kWh", "E_del45_annual_kWh", "shield_water_h"]


def grid_path(purpose: str, cluster_id: int) -> Path:
    return OUT_DIR / f"plain_baseline_{purpose}_r{cluster_id}.csv"


def build_plain_grid(cluster_id: int, purpose: str, flows=FLOW_GRID) -> pd.DataFrame:
    rows = []
    bounds = geometry_bounds()
    for year in roles.years_for(purpose):
        w = roles.load_point_year("uttarakhand", cluster_id, CURRENT_MEDOIDS[cluster_id], year, purpose)
        for fl in flows:
            dv = to_design(0.02, 0.05, float(fl), "staggered")        # geometry irrelevant for the plain tank
            o = run_case("uttarakhand", cluster_id, None, dv, design_bounds=bounds, weather=w,
                         tank_model="multinode", compute_tes_metrics=True, night_day_stride=0)
            m, t = o["metrics"], o["tes"]
            rows.append({"regime": cluster_id, "year": year, "flow": float(fl),
                         "useful_energy_kWh": m["useful_energy_kWh"], "E_del45_window_kWh": t["E_del45_window_kWh"],
                         "E_del45_annual_kWh": t["E_del45_annual_kWh"], "shield_water_h": m["shield_water_h"]})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(grid_path(purpose, cluster_id), index=False)
    return df


@lru_cache(maxsize=None)
def _interpolators(purpose: str, cluster_id: int, year: int):
    df = pd.read_csv(grid_path(purpose, cluster_id))
    g = df[df.year == year].sort_values("flow")
    return {m: PchipInterpolator(g["flow"].to_numpy(), g[m].to_numpy()) for m in METRICS}


def plain_at(cluster_id: int, year: int, flow: float, purpose: str = None) -> dict:
    """Water-only tank metrics at `flow` (interpolated)."""
    purpose = purpose or roles.role_of(year)
    return {m: float(f(flow)) for m, f in _interpolators(purpose, cluster_id, year).items()}
