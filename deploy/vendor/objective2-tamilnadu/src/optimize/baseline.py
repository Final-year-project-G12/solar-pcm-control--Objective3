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
from src.io_utils import dt_for, solver_overrides_for
from src.simulation.run_case import run_case
from src.weather import roles
from src.weather.inventory import CURRENT_MEDOIDS
from config import STATE

OUT_DIR = Path(__file__).resolve().parents[2] / "results"
FLOW_GRID = np.round(np.linspace(0.010, 0.050, 17), 6)
METRICS = ["useful_energy_kWh", "E_del45_window_kWh", "E_del45_annual_kWh", "shield_water_h"]


def grid_path(purpose: str, cluster_id: int) -> Path:
    return OUT_DIR / f"plain_baseline_{purpose}_r{cluster_id}.csv"


def build_plain_grid(cluster_id: int, purpose: str, flows=FLOW_GRID) -> pd.DataFrame:
    rows = []
    bounds = geometry_bounds()
    for year in roles.years_for(purpose):
        w = roles.load_point_year(STATE, cluster_id, CURRENT_MEDOIDS[cluster_id], year, purpose)
        for fl in flows:
            dv = to_design(0.02, 0.05, float(fl), "staggered")        # geometry irrelevant for the plain tank
            o = run_case(STATE, cluster_id, None, dv, design_bounds=bounds, weather=w,
                         tank_model="multinode", compute_tes_metrics=True, night_day_stride=0,
                         system_config_overrides=solver_overrides_for(purpose))
            m, t = o["metrics"], o["tes"]
            rows.append({"regime": cluster_id, "year": year, "flow": float(fl), "dt_s": dt_for(purpose),
                         "useful_energy_kWh": m["useful_energy_kWh"], "E_del45_window_kWh": t["E_del45_window_kWh"],
                         "E_del45_annual_kWh": t["E_del45_annual_kWh"], "shield_water_h": m["shield_water_h"]})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(grid_path(purpose, cluster_id), index=False)
    return df


@lru_cache(maxsize=None)
def _interpolators(purpose: str, cluster_id: int, year: int):
    path = grid_path(purpose, cluster_id)
    df = pd.read_csv(path)
    # g_window and friends are DIFFERENCES against this grid, so the grid must
    # have been built at the same timestep as the PCM run being differenced.
    # Grids written before dt_for() existed have no dt_s column; they were all
    # built at the search step, which is what "explore" still uses.
    want = dt_for(purpose)
    have = float(df["dt_s"].iloc[0]) if "dt_s" in df.columns else dt_for("explore")
    if abs(have - want) > 1e-9:
        raise RuntimeError(
            f"{path.name} was built at dt={have} s but purpose '{purpose}' is "
            f"evaluated at dt={want} s. Differencing across two discretizations "
            f"would put the dt error straight into g_window. Rebuild it: "
            f"python scripts/g8_build_plain_baseline.py {cluster_id}"
        )
    g = df[df.year == year].sort_values("flow")
    return {m: PchipInterpolator(g["flow"].to_numpy(), g[m].to_numpy()) for m in METRICS}


def plain_at(cluster_id: int, year: int, flow: float, purpose: str = None) -> dict:
    """Water-only tank metrics at `flow` (interpolated)."""
    purpose = purpose or roles.role_of(year)
    return {m: float(f(flow)) for m, f in _interpolators(purpose, cluster_id, year).items()}
