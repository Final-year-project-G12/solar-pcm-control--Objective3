"""
src/design/space.py
======================
O2 G8 (2026-10-26) - the APPROVED design space (configs/design_space.yaml) as code.

Four design variables: PCM thickness (= sphere inner radius), capsule count (set through the PCM outer volume
fraction), flow rate, arrangement. `to_design()` turns a point of that space into a DesignVector and
`geometry_bounds()` gives the permissive bounds dict the geometry engine needs (configs/design_space.yaml, geometry_engine). Geometry feasibility is still checked EXACTLY by
src/design/constraints.check_design, never learned.
"""

import math
from functools import lru_cache
from pathlib import Path

import yaml

from src.design.schema import DesignVector
from src.io_utils import load_design_bounds, load_system_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "design_space.yaml"
ARRANGEMENTS = ("single-layer", "staggered", "radial")


@lru_cache(maxsize=1)
def load_space() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def wall_m() -> float:
    return load_system_config()["pcm_integration"]["capsule_wall_thickness_m"]


def v_tank_m3() -> float:
    return load_system_config()["tank"]["volume_L"] / 1000.0


def capsule_outer_volume_m3(r_in_m: float) -> float:
    return 4.0 / 3.0 * math.pi * (r_in_m + wall_m()) ** 3


def count_range(r_in_m: float):
    sp = load_space()["variables"]["pcm_volume_fraction"]
    v = capsule_outer_volume_m3(r_in_m)
    return math.ceil(sp["min"] * v_tank_m3() / v - 1e-9), math.floor(sp["max"] * v_tank_m3() / v + 1e-9)


def count_for_fraction(r_in_m: float, fraction: float) -> int:
    n_lo, n_hi = count_range(r_in_m)
    n = round(fraction * v_tank_m3() / capsule_outer_volume_m3(r_in_m))
    return int(min(max(n, n_lo), n_hi))


def to_design(r_in_m: float, fraction: float, flow_kg_s: float, arrangement: str) -> DesignVector:
    return DesignVector(capsule_diameter_m=2.0 * (r_in_m + wall_m()),
                        n_capsule=count_for_fraction(r_in_m, fraction),
                        flow_rate_kg_s=float(flow_kg_s), capsule_arrangement=arrangement)


def geometry_bounds() -> dict:
    """Bounds dict for check_design: wide enough not to bind (the space is enforced by to_design()),
    the fraction cap and flow range kept at the approved values."""
    return load_design_bounds()


def in_space(r_in_m, fraction, flow, arrangement) -> bool:
    v = load_space()["variables"]
    return (v["inner_radius_m"]["min"] - 1e-12 <= r_in_m <= v["inner_radius_m"]["max"] + 1e-12
            and v["pcm_volume_fraction"]["min"] - 1e-12 <= fraction <= v["pcm_volume_fraction"]["max"] + 1e-12
            and v["flow_rate_kg_s"]["min"] - 1e-12 <= flow <= v["flow_rate_kg_s"]["max"] + 1e-12
            and arrangement in v["arrangement"])
