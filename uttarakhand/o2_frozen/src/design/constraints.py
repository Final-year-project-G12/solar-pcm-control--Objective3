"""
src/design/constraints.py
===========================
Wraps geometry.compute_geometry() with the geometry-engine bounds
(`geometry_engine` block of configs/design_space.yaml): range checks (capsule diameter, capsule count, derived thickness) so every
generated design gets ONE valid/invalid verdict plus a single reason code,
before any physics is run. The simulator never sees a geometrically
invalid design.

Reason-code precedence (first hit wins, matches the framework doc's list):
  bounds_violation -> overlap -> volume_exceeded -> passage_blocked ->
  pressure_drop_limit -> flow_out_of_range -> (valid)
"""

from src.design.schema import DesignVector
from src.design.geometry import compute_geometry
from src.io_utils import load_system_config, load_design_bounds


def _in_bounds(value, bounds):
    return bounds["min"] - 1e-12 <= value <= bounds["max"] + 1e-12


def check_design(design: DesignVector, system_config: dict = None,
                  design_bounds: dict = None) -> dict:
    system_config = system_config or load_system_config()
    design_bounds = design_bounds or load_design_bounds()

    # ---- 1. raw variable-range checks (before any geometry math) -------
    if design.capsule_shape not in design_bounds["capsule_shape"]:
        return _reject(design, "bounds_violation", f"shape {design.capsule_shape} not in "
                                                      f"{design_bounds['capsule_shape']}")
    if design.capsule_arrangement not in design_bounds["capsule_arrangement"]:
        return _reject(design, "bounds_violation",
                        f"arrangement {design.capsule_arrangement} not in "
                        f"{design_bounds['capsule_arrangement']}")
    if not _in_bounds(design.capsule_diameter_m, design_bounds["capsule_diameter_m"]):
        return _reject(design, "bounds_violation", "capsule_diameter_m out of range")

    count_bounds = design_bounds["capsule_count"]
    if not (count_bounds["min"] <= design.n_capsule <= count_bounds["max"]):
        return _reject(design, "bounds_violation", "n_capsule out of range")
    if design.n_capsule != int(design.n_capsule):
        return _reject(design, "bounds_violation", "n_capsule must be an integer")

    thickness_m = design.capsule_diameter_m / 2.0
    if not _in_bounds(thickness_m, design_bounds["pcm_thickness_m"]):
        return _reject(design, "bounds_violation", "derived pcm_thickness_m out of range")

    # ---- 2. geometry + hydraulics (may itself reject) -------------------
    geom = compute_geometry(design, system_config, design_bounds)
    geom["design"] = design.as_dict()
    geom["notes"] = None
    return geom


def _reject(design: DesignVector, reason: str, note: str) -> dict:
    return {
        "design": design.as_dict(),
        "valid": False,
        "reason": reason,
        "notes": note,
    }
