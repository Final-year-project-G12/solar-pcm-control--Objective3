"""
Phase O3-0 — contract loader and validator.

Loads obj3_environment_contract_tamilnadu_v3.json, validates internal
consistency, and refuses to proceed if anything is mismatched.
"""

import hashlib
import json
import pathlib
import sys
from typing import Any


def load_contract(contract_path: str | pathlib.Path) -> dict:
    """Load and return the contract JSON.  Raises FileNotFoundError if missing."""
    path = pathlib.Path(contract_path)
    if not path.exists():
        raise FileNotFoundError(
            f"O3 contract not found at {path}. "
            "Re-run src/handoff/build_obj3_contract_tamilnadu.py in the O2 tree."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def contract_sha256(contract_path: str | pathlib.Path) -> str:
    data = pathlib.Path(contract_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def validate_contract(c: dict) -> list[str]:
    """Return a list of error strings.  Empty list means contract is valid."""
    errors: list[str] = []

    # --- version ---------------------------------------------------------
    ver = c.get("contract_version", "")
    if not ver.startswith("obj3_contract_v3"):
        errors.append(
            f"Wrong contract version '{ver}'. O3 requires obj3_contract_v3.x. "
            "The 2026-10-27 contract (v2.0) is superseded and must NOT be used."
        )

    # --- hard limits present and sane ------------------------------------
    gl = c.get("global_limits", {})
    _check_field(errors, gl, "delivery_target_temp_C", float, lo=40.0, hi=55.0)
    _check_field(errors, gl, "max_safe_water_temp_C",  float, lo=70.0, hi=80.0)
    _check_field(errors, gl, "max_safe_pcm_temp_C",    float, lo=60.0, hi=70.0)
    _check_field(errors, gl, "flow_min_kg_s",          float, lo=0.005, hi=0.02)
    _check_field(errors, gl, "flow_max_kg_s",          float, lo=0.03,  hi=0.06)

    # guard-band thresholds must be below hard limits
    for rd in c.get("static_design_per_regime", []):
        for d in rd.get("held_out_2011_2015", []):
            pcm_max = d.get("t_pcm_max", 0.0)
            water_max = d.get("t_water_max", 0.0)
            if pcm_max > gl.get("max_safe_pcm_temp_C", 65.0) + 0.1:
                errors.append(
                    f"Regime {rd.get('regime_id')} held-out t_pcm_max {pcm_max:.1f} "
                    f"exceeds hard limit {gl.get('max_safe_pcm_temp_C')}"
                )
            if water_max > gl.get("max_safe_water_temp_C", 75.0) + 0.1:
                errors.append(
                    f"Regime {rd.get('regime_id')} held-out t_water_max {water_max:.1f} "
                    f"exceeds hard limit {gl.get('max_safe_water_temp_C')}"
                )

    # --- fixed system specs (100 L / 2.5 m2) ----------------------------
    fs = c.get("fixed_system", {})
    if abs(fs.get("tank_volume_L", 0) - 100.0) > 0.5:
        errors.append(
            f"Tank volume is {fs.get('tank_volume_L')} L — must be 100 L. "
            "The 50 L contract is superseded."
        )
    if abs(fs.get("collector_area_m2", 0) - 2.5) > 0.05:
        errors.append(
            f"Collector area is {fs.get('collector_area_m2')} m² — must be 2.5 m²."
        )

    # --- at least one regime with a selected PCM -------------------------
    designs = c.get("static_design_per_regime", [])
    if not designs:
        errors.append("No static_design_per_regime entries found.")
    for rd in designs:
        if not rd.get("selected_pcm"):
            errors.append(f"Regime {rd.get('regime_id')} has no selected_pcm.")
        if not rd.get("n_capsules"):
            errors.append(f"Regime {rd.get('regime_id')} has n_capsules=0 or missing.")

    # --- held-out test must have been run --------------------------------
    held_out_found = any(rd.get("held_out_2011_2015") for rd in designs)
    if not held_out_found:
        errors.append(
            "No held_out_2011_2015 data found in any regime. "
            "Run g10 held-out test before starting O3."
        )

    return errors


def assert_contract_valid(contract_path: str | pathlib.Path) -> dict:
    """Load + validate.  Raises RuntimeError with all errors if invalid."""
    c = load_contract(contract_path)
    errors = validate_contract(c)
    if errors:
        msg = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise RuntimeError(
            f"O3 contract validation failed ({len(errors)} error(s)):\n{msg}"
        )
    return c


# --- helpers -------------------------------------------------------------

def _check_field(errors: list, d: dict, key: str, typ: type,
                 lo: float | None = None, hi: float | None = None) -> None:
    if key not in d:
        errors.append(f"Missing required field '{key}' in global_limits.")
        return
    val = d[key]
    if not isinstance(val, (int, float)):
        errors.append(f"'{key}' must be numeric, got {type(val).__name__}.")
        return
    if lo is not None and val < lo:
        errors.append(f"'{key}' = {val} is below expected minimum {lo}.")
    if hi is not None and val > hi:
        errors.append(f"'{key}' = {val} is above expected maximum {hi}.")


def get_regime_design(contract: dict, regime_id: int) -> dict:
    """Return the static design block for one regime.  Raises KeyError if not found."""
    for rd in contract.get("static_design_per_regime", []):
        if rd.get("regime_id") == regime_id:
            return rd
    raise KeyError(f"regime_id={regime_id} not found in contract.")
