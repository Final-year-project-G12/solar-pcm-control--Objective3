"""
Phase O3-0 — contract loader and validator (Rajasthan).

Loads obj3_environment_contract_rajasthan_v3.json, validates internal consistency and refuses
to proceed if anything is mismatched.

Same structure, function names and error style as the Tamil Nadu loader. What differs, on purpose:
  * the contract must say state == "rajasthan";
  * the fixed system is the Rajasthan one: 50 L tank / 1.5 m2 collector (TN is 100 L / 2.5 m2);
  * the flow envelope and temperature limits are checked against the Rajasthan values of the
    O2 shared config (0.010-0.050 kg/s, 45 / 72 / 75 / 62 / 65 C), not TN's;
  * the held-out limit check looks at the CHOSEN design rows. The held-out table also holds
    'best_of_pcm' alternatives that O2 did not release; they are reported, not enforced;
  * a provenance check re-hashes the O2 source files the contract was built from, so a stale
    contract (O2 result changed after the contract was built) is caught (plan section 2:
    "the environment refuses to start if a contract hash, config hash ... check fails").
"""

import hashlib
import json
import pathlib

# --- Rajasthan fixed system (O2 configs/system_config_shared.yaml) -------------------------
RJ_TANK_VOLUME_L = 50.0
RJ_COLLECTOR_AREA_M2 = 1.5
RJ_FLOW_MIN = 0.010
RJ_FLOW_MAX = 0.050
RJ_N_REGIMES = 3


def load_contract(contract_path: str | pathlib.Path) -> dict:
    """Load and return the contract JSON.  Raises FileNotFoundError if missing."""
    path = pathlib.Path(contract_path)
    if not path.exists():
        raise FileNotFoundError(
            f"O3 contract not found at {path}. "
            "Re-run `python -m src.handoff.build_obj3_contract_rajasthan_v3` in the O2 Rajasthan tree "
            "and copy results/obj3_environment_contract_rajasthan_v3.json into contracts/."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def contract_sha256(contract_path: str | pathlib.Path) -> str:
    data = pathlib.Path(contract_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def validate_contract(c: dict, check_provenance: bool = True) -> list[str]:
    """Return a list of error strings.  Empty list means contract is valid."""
    errors: list[str] = []

    # --- identity / version ----------------------------------------------
    if c.get("state") != "rajasthan":
        errors.append(f"Contract state is '{c.get('state')}', expected 'rajasthan'.")
    ver = c.get("contract_version", "")
    if not ver.startswith("obj3_contract_v3"):
        errors.append(
            f"Wrong contract version '{ver}'. O3 requires obj3_contract_v3.x. "
            "The v2 Rajasthan file (no version field, `regimes`, pump_flow_*) is the OLD schema."
        )

    # --- hard limits present and sane ------------------------------------
    gl = c.get("global_limits", {})
    _check_field(errors, gl, "delivery_target_temp_C", float, lo=40.0, hi=55.0)
    _check_field(errors, gl, "max_safe_water_temp_C",  float, lo=70.0, hi=80.0)
    _check_field(errors, gl, "max_safe_pcm_temp_C",    float, lo=60.0, hi=70.0)
    _check_field(errors, gl, "flow_min_kg_s",          float, lo=0.005, hi=0.02)
    _check_field(errors, gl, "flow_max_kg_s",          float, lo=0.03,  hi=0.06)
    if abs(gl.get("flow_min_kg_s", 0.0) - RJ_FLOW_MIN) > 1e-9 or abs(gl.get("flow_max_kg_s", 0.0) - RJ_FLOW_MAX) > 1e-9:
        errors.append(
            f"Flow envelope {gl.get('flow_min_kg_s')}-{gl.get('flow_max_kg_s')} kg/s is not the Rajasthan "
            f"envelope {RJ_FLOW_MIN}-{RJ_FLOW_MAX} kg/s."
        )

    # held-out limit check on the released (chosen) designs
    for rd in c.get("static_design_per_regime", []):
        for d in rd.get("held_out_2011_2015", []):
            if d.get("role", "chosen") != "chosen":
                continue
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

    # --- fixed system specs (50 L / 1.5 m2) ------------------------------
    fs = c.get("fixed_system", {})
    if abs(fs.get("tank_volume_L", 0) - RJ_TANK_VOLUME_L) > 0.5:
        errors.append(
            f"Tank volume is {fs.get('tank_volume_L')} L — Rajasthan must be {RJ_TANK_VOLUME_L:.0f} L "
            "(the 100 L / 2.5 m2 pairing is Tamil Nadu's)."
        )
    if abs(fs.get("collector_area_m2", 0) - RJ_COLLECTOR_AREA_M2) > 0.05:
        errors.append(
            f"Collector area is {fs.get('collector_area_m2')} m² — Rajasthan must be {RJ_COLLECTOR_AREA_M2} m²."
        )

    # --- shield matches limits -------------------------------------------
    sh = c.get("safety_shield", {})
    if sh.get("bypass_water_C", 0) >= gl.get("max_safe_water_temp_C", 75.0):
        errors.append("Shield water guard band is not below the water hard limit.")
    if sh.get("bypass_pcm_C", 0) >= gl.get("max_safe_pcm_temp_C", 65.0):
        errors.append("Shield PCM guard band is not below the PCM hard limit.")

    # --- three regimes, each with a selected PCM -------------------------
    designs = c.get("static_design_per_regime", [])
    if not designs:
        errors.append("No static_design_per_regime entries found.")
    ids = sorted(rd.get("regime_id") for rd in designs)
    if ids != list(range(RJ_N_REGIMES)):
        errors.append(f"Expected regimes {list(range(RJ_N_REGIMES))}, got {ids}.")
    for rd in designs:
        if not rd.get("selected_pcm"):
            errors.append(f"Regime {rd.get('regime_id')} has no selected_pcm.")
        if not rd.get("n_capsules"):
            errors.append(f"Regime {rd.get('regime_id')} has n_capsules=0 or missing.")
        flow = rd.get("nominal_flow_kg_s", 0.0)
        if not (RJ_FLOW_MIN - 1e-12 <= flow <= RJ_FLOW_MAX + 1e-12):
            errors.append(f"Regime {rd.get('regime_id')} nominal flow {flow} is outside the flow envelope.")

    # --- held-out test must have been run (by O2) ------------------------
    held_out_found = any(rd.get("held_out_2011_2015") for rd in designs)
    if not held_out_found:
        errors.append(
            "No held_out_2011_2015 data found in any regime. "
            "Run g10 held-out test before starting O3."
        )

    # --- provenance: the O2 files the contract was built from must be unchanged ---
    if check_provenance:
        errors.extend(check_provenance_hashes(c))

    return errors


def check_provenance_hashes(c: dict) -> list[str]:
    """Re-hash the O2 source files named in `provenance` and compare. Returns error strings.

    If the O2 tree cannot be found (e.g. contract copied to another machine) the check is
    skipped silently; the contract's own SHA-256 is still recorded by phase 0.
    """
    import config_O2

    errs: list[str] = []
    prov = c.get("provenance", {})
    o2 = config_O2.O2_ROOT
    if not o2.exists():
        return errs

    def _cmp(rel: str, expected: str) -> None:
        p = o2 / rel
        if not p.exists():
            errs.append(f"Provenance file missing in O2 tree: {rel}")
            return
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != expected:
            errs.append(f"Provenance mismatch for {rel}: contract has {expected[:12]}..., file is {got[:12]}... "
                        "(an O2 file changed after the contract was built; rebuild the contract).")

    for name, h in prov.get("selection", {}).items():
        _cmp(f"results/{name}", h)
    for name in ("g10_heldout_results.csv", "held_out_test.lock", "wp6_band.csv"):
        if name in prov:
            _cmp(f"results/{name}", prov[name])
    for name in ("configs/system_config_shared.yaml", "configs/design_space.yaml", "configs/states/rajasthan.yaml"):
        if name in prov:
            _cmp(name, prov[name])
    return errs


def assert_contract_valid(contract_path: str | pathlib.Path, check_provenance: bool = True) -> dict:
    """Load + validate.  Raises RuntimeError with all errors if invalid."""
    c = load_contract(contract_path)
    errors = validate_contract(c, check_provenance=check_provenance)
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
