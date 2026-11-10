"""
Phase O3-0 tests — contract loader and validator.

Run with: pytest tests/test_phase0_audit.py -v
"""

import pathlib
import pytest
from o3.utils.contract_loader import load_contract, validate_contract, assert_contract_valid, get_regime_design

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"


def test_contract_exists():
    assert CONTRACT_PATH.exists(), (
        f"Contract not found at {CONTRACT_PATH}. "
        "Copy obj3_environment_contract_tamilnadu_v3.json from the O2 tree."
    )


def test_contract_loads():
    c = load_contract(CONTRACT_PATH)
    assert isinstance(c, dict)
    assert "contract_version" in c


def test_contract_version_v3():
    c = load_contract(CONTRACT_PATH)
    assert c["contract_version"].startswith("obj3_contract_v3"), (
        "Contract must be v3.x. The 2026-10-27 v2 contract is superseded."
    )


def test_tank_volume_100L():
    c = load_contract(CONTRACT_PATH)
    vol = c["fixed_system"]["tank_volume_L"]
    assert abs(vol - 100.0) < 0.5, f"Tank must be 100 L, got {vol}"


def test_collector_area_2p5():
    c = load_contract(CONTRACT_PATH)
    area = c["fixed_system"]["collector_area_m2"]
    assert abs(area - 2.5) < 0.05, f"Collector must be 2.5 m², got {area}"


def test_safety_limits_sane():
    c = load_contract(CONTRACT_PATH)
    gl = c["global_limits"]
    assert gl["max_safe_water_temp_C"] <= 80.0
    assert gl["max_safe_pcm_temp_C"]   <= 70.0
    assert gl["delivery_target_temp_C"] < gl["max_safe_water_temp_C"]


def test_held_out_test_present():
    c = load_contract(CONTRACT_PATH)
    for rd in c["static_design_per_regime"]:
        rows = rd.get("held_out_2011_2015", [])
        assert len(rows) > 0, (
            f"Regime {rd['regime_id']} has no held_out_2011_2015 data. "
            "Run g10 before starting O3."
        )


def test_held_out_temps_within_limits():
    c = load_contract(CONTRACT_PATH)
    gl = c["global_limits"]
    for rd in c["static_design_per_regime"]:
        for row in rd.get("held_out_2011_2015", []):
            assert row["t_pcm_max"]   <= gl["max_safe_pcm_temp_C"]   + 0.1, \
                f"Regime {rd['regime_id']} PCM exceeded limit: {row['t_pcm_max']}"
            assert row["t_water_max"] <= gl["max_safe_water_temp_C"] + 0.1, \
                f"Regime {rd['regime_id']} water exceeded limit: {row['t_water_max']}"


def test_three_regimes():
    c = load_contract(CONTRACT_PATH)
    regime_ids = [rd["regime_id"] for rd in c["static_design_per_regime"]]
    assert 0 in regime_ids and 1 in regime_ids and 2 in regime_ids, \
        f"Expected regimes 0,1,2 — got {regime_ids}"


def test_full_validation_passes():
    c = load_contract(CONTRACT_PATH)
    errors = validate_contract(c)
    assert errors == [], f"Contract validation errors:\n" + "\n".join(errors)


def test_assert_contract_valid_no_raise():
    # should not raise
    c = assert_contract_valid(CONTRACT_PATH)
    assert c is not None


def test_get_regime_design():
    c = load_contract(CONTRACT_PATH)
    for rid in [0, 1, 2]:
        rd = get_regime_design(c, rid)
        assert rd["regime_id"] == rid
        assert rd["selected_pcm"] in ("PureTemp 58", "RT57HC")
