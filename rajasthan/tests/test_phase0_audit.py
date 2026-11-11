"""
Phase O3-0 tests — contract loader and validator (Rajasthan).

Run with: pytest tests/test_phase0_audit.py -v
"""

import copy
import pathlib
import pytest

import config_O2
from o3.utils.contract_loader import (load_contract, validate_contract, assert_contract_valid, get_regime_design,
                                      contract_sha256)

CONTRACT_PATH = config_O2.CONTRACT_FILE


def test_contract_exists():
    assert CONTRACT_PATH.exists(), (
        f"Contract not found at {CONTRACT_PATH}. Copy obj3_environment_contract_rajasthan_v3.json from the O2 tree.")


def test_contract_loads():
    c = load_contract(CONTRACT_PATH)
    assert isinstance(c, dict)
    assert "contract_version" in c


def test_contract_is_rajasthan():
    assert load_contract(CONTRACT_PATH)["state"] == "rajasthan"


def test_contract_version_v3():
    c = load_contract(CONTRACT_PATH)
    assert c["contract_version"].startswith("obj3_contract_v3"), "Contract must be v3.x (the v2 file has no version field)."


def test_tank_volume_50L():
    vol = load_contract(CONTRACT_PATH)["fixed_system"]["tank_volume_L"]
    assert abs(vol - 50.0) < 0.5, f"Rajasthan tank must be 50 L, got {vol}"


def test_collector_area_1p5():
    area = load_contract(CONTRACT_PATH)["fixed_system"]["collector_area_m2"]
    assert abs(area - 1.5) < 0.05, f"Rajasthan collector must be 1.5 m2, got {area}"


def test_flow_envelope_is_rajasthan():
    gl = load_contract(CONTRACT_PATH)["global_limits"]
    assert (gl["flow_min_kg_s"], gl["flow_max_kg_s"]) == (0.01, 0.05)


def test_safety_limits_sane():
    gl = load_contract(CONTRACT_PATH)["global_limits"]
    assert gl["max_safe_water_temp_C"] <= 80.0
    assert gl["max_safe_pcm_temp_C"] <= 70.0
    assert gl["delivery_target_temp_C"] < gl["max_safe_water_temp_C"]


def test_held_out_test_present():
    for rd in load_contract(CONTRACT_PATH)["static_design_per_regime"]:
        assert len(rd.get("held_out_2011_2015", [])) > 0, f"Regime {rd['regime_id']} has no held_out_2011_2015 data."


def test_held_out_temps_within_limits_for_chosen_designs():
    c = load_contract(CONTRACT_PATH)
    gl = c["global_limits"]
    for rd in c["static_design_per_regime"]:
        rows = [r for r in rd["held_out_2011_2015"] if r["role"] == "chosen"]
        assert len(rows) == 5, "chosen design must have 5 held-out years"
        for row in rows:
            assert row["t_pcm_max"] <= gl["max_safe_pcm_temp_C"] + 0.1
            assert row["t_water_max"] <= gl["max_safe_water_temp_C"] + 0.1


def test_three_regimes():
    ids = [rd["regime_id"] for rd in load_contract(CONTRACT_PATH)["static_design_per_regime"]]
    assert sorted(ids) == [0, 1, 2]


def test_cluster0_small_sample_policy_recorded():
    rd = get_regime_design(load_contract(CONTRACT_PATH), 0)
    assert rd["o1_candidate_pool_status"] == "undersized"
    assert "small_sample_policy" in rd


def test_full_validation_passes():
    errors = validate_contract(load_contract(CONTRACT_PATH))
    assert errors == [], "Contract validation errors:\n" + "\n".join(errors)


def test_assert_contract_valid_no_raise():
    assert assert_contract_valid(CONTRACT_PATH) is not None


def test_get_regime_design():
    c = load_contract(CONTRACT_PATH)
    expect = {0: "OM55", 1: "PureTemp 60", 2: "PureTemp 58"}
    for rid in [0, 1, 2]:
        rd = get_regime_design(c, rid)
        assert rd["regime_id"] == rid
        assert expect[rid] in rd["selected_pcm"]
        assert rd["n_capsules"] == 52


# --- the validator must REJECT the Tamil Nadu / old-schema mistakes -----------------------

def test_validator_rejects_tn_system_size():
    c = copy.deepcopy(load_contract(CONTRACT_PATH))
    c["fixed_system"]["tank_volume_L"], c["fixed_system"]["collector_area_m2"] = 100.0, 2.5
    errs = validate_contract(c)
    assert any("Tank volume" in e for e in errs) and any("Collector area" in e for e in errs)


def test_validator_rejects_old_version_and_wrong_state():
    c = copy.deepcopy(load_contract(CONTRACT_PATH))
    c["contract_version"], c["state"] = "", "tamilnadu"
    errs = validate_contract(c)
    assert any("version" in e for e in errs) and any("state" in e for e in errs)


def test_validator_rejects_flow_outside_envelope():
    c = copy.deepcopy(load_contract(CONTRACT_PATH))
    c["static_design_per_regime"][0]["nominal_flow_kg_s"] = 0.2
    assert any("flow" in e.lower() for e in validate_contract(c))


def test_provenance_detects_changed_o2_file():
    c = copy.deepcopy(load_contract(CONTRACT_PATH))
    c["provenance"]["wp6_band.csv"] = "0" * 64
    assert any("Provenance mismatch" in e for e in validate_contract(c))


def test_copy_equals_o2_original():
    if config_O2.O2_CONTRACT_V3.exists():
        assert contract_sha256(CONTRACT_PATH) == contract_sha256(config_O2.O2_CONTRACT_V3)
