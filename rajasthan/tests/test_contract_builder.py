"""
Tests for the O2-side builder src/handoff/build_obj3_contract_rajasthan_v3.py:
it must only restructure — every design number equals its O2 source.
"""
import json

import config_O2


def test_builder_verifies_against_sources():
    from src.handoff import build_obj3_contract_rajasthan_v3 as b
    checked = b.verify_against_sources()
    assert len(checked) >= 20


def test_v3_numbers_equal_v2_numbers():
    v2 = json.loads((config_O2.O2_RESULTS_DIR / "obj3_environment_contract_rajasthan_v2.json").read_text(encoding="utf-8"))
    v3 = json.loads(config_O2.CONTRACT_FILE.read_text(encoding="utf-8"))
    for r2, d3 in zip(v2["regimes"], v3["static_design_per_regime"]):
        ch = r2["selection"]["chosen"]
        assert (d3["selected_pcm"], d3["n_capsules"], d3["nominal_flow_kg_s"], d3["capsule_inner_radius_m"]) == \
               (ch["pcm"], ch["N"], ch["flow"], ch["R_in_m"])
        assert d3["held_out_2011_2015"] == r2["held_out_2011_2015"]
    for a, b in (("delivery_temp_target_C", "delivery_target_temp_C"), ("max_water_temp_C", "max_safe_water_temp_C"),
                 ("max_pcm_temp_C", "max_safe_pcm_temp_C"), ("pump_flow_min_kg_s", "flow_min_kg_s"),
                 ("pump_flow_max_kg_s", "flow_max_kg_s")):
        assert v2["global_limits"][a] == v3["global_limits"][b]
