"""
src/handoff/build_obj3_contract_rajasthan_v3.py
================================================
Objective 3 contract, schema v3, for RAJASTHAN. Writes
results/obj3_environment_contract_rajasthan_v3.json.

WHY THIS FILE EXISTS (decided with the student, 2026-11-11, option (a))
  The Objective 3 code was written against the Tamil Nadu v3 schema
  (obj3_contract_v3.x: contract_version, static_design_per_regime, fixed_system,
  delivery_target_temp_C, flow_min/max_kg_s, ...). Rajasthan only had the v2 file
  (results/obj3_environment_contract_rajasthan_v2.json: no version field, `regimes`,
  delivery_temp_target_C, pump_flow_*). This builder RE-EXPRESSES the existing Rajasthan
  results in the v3 schema. It is a pure restructure:

    * no O2 result value is changed, recomputed, re-rounded or re-simulated;
    * every number is read from a file under results/ or configs/ (the v2 contract, the
      g10 selection logs, g10_heldout_results.csv, wp6_band.csv, held_out_test.lock,
      configs/*.yaml);
    * the Rajasthan numbers stay Rajasthan numbers: 50 L tank, 1.5 m2 collector,
      three regimes, flow envelope 0.010-0.050 kg/s, 5 held-out years (2011-2015).
      The Tamil Nadu 100 L / 2.5 m2 pairing is NOT used anywhere.

  `verify_against_sources()` re-reads the sources and checks every copied number against
  its origin; it is called at the end of build() and by tests/test_contract_builder in O3.

Run (from the O2 Rajasthan root):
    python -m src.handoff.build_obj3_contract_rajasthan_v3
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import BASE_DIR, RESULTS_DIR
from src.io_utils import load_system_config, load_state_config, load_design_bounds

OUT = RESULTS_DIR
V2 = OUT / "obj3_environment_contract_rajasthan_v2.json"
NEW = OUT / "obj3_environment_contract_rajasthan_v3.json"
CONTRACT_VERSION = "obj3_contract_v3.0_rajasthan_2026-11-11"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_v2() -> dict:
    return json.loads(V2.read_text(encoding="utf-8"))


def build() -> Path:
    v2 = _load_v2()
    cfg = load_system_config()
    state = load_state_config("rajasthan")
    bounds = load_design_bounds()
    held = pd.read_csv(OUT / "g10_heldout_results.csv")
    band = pd.read_csv(OUT / "wp6_band.csv")
    lock = json.loads((OUT / "held_out_test.lock").read_text(encoding="utf-8"))

    gl2 = v2["global_limits"]
    sk = v2["control_skeleton"]
    shield2 = sk["safety_shield"]

    c = {
        "state": "rajasthan",
        "contract_version": CONTRACT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validated_simulator_version": v2["simulator_version"],
        "source": ("Objective 2 Rajasthan pipeline results (results/g10_selection_r*.json, g10_heldout_results.csv, "
                   "wp6_band.csv, held_out_test.lock, obj3_environment_contract_rajasthan_v2.json, configs/*.yaml). "
                   "Built by src/handoff/build_obj3_contract_rajasthan_v3.py; restructure only, no value changed."),
        "supersession_note": ("RE-EXPRESSES results/obj3_environment_contract_rajasthan_v2.json in the v3 schema that the "
                              "Objective 3 code reads. Same designs, same limits, same held-out results; only the field "
                              "layout and names changed. The older results/obj3_environment_contract_rajasthan.json "
                              "(legacy Phase-7 deployables, UTC-indexed demand, single-node tank) stays superseded."),
        "global_limits": {
            "delivery_target_temp_C": gl2["delivery_temp_target_C"],
            "max_safe_water_temp_C": gl2["max_water_temp_C"],
            "max_safe_pcm_temp_C": gl2["max_pcm_temp_C"],
            "max_pressure_bar": gl2["max_pressure_bar"],
            "min_irradiance_cutoff_Wm2": gl2["min_irradiance_cutoff_Wm2"],
            "flow_min_kg_s": gl2["pump_flow_min_kg_s"],
            "flow_max_kg_s": gl2["pump_flow_max_kg_s"],
            "pump_efficiency": cfg["pump"]["efficiency"],
            "note": ("HARD limits (renamed from the v2 keys delivery_temp_target_C / max_water_temp_C / max_pcm_temp_C / "
                     "pump_flow_*). The controller may never command outside the flow envelope and may never drive "
                     "water above max_safe_water_temp_C or PCM above max_safe_pcm_temp_C."),
        },
        "fixed_system": {
            "tank_volume_L": cfg["tank"]["volume_L"],
            "tank_height_to_diameter_ratio": cfg["tank"]["height_to_diameter_ratio"],
            "tank_insulation_m": cfg["tank"]["insulation_m"],
            "tank_U_W_m2K": cfg["tank"]["U_tank_W_m2K"],
            "collector_area_m2": cfg["collector"]["area_m2"],
            "collector_fr_tau_alpha": cfg["collector"]["fr_tau_alpha"],
            "collector_fr_ul_W_m2K": cfg["collector"]["fr_ul_W_m2K"],
            "collector_flow_dependent_fr_enabled": cfg["collector"]["flow_dependent_fr"]["enabled"],
            "capsule_wall_material": cfg["pcm_integration"]["capsule_wall_material"],
            "capsule_wall_thickness_m": cfg["pcm_integration"]["capsule_wall_thickness_m"],
            "liquid_convection_enhancement": cfg["pcm_integration"]["liquid_convection_enhancement"],
            "melting_half_width_K": cfg["pcm_integration"]["melting_half_width_K"],
            "n_nodes": 10,
            "stack_anchor": "center",
            "WARNING_tank_collector": ("Rajasthan is 50 L / 1.5 m2 (33.3 L/m2, below the 40-100 L/m2 band of IS 12976:2023 - "
                                       "a standing O2 limitation). Tamil Nadu is 100 L / 2.5 m2. Never mix the two."),
        },
    }

    # ---- per regime (v3 flat layout) -------------------------------------------------------
    designs = []
    for r2 in v2["regimes"]:
        k = r2["regime_id"]
        sel = r2["selection"]
        ch = sel["chosen"]
        sreg = next(x for x in state["regimes"] if int(x["cluster_id"]) == k)
        wall = cfg["pcm_integration"]["capsule_wall_thickness_m"]
        h = held[held.regime == k]
        designs.append({
            "regime_id": k,
            "label": sreg["label"],
            "medoid_point_id": v2["weather_sequences"]["points"][str(k)],
            "n_points": sreg["n_points"],
            "population_covered": sreg["population_covered"],
            "T_mains_est_C": sreg["T_mains_est_C"],
            "o1_pcm_shortlist": sreg["pcm_shortlist"],
            "o1_candidate_pool_status": sreg.get("candidate_pool_status"),
            "o1_n_survivors_in_cluster": sreg.get("n_survivors_in_cluster"),
            "o1_kendalls_w_cluster": sreg.get("kendalls_w_cluster"),
            "selected_pcm": ch["pcm"],
            "selected_pcm_o1_consensus_rank": ch["pcm_rank"],
            "capsule_inner_radius_m": ch["R_in_m"],
            "capsule_inner_diameter_mm": 2.0 * ch["R_in_m"] * 1000.0,
            "capsule_outer_diameter_mm": 2.0 * (ch["R_in_m"] + wall) * 1000.0,
            "pcm_volume_fraction": ch["fraction"],
            "n_capsules": ch["N"],
            "arrangement": ch["arrangement"],
            "nominal_flow_kg_s": ch["flow"],
            "pcm_mass_kg": ch["pcm_mass_kg"],
            "select_year_metrics": {
                "g_window_kWh_yr": ch["g_window"], "g_annual_kWh_yr": ch["g_annual"],
                "d_overheat_h": ch["d_overheat"], "t_pcm_max_C": ch["t_pcm_max"], "t_water_max_C": ch["t_water_max"],
            },
            "repeatability_band": dict(r2["wp6_band"]),
            "selection_funnel": {
                "n_candidates": sel["n_candidates"],
                "n_S0_indistinguishable_on_g_window": sel["n_S0_indistinguishable_on_g_window"],
                "n_S1": sel["n_S1"], "n_S2": sel["n_S2"],
                "note": ("Tie-breaks were declared before results were seen (plan WP6). Designs differing by less than the "
                         "repeatability band are treated as tied, then overheating, PCM mass, capsule count and temperature "
                         "margin decide."),
            },
            "selection_fingerprint_sha256": sel["fingerprint_sha256"],
            "alternatives_best_per_pcm": sel["alternatives_best_per_pcm"],
            "water_only_twin": r2["water_only_twin"],
            "held_out_2011_2015": r2["held_out_2011_2015"],
        })
    # Cluster 0: O1 candidate pool is undersized (n = 4). Recorded, not hidden (memory: accept calibrated-kappa policy).
    for d in designs:
        if d["o1_candidate_pool_status"] == "undersized":
            d["small_sample_policy"] = (
                f"O1 candidate pool for regime {d['regime_id']} is UNDERSIZED (n={d['o1_n_survivors_in_cluster']} survivors). "
                "Policy (decided 2026-09-17): accept the calibrated-kappa ranking, report the caveat. O3 treats the design "
                "as given; it does not re-rank PCMs.")
    c["static_design_per_regime"] = designs

    # ---- held-out summary (computed counts, no values altered) -------------------------------
    chosen = held[held.role == "chosen"]
    c["held_out_test"] = {
        "status": "COMPLETED AND LOCKED by O2 - O3 must not tune against these years",
        "years": sorted(int(y) for y in chosen.year.unique()),
        "run_id": lock["run_id"],
        "locked_utc": lock["utc"],
        "selection_fingerprint": lock["selection_fingerprint"],
        "n_designs_tested": int(held.groupby(["regime", "role"]).ngroups),
        "coverage": ("The chosen design of each regime plus the best design of each other O1 PCM, over 5 held-out years. "
                     "Rows with role 'best_of_pcm' are alternatives, not released designs."),
        "gates_chosen_designs": {
            "rows": int(len(chosen)),
            "pcm_peak_le_65C_rows": int((chosen.t_pcm_max <= gl2["max_pcm_temp_C"]).sum()),
            "water_peak_le_75C_rows": int((chosen.t_water_max <= gl2["max_water_temp_C"]).sum()),
            "g_window_positive_rows": int((chosen.g_window > 0).sum()),
            "gain_serve_frac_45_positive_rows": int((chosen.gain_serve_frac_45_mean > 0).sum()),
            "note": ("g_window is the O2 gain in delivered energy to 45 C in the 18:00-09:00 window versus the water-only twin. "
                     "gain_serve_frac_45_mean is NOT positive in every row: PCM does not extend the time at 45 C in the "
                     "evening in every regime (plan section 0). Counts are computed from g10_heldout_results.csv."),
        },
    }
    c["weather_sequences"] = v2["weather_sequences"]
    c["weather_sequences"]["enforcement"] = "src/weather/roles.py refuses to load a year whose role differs from the stated purpose"
    c["weather_sequences"]["medoid_points"] = v2["weather_sequences"]["points"]
    c["time_step_s"] = v2["time_step_s"]
    c["time_step_confirm_s"] = 37.5
    c["time_step_confirm_note"] = "g2 convergence studies: results/g2_timestep_convergence_150_75_37p5.csv and ..._300_150_75.csv"
    c["demand_profile_file"] = v2["demand_profile_file"]
    c["demand_daily_total_L"] = state["demand_profile"]["daily_total_L"]

    c["action_space"] = {
        "discrete_version": {
            "actions": sk["actions"],
            "action_notes": sk["action_notes"],
            "note": ("In the O2 kernel the only controllable element is whether the collector loop circulates; 'discharge' is "
                     "passive PCM-water exchange and is physically identical to bypass at the kernel level."),
        },
        "hybrid_version": {
            "mode": sk["actions"],
            "continuous": {"flow_rate_kg_s": "within the flow envelope in global_limits"},
            "note": sk["recommended_action_space"],
        },
        "recommended": "discrete_version",
        "recommendation_note": ("The v2 text recommended the continuous-flow hybrid; the model's F_R is flow-independent "
                                "(deferred_future_work), so a flow action would reward a model artefact. v3 therefore "
                                "recommends the discrete version until F_R(flow) is enabled as a declared physics change."),
        "hard_rule": sk["hard_rule"],
    }
    c["reset_scenarios"] = v2["reset_scenarios"]
    c["acceptance_test_before_drl_training"] = v2["acceptance_test_before_drl_training"]
    c["dynamic_state_schema"] = {
        "description": v2["dynamic_state_schema"]["description"],
        "fields": v2["dynamic_state_schema"]["fields"],
        "state_vector_order": sk["state_vector"],
        "stratification_note": sk["observation_notes"],
    }
    c["safety_shield"] = {
        "enabled": bool(cfg["safety_shield"]["enabled"]),
        "bypass_water_C": cfg["safety_shield"]["bypass_water_C"],
        "bypass_pcm_C": cfg["safety_shield"]["bypass_pcm_C"],
        "guard_band_K": cfg["safety"]["max_water_temp_C"] - cfg["safety_shield"]["bypass_water_C"],
        "enforced_in": "src/simulation/tank_model_multinode.py - inside the numba kernel, not in the policy",
        "note": shield2["note_v2"] + " " + shield2["rationale"],
        "pump_off_below_irradiance_Wm2": shield2["pump_off_below_irradiance_Wm2"],
        "dry_run": shield2["dry_run"],
        "sensor_failure": shield2["sensor_failure"],
    }
    c["reward_function"] = {
        "status": v2["reward_function"]["status"],
        "formula": v2["reward_function"]["formula"],
        "term_definitions": v2["reward_function"]["penalty_term_definitions"],
        "removed_terms": v2["reward_function"]["removed_terms"],
        "normalisation": v2["reward_function"]["normalisation"],
    }
    c["design_space_caveats"] = {
        "NOT_A_GLOBAL_OPTIMUM": True,
        "statement": "The designs are the best found WITHIN the declared design space (configs/design_space.yaml).",
        "searched_bounds": {
            "capsule_inner_radius_m": [0.010, 0.040],
            "pcm_volume_fraction": [bounds["pcm_volume_fraction"]["min"], bounds["pcm_volume_fraction"]["max"]],
            "flow_kg_s": [bounds["flow_rate_kg_s"]["min"], bounds["flow_rate_kg_s"]["max"]],
            "arrangement": bounds["capsule_arrangement"],
        },
        "chosen_fractions": {str(d["regime_id"]): d["pcm_volume_fraction"] for d in designs},
        "chosen_inner_radius_m": {str(d["regime_id"]): d["capsule_inner_radius_m"] for d in designs},
        "binding_bound_note": ("Chosen PCM fractions sit at or just under the 30 % cap and R_in at or just under the 40 mm "
                               "bound (plan section 10: the 30 % cap was the only binding bound). Memory 2026-11-10: B1 bounds "
                               "still bind."),
        "flow_optimum_may_be_artefact": ("collector.flow_dependent_fr.enabled is FALSE, so F_R is flow-independent and a "
                                         "flow optimum is a model artefact."),
    }
    c["temperature_specification"] = {
        "o2_delivery_target_temp_C": cfg["delivery"]["target_temp_C"],
        "o2_max_pcm_temp_C": cfg["safety"]["max_pcm_temp_C"],
        "o2_max_water_temp_C": cfg["safety"]["max_water_temp_C"],
        "o1_Tm_target_C": state["regimes"][0].get("o1_Tm_target_C"),
        "note": ("O2 MEASURES delivered energy at 45 C. O1 sized latent heat for a higher delivery temperature "
                 "(plan section 10 item: O1 shared config 60 C vs O2 45 C). An open service-temperature mismatch, not an O3 issue."),
    }
    c["verification_status"] = {
        "simulator_verification": "see O2 docs/PROGRESS.md and results/g3_metrics_report.csv, g4_multinode_report.csv (Gate-1 residual in g10_heldout_results.csv)",
        "model_consistency_cross_check": "OpenFOAM checks done (memory 2026-11-10); cfd_closure_v1 is UNVALIDATED (results/b2_closure_v1_check.csv)",
        "experimental_validation": {"status": "NOT DONE", "scope": "Objective 4"},
        "wording_rule": "Say 'simulator verification' and 'model consistency check', never 'validated against experiment'.",
    }
    c["deferred_future_work"] = v2["deferred_future_work"]
    c["objective_boundary"] = ("Objective 1 selected the PCM shortlist. Objective 2 selected the released designs. Objective 3 "
                               "may not change PCM, capsule geometry or count, tank volume, collector area, safety limits or physics.")
    c["provenance"] = {
        "obj3_environment_contract_rajasthan_v2.json": _sha(V2),
        "selection": {f"g10_selection_r{k}.json": _sha(OUT / f"g10_selection_r{k}.json") for k in range(3)},
        "g10_heldout_results.csv": _sha(OUT / "g10_heldout_results.csv"),
        "held_out_test.lock": _sha(OUT / "held_out_test.lock"),
        "wp6_band.csv": _sha(OUT / "wp6_band.csv"),
        "configs/system_config_shared.yaml": _sha(BASE_DIR / "configs" / "system_config_shared.yaml"),
        "configs/design_space.yaml": _sha(BASE_DIR / "configs" / "design_space.yaml"),
        "configs/states/rajasthan.yaml": _sha(BASE_DIR / "configs" / "states" / "rajasthan.yaml"),
    }
    NEW.write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
    verify_against_sources(c)
    return NEW


def verify_against_sources(c: dict | None = None) -> list[str]:
    """Check that every design number copied into the v3 contract equals its source value. Raises on mismatch."""
    c = c or json.loads(NEW.read_text(encoding="utf-8"))
    v2 = _load_v2()
    cfg = load_system_config()
    held = pd.read_csv(OUT / "g10_heldout_results.csv")
    checked = []
    for d in c["static_design_per_regime"]:
        k = d["regime_id"]
        ch = json.loads((OUT / f"g10_selection_r{k}.json").read_text(encoding="utf-8"))["chosen"]
        for src, dst in (("pcm", "selected_pcm"), ("R_in_m", "capsule_inner_radius_m"), ("fraction", "pcm_volume_fraction"),
                         ("N", "n_capsules"), ("flow", "nominal_flow_kg_s"), ("arrangement", "arrangement"),
                         ("pcm_mass_kg", "pcm_mass_kg")):
            assert ch[src] == d[dst], f"regime {k}: {dst} {d[dst]} != g10 selection {src} {ch[src]}"
            checked.append(f"r{k}.{dst}")
        h = held[held.regime == k].round(4).to_dict("records")
        assert [r["year"] for r in h] == [r["year"] for r in d["held_out_2011_2015"]], "held-out year order differs"
        for a, b in zip(h, d["held_out_2011_2015"]):          # v2 drops the plain_* columns; compare the shared keys
            for x in b:
                assert a[x] == b[x] or (isinstance(b[x], float) and abs(a[x] - b[x]) < 1e-9), f"held-out {x} differs"
    gl2, gl = v2["global_limits"], c["global_limits"]
    for a, b in (("delivery_temp_target_C", "delivery_target_temp_C"), ("max_water_temp_C", "max_safe_water_temp_C"),
                 ("max_pcm_temp_C", "max_safe_pcm_temp_C"), ("pump_flow_min_kg_s", "flow_min_kg_s"),
                 ("pump_flow_max_kg_s", "flow_max_kg_s")):
        assert gl2[a] == gl[b], f"{b} differs from v2 {a}"
        checked.append(b)
    assert c["fixed_system"]["tank_volume_L"] == cfg["tank"]["volume_L"] == 50.0
    assert c["fixed_system"]["collector_area_m2"] == cfg["collector"]["area_m2"] == 1.5
    return checked


if __name__ == "__main__":
    p = build()
    print("written", p)
    print("verified", len(verify_against_sources()), "copied fields against their sources")
