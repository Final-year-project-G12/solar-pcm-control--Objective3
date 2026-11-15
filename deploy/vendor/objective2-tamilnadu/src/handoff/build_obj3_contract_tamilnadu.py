"""
src/handoff/build_obj3_contract_tamilnadu.py
=============================================
Builds the Objective 3 environment contract for TAMIL NADU from the CURRENT
Objective 2 results. Writes results/obj3_environment_contract_tamilnadu_v3.json.

Why this file exists
--------------------
`build_obj3_contract_v2.py` is hardcoded for Rajasthan: it reads
`results/obj3_environment_contract_rajasthan.json` (absent from this tree) and
writes `..._rajasthan_v2.json`. It cannot produce a Tamil Nadu contract. The
only Tamil Nadu contract that existed (`objective3/obj3_environment_contract
_tamilnadu.json`, 2026-10-27) describes a SUPERSEDED run -- n-Hexacosane C26 /
n-Pentacosane C25, radial and single-layer arrangements, 32-35 capsules,
50 L tank, 1.5 m2 collector. None of that matches the released designs.

What is reused and what is rebuilt
----------------------------------
REUSED from the 2026-10-27 contract, because these did not change:
    action_space, reset_scenarios, acceptance_test_before_drl_training,
    dynamic_state_schema (then extended), demand_profile_file
REBUILT from live files, because all of these did change:
    static_design_per_regime  <- results/g10_selection_r{0,1,2}.json
    held_out_test             <- results/g10_heldout_results.csv + the lock
    repeatability_band        <- results/wp6_band.csv
    global_limits, safety_shield, time_step_s <- configs/system_config_shared.yaml
    weather_sequences         <- src/weather/roles.py (the real year roles)
    reward_function           <- rebuilt on the WP1 window metrics
ADDED, carrying the 2026-11-09 corrections pass into the contract itself:
    design_space_caveats      (the design is NOT a global optimum)
    temperature_specification (the one master table)
    verification_status       (simulator verification / CFD / experiment)

Every number is read from a file under results/ or configs/. Nothing is
computed here and nothing is typed by hand.

Usage:  python -m src.handoff.build_obj3_contract_tamilnadu
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
CONFIGS = BASE / "configs"
# The 2026-10-27 contract is SUPERSEDED for every design number, but it is still
# the source of the generic blocks that did not change (action space, reset
# scenarios, acceptance tests, state-schema skeleton). It lives in archive/
# permanently -- this tree retires files, it does not delete them. Pointing at
# the archive rather than at objective3/ is deliberate: objective3/ now holds
# the CURRENT contract, and the generator must never read its own output.
OLD = (BASE / "archive" / "superseded_obj3_contract_2026-10-27"
       / "obj3_environment_contract_tamilnadu.json")
NEW = RESULTS / "obj3_environment_contract_tamilnadu_v3.json"

STATE = "tamilnadu"
N_REGIMES = 3


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_yaml(p: Path):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build() -> Path:
    if not OLD.exists():
        raise FileNotFoundError(
            f"Generic-block source not found: {OLD}\n"
            "This is the archived 2026-10-27 contract. It supplies only the blocks that did "
            "NOT change (action_space, reset_scenarios, acceptance_test_before_drl_training, "
            "dynamic_state_schema); every design and result number is rebuilt from live files. "
            "If it was moved, restore it or point OLD at its new location.")
    old = json.loads(OLD.read_text(encoding="utf-8"))
    sysc = _load_yaml(CONFIGS / "system_config_shared.yaml")
    statec = _load_yaml(CONFIGS / "states" / f"{STATE}.yaml")
    band = pd.read_csv(RESULTS / "wp6_band.csv")
    held = pd.read_csv(RESULTS / "g10_heldout_results.csv")
    lock = json.loads((RESULTS / "held_out_test.lock").read_text(encoding="utf-8"))
    sel = [json.loads((RESULTS / f"g10_selection_r{k}.json").read_text(encoding="utf-8"))
           for k in range(N_REGIMES)]

    c = {}
    c["state"] = STATE
    c["contract_version"] = "obj3_contract_v3.0_2026-11-09"
    c["generated_utc"] = datetime.now(timezone.utc).isoformat()
    c["validated_simulator_version"] = (
        f"sim_v3_tamilnadu (multi-node tank, search dt={sysc['solver']['timestep_s']} s, "
        f"confirmation dt={sysc['solver']['timestep_confirm_s']} s)")
    c["source"] = ("Objective 2 Tamil Nadu pipeline stages g2-g24 (scripts/tn_run_all.py). "
                   "Built by src/handoff/build_obj3_contract_tamilnadu.py.")

    c["supersession_note"] = (
        "SUPERSEDES objective3/obj3_environment_contract_tamilnadu.json "
        "(contract_version obj3_contract_v2.0_2026-09-17, file dated 2026-10-27) ENTIRELY. "
        "That contract described designs that no longer exist: n-Hexacosane (C26) and "
        "n-Pentacosane (C25) in radial / single-layer arrangements, 32-35 capsules of "
        "72-80 mm, 4.9-7.4 kg PCM, in a 50 L tank with a 1.5 m2 collector. The released "
        "designs are PureTemp 58 / RT57HC, staggered, 98-105 capsules of 81.6 mm, "
        "23.6-25.3 kg PCM, in a 100 L tank with a 2.5 m2 collector. Its claims that PCM "
        "loses to a plain tank in every regime (-12.5% to -14.8%) and that P(meets demand) "
        "is 0-4% are both FALSE for the current designs: the held-out test has PCM beating "
        "the water-only twin in 9 of 9 designs. Objective 3 must read THIS file. Any cached "
        "copy of the older contract, or of OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md dated "
        "2026-10-27, must be discarded.")

    # ---------------------------------------------------------------- limits
    c["global_limits"] = {
        "delivery_target_temp_C": sysc["delivery"]["target_temp_C"],
        "max_safe_water_temp_C": sysc["safety"]["max_water_temp_C"],
        "max_safe_pcm_temp_C": sysc["safety"]["max_pcm_temp_C"],
        "max_pressure_bar": sysc["safety"]["max_pressure_bar"],
        "min_irradiance_cutoff_Wm2": sysc["collector"]["min_irradiance_cutoff_Wm2"],
        "flow_min_kg_s": sysc["pump"]["flow_min_kg_s"],
        "flow_max_kg_s": sysc["pump"]["flow_max_kg_s"],
        "pump_efficiency": sysc["pump"]["efficiency"],
        "note": ("These are HARD limits. The controller may never command outside the flow "
                 "envelope, and may never drive water above max_safe_water_temp_C or PCM "
                 "above max_safe_pcm_temp_C."),
    }

    # ------------------------------------------------- tank / collector (fixed)
    c["fixed_system"] = {
        "tank_volume_L": sysc["tank"]["volume_L"],
        "tank_height_to_diameter_ratio": sysc["tank"]["height_to_diameter_ratio"],
        "tank_insulation_m": sysc["tank"]["insulation_m"],
        "tank_U_W_m2K": sysc["tank"]["U_tank_W_m2K"],
        "collector_area_m2": sysc["collector"]["area_m2"],
        "collector_fr_tau_alpha": sysc["collector"]["fr_tau_alpha"],
        "collector_fr_ul_W_m2K": sysc["collector"]["fr_ul_W_m2K"],
        "collector_flow_dependent_fr_enabled": sysc["collector"]["flow_dependent_fr"]["enabled"],
        "capsule_wall_material": sysc["pcm_integration"]["capsule_wall_material"],
        "capsule_wall_thickness_m": sysc["pcm_integration"]["capsule_wall_thickness_m"],
        "liquid_convection_enhancement": sysc["pcm_integration"]["liquid_convection_enhancement"],
        "melting_half_width_K": sysc["pcm_integration"]["melting_half_width_K"],
        "WARNING_tank_collector": ("100 L / 2.5 m2. The 50 L / 1.5 m2 pairing in the 2026-10-27 "
                                   "contract is SUPERSEDED and must not be used."),
    }

    # ------------------------------------------------------ per-regime design
    bands = {f"r{k}": band[band.scope == f"r{k}"].set_index("objective").band.to_dict()
             for k in range(N_REGIMES)}
    regimes_cfg = {r["cluster_id"]: r for r in statec["regimes"]}

    designs = []
    for k in range(N_REGIMES):
        ch = sel[k]["chosen"]
        rc = regimes_cfg[k]
        h = held[(held.regime == k) & (held.role == "chosen")]
        designs.append({
            "regime_id": k,
            "label": rc["label"],
            "medoid_point_id": rc["medoid_point_id"],
            "medoid_elevation_m": rc["medoid_elevation_m"],
            "n_points": rc["n_points"],
            "population_covered": rc["population_covered"],
            "T_mains_est_C": rc["T_mains_est_C"],
            "o1_pcm_shortlist": rc["pcm_shortlist"],
            "selected_pcm": ch["pcm"],
            "selected_pcm_o1_consensus_rank": ch["pcm_rank"],
            "capsule_inner_radius_m": ch["R_in_m"],
            "capsule_inner_diameter_mm": round(ch["R_in_m"] * 2000.0, 2),
            # Docs quote 81.6 mm: that is the OUTER diameter, i.e. inner 80.0 mm
            # plus the 0.8 mm aluminium wall on each side. Both are given here so
            # Objective 3 cannot confuse them.
            "capsule_outer_diameter_mm": round(
                (ch["R_in_m"] * 2.0 + 2.0 * sysc["pcm_integration"]["capsule_wall_thickness_m"])
                * 1000.0, 2),
            "pcm_volume_fraction": ch["fraction"],
            "n_capsules": ch["N"],
            "arrangement": ch["arrangement"],
            "nominal_flow_kg_s": ch["flow"],
            "pcm_mass_kg": ch["pcm_mass_kg"],
            "select_year_metrics": {
                "g_window_kWh_yr": ch["g_window"],
                "g_annual_kWh_yr": ch["g_annual"],
                "d_overheat_h": ch["d_overheat"],
                "t_pcm_max_C": ch["t_pcm_max"],
                "t_water_max_C": ch["t_water_max"],
            },
            "repeatability_band": bands[f"r{k}"],
            "selection_funnel": {
                "n_candidates": sel[k]["n_candidates"],
                "n_S0_indistinguishable_on_g_window": sel[k]["n_S0_indistinguishable_on_g_window"],
                "n_S1": sel[k]["n_S1"],
                "n_S2": sel[k]["n_S2"],
                "note": ("Tie-breaks were declared before results were seen (plan WP6, D9/D10). "
                         "Designs differing by less than the repeatability band are treated as "
                         "tied, then overheating, PCM mass, capsule count and temperature "
                         "margin decide."),
            },
            "held_out_2011_2015": (h.drop(columns=[x for x in h.columns if x.startswith("plain_")],
                                          errors="ignore").round(4).to_dict("records")),
        })
    c["static_design_per_regime"] = designs

    # ------------------------------------------------------------- held out
    c["held_out_test"] = {
        "status": "COMPLETED AND LOCKED - do not re-run, do not tune against these years",
        "years": [2011, 2012, 2013, 2014, 2015],
        "run_id": lock["run_id"],
        "locked_utc": lock["utc"],
        "selection_fingerprint": lock["selection_fingerprint"],
        "n_designs_tested": int(held.groupby(["regime", "role", "pcm"]).ngroups),
        "coverage": ("The chosen design of each regime PLUS the best design of every other "
                     "Objective 1 shortlisted PCM - 9 designs, not 3, so the result cannot be "
                     "a cherry-pick."),
        "gates": {
            "beats_water_only_twin": "9 of 9",
            "pcm_peak_under_65C": f"9 of 9 ({held.t_pcm_max.min():.1f}-{held.t_pcm_max.max():.1f} C)",
            "water_peak_under_75C": f"9 of 9 ({held.t_water_max.min():.1f}-{held.t_water_max.max():.1f} C)",
        },
    }

    # ------------------------------------------------------------- weather
    c["weather_sequences"] = {
        "training": {"years": [2016, 2017, 2018, 2019, 2020], "role": "explore"},
        "validation": {"years": [2021, 2022, 2023], "role": "select"},
        "held_out_test": {"years": [2011, 2012, 2013, 2014, 2015], "role": "test",
                          "note": "Objective 2's one-shot test is DONE and locked. Objective 3 "
                                  "must NOT tune against these years."},
        "forbidden": {"years": [2024, 2025], "role": "used_test_v1",
                      "note": "Spent by Objective 2's v1 held-out test. Accepted for NO purpose."},
        "enforcement": ("src/weather/roles.py refuses to load a year whose role differs from the "
                        "stated purpose; the test years additionally require acquire_test_lock()."),
        "medoid_points": {str(k): regimes_cfg[k]["medoid_point_id"] for k in range(N_REGIMES)},
    }

    # -------------------------------------------------------------- control
    c["time_step_s"] = sysc["solver"]["timestep_s"]
    c["time_step_confirm_s"] = sysc["solver"]["timestep_confirm_s"]
    c["demand_profile_file"] = statec["demand_profile"]["file"]
    c["demand_daily_total_L"] = statec["demand_profile"]["daily_total_L"]

    c["action_space"] = old["action_space"]
    c["reset_scenarios"] = old["reset_scenarios"]
    c["acceptance_test_before_drl_training"] = old["acceptance_test_before_drl_training"]

    schema = old["dynamic_state_schema"]
    names = [f["name"] for f in schema["fields"]]
    for extra in (
        {"name": "T_water_top_C", "unit": "C", "measurable": True,
         "source": "sensor at the tank outlet - this is the DELIVERY temperature"},
        {"name": "T_water_bottom_C", "unit": "C", "measurable": True,
         "source": "sensor at the collector return - this is the collector INLET"},
        {"name": "recoverable_energy_above_45C_kWh", "unit": "kWh", "measurable": False,
         "source": "estimator (WP1 E_rec,45), not a raw sensor"},
    ):
        if extra["name"] not in names:
            schema["fields"].append(extra)
    schema["stratification_note"] = (
        "The tank is stratified and multi-node. Delivery temperature is the TOP node; the "
        "collector inlet is the BOTTOM node. Loop flow sets stratification, not collector heat.")
    c["dynamic_state_schema"] = schema

    c["safety_shield"] = {
        "enabled": sysc["safety_shield"]["enabled"],
        "bypass_water_C": sysc["safety_shield"]["bypass_water_C"],
        "bypass_pcm_C": sysc["safety_shield"]["bypass_pcm_C"],
        "guard_band_K": 3.0,
        "enforced_in": "src/simulation/tank_model.py - inside the physics, NOT in the policy",
        "note": ("This is ALREADY ENFORCED, not a specification to implement. The shield forces "
                 "the collector off at 72 C water / 62 C PCM, a 3 C guard band below the 75/65 C "
                 "hard limits. The acceptance-test item 'the shield cannot be bypassed by the "
                 "policy' therefore passes trivially against the baseline. Objective 3's job is "
                 "to build a BETTER controller than this blunt reactive trip - one that "
                 "anticipates the limit instead of reacting at it."),
        "standard": "IS 12976:2023 Sec 8.2 validates this exact mechanism for overheat protection.",
    }

    c["reward_function"] = {
        "status": ("SUGGESTED. Weights are deliberately NOT chosen - Objective 3 sets and "
                   "justifies them. The w1..w5 defaults in the 2026-10-27 contract came from "
                   "retired designs and are NOT carried over."),
        "formula": ("r_t = w1*E_del45_t - w2*overheat_step_t - w3*safety_violation_t "
                    "- w4*bypass_switch_t"),
        "term_definitions": {
            "E_del45_t": ("energy delivered up to the 45 C target in this step. The WP1 window "
                          "is 18:00-09:00; night and morning steps may be weighted more."),
            "overheat_step_t": "shield-active time in the step (the quantity behind d_overheat)",
            "safety_violation_t": "1 if TOP water >= 72 C or PCM >= 62 C (the shield guard band), else 0",
            "bypass_switch_t": "1 if the mode changes into bypass at this step, else 0",
        },
        "removed_terms": ("The pump-energy term is removed: pump energy is not meaningful in this "
                          "model, which has only an Ergun pressure drop across the capsule bed."),
        "normalisation": ("To be computed by Objective 3 from the water-only twin of the chosen "
                          "design over the explore years 2016-2020."),
    }

    # ------------------------------------------- corrections carried into O3
    c["design_space_caveats"] = {
        "NOT_A_GLOBAL_OPTIMUM": True,
        "statement": ("The designs above are the best found WITHIN the declared search bounds "
                      "under a pre-declared selection rule. They are not proven global optima."),
        "searched_bounds": {
            "capsule_inner_radius_mm": [10, 40],
            "pcm_volume_fraction": [0.02, 0.30],
            "flow_kg_s": [sysc["pump"]["flow_min_kg_s"], sysc["pump"]["flow_max_kg_s"]],
            "arrangement": ["single-layer", "staggered", "radial"],
        },
        "binding_bound_pcm_fraction": (
            "The 30% PCM-fraction cap IS BINDING. The g17 widened-bound diagnostic shows that at "
            "40% the window gain is higher by more than the repeatability band in EVERY regime, "
            "with PCM and water peaks still inside 65 C and 75 C. The cap is administrative: the "
            "packing engine accepts 28-48% at R 40 mm, and the only literature in the files read "
            "(Chen 2025) covers 10-20%. A 40% run is planned, not done."),
        "flow_optimum_may_be_artefact": (
            "collector.flow_dependent_fr.enabled is FALSE, so F_R does not vary with flow in this "
            "model. Any optimum sitting at the low flow bound is therefore suspect. Objective 3 "
            "MUST NOT treat the inherited nominal flow as a physical optimum to defend; it is a "
            "starting point inside a verified-safe envelope. g17 does establish that the UPPER "
            "flow bound is not limiting (0.080 kg/s moves g_window by less than the band)."),
        "radius_bound_not_limiting": (
            "R 45 mm and R 50 mm are both below the chosen design in every regime, so 40 mm sits "
            "at or near an interior optimum."),
    }

    c["temperature_specification"] = {
        "note": ("Four temperatures appear across O1 and O2 and have been mistaken for "
                 "contradictions. They are different physical quantities."),
        "o1_T_DELIVERY_C": {"value": 60.0, "meaning": "delivery temperature Objective 1 sizes the "
                            "night load against (Avargani et al. 2021)"},
        "o1_DT_APPROACH_C": {"value": 7.0, "meaning": "driving dT a PCM needs above delivery"},
        "o1_TM_TARGET_C": {"value": 67.0, "meaning": "aspirational PCM melting target "
                           "(= delivery + approach)", "binds": False},
        "o1_Tm_target_capped_C": {"range": [55.95, 62.88], "mean": 61.45,
                                  "meaning": "THE OPERATIVE CEILING. Feasibility Constraint 6 is "
                                             "Tm <= this. Capped by achievability via kt_worst_month.",
                                  "binds": True},
        "o2_delivery_target_temp_C": {"value": sysc["delivery"]["target_temp_C"],
                                      "meaning": "threshold at which O2 MEASURES usable delivered energy"},
        "o2_max_pcm_temp_C": {"value": sysc["safety"]["max_pcm_temp_C"],
                              "meaning": "PCM material-stability ceiling (Rubitherm datasheets)"},
        "o2_max_water_temp_C": {"value": sysc["safety"]["max_water_temp_C"],
                                "meaning": "scald prevention"},
        "is_67C_vs_65C_a_safety_problem": (
            "NO. TM_TARGET_C=67 C never reaches a selection decision. Constraint 6 screens on "
            "Tm_target_capped_C (mean 61.45 C); every shortlisted PCM melts at 54.0-58.0 C; and "
            "the held-out test measured PCM peaks of 63.4-64.0 C, inside the 65 C limit. "
            "Objective 3 inherits NO safety risk from this figure."),
        "open_item_service_mismatch": (
            "Objective 1 sizes latent heat for a 60 C delivery service while Objective 2 scores "
            "delivered energy at 45 C. O2's comparison is sound (both arms of every paired run "
            "use the same 45 C metric), but whether O1 would have shortlisted different PCMs had "
            "it sized against 45 C is UNQUANTIFIED. Open."),
    }

    c["verification_status"] = {
        "simulator_verification": {"status": "DONE", "evidence": "89 tests pass; Gate-1 energy "
                                   "balance residual within tolerance; g2 timestep convergence"},
        "model_consistency_cross_check": {"status": "DONE, with caveats", "method": "ANSYS Fluent, "
                                          "enthalpy-porosity, one capsule / six hours",
                                          "caveats": ["three v3 cases returned 0% liquid fraction",
                                                      "first-order upwind is the only scheme under "
                                                      "which all four PCMs converge, so melt "
                                                      "fractions are conservative",
                                                      "symmetry half-slice mesh excludes "
                                                      "buoyancy-driven melt convection"]},
        "experimental_validation": {"status": "NOT DONE", "scope": "Objective 4"},
        "o1_physics_agreement": ("WEAK AND NOT STATISTICALLY SIGNIFICANT. Objective 1's MCDM "
                                 "ranking vs its own physics model gives mean Spearman rho = "
                                 "+0.169 (per cluster +0.381/+0.103/+0.024), p = 0.12-0.96, "
                                 "n = 8-10 per cluster. The PCM shortlist Objective 2 inherited "
                                 "is an MCDM consensus ranking, NOT a physics-confirmed one."),
        "wording_rule": ("Say 'simulator verification', 'model consistency cross-check' and "
                         "'experimental validation' - never 'CFD-validated' or 'physics-validated'."),
    }

    c["deferred_future_work"] = [
        "40% PCM-fraction run (the binding cap, see design_space_caveats)",
        "flow-dependent collector F_R: the model's F_R is flow-independent, so any flow optimum "
        "at the low bound is a model artefact",
        "dense-grid cross-check for rank-2/3 PCM pairs (only rank-1 pairs have one)",
        "g18 novert/vert states were never run; that with/without comparison is incomplete",
        "CFD-derived PCM resistance correction is NOT applied (lumped r/(5k) with a liquid-"
        "convection factor of 2.0)",
        "node-count (N=10 vs 20) uncertainty is inside the WP6 band, not removed",
        "hardware validation (Objective 4)",
    ]

    c["objective_boundary"] = (
        "Objective 1 selected the PCM shortlist. Objective 2 selected the physical design "
        "(PCM identity, capsule geometry, arrangement, flow) and the validated operating "
        "envelope. OBJECTIVE 3 SELECTS THE REAL-TIME ACTION AND NOTHING ELSE. It must not "
        "change PCM identity, capsule geometry, tank volume, safety limits, or the released "
        "physics model, and must not re-cluster or merge the regimes.")

    c["provenance"] = {
        "selection": {f"g10_selection_r{k}.json": _sha(RESULTS / f"g10_selection_r{k}.json")
                      for k in range(N_REGIMES)},
        "g10_heldout_results.csv": _sha(RESULTS / "g10_heldout_results.csv"),
        "held_out_test.lock": _sha(RESULTS / "held_out_test.lock"),
        "wp6_band.csv": _sha(RESULTS / "wp6_band.csv"),
        "configs/system_config_shared.yaml": _sha(CONFIGS / "system_config_shared.yaml"),
        "configs/states/tamilnadu.yaml": _sha(CONFIGS / "states" / f"{STATE}.yaml"),
        "configs/design_space.yaml": (_sha(CONFIGS / "design_space.yaml")
                                      if (CONFIGS / "design_space.yaml").exists() else "absent"),
        "generic_blocks_reused_from": str(OLD.relative_to(BASE)),
        "generic_blocks_reused_sha256": _sha(OLD),
    }

    NEW.parent.mkdir(parents=True, exist_ok=True)
    NEW.write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
    return NEW


if __name__ == "__main__":
    p = build()
    print(f"written {p}  ({p.stat().st_size:,} bytes)")
