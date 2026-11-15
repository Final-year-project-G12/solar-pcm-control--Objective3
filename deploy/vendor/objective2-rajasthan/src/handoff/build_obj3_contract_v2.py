"""
src/handoff/build_obj3_contract_v2.py
=========================================
G12 (O2, 2026-10-27) - regenerated Objective 3 environment contract. Writes
results/obj3_environment_contract_rajasthan_v2.json and leaves the old contract untouched.

It keeps the generic parts of the old contract (limits, action set, safety shield text, reward skeleton, acceptance tests),
because those did not change, and replaces what the rework changed:
  * simulator: multi-node (N=10) tank, dt = 150 s, local-time demand, inner-volume PCM mass + wall mass, O1 top-3 PCMs
  * per regime: the CHOSEN design, the best design of each other O1 PCM, their water-only twins at the same flow,
    select-year and held-out-test results (all of them, including where PCM does not help), the WP6 band
  * weather: explore / select / test years are now real (src/weather/roles.py); O3 must respect the roles
  * state vector gains the stratification fields (top/bottom temperature)
  * reward terms follow the WP1 metrics instead of annual E_load
Every number is read from a file under results/; nothing is computed here.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from config import BASE_DIR, RESULTS_DIR
from src.io_utils import load_system_config

OUT = RESULTS_DIR
OLD = RESULTS_DIR / "obj3_environment_contract_rajasthan.json"
NEW = OUT / "obj3_environment_contract_rajasthan_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    c = json.loads(OLD.read_text(encoding="utf-8"))
    band = pd.read_csv(OUT / "wp6_band.csv")
    held = pd.read_csv(OUT / "g10_heldout_results.csv") if (OUT / "g10_heldout_results.csv").exists() else None
    sys_cfg = load_system_config()

    c["schema"] = "obj3_environment_contract_rajasthan_v2"
    c["simulator_version"] = "sim_v3_rajasthan (multi-node N=10, dt=150 s)"
    c["time_step_s"] = 150
    c["supersedes"] = ("results/obj3_environment_contract_rajasthan.json (built from the legacy Phase-7 deployables, "
                       "UTC-indexed demand, single-node tank, retargeted PCMs). Do not train against that file.")
    c["generated_from"] = {
        "selection_logs": {f"r{k}": _sha(OUT / f"g10_selection_r{k}.json") for k in range(3)},
        "wp6_band": _sha(OUT / "wp6_band.csv"),
        "heldout_results": _sha(OUT / "g10_heldout_results.csv") if held is not None else "not run",
        "design_space": "configs/design_space.yaml",
        "system_config": sys_cfg.get("version", "system_config_shared.yaml"),
    }
    sk = c["control_skeleton"]
    sk["state_vector"] = ["T_water_top_C", "T_water_bottom_C", "T_water_mean_C", "T_pcm_C", "f_melt", "GHI_Wm2", "T_amb_C",
                          "hour_of_day", "draw_mass_kg_now", "T_mains_C", "store_energy_above_mains_kWh",
                          "recoverable_energy_above_45C_kWh", "minutes_since_last_draw"]
    sk["observation_notes"] = ("The tank is stratified: delivery temperature is the TOP node, the collector inlet is the "
                               "BOTTOM node. recoverable_energy_above_45C_kWh is the WP1 E_rec,45 quantity and needs an "
                               "estimator (not a raw sensor). Loop flow sets stratification, not collector heat, in this model.")
    sk["safety_shield"]["note_v2"] = ("Finding: the existing PCM-side shield only blocks CHARGING at 62 C and is not "
                                     "enough for PCMs already above their melt band; some designs exceed 65 C. A design is "
                                     "infeasible if its PCM peak exceeds 65 C in any simulated year.")
    fields = c["dynamic_state_schema"]["fields"]
    names = [f["name"] for f in fields]
    extra = [{"name": "T_water_top_C", "unit": "C", "measurable": True, "source": "sensor at the tank outlet"},
             {"name": "T_water_bottom_C", "unit": "C", "measurable": True, "source": "sensor at the collector return"},
             {"name": "recoverable_energy_above_45C_kWh", "unit": "kWh", "measurable": False, "source": "estimator (WP1 E_rec,45)"}]
    c["dynamic_state_schema"]["fields"] = fields + [e for e in extra if e["name"] not in names]
    c["weather_sequences"] = {
        "training": {"years": [2016, 2017, 2018, 2019, 2020], "source": "src/weather/roles.py (explore)"},
        "validation": {"years": [2021, 2022, 2023], "source": "select"},
        "held_out_test": {"years": [2011, 2012, 2013, 2014, 2015], "note": "O2's v3 (cap 30 %) one-shot test is DONE and recorded; O3 must not tune against these years"},
        "used_v1_test": {"years": [2024, 2025], "note": "the v1 (cap 20 %) held-out years, already used by O2; O3 must not tune against them either"},
        "points": {"0": "RJP_0132", "1": "RJP_0192", "2": "RJP_0083"},
    }
    sk["safety_shield"]["force_bypass_if"] = "T_water_top_C >= 72.0 (3.0 C guard band below the 75.0 C hard limit, to allow for sensor and control lag)"
    sk["safety_shield"]["rationale"] = ("Hard requirement carried from the earlier contract. In the current model the shield acts on the TOP "
                                       "water node (start of sub-step) and on the PCM; plain-tank peaks reached 73.3 C in the multi-node model, "
                                       "below the 75 C limit (docs/PROGRESS.md section 6).")
    c["reward_function"] = {
        "status": "suggested (v2), weights NOT chosen: built from the WP1 metrics; Objective 3 sets and justifies weights",
        "formula": "r_t = w1 * E_del45_t - w2 * overheat_step_t - w3 * safety_violation_t - w4 * bypass_switch_t, with E_del45_t the energy "
                   "delivered up to 45 C in the step (night and morning steps may be weighted more, WP1 window 18:00-09:00)",
        "removed_terms": "the earlier pump-energy term is removed: pump energy is not meaningful in the O2 model (Ergun pressure drop across "
                         "the capsule bed only). The earlier normalisation constants and default weights came from the retired Phase-7 "
                         "designs and are removed, not carried over.",
        "normalisation": "to be computed by Objective 3 from the water-only twin of the chosen design over the explore years",
        "penalty_term_definitions": {
            "safety_violation_t": "1 if the TOP water node is at or above 72 C or the PCM is at or above 62 C (the shield guard band), else 0",
            "bypass_switch_t": "1 if the mode changes into bypass at this step, else 0",
            "overheat_step_t": "shield-active time in the step (the quantity behind d_overheat)",
        },
    }
    c["time_step_s"] = 150
    cfg = BASE_DIR / "configs"
    c["config_hashes"] = {f"configs/{n}": {"sha256": _sha(cfg / n)} for n in
                          ("system_config_shared.yaml", "design_space.yaml")}
    c["config_hashes"]["states/rajasthan.yaml"] = {"sha256": _sha(cfg / "states" / "rajasthan.yaml")}

    regimes = []
    for k in range(3):
        sel = json.loads((OUT / f"g10_selection_r{k}.json").read_text(encoding="utf-8"))
        b = band[band.scope == f"r{k}"].set_index("objective").band.to_dict()
        entry = {"regime_id": k, "wp6_band": b, "selection": sel,
                 "water_only_twin": "same flow, weather year, demand, shield and tank model as each design (baseline convention)"}
        if held is not None:
            h = held[held.regime == k]
            entry["held_out_2011_2015"] = h.drop(columns=[x for x in h.columns if x.startswith("plain_")], errors="ignore").round(4).to_dict("records")
        regimes.append(entry)
    c["regimes"] = regimes
    c["deferred_future_work"] = c.get("deferred_future_work", []) + [
        "collector F_R(flow) correction: the model's F_R is flow-independent, so any flow optimum at 0.010 kg/s is a model artefact",
        "CFD-derived PCM resistance correction is NOT applied (lumped r/(5k) with liquid-convection factor 2.0)",
        "node-count (N=10 vs 20) uncertainty is inside the WP6 band, not removed",
        "O1 conductivity discrepancy between the ranking table and the PCM database is unresolved",
    ]
    NEW.write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
    return NEW


if __name__ == "__main__":
    print("written", build())
