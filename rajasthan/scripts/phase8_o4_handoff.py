"""
Phase O3-8 — Objective 4 (hardware) handoff specification (Rajasthan).

Produces the deliverables the implementation plan requires before any embedded deployment:

  - sensor list (what O4 must instrument to reconstruct the 13-field contract observation);
  - sampling interval and actuator interface (how the policy drives hardware);
  - fail-safe behaviour (the safety shield, restated as embedded firmware logic that runs independently
    of the learned policy);
  - embedded policy export (a portable copy of the frozen policy + SHA);
  - a deployment test plan.

EXIT GATE (from the plan): no embedded deployment until hardware-in-the-loop tests pass. This script
only produces the *specification*; it explicitly does not authorise field deployment.

Run from the rajasthan O3 root:
    python scripts/phase8_o4_handoff.py --algo ppo --regime 1
(--n-jobs / --n-envs are accepted for uniformity and unused: this phase reads files and writes a spec.)
"""

import argparse
import datetime
import json
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args  # noqa: E402
from o3.utils.models import find_model, sha256_of_file  # noqa: E402

CONTRACT_PATH = config_O2.CONTRACT_FILE
RESULTS_DIR = config_O2.RESULTS_DIR


# Sensors required to reconstruct the 13-field contract observation (obs layout V2) at run time.
SENSOR_LIST = [
    {"obs_index": 0, "signal": "T_water_top", "unit": "degC", "source": "tank top thermistor/RTD (delivery node)",
     "range": [0, 100], "sample_rate_s": 10, "notes": "the shield acts on this node"},
    {"obs_index": 1, "signal": "T_water_bottom", "unit": "degC", "source": "tank bottom RTD (collector return)",
     "range": [0, 100], "sample_rate_s": 10, "notes": "collector inlet node"},
    {"obs_index": 2, "signal": "T_water_mean", "unit": "degC", "source": "derived (mass-weighted mean of node/probe readings)",
     "range": [0, 100], "sample_rate_s": 10, "notes": "needs >= 3 probes along the tank, or an estimator"},
    {"obs_index": 3, "signal": "T_pcm", "unit": "degC", "source": "estimator (no direct PCM sensor assumed in the contract)",
     "range": [0, 100], "sample_rate_s": 10, "notes": "capsule-embedded thermocouple would make it measurable (an O4 choice)"},
    {"obs_index": 4, "signal": "f_melt", "unit": "fraction", "source": "estimator (enthalpy from T_pcm) — NOT directly measurable",
     "range": [0, 1], "sample_rate_s": 10, "notes": "state estimator, not a sensor; see fail-safe if it diverges"},
    {"obs_index": 5, "signal": "GHI", "unit": "W/m2", "source": "pyranometer (or LDR calibrated against one)",
     "range": [0, 1400], "sample_rate_s": 60, "notes": "the simulator uses the hourly MEAN (mild look-ahead); the live controller must use a nowcast"},
    {"obs_index": 6, "signal": "T_ambient", "unit": "degC", "source": "shielded ambient RTD", "range": [-10, 55],
     "sample_rate_s": 60, "notes": "dry-bulb air temperature"},
    {"obs_index": 7, "signal": "hour_of_day", "unit": "h", "source": "RTC", "range": [0, 23], "sample_rate_s": 60,
     "notes": "LOCAL clock (the O2 demand curve is local-time indexed)"},
    {"obs_index": 8, "signal": "draw_mass_now", "unit": "kg/h", "source": "draw-line flow meter", "range": [0, 100],
     "sample_rate_s": 10, "notes": "hot-water draw"},
    {"obs_index": 9, "signal": "T_mains", "unit": "degC", "source": "inlet RTD", "range": [0, 45], "sample_rate_s": 60,
     "notes": "cold mains inlet temperature (O2 uses a per-regime constant)"},
    {"obs_index": 10, "signal": "store_energy_above_mains_kWh", "unit": "kWh", "source": "estimator (T_water, T_pcm vs T_mains)",
     "range": [0, 5], "sample_rate_s": 10, "notes": "estimator, not a sensor"},
    {"obs_index": 11, "signal": "recoverable_energy_above_45C_kWh", "unit": "kWh", "source": "estimator (WP1 E_rec,45)",
     "range": [0, 3], "sample_rate_s": 10, "notes": "estimator, not a sensor"},
    {"obs_index": 12, "signal": "minutes_since_last_draw", "unit": "min", "source": "derived from draw sensor history",
     "range": [0, 1440], "sample_rate_s": 60, "notes": "controller state"},
]

ACTUATOR_INTERFACE = [
    {"actuator": "circulation_pump", "type": "relay / VFD enable",
     "commands": {"CHARGE": "pump ON at the design flow", "IDLE": "pump OFF", "BYPASS": "pump OFF"},
     "nominal_flow_source": "contract.static_design_per_regime[regime].nominal_flow_kg_s "
                            "(constant per run; the O2 collector model is flow-independent, so flow is not a control variable)"},
    {"actuator": "collector_bypass_valve", "type": "3-way motorised valve",
     "commands": {"CHARGE": "collector in loop", "IDLE": "collector in loop (pump off)", "BYPASS": "collector isolated"},
     "notes": "BYPASS and the water-side shield both drive this valve to isolate. In the O2 kernel IDLE and BYPASS are physically identical."},
]


def build_failsafe(contract: dict) -> dict:
    """Restate the safety shield as embedded firmware logic (runs regardless of the learned policy)."""
    gl = contract.get("global_limits", {})
    shield = contract.get("safety_shield", {})
    return {
        "principle": "The shield is a hardware-level interlock independent of the learned policy. It evaluates every "
                     "control cycle and overrides the policy command when a guard band is crossed.",
        "hard_limits": {"max_water_temp_C": gl.get("max_safe_water_temp_C", 75.0), "max_pcm_temp_C": gl.get("max_safe_pcm_temp_C", 65.0)},
        "guard_bands": {"bypass_water_C": shield.get("bypass_water_C", 72.0), "bypass_pcm_C": shield.get("bypass_pcm_C", 62.0)},
        "flow_envelope_kg_s": [gl.get("flow_min_kg_s"), gl.get("flow_max_kg_s")],
        "pump_off_below_irradiance_Wm2": shield.get("pump_off_below_irradiance_Wm2", 10.0),
        "rules": [
            "IF T_water_top >= bypass_water_C: force BYPASS (isolate collector, pump OFF).",
            "IF T_pcm >= bypass_pcm_C AND T_water > T_pcm: block the charge heat path.",
            "IF GHI < pump_off_below_irradiance_Wm2: pump OFF.",
            "IF pump commanded ON and the flow sensor reads ~0 (dry run): stop the pump, raise a fault.",
            "IF any sensor stale > stale_timeout_s: fail to BYPASS (safe default), raise a fault.",
            "IF the f_melt estimator diverges from the T_pcm-implied enthalpy: fall back to reactive rule B1 and raise an alarm.",
            "IF a hard limit is exceeded despite the shield: latch fault, pump OFF, alarm, require manual reset.",
        ],
        "stale_timeout_s": 120,
        "control_cycle_s": 3600,
        "control_cycle_note": "The policy decides once per HOUR (plan 3.0). Sensors are sampled faster; the shield runs every 10 s.",
    }


def export_policy(algo: str, regime_id: int, zip_src: pathlib.Path) -> dict:
    """Copy the frozen policy into the handoff bundle and record its SHA."""
    handoff = RESULTS_DIR / "obj4_handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    out = {"algo": algo, "regime_id": regime_id, "source_checkpoint": str(zip_src)}
    dst = handoff / f"policy_{algo}_r{regime_id}.zip"
    shutil.copy2(zip_src, dst)
    try:
        out["exported_to"] = str(dst.relative_to(config_O2.O3_ROOT))
    except ValueError:                      # results redirected outside the repo (O3_RESULTS_DIR)
        out["exported_to"] = str(dst)
    out["sha256"] = sha256_of_file(dst)
    try:
        import torch  # noqa: F401
        out["onnx_export"] = "supported — run torch.onnx.export on the policy network for MCU/edge inference (not auto-run here)"
    except Exception:
        out["onnx_export"] = "torch not available in this env"
    return out


def deployment_test_plan() -> list[dict]:
    return [
        {"step": 1, "name": "Bench sensor calibration", "gate": "every sensor within datasheet tolerance vs reference"},
        {"step": 2, "name": "Actuator dry-run", "gate": "pump + valve respond to all 3 commands, measured latency < control cycle"},
        {"step": 3, "name": "Shield interlock test (policy disabled)", "gate": "forced over-temp triggers BYPASS and latches fault without the policy"},
        {"step": 4, "name": "Software-in-the-loop replay on hardware controller",
         "gate": "controller reproduces simulator actions bit-for-bit on logged weather"},
        {"step": 5, "name": "Hardware-in-the-loop (HIL) with emulated tank", "gate": "closed-loop stable, zero hard-limit breaches over a stress week"},
        {"step": 6, "name": "Instrumented field trial",
         "gate": "measured delivery >= B0 baseline, zero safety violations — THIS is the first real experimental validation (Objective 4)"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"])
    parser.add_argument("--regime", type=int, default=1)
    add_parallel_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase O3-8 — Objective 4 Hardware Handoff Specification (Rajasthan)")
    print(f"Algorithm: {args.algo}  |  Regime: {args.regime}")
    print("=" * 70)

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    model = find_model(args.algo, args.regime)
    if model is None:
        print(f"[WARN] No trained {args.algo} model for regime {args.regime}; spec will carry a policy-export warning.")
        policy_export = {"algo": args.algo, "regime_id": args.regime, "exported_to": None, "sha256": None,
                         "warning": "No trained checkpoint found — run phase O3-4 first."}
    else:
        policy_export = export_policy(args.algo, args.regime, model)

    rd = next(d for d in contract["static_design_per_regime"] if d["regime_id"] == args.regime)
    spec = {
        "spec_version": "obj4_handoff_v2_rajasthan",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": contract.get("state", "rajasthan"),
        "regime_id": args.regime,
        "design": {"pcm": rd["selected_pcm"], "n_capsules": rd["n_capsules"], "flow_kg_s": rd["nominal_flow_kg_s"],
                   "tank_L": contract["fixed_system"]["tank_volume_L"], "collector_m2": contract["fixed_system"]["collector_area_m2"]},
        "source_contract": CONTRACT_PATH.name,
        "status": "SPECIFICATION ONLY — NOT AUTHORISED FOR FIELD DEPLOYMENT",
        "exit_gate": "No embedded deployment until hardware-in-the-loop tests pass (deployment_test_plan step 5).",
        "sampling": {"control_cycle_s": 3600,
                     "observation_assembly": "assemble the 13-field V2 vector each decision; normalise with the FIXED scale "
                                             "constants in o3/environment/obs.py (never live statistics)"},
        "sensor_list": SENSOR_LIST,
        "actuator_interface": ACTUATOR_INTERFACE,
        "fail_safe": build_failsafe(contract),
        "policy_export": policy_export,
        "deployment_test_plan": deployment_test_plan(),
        "provenance_note": "Thermal state (T_pcm, f_melt, stored and recoverable energy) is estimated, not measured. Simulation "
                           "results do not constitute experimental validation; that is Objective 4's purpose.",
    }

    handoff = RESULTS_DIR / "obj4_handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    spec_path = handoff / f"obj4_hardware_interface_spec_r{args.regime}.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[INFO] Hardware interface spec -> {spec_path}")

    md_path = handoff / f"obj4_deployment_test_plan_r{args.regime}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Objective 4 Deployment Test Plan — {spec['state']} regime {args.regime}\n\n")
        f.write(f"**Policy:** {args.algo}  \n**Status:** {spec['status']}  \n**Exit gate:** {spec['exit_gate']}\n\n")
        f.write("| Step | Test | Pass gate |\n|---|---|---|\n")
        for s in spec["deployment_test_plan"]:
            f.write(f"| {s['step']} | {s['name']} | {s['gate']} |\n")
        f.write("\n## Fail-safe (hardware interlock, independent of the policy)\n\n")
        for r in spec["fail_safe"]["rules"]:
            f.write(f"- {r}\n")
    print(f"[INFO] Deployment test plan -> {md_path}")
    print("\n[OK]   Phase O3-8 handoff spec complete.")
    print("       NOTE: specification only — hardware validation is Objective 4.")


if __name__ == "__main__":
    main()
