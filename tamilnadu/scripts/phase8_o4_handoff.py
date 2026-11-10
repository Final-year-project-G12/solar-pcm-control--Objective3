"""
Phase O3-8 — Objective 4 (hardware) handoff specification.

Produces the deliverables the implementation plan requires before any
embedded deployment:

  - sensor list (what O4 must instrument to reconstruct the V1 observation);
  - sampling interval and actuator interface (how the policy drives hardware);
  - fail-safe behaviour (the safety shield, restated as embedded firmware
    logic that runs independently of the learned policy);
  - embedded policy export (a portable copy of the frozen policy + SHA);
  - a deployment test plan.

EXIT GATE (from the plan): no embedded deployment until hardware-in-the-loop
tests pass. This script only produces the *specification*; it explicitly does
not authorise field deployment.

Run from the objective3 root:
    python scripts/phase8_o4_handoff.py --algo recurrent --regime 0
"""

import argparse
import json
import pathlib
import shutil
import sys
import datetime

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"
HANDOFF_DIR   = REPO_ROOT / "results" / "obj4_handoff"


# Sensors required to reconstruct the 11-dim V1 observation at run time.
SENSOR_LIST = [
    {"obs_index": 0,  "signal": "GHI",            "unit": "W/m2",
     "source": "pyranometer (plane-of-array)",  "range": [0, 1400],
     "sample_rate_s": 60,  "notes": "global horizontal irradiance"},
    {"obs_index": 1,  "signal": "T_ambient",      "unit": "degC",
     "source": "shielded ambient RTD",          "range": [-10, 55],
     "sample_rate_s": 60,  "notes": "dry-bulb air temperature"},
    {"obs_index": 2,  "signal": "T_water_top",    "unit": "degC",
     "source": "tank top thermistor/RTD",        "range": [0, 100],
     "sample_rate_s": 10,  "notes": "delivery-node water temperature"},
    {"obs_index": 3,  "signal": "T_pcm",          "unit": "degC",
     "source": "capsule-embedded thermocouple",  "range": [0, 100],
     "sample_rate_s": 10,  "notes": "representative PCM capsule temperature"},
    {"obs_index": 4,  "signal": "f_melt",         "unit": "fraction",
     "source": "estimator (enthalpy from T_pcm) — NOT directly measurable",
     "range": [0, 1],      "sample_rate_s": 10,
     "notes": "state estimator, not a sensor; see fail-safe if estimator diverges"},
    {"obs_index": 5,  "signal": "demand_flow",    "unit": "L/h",
     "source": "draw-line flow meter",           "range": [0, 100],
     "sample_rate_s": 10,  "notes": "hot-water draw rate"},
    {"obs_index": 6,  "signal": "T_mains",        "unit": "degC",
     "source": "inlet RTD",                       "range": [0, 45],
     "sample_rate_s": 60,  "notes": "cold mains inlet temperature"},
    {"obs_index": 7,  "signal": "hour_of_day",    "unit": "h",
     "source": "RTC",                             "range": [0, 23],
     "sample_rate_s": 60,  "notes": "local clock"},
    {"obs_index": 8,  "signal": "day_of_year",    "unit": "day",
     "source": "RTC",                             "range": [0, 365],
     "sample_rate_s": 3600, "notes": "local calendar"},
    {"obs_index": 9,  "signal": "prev_action",    "unit": "enum",
     "source": "controller state",               "range": [0, 2],
     "sample_rate_s": 3600, "notes": "last command issued (0=CHARGE,1=IDLE,2=BYPASS)"},
    {"obs_index": 10, "signal": "T_water_margin", "unit": "degC",
     "source": "derived (max_water_C - T_water)", "range": [-5, 75],
     "sample_rate_s": 10,  "notes": "safety margin, computed from T_water_top"},
]

ACTUATOR_INTERFACE = [
    {"actuator": "circulation_pump", "type": "relay / VFD enable",
     "commands": {"CHARGE": "pump ON at nominal flow", "IDLE": "pump OFF",
                  "BYPASS": "pump OFF"},
     "nominal_flow_source": "contract.static_design_per_regime[regime].nominal_flow_kg_s"},
    {"actuator": "collector_bypass_valve", "type": "3-way motorised valve",
     "commands": {"CHARGE": "collector in loop", "IDLE": "collector in loop (pump off)",
                  "BYPASS": "collector isolated"},
     "notes": "BYPASS and the water-side shield both drive this valve to isolate"},
]


def load_model_path(algo: str, regime_id: int) -> pathlib.Path | None:
    from scripts.phase6_locked_eval import load_model
    try:
        return load_model(algo, regime_id)
    except FileNotFoundError:
        return None


def build_failsafe(contract: dict) -> dict:
    """Restate the safety shield as embedded firmware logic (runs regardless
    of the learned policy — the policy can never disable it)."""
    gl = contract.get("global_limits", {})
    shield = contract.get("safety_shield", {})
    return {
        "principle": "The shield is a hardware-level interlock independent of "
                     "the learned policy. It evaluates every control cycle and "
                     "overrides the policy command when a guard band is crossed.",
        "hard_limits": {
            "max_water_temp_C": gl.get("max_safe_water_temp_C", 75.0),
            "max_pcm_temp_C":   gl.get("max_safe_pcm_temp_C", 65.0),
        },
        "guard_bands": {
            "bypass_water_C": shield.get("bypass_water_C", 72.0),
            "bypass_pcm_C":   shield.get("bypass_pcm_C", 62.0),
        },
        "rules": [
            "IF T_water >= bypass_water_C: force BYPASS (isolate collector, pump OFF).",
            "IF T_pcm >= bypass_pcm_C AND T_water > T_pcm: block charge heat path.",
            "IF any sensor stale > stale_timeout_s: fail to BYPASS (safe default).",
            "IF f_melt estimator diverges from T_pcm-implied enthalpy: fall back "
            "to reactive rule B1 and raise an alarm.",
            "IF hard limit exceeded despite shield: latch fault, pump OFF, "
            "alarm, require manual reset.",
        ],
        "stale_timeout_s": 120,
        "control_cycle_s": 10,
    }


def export_policy(algo: str, regime_id: int, model_path: pathlib.Path) -> dict:
    """Copy the frozen policy into the handoff bundle and record its SHA."""
    from scripts.phase6_locked_eval import sha256_of_file
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    zip_src = model_path.with_suffix(".zip")
    out = {"algo": algo, "regime_id": regime_id,
           "source_checkpoint": str(zip_src.relative_to(REPO_ROOT))}
    if zip_src.exists():
        dst = HANDOFF_DIR / f"policy_{algo}_r{regime_id}.zip"
        shutil.copy2(zip_src, dst)
        out["exported_to"] = str(dst.relative_to(REPO_ROOT))
        out["sha256"] = sha256_of_file(dst)
        # best-effort ONNX export for embedded inference
        try:
            import torch  # noqa: F401
            out["onnx_export"] = "supported — run torch.onnx.export on the " \
                "policy network for MCU/edge inference (not auto-run here)"
        except Exception:
            out["onnx_export"] = "torch not available in this env"
    else:
        out["exported_to"] = None
        out["sha256"] = None
        out["warning"] = "No trained checkpoint found — train phase O3-4 first."
    return out


def deployment_test_plan() -> list[dict]:
    return [
        {"step": 1, "name": "Bench sensor calibration",
         "gate": "every sensor within datasheet tolerance vs reference"},
        {"step": 2, "name": "Actuator dry-run",
         "gate": "pump + valve respond to all 3 commands, measured latency < control_cycle_s"},
        {"step": 3, "name": "Shield interlock test (policy disabled)",
         "gate": "forced over-temp triggers BYPASS and latches fault without the policy"},
        {"step": 4, "name": "Software-in-the-loop replay on hardware controller",
         "gate": "controller reproduces simulator actions bit-for-bit on logged weather"},
        {"step": 5, "name": "Hardware-in-the-loop (HIL) with emulated tank",
         "gate": "closed-loop stable, zero hard-limit breaches over a stress week"},
        {"step": 6, "name": "Instrumented field trial",
         "gate": "measured delivery >= B0 baseline, zero safety violations — "
                 "THIS is the first real experimental validation (Objective 4)"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",   type=str, default="recurrent",
                        choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"])
    parser.add_argument("--regime", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase O3-8 — Objective 4 Hardware Handoff Specification")
    print(f"Algorithm: {args.algo}  |  Regime: {args.regime}")
    print("=" * 70)

    with open(CONTRACT_PATH) as f:
        contract = json.load(f)

    model_path = load_model_path(args.algo, args.regime)
    if model_path is None:
        print(f"[WARN] No trained {args.algo} model for regime {args.regime}; "
              "spec will be written with a policy-export warning.")
        policy_export = {"algo": args.algo, "regime_id": args.regime,
                         "exported_to": None, "sha256": None,
                         "warning": "No trained checkpoint found."}
    else:
        policy_export = export_policy(args.algo, args.regime, model_path)

    spec = {
        "spec_version": "obj4_handoff_v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": contract.get("state", "tamilnadu"),
        "regime_id": args.regime,
        "source_contract": CONTRACT_PATH.name,
        "status": "SPECIFICATION ONLY — NOT AUTHORISED FOR FIELD DEPLOYMENT",
        "exit_gate": "No embedded deployment until hardware-in-the-loop tests "
                     "pass (deployment_test_plan step 5).",
        "sampling": {"control_cycle_s": 10,
                     "observation_assembly": "assemble 11-dim V1 vector each cycle; "
                     "normalise with TRAINING-set statistics only (never live stats)"},
        "sensor_list": SENSOR_LIST,
        "actuator_interface": ACTUATOR_INTERFACE,
        "fail_safe": build_failsafe(contract),
        "policy_export": policy_export,
        "deployment_test_plan": deployment_test_plan(),
        "provenance_note": "Thermal state (f_melt) is estimated, not measured. "
                           "Simulation results do not constitute experimental "
                           "validation; that is Objective 4's purpose.",
    }

    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    spec_path = HANDOFF_DIR / f"obj4_hardware_interface_spec_r{args.regime}.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[INFO] Hardware interface spec -> {spec_path}")

    # human-readable deployment test plan
    md_path = HANDOFF_DIR / f"obj4_deployment_test_plan_r{args.regime}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Objective 4 Deployment Test Plan — {spec['state']} "
                f"regime {args.regime}\n\n")
        f.write(f"**Policy:** {args.algo}  \n")
        f.write(f"**Status:** {spec['status']}  \n")
        f.write(f"**Exit gate:** {spec['exit_gate']}\n\n")
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
