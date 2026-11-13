"""
scripts/m11_o4_handoff.py -- Objective 4 hand-off package (Tamil Nadu section 16.4 equivalent), Uttarakhand.
SIMULATION-ONLY specification derived from the frozen contract and the frozen policies; it claims nothing about
hardware. The plan (section 11) keeps the O4 hand-off as a later task: this package is the input O4 starts from.

Per PCM regime (0-2): results/o4_handoff/r{k}/
  hardware_interface_spec.json   sensor list (contract dynamic_state_schema + the measurable flag; estimator-only
                                 fields marked), sampling interval, the single actuator (collector-loop pump on/off),
                                 the O2 shield restated as an INDEPENDENT hardware interlock, hard limits, fail-safe
                                 behaviour, the observation layout and normalisation the policy expects
  policy_r{k}.zip                copy of the frozen primary PPO policy (+ normalisation.json) with SHA-256
  deployment_test_plan.md        staged test plan ending at hardware-in-the-loop; no embedded deployment before HIL
Regime 3 gets no policy (water-only by O2's verdict): its spec says "always enable, interlock only".
Usage: python scripts/m11_o4_handoff.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3 import o2_bridge as ob                     # noqa: E402
from o3.env import OBS_FIELDS                      # noqa: E402

PLAN = """# Objective 4 deployment test plan - Uttarakhand regime {k} (SIMULATION-DERIVED, NOT VALIDATED)

Policy `{pol}` (SHA-256 `{sha}`), observation layout and normalisation in `hardware_interface_spec.json`.
**No embedded deployment until step 6 passes.**

1. **Bench interlock test (no policy).** Wire the independent interlock: pump OFF whenever the top-of-tank sensor reads
   >= {bw} C, PCM-side exchange cut at {bp} C if a PCM sensor exists; hard limits 75 C water / 65 C PCM. Verify by heating
   the sensor, not by software.
2. **Sensor commissioning.** Calibrate every `measurable: true` sensor; log 7 days at the sampling interval; check
   ranges against the simulator's training distribution (normalisation file).
3. **Estimator.** T_pcm, f_melt, stored and recoverable energy are NOT measurable (contract). Build and validate an
   estimator from tank temperatures before the policy may use them; until then feed their training means (the O3
   `pcm_dropout` sensor-fault scenario measured the cost of doing so).
4. **Shadow mode.** Run the policy in parallel with always-on operation (B0) without actuating; log both decisions.
5. **Software-in-the-loop replay of the shadow log** through the O3 simulator (`app/streamlit_app.py`, re-run tab).
6. **Hardware-in-the-loop.** Policy actuates the pump on the bench rig with the interlock active; pass = zero hard-limit
   events and delivery not worse than B0 by more than the O3 validation gap ({gap:.0f} kWh/yr window).

Simulation result to beat (O3, validation 2021-2023): B0 delivers more window energy than this policy; the policy's
benefit is fewer shield hours. Deploy only if the project values that trade-off.
"""


def main():
    cfg = o3c.load()
    c = ob.load_contract()
    lim = c["global_limits"]
    fields = c["dynamic_state_schema"]["fields"]
    sh = ob.load_system_config()["safety_shield"]             # O2's own shield thresholds (source of truth)
    assert sh["enabled"]
    root = o3c.RESULTS / "o4_handoff"
    for k in cfg["regimes"]["pcm"] + cfg["regimes"]["water_only"]:
        out = root / f"r{k}"
        out.mkdir(parents=True, exist_ok=True)
        d = ob.chosen_design(k)
        spec = {
            "label": "SIMULATION-DERIVED SPECIFICATION - NOT HARDWARE-VALIDATED",
            "state": "uttarakhand", "regime": k, "medoid": __import__("o3.roles_o3", fromlist=["x"]).MEDOIDS[k],
            "design": {kk: d[kk] for kk in ("pcm", "N", "flow", "fraction", "arrangement")},
            "control_period_s": cfg["control"]["interval_s"],
            "sensor_sampling_interval_s": 60, "sensor_sampling_note": "O3 choice: 1-min samples, hourly mean fed to "
                                                                       "the policy (it was trained on hourly means)",
            "sensors": [{"name": f["name"], "unit": f["unit"], "measurable": f["measurable"], "source": f["source"],
                         "role": "sensor" if f["measurable"] else "ESTIMATOR REQUIRED (no direct sensor)"}
                        for f in fields],
            "actuators": [{"name": "collector_loop_pump", "type": "on/off", "commands": {"0": "bypass (pump off)",
                                                                                     "1": "enable (pump on)"},
                           "fixed_flow_kg_s": d["flow"], "flow_envelope_kg_s": [lim["pump_flow_min_kg_s"],
                                                                              lim["pump_flow_max_kg_s"]],
                           "note": "the policy never sets flow (plan 3.3)"}],
            "independent_hardware_interlock": {
                "water": f"pump OFF while top-of-tank >= {sh['bypass_water_C']} C",
                "pcm": f"PCM-side exchange cut at {sh['bypass_pcm_C']} C when the water is hotter (needs a PCM sensor "
                       "or estimator)",
                "source": "O2 configs/system_config_shared.yaml safety_shield (same values the kernel uses)",
                "hard_limits_C": {"water": lim["max_water_temp_C"], "pcm": lim["max_pcm_temp_C"]},
                "rule": "implemented in hardware/firmware below the policy; the policy can never override it"},
            "fail_safe": {"sensor_or_policy_fault": "fall back to B0 (always enable) under the interlock",
                          "stale_weather": "persistence of the last valid value (as the O3 live demo)",
                          "watchdog_s": 2 * cfg["control"]["interval_s"]},
            "o3_config_sha256": o3c.config_hash(),
            "contract_sha256": ob.sha256_file(ob.CONTRACT_PATH)}
        if k in cfg["regimes"]["pcm"]:
            fz = frozen.load_frozen(k)
            p = fz["policies"][f"seed{fz['primary_seed']}"]
            src = frozen.frozen_dir(k) / p["file"]
            dst = out / f"policy_r{k}.zip"
            shutil.copy(src, dst)
            shutil.copy(frozen.frozen_dir(k) / "normalisation.json", out / "normalisation.json")
            assert ob.sha256_file(dst) == p["sha256"]
            spec["policy"] = {"file": dst.name, "sha256": p["sha256"], "algorithm": "PPO (Stable-Baselines3 2.4.1, "
                              "MlpPolicy [64, 64])", "weight_set": fz["weight_set"], "seed": fz["primary_seed"],
                              "observation_fields": OBS_FIELDS, "hour_encoding": "sin/cos of 2*pi*hour/24",
                              "normalisation": "normalisation.json: (x - obs_mean) / obs_std",
                              "normalisation_sha256": fz["normalisation_sha256"],
                              "inference": "deterministic argmax; latency in results/m10_deployment_r{k}.json"}
            gap = fz["validation_seed_means"]["E_del_window_kWh"]
            b0 = __import__("pandas").read_csv(o3c.RESULTS / f"m3_baselines_r{k}.csv")
            b0 = b0[(b0.controller == "B0") & (b0.purpose == "validate")].E_del_window_kWh.mean()
            (out / "deployment_test_plan.md").write_text(PLAN.format(k=k, pol=dst.name, sha=p["sha256"], gap=b0 - gap,
                                                                     bw=sh["bypass_water_C"], bp=sh["bypass_pcm_C"]),
                                                         encoding="utf-8")
        else:
            spec["policy"] = None
            spec["note"] = "Regime 3: O2 found the PCM gives no gain (4 capsules); always enable under the interlock."
        (out / "hardware_interface_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        print(f"r{k}: {out}")


if __name__ == "__main__":
    main()
