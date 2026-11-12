"""
Phase O3-9 — deployment metrics (plan section 18.2), Rajasthan. Validation years only; simulation only.

For a Gen-2 run (default: the run selected by gen2_study.py for this regime) and the baselines B0 / B1 / twin:

  * thermal efficiency  = energy delivered to 45 C / solar energy incident on the collector (1.5 m2 x GHI)
  * demand coverage     = E_del45 / (E_del45 + E_unmet)   (the share of the 45 C demand met by the solar store)
  * energy savings      = extra delivered heat versus the rule-based baseline B1 (= backup heat the heater need not supply)
  * inference cost      = ms per decision (single thread, this CPU), parameter count, fp32 weight size, checkpoint size
  * sensor faults       = 7 scenarios that corrupt the OBSERVATION only (physics untouched): nominal, pyranometer dead,
                          pyranometer stuck, top-water sensor bias +2 C, 2 % sensor noise, PCM-sensor dropout, combined.
                          The shield acts on the true state inside the kernel, so hard-limit safety cannot depend on these.

    python scripts/phase9_deployment_metrics.py --regime 1 --algo dqn --n-jobs 6
    python scripts/phase9_deployment_metrics.py --regime 1 --run results/gen2/<tag>

Outputs: results/deployment_metrics_r<regime>.json, results/phase9_sensor_faults_r<regime>.csv.  --n-envs unused.
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.utils import weather_roles  # noqa: E402
from o3.agents.gen2 import evaluate_run, evaluate_baseline_as, gen2_dir, run_model_path  # noqa: E402

FAULTS = ["nominal", "pyranometer_dead", "pyranometer_stuck", "water_sensor_bias_p2C", "sensor_noise_2pct", "pcm_sensor_dropout", "combined"]
I_TTOP, I_TPCM, I_FMELT, I_GHI = 0, 3, 4, 5          # V2 / V3 observation indices


def corrupt(obs: np.ndarray, fault: str, rng, mem: dict) -> np.ndarray:
    o = obs.copy()
    if fault in ("pyranometer_dead", "combined"):
        o[I_GHI] = 0.0
    if fault == "pyranometer_stuck":
        o[I_GHI] = 0.5                                  # reads 500 W/m2 day and night
    if fault in ("water_sensor_bias_p2C", "combined"):
        o[I_TTOP] += 2.0 / 40.0
    if fault in ("sensor_noise_2pct", "combined"):
        o[:13] = o[:13] * (1.0 + rng.normal(0.0, 0.02, 13))
    if fault == "pcm_sensor_dropout":                   # estimator frozen at its first value
        mem.setdefault("pcm", (o[I_TPCM], o[I_FMELT]))
        o[I_TPCM], o[I_FMELT] = mem["pcm"]
    return np.clip(o, -5.0, 5.0).astype(np.float32)


def fault_job(policy: str, regime_id: int, year: int, fault: str, run_dir: str | None = None) -> dict:
    """One (policy, year, fault) episode. policy: 'run' (the Gen-2 model in run_dir) or a baseline key (b0, b1)."""
    from o3.environment.pcm_env import PCMEnv
    from o3.evaluation.rollout import ModelPolicy, ALGO_KIND, ALGO_LABEL, load_model, make_baseline
    kw, name = {}, policy.upper()
    if policy == "run":
        import torch
        torch.set_num_threads(1)
        meta = json.loads((pathlib.Path(run_dir) / "run_meta.json").read_text())
        pol = ModelPolicy(load_model(meta["algo"], run_model_path(pathlib.Path(run_dir))), ALGO_KIND[meta["algo"]])
        kw = dict(obs_version=meta["obs_version"], control_step_min=meta["control_step_min"], reward_weights={"shield": meta["shield_weight"]})
        name = ALGO_LABEL[meta["algo"]]
    else:
        pol = make_baseline(policy)
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=regime_id, year=year, **kw)
    rng, mem = np.random.default_rng([year, FAULTS.index(fault)]), {}
    obs, _ = env.reset(year=year)
    pol.reset()
    done = False
    while not done:
        obs, _, done, _, _ = env.step(pol.act(corrupt(obs, fault, rng, mem)))
    s = env.episode_summary()
    return {"policy": name, "fault": fault, "year": year, "E45_kWh": s["E_del45_annual_kWh"], "window_kWh": s["E_del45_window_kWh"],
            "shield_h": s["shield_water_h"] + s["shield_pcm_h"], "violations": s["n_safety_violations"],
            "hours_over_limit": s["hours_over_limit"], "t_top_max_C": s["t_top_max_C"]}


def inference_cost(run_dir: pathlib.Path) -> dict:
    import torch
    from o3.evaluation.rollout import load_model
    torch.set_num_threads(1)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    path = run_model_path(run_dir)
    model = load_model(meta["algo"], path)
    from o3.environment import obs as O
    x = np.zeros(O.OBS_DIMS[meta["obs_version"]], dtype=np.float32)
    for _ in range(50):
        model.predict(x, deterministic=True)
    t = time.perf_counter()
    n = 2000
    for _ in range(n):
        model.predict(x, deterministic=True)
    ms = (time.perf_counter() - t) / n * 1000.0
    net = model.q_net if hasattr(model, "q_net") else (model.actor if hasattr(model, "actor") else model.policy)
    n_par = int(sum(p.numel() for p in net.parameters()))
    return {"ms_per_decision": ms, "policy_parameters": n_par, "fp32_weights_kB": n_par * 4 / 1000.0,
            "checkpoint_kB": path.stat().st_size / 1000.0, "control_period_s": meta["control_step_min"] * 60,
            "note": "single thread on the development x86 CPU; only the acting network is counted (not target / critic networks)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--algo", default="dqn")
    ap.add_argument("--heater-efficiency", type=float, default=1.0,
                    help="ASSUMPTION (no project file gives one): backup-heater heat per unit of input energy. 1.0 = resistive element "
                         "(1 kWh electricity -> 1 kWh heat); gas ~0.85-0.9; a heat pump would be >1. Reported with a sensitivity table.")
    ap.add_argument("--run", default=None, help="Gen-2 run directory (default: the selected run for this regime, 60-min control)")
    add_parallel_args(ap)
    a = ap.parse_args()
    n_jobs, _ = resolve(a)
    if a.run:
        run_dir = pathlib.Path(a.run)
    else:
        sel = json.loads((gen2_dir() / f"gen2_selected_{a.algo}.json").read_text()).get(f"r{a.regime}_60min")
        if not sel:
            print(f"[FAIL] no selected Gen-2 {a.algo} run for regime {a.regime}; run scripts/gen2_study.py or pass --run")
            sys.exit(1)
        run_dir = gen2_dir() / sel["tag"]
    meta = json.loads((run_dir / "run_meta.json").read_text())
    regime, cad = meta["regime_id"], meta["control_step_min"]
    print(f"Phase O3-9 deployment metrics | regime {regime} | run {meta['tag']} | n_jobs {n_jobs}")

    jobs = [dict(run_dir=str(run_dir), year=y) for y in VAL_YEARS]
    rows = run_parallel(evaluate_run, jobs, n_jobs=n_jobs)
    bjobs = [dict(baseline="b0" if b == "twin" else b, regime_id=regime, year=y, control_step_min=cad, water_only=(b == "twin"))
             for b in ("twin", "b0", "b1") for y in VAL_YEARS]
    rows += run_parallel(evaluate_baseline_as, bjobs, n_jobs=n_jobs)
    area = json.loads(config_O2.CONTRACT_FILE.read_text(encoding="utf-8"))["fixed_system"]["collector_area_m2"]
    incident = float(np.mean([weather_roles.load_weather(regime, y)["GHI_Wm2"].sum() / 1000.0 * area for y in VAL_YEARS]))
    pol = {}
    for name in dict.fromkeys(r["policy"] for r in rows):
        rs = [r for r in rows if r["policy"] == name]
        e45, un = float(np.mean([r["total_delivered_kWh"] for r in rs])), float(np.mean([r["total_unmet_kWh"] for r in rs]))
        pol[name] = {"E_del45_kWh_yr": e45, "thermal_efficiency_pct": 100 * e45 / incident, "demand_coverage_pct": 100 * e45 / (e45 + un),
                     "shield_h_yr": float(np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in rs])),
                     "violations": int(sum(r["n_safety_violations"] for r in rs)),
                     "backup_heat_kWh_yr": un}                  # the 45 C demand the solar store did not meet: what the backup heater supplies
    b1 = pol["B1_reactive_rule"]["E_del45_kWh_yr"]
    for v in pol.values():
        v["extra_heat_vs_B1_kWh_yr"] = v["E_del45_kWh_yr"] - b1
        v["extra_heat_vs_B1_pct"] = 100 * (v["E_del45_kWh_yr"] / b1 - 1)
        ub1 = pol["B1_reactive_rule"]["backup_heat_kWh_yr"]
        v["backup_heat_avoided_vs_B1_kWh_yr"] = ub1 - v["backup_heat_kWh_yr"]
        # backup input energy avoided = heat avoided / heater efficiency (assumed); sensitivity over a plausible range
        v["backup_energy_avoided_vs_B1_kWh_yr_at_efficiency"] = {str(e): v["backup_heat_avoided_vs_B1_kWh_yr"] / e for e in (0.85, 1.0, 3.0)}
        v["backup_energy_avoided_vs_B1_kWh_yr"] = v["backup_heat_avoided_vs_B1_kWh_yr"] / a.heater_efficiency

    frows = run_parallel(fault_job, [dict(policy=p, regime_id=regime, year=y, fault=f, run_dir=str(run_dir))
                                     for p in ("run", "b1") for f in FAULTS for y in VAL_YEARS], n_jobs=n_jobs)
    res = config_O2.RESULTS_DIR
    with open(res / f"phase9_sensor_faults_r{regime}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frows[0]))
        w.writeheader()
        w.writerows(frows)
    faults = {}
    for p in dict.fromkeys(r["policy"] for r in frows):
        nom = float(np.mean([r["E45_kWh"] for r in frows if r["policy"] == p and r["fault"] == "nominal"]))
        faults[p] = {f: {"E45_kWh": float(np.mean([r["E45_kWh"] for r in frows if r["policy"] == p and r["fault"] == f])),
                         "loss_pct": 100 * (1 - float(np.mean([r["E45_kWh"] for r in frows if r["policy"] == p and r["fault"] == f])) / nom),
                         "shield_h": float(np.mean([r["shield_h"] for r in frows if r["policy"] == p and r["fault"] == f])),
                         "violations": int(sum(r["violations"] for r in frows if r["policy"] == p and r["fault"] == f))} for f in FAULTS}
    out = {"regime_id": regime, "run": meta, "years": VAL_YEARS, "incident_solar_kWh_yr": incident, "collector_area_m2": area,
           "heater_efficiency_ASSUMED": a.heater_efficiency,
           "heater_note": "Backup-heater energy avoided = backup heat avoided / heater efficiency. The efficiency is an assumption (not in any project file); "
                          "1.0 = resistive electric element. Sensitivity (0.85 gas-like, 1.0 resistive, 3.0 heat pump) is stored per policy.",
           "policies": pol, "inference": inference_cost(run_dir), "sensor_faults": faults,
           "label": "Software-in-the-loop simulation. Not field validation."}
    (res / f"deployment_metrics_r{regime}.json").write_text(json.dumps(out, indent=2))

    print(f"\nincident solar on {area} m2: {incident:.0f} kWh/yr")
    for n, v in pol.items():
        print(f"  {n:20s} eff {v['thermal_efficiency_pct']:5.1f} %  coverage {v['demand_coverage_pct']:5.1f} %  "
              f"vs B1 {v['extra_heat_vs_B1_kWh_yr']:+7.1f} kWh ({v['extra_heat_vs_B1_pct']:+.1f} %)  backup energy avoided {v['backup_energy_avoided_vs_B1_kWh_yr']:+6.1f} kWh (eff {a.heater_efficiency:g}, assumed)  shield {v['shield_h_yr']:5.0f} h  viol {v['violations']}")
    i = out["inference"]
    print(f"  inference: {i['ms_per_decision']:.2f} ms/decision, {i['policy_parameters']:,} parameters, {i['fp32_weights_kB']:.0f} kB weights")
    for p, d in faults.items():
        worst = max(FAULTS[1:], key=lambda f: d[f]["loss_pct"])
        print(f"  faults {p}: worst {worst} {d[worst]['loss_pct']:+.1f} % delivery; violations in any scenario: {sum(x['violations'] for x in d.values())}")
    print(f"-> {res / f'deployment_metrics_r{regime}.json'}")


if __name__ == "__main__":
    main()
