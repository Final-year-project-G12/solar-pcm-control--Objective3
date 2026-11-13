"""
scripts/m10_deployment_metrics.py -- deployment metrics (Tamil Nadu section 18.2 equivalents) for Uttarakhand.
VALIDATION years 2021-2023 only, fully_solid, deterministic demand; frozen policies, no test year.

  thermal efficiency     annual delivered energy (to 50 C) / solar incident on the collector (GHI x area_m2)
  collector efficiency   collector heat into the tank / solar incident
  demand coverage        annual delivered energy / energy needed to heat every drawn kg from mains to 50 C
  savings vs B1          delivered(controller) - delivered(B1), annual and window; backup-heater electricity avoided
                         = delta / eta_heater with eta_heater = 1.0 (resistive element: O3 ASSUMPTION, no O2 value)
  latency / footprint    PPO deterministic predict() per decision (median, p95 over 8760 calls, 1 thread),
                         parameter count, fp32 weight bytes, checkpoint size
  sensor faults          observation corrupted, physics untouched (the shield acts on the true state):
                         pyranometer_dead (GHI = 0), pyranometer_stuck (GHI frozen at the first daylight reading),
                         water_temp_bias_+2C (top / bottom / mean), sensor_noise (temperatures N(0, 0.5 C), GHI x
                         N(1, 0.05), T_amb N(0, 0.5 C)), pcm_dropout (T_pcm, f_melt, stored / recoverable energy held at
                         their training means), combined (bias + noise + pcm dropout); for PPO (primary seed), B1, B2
Writes results/m10_deployment_r{k}.json and results/m10_sensor_faults_r{k}.csv
Usage: python scripts/m10_deployment_metrics.py --regime 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                                       # noqa: E402
from o3 import frozen                                              # noqa: E402
from o3 import o2_bridge as ob                                     # noqa: E402
from o3 import roles_o3                                            # noqa: E402
from o3.controllers import I, B0AlwaysOn, B2Forecast               # noqa: E402
from o3.evaluate import episode_metrics, run_episode               # noqa: E402

ETA_HEATER = 1.0
TEMPS = ["T_water_top_C", "T_water_bottom_C", "T_water_mean_C"]
PCM = ["T_pcm_C", "f_melt", "store_energy_above_mains_kWh", "recoverable_energy_above_target_kWh"]
SCENARIOS = ["none", "pyranometer_dead", "pyranometer_stuck", "water_temp_bias_+2C", "sensor_noise", "pcm_dropout",
             "combined"]


class Faulty:
    """Wrap a controller: corrupt the RAW observation, re-normalise with the frozen statistics, pass both on."""

    def __init__(self, inner, scenario, mu, sd, seed=0):
        self.inner, self.sc, self.mu, self.sd = inner, scenario, mu, sd
        self.name, self.needs_norm = inner.name, inner.needs_norm
        self.rng = np.random.default_rng(seed)
        self.stuck = None

    def bind(self, env):
        if hasattr(self.inner, "bind"):
            self.inner.bind(env)

    def act(self, raw, norm=None):
        r = np.array(raw, dtype=np.float64)
        sc = self.sc
        if sc == "pyranometer_dead":
            r[I["GHI_Wm2"]] = 0.0
        if sc == "pyranometer_stuck":
            if self.stuck is None and r[I["GHI_Wm2"]] > 0:
                self.stuck = r[I["GHI_Wm2"]]
            if self.stuck is not None:
                r[I["GHI_Wm2"]] = self.stuck
        if sc in ("water_temp_bias_+2C", "combined"):
            for f in TEMPS:
                r[I[f]] += 2.0
        if sc in ("sensor_noise", "combined"):
            for f in TEMPS:
                r[I[f]] += self.rng.normal(0, 0.5)
            r[I["T_amb_C"]] += self.rng.normal(0, 0.5)
            r[I["GHI_Wm2"]] = max(r[I["GHI_Wm2"]] * (1 + self.rng.normal(0, 0.05)), 0.0)
        if sc in ("pcm_dropout", "combined"):
            for f in PCM:
                r[I[f]] = self.mu[I[f]]
        r = r.astype(np.float32)
        return self.inner.act(r, ((r - self.mu) / self.sd).astype(np.float32))


def footprint(policy, mu, sd, k):
    torch.set_num_threads(1)
    m = policy.model
    n_par = int(sum(p.numel() for p in m.policy.parameters()))
    env_obs = []
    from o3.env import UKPCMEnv
    env = UKPCMEnv(k, "validate", [2021], ["fully_solid"])
    o, _ = env.reset(options={"year": 2021, "scenario": "fully_solid"})
    for _ in range(8760):
        env_obs.append(o)
        o, *_ = env.step(1)
    t = []
    for o in env_obs:
        t0 = time.perf_counter()
        m.predict(o, deterministic=True)
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    return {"latency_ms_median": float(np.median(t)), "latency_ms_p95": float(np.quantile(t, 0.95)),
            "n_parameters": n_par, "weights_fp32_bytes": n_par * 4,
            "checkpoint_bytes": int(Path(policy.path).stat().st_size), "threads": 1,
            "note": "x86 laptop CPU, measured while other evaluation jobs were running (conservative)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, required=True)
    k = ap.parse_args().regime
    cfg = o3c.load()
    yrs, h = cfg["years"]["validate"], o3c.config_hash()
    has_ppo = k in cfg["regimes"]["pcm"]
    norm = o3c.load_normalisation(k)
    mu, sd = np.asarray(norm["obs_mean"], np.float32), np.asarray(norm["obs_std"], np.float32)
    b2p = o3c.RESULTS / f"b2_params_r{k}.json"
    ctrls = {"B0": B0AlwaysOn(), "B1": frozen.b1(k)}
    if b2p.exists():
        ctrls["B2"] = B2Forecast(**json.loads(b2p.read_text())["params"])
    if has_ppo:
        ctrls["PPO"] = frozen.primary_policy(k)

    case = None
    per = {n: [] for n in ctrls}
    twin = []
    for y in yrs:
        w = roles_o3.load_weather(k, y, "validate")
        case = case or ob.build_kernel_case(k, w.iloc[:1])
        area = case.system_config["collector"]["area_m2"]
        incident_kWh = w["GHI_Wm2"].sum() * area / 1000.0
        for n, c in ctrls.items():
            df, _ = run_episode(c, k, y, "fully_solid", "validate")
            m = episode_metrics(df)
            need_kWh = (df["draw_kg"] * case.P[0] * (case.target_C - case.mains_C)).sum() / 3.6e6
            per[n].append({"year": y, "E_del_annual_kWh": m["E_del_annual_kWh"], "E_del_window_kWh": m["E_del_window_kWh"],
                           "collector_kWh": m["collector_kWh"], "incident_kWh": incident_kWh, "demand_kWh": need_kWh,
                           "overheat_h": m["overheat_h"], "hours_over_limit": m["hours_over_limit"]})
    out = {"regime": k, "years": yrs, "collector_area_m2": area, "eta_heater_assumed": ETA_HEATER,
           "o3_config_sha256": h, "controllers": {}}
    b1 = pd.DataFrame(per["B1"])
    for n, rows in per.items():
        d = pd.DataFrame(rows)
        out["controllers"][n] = {
            "thermal_efficiency": float(d.E_del_annual_kWh.sum() / d.incident_kWh.sum()),
            "collector_efficiency": float(d.collector_kWh.sum() / d.incident_kWh.sum()),
            "demand_coverage": float(d.E_del_annual_kWh.sum() / d.demand_kWh.sum()),
            "E_del_annual_kWh": float(d.E_del_annual_kWh.mean()),
            "savings_vs_B1_annual_kWh": float(d.E_del_annual_kWh.mean() - b1.E_del_annual_kWh.mean()),
            "savings_vs_B1_window_kWh": float(d.E_del_window_kWh.mean() - b1.E_del_window_kWh.mean()),
            "backup_electricity_avoided_vs_B1_kWh": float((d.E_del_annual_kWh.mean() - b1.E_del_annual_kWh.mean()) / ETA_HEATER),
            "overheat_h": float(d.overheat_h.mean()), "hours_over_limit": float(d.hours_over_limit.sum())}
    if has_ppo:
        out["ppo_footprint"] = footprint(ctrls["PPO"], mu, sd, k)

    rows = []
    for n in [x for x in ("PPO", "B1", "B2") if x in ctrls]:
        for sc in SCENARIOS:
            for y in yrs:
                c = ctrls[n] if sc == "none" else Faulty(ctrls[n], sc, mu, sd, seed=y)
                df, _ = run_episode(c, k, y, "fully_solid", "validate")
                m = episode_metrics(df)
                rows.append({"regime": k, "controller": n, "scenario": sc, "year": y, **{x: m[x] for x in (
                    "E_del_window_kWh", "E_del_annual_kWh", "overheat_h", "hours_over_limit", "t_water_max_C",
                    "t_pcm_max_C", "bypass_hours", "bypass_switches")}})
    f = pd.DataFrame(rows).assign(o3_config_sha256=h)
    base = f[f.scenario == "none"].groupby("controller").E_del_annual_kWh.mean()
    f["d_annual_vs_nominal_pct"] = (f.E_del_annual_kWh / f.controller.map(base) - 1) * 100
    f.to_csv(o3c.RESULTS / f"m10_sensor_faults_r{k}.csv", index=False)
    out["sensor_faults_max_hours_over_limit"] = float(f.hours_over_limit.max())
    out["sensor_faults_worst_annual_change_pct"] = {
        c: float(g.groupby("scenario").d_annual_vs_nominal_pct.mean().min()) for c, g in f.groupby("controller")}
    (o3c.RESULTS / f"m10_deployment_r{k}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({n: {kk: round(v, 3) for kk, v in d.items()} for n, d in out["controllers"].items()}, indent=1))
    if has_ppo:
        print(out["ppo_footprint"])
    print(f.groupby(["controller", "scenario"])[["d_annual_vs_nominal_pct", "overheat_h", "hours_over_limit"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
