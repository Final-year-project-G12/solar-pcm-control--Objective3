"""
Phase O3-9 — Deployment metrics: thermal efficiency, energy savings vs the
baseline controllers, inference latency / memory footprint, and robustness
under SENSOR faults (corrupted observations, true physics untouched).

Key distinction from the phase-5 robustness suite: phase 5 perturbs the
*world* (weather noise/bias, initial PCM state); this phase perturbs the
*sensor channel the policy reads*. The safety shield acts on the true state
inside the environment, so hard safety holds by construction even when the
policy is fed garbage — what degrades is delivery / shield efficiency.

Run from the objective3 root (regime 0 by default):
    python scripts/phase9_deployment_metrics.py

Outputs:
    results/deployment_metrics_r{regime}.json
    results/phase9_sensor_faults_r{regime}.csv
    results/plots/R2/deployment_metrics.html      (4 Plotly charts)
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"

# validated palette — algorithms/controllers (categorical, fixed order)
C = {"B0": "#8f8e88", "B1": "#b5b4ae", "B2": "#d2d1cb",
     "PPO": "#eb6834", "DQN": "#2a78d6", "SAC": "#1baf7a"}

COLLECTOR_AREA_M2_DEFAULT = 2.5


# ------------------------------------------------------------------ helpers --
def load_comparison(regime: int) -> dict:
    """policy -> one row (year 2021) from phase4_comparison CSV."""
    import csv
    rows = {}
    with open(RESULTS_DIR / f"phase4_comparison_r{regime}.csv") as f:
        for r in csv.DictReader(f):
            if int(r["year"]) == 2021:
                rows[r["policy"]] = {k: (float(v) if v not in ("", None) else 0.0)
                                     for k, v in r.items()
                                     if k not in ("policy",)}
    return rows


def annual_ghi_kwh_m2(regime: int, year: int = 2021) -> float:
    from o3.environment.pcm_env import PCMEnv
    env = PCMEnv(CONTRACT_PATH, regime_id=regime, year=year)
    ghi_wh = float(env._weather["GHI_Wm2"].sum())      # 1-h steps -> Wh/m2
    return ghi_wh / 1000.0


def collector_area(contract: dict) -> float:
    fs = contract.get("fixed_system", {})
    for k in ("collector_area_m2", "A_collector_m2", "collector_area"):
        if k in fs:
            return float(fs[k])
    return COLLECTOR_AREA_M2_DEFAULT


def load_model(algo: str, regime: int, tag: str | None = None):
    base = {"ppo": "ppo", "dqn": "dqn", "sac": "sac"}[algo]
    rdir = f"r{regime}_{tag}" if tag else f"r{regime}"
    p = RESULTS_DIR / base / rdir / "best_model" / "best_model"
    if not p.with_suffix(".zip").exists():
        p = RESULTS_DIR / base / rdir / f"{base}_final"
    if algo == "ppo":
        from stable_baselines3 import PPO; return PPO.load(str(p)), p.with_suffix(".zip")
    if algo == "dqn":
        from stable_baselines3 import DQN; return DQN.load(str(p)), p.with_suffix(".zip")
    from stable_baselines3 import SAC; return SAC.load(str(p)), p.with_suffix(".zip")


# -------------------------------------------- 1+2: efficiency and savings ---
def efficiency_and_savings(regime: int) -> dict:
    contract = json.load(open(CONTRACT_PATH))
    area = collector_area(contract)
    ghi = annual_ghi_kwh_m2(regime)          # kWh/m2/yr
    solar_kwh = ghi * area                   # kWh incident on collector per yr
    rows = load_comparison(regime)

    out = {"annual_ghi_kwh_m2": round(ghi, 1),
           "collector_area_m2": area,
           "annual_solar_on_collector_kwh": round(solar_kwh, 1),
           "controllers": {}}
    short = {"B0_fixed_charge": "B0", "B1_reactive_rule": "B1",
             "B2_forecast_rule": "B2", "PPO": "PPO", "DQN": "DQN", "SAC": "SAC"}
    for name, r in rows.items():
        s = short.get(name, name)
        deliv = r["total_delivered_kWh"]; unmet = r["total_unmet_kWh"]
        shields = r["total_shield_water"] + r["total_shield_pcm"]
        out["controllers"][s] = {
            "delivered_kWh": round(deliv, 1),
            "unmet_kWh": round(unmet, 1),
            "shields": int(shields),
            # solar-to-load thermal efficiency: delivered / incident solar
            "thermal_efficiency_pct": round(100.0 * deliv / solar_kwh, 2),
            # demand coverage: delivered / (delivered + unmet)
            "demand_coverage_pct": round(100.0 * deliv / (deliv + unmet), 2),
        }
    # savings vs baselines (backup-heater framing: unmet is met electrically)
    for drl in ("DQN", "SAC"):
        if drl not in out["controllers"]:
            continue
        d = out["controllers"][drl]
        for base in ("B0", "B1"):
            b = out["controllers"].get(base)
            if not b:
                continue
            d[f"delivery_vs_{base}_pct"] = round(
                100.0 * d["delivered_kWh"] / b["delivered_kWh"], 1)
            d[f"backup_energy_saved_vs_{base}_kWh"] = round(
                b["unmet_kWh"] - d["unmet_kWh"], 1)
            d[f"shield_reduction_vs_{base}_pct"] = round(
                100.0 * (b["shields"] - d["shields"]) / b["shields"], 1)
    return out


# ------------------------------------------------- 3: latency and memory ----
def latency_and_memory(regime: int, n_iter: int = 2000,
                        obs_version: str = "V1", forecast_mode: str = "perfect",
                        tag: str | None = None) -> dict:
    from o3.environment.gym_wrapper import PCMGymEnv
    env = PCMGymEnv(CONTRACT_PATH, regime_id=regime, fixed_year=2021,
                     obs_version=obs_version, forecast_mode=forecast_mode)
    obs, _ = env.reset()
    res = {}
    for algo in ("dqn", "sac"):
        try:
            model, zpath = load_model(algo, regime, tag=tag)
        except FileNotFoundError:
            print(f"  [skip] no {algo} model for regime {regime} tag={tag}")
            continue
        # warm-up
        for _ in range(50):
            model.predict(obs, deterministic=True)
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model.predict(obs, deterministic=True)
            ts.append((time.perf_counter() - t0) * 1e6)   # microseconds
        ts.sort()
        n_params = sum(p.numel() for p in model.policy.parameters())
        bytes_fp32 = sum(p.numel() * 4 for p in model.policy.parameters())
        res[algo.upper()] = {
            "latency_mean_us": round(statistics.fmean(ts), 1),
            "latency_p50_us": round(ts[len(ts)//2], 1),
            "latency_p95_us": round(ts[int(len(ts)*0.95)], 1),
            "latency_max_us": round(ts[-1], 1),
            "n_params": int(n_params),
            "weights_fp32_kB": round(bytes_fp32 / 1024, 1),
            "model_zip_kB": round(zpath.stat().st_size / 1024, 1),
        }
    res["note"] = ("Measured single-threaded on the development x86 CPU. "
                   "Control period is 1 hour, so even a 100x slower embedded "
                   "target leaves >6 orders of magnitude of headroom. "
                   "Weights are MLP fp32; see results/obj4_handoff for the "
                   "target interface spec.")
    return res


# ---------------------------------------------- 4: sensor-fault robustness --
FAULTS = {
    "nominal":        None,
    "ghi_dropout":    ("set",   0, 0.0),    # pyranometer reads 0
    "ghi_stuck":      ("stuck", 0, None),   # pyranometer frozen at first value
    "tw_bias_+2C":    ("add",   2, 2.0/40), # water sensor +2 degC (normalised)
    "tw_noise":       ("noise", 2, 0.05),   # water sensor gaussian noise
    "tpcm_dropout":   ("set",   3, 0.0),    # PCM sensor reads 0
    "combined_fault": ("multi", None, None),
}


def _corrupt(obs, fault, state, rng):
    if fault is None:
        return obs
    kind, idx, val = fault
    o = obs.copy()
    if kind == "set":
        o[idx] = val
    elif kind == "add":
        o[idx] += val
    elif kind == "noise":
        o[idx] += rng.normal(0.0, val)
    elif kind == "stuck":
        if state.get("stuck") is None:
            state["stuck"] = float(o[idx])
        o[idx] = state["stuck"]
    elif kind == "multi":
        o[0] += rng.normal(0.0, 0.05)       # noisy pyranometer
        o[2] += 2.0 / 40                    # biased water sensor
    return o


def sensor_fault_suite(regime: int, year: int = 2021,
                        obs_version: str = "V1", forecast_mode: str = "perfect",
                        tag: str | None = None) -> list[dict]:
    from o3.environment.gym_wrapper import PCMGymEnv, PCMContinuousGymEnv
    rows = []
    for algo in ("dqn", "sac"):
        try:
            model, _ = load_model(algo, regime, tag=tag)
        except FileNotFoundError:
            print(f"  [skip] no {algo} model for regime {regime} tag={tag}")
            continue
        EnvCls = PCMContinuousGymEnv if algo == "sac" else PCMGymEnv
        for fname, fault in FAULTS.items():
            env = EnvCls(CONTRACT_PATH, regime_id=regime, fixed_year=year,
                         obs_version=obs_version, forecast_mode=forecast_mode)
            obs, _ = env.reset()
            rng = np.random.default_rng(7)
            st = {"stuck": None}
            deliv = unmet = 0.0
            sw = sp = sv = 0
            done = False
            while not done:
                a, _ = model.predict(_corrupt(obs, fault, st, rng),
                                     deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
                deliv += info["E_delivered_Wh"]; unmet += info["E_unmet_Wh"]
                sw += int(info["n_shield_water"]); sp += int(info["n_shield_pcm"])
                sv += int(info["n_safety_violations"])
            rows.append({"algo": algo.upper(), "fault": fname,
                         "delivered_kWh": round(deliv/1000, 1),
                         "unmet_kWh": round(unmet/1000, 1),
                         "shields": sw + sp, "safety_violations": sv})
            print(f"  {algo.upper():4s} {fname:16s} del={deliv/1000:7.1f} kWh "
                  f"shields={sw+sp:6d} safety={sv}")
    return rows


# ----------------------------------------------------------------- plots ----
def make_plots(eff: dict, lat: dict, faults: list[dict], out_html: pathlib.Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    order = [c for c in ("B0", "B1", "B2", "PPO", "DQN", "SAC")
             if c in eff["controllers"]]
    ctr = eff["controllers"]

    figs = []

    # 8 — thermal efficiency + demand coverage
    f = go.Figure()
    f.add_bar(name="solar→load thermal efficiency",
              x=order, y=[ctr[c]["thermal_efficiency_pct"] for c in order],
              marker_color=[C[c] for c in order])
    f.add_bar(name="demand coverage",
              x=order, y=[ctr[c]["demand_coverage_pct"] for c in order],
              marker_color=[C[c] for c in order], opacity=0.45)
    f.update_layout(
        title=f"8 · Thermal efficiency and demand coverage (%, regime 0, 2021 · "
              f"GHI {eff['annual_ghi_kwh_m2']} kWh/m²·yr × {eff['collector_area_m2']} m²)",
        barmode="group", yaxis_title="%", template="plotly_white", height=380)
    figs.append(f)

    # 9 — energy savings vs baselines (backup energy saved + delivery delta)
    drls = [c for c in ("DQN", "SAC") if c in ctr]
    f = make_subplots(rows=1, cols=2, subplot_titles=(
        "backup-heater energy saved vs baseline (kWh/yr)",
        "shield reduction vs baseline (%)"))
    for i, base in enumerate(("B0", "B1")):
        f.add_bar(name=f"vs {base}", x=drls,
                  y=[ctr[c].get(f"backup_energy_saved_vs_{base}_kWh", 0) for c in drls],
                  marker_color=["#8f8e88", "#2a78d6"][i], row=1, col=1)
        f.add_bar(name=f"vs {base} ", x=drls, showlegend=False,
                  y=[ctr[c].get(f"shield_reduction_vs_{base}_pct", 0) for c in drls],
                  marker_color=["#8f8e88", "#2a78d6"][i], row=1, col=2)
    f.update_layout(title="9 · Energy savings and protective-trip reduction vs "
                          "baseline controllers", barmode="group",
                    template="plotly_white", height=380)
    figs.append(f)

    # 10 — inference latency & memory
    algos = [a for a in ("DQN", "SAC") if a in lat]
    f = make_subplots(rows=1, cols=2, subplot_titles=(
        "inference latency per decision (µs, lower is better)",
        "memory footprint (kB)"))
    f.add_bar(x=algos, y=[lat[a]["latency_mean_us"] for a in algos],
              name="mean", marker_color=[C[a] for a in algos],
              error_y=dict(type="data",
                           array=[lat[a]["latency_p95_us"] - lat[a]["latency_mean_us"]
                                  for a in algos], visible=True),
              row=1, col=1)
    f.add_bar(x=algos, y=[lat[a]["weights_fp32_kB"] for a in algos],
              name="weights (fp32)", marker_color=[C[a] for a in algos], row=1, col=2)
    f.add_bar(x=algos, y=[lat[a]["model_zip_kB"] for a in algos],
              name="checkpoint .zip", marker_color=[C[a] for a in algos],
              opacity=0.45, row=1, col=2)
    f.update_layout(title="10 · Inference latency and memory footprint "
                          "(dev CPU, single thread; control period = 1 h)",
                    barmode="group", template="plotly_white", height=380)
    figs.append(f)

    # 11 — sensor-fault robustness
    fault_names = list(dict.fromkeys(r["fault"] for r in faults))
    f = make_subplots(rows=1, cols=2, subplot_titles=(
        "delivered energy under sensor faults (kWh/yr)",
        "shield activations under sensor faults (/yr)"))
    fault_algos = [a for a in ("DQN", "SAC") if any(r["algo"] == a for r in faults)]
    for algo in fault_algos:
        sub = [r for r in faults if r["algo"] == algo]
        f.add_bar(name=algo, x=fault_names,
                  y=[next(r["delivered_kWh"] for r in sub if r["fault"] == fn)
                     for fn in fault_names],
                  marker_color=C[algo], row=1, col=1)
        f.add_bar(name=algo + " ", showlegend=False, x=fault_names,
                  y=[next(r["shields"] for r in sub if r["fault"] == fn)
                     for fn in fault_names],
                  marker_color=C[algo], row=1, col=2)
    nv = sum(r["safety_violations"] for r in faults)
    f.update_layout(title=f"11 · Robustness under sensor faults — observation "
                          f"corrupted, physics untouched · hard-safety "
                          f"violations across all scenarios: {nv}",
                    barmode="group", template="plotly_white", height=400)
    f.update_xaxes(tickangle=30)
    figs.append(f)

    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>O3 Deployment Metrics</title>',
             '<style>body{background:#f9f9f7;margin:0;padding:16px;'
             'font-family:system-ui,sans-serif;color:#0b0b0b}h1{font-size:20px}'
             '.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);'
             'border-radius:10px;margin:14px 0;padding:8px}</style></head><body>',
             '<h1>Objective 3 — deployment metrics (efficiency · savings · '
             'latency · sensor faults)</h1>']
    for i, f in enumerate(figs):
        inc = "cdn" if i == 0 else False
        parts.append('<div class="card">' +
                     f.to_html(full_html=False, include_plotlyjs=inc) + '</div>')
    parts.append('</body></html>')
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(parts), encoding="utf-8")


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--skip-faults", action="store_true")
    ap.add_argument("--obs-version", default="V1", choices=["V1", "V2", "V3"])
    ap.add_argument("--forecast-mode", default="perfect",
                    choices=["perfect", "smart", "none"])
    ap.add_argument("--tag", default=None,
                    help="evaluate results/<algo>/r<regime>_<tag>/ instead of "
                         "the canonical r<regime>/ dir")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase O3-9 — Deployment metrics")
    print(f"obs={args.obs_version}  forecast={args.forecast_mode}  tag={args.tag}")
    print("=" * 60)

    print("\n--- 1+2. Thermal efficiency & energy savings ---")
    eff = efficiency_and_savings(args.regime)
    for c, d in eff["controllers"].items():
        print(f"  {c:4s} η={d['thermal_efficiency_pct']:5.2f}%  "
              f"coverage={d['demand_coverage_pct']:5.2f}%")

    print("\n--- 3. Inference latency & memory ---")
    lat = latency_and_memory(args.regime, obs_version=args.obs_version,
                              forecast_mode=args.forecast_mode, tag=args.tag)
    for a in ("DQN", "SAC"):
        if a not in lat:
            continue
        d = lat[a]
        print(f"  {a:4s} mean={d['latency_mean_us']:7.1f}us "
              f"p95={d['latency_p95_us']:7.1f}us  params={d['n_params']:,} "
              f"weights={d['weights_fp32_kB']}kB zip={d['model_zip_kB']}kB")

    faults = []
    if not args.skip_faults:
        print("\n--- 4. Sensor-fault robustness (obs corrupted, physics true) ---")
        faults = sensor_fault_suite(args.regime, obs_version=args.obs_version,
                                     forecast_mode=args.forecast_mode, tag=args.tag)
        import csv
        suffix = f"_{args.tag}" if args.tag else ""
        fp = RESULTS_DIR / f"phase9_sensor_faults_r{args.regime}{suffix}.csv"
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(faults[0].keys()))
            w.writeheader(); w.writerows(faults)
        print(f"[INFO] sensor-fault CSV -> {fp}")

    out = {"efficiency_and_savings": eff, "latency_memory": lat,
           "sensor_faults": faults}
    suffix = f"_{args.tag}" if args.tag else ""
    jp = RESULTS_DIR / f"deployment_metrics_r{args.regime}{suffix}.json"
    jp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[INFO] metrics JSON -> {jp}")

    html = RESULTS_DIR / "plots" / "R2" / f"deployment_metrics{suffix}.html"
    make_plots(eff, lat, faults, html)
    print(f"[INFO] plots -> {html}")
    print("\n[OK]   Phase O3-9 complete.")


if __name__ == "__main__":
    main()
