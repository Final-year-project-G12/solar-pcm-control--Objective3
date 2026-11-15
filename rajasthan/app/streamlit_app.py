"""
Objective 3 (Rajasthan) — Streamlit portal, same design as tamilnadu/scripts/streamlit_portal.py.   Run:  streamlit run app/streamlit_app.py

    Software-in-the-loop simulation. Not field validation.

The user interface is app/portal_ui.py (shared with the other states); this file is the Rajasthan adapter: it runs the frozen Objective-2
simulator (PCMEnv) with a baseline or a trained agent, records every hour in the portal's record format, and supplies the deployment
metrics. Replay uses train / validation years only (the held-out test years are never offered and the loader refuses them); LIVE uses
Open-Meteo at the regime's medoid point. Nothing here trains and the O3 test years are never touched.
The previous Rajasthan demo is kept as app/classic_app.py.
"""
import datetime
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))
import config_O2  # noqa: E402

config_O2.setup_paths()

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from o3.demo.live_adapter import BANNER, MEDOIDS  # noqa: E402
from o3.demo.replay import _policy  # noqa: E402
from o3.environment.pcm_env import PCMEnv  # noqa: E402
from o3.utils import weather_roles  # noqa: E402
import portal_ui  # noqa: E402

st.set_page_config(page_title="O3 Rajasthan portal", layout="wide")

RES = config_O2.RESULTS_DIR
RERUN = RES / "rerun_2026-11-13" / "gen2"
ORIG = RES / "gen2"
IST_MIN = 330                      # weather rows start at 00:00 UTC; Rajasthan local time is UTC+5:30
ALWAYS_ON = "b0"
BASELINES = {"b0": "B0 always-on pump", "b1": "B1 reactive rule", "b2": "B2 forecast rule"}


# ------------------------------------------------------------------------------------------------ which trained runs to offer
def _summary(root: pathlib.Path, algo: str) -> pd.DataFrame | None:
    p = root / f"gen2_summary_{algo}.csv"
    return pd.read_csv(p) if p.exists() else None


def _selected_runs(regime: int) -> dict:
    """{'DQN': (root, tag), 'SAC': (root, tag)} for this regime.
    DQN: the released run recorded in results/gen2 (the one the locked test used) for regime 1, else the re-run's selection.
    SAC: recomputed from the summary under the annual-only rule (zero violations -> annual delivery >= 95 % of B0 -> maximum shield
    cut), because results/gen2/gen2_selected_sac.json for regime 1 was written under the old rule and still lists an always-charge copy."""
    out = {}
    for root in ((ORIG, RERUN) if regime == 1 else (RERUN, ORIG)):
        f = root / "gen2_selected_dqn.json"
        if f.exists():
            d = json.loads(f.read_text()).get(f"r{regime}_60min")
            if d and (root / d["tag"] / "run_meta.json").exists():
                out["DQN"] = (root, d["tag"])
                break
    for root in ((ORIG, RERUN) if regime == 1 else (RERUN, ORIG)):
        s = _summary(root, "sac")
        if s is None:
            continue
        r = s[(s.regime_id == regime) & (s.control_step_min == 60) & (s.violations == 0) & (s.pct_of_B0_delivery >= 95)]
        r = r[[(root / t / "run_meta.json").exists() for t in r.tag]]
        if len(r):
            out["SAC"] = (root, r.sort_values("shield_cut_vs_B0_pct", ascending=False).iloc[0].tag)
            break
    return out


def _model_tuple(root: pathlib.Path, tag: str) -> tuple:
    from o3.agents.gen2 import run_model_path
    m = json.loads((root / tag / "run_meta.json").read_text())
    return (m["algo"], str(run_model_path(root / tag)), m["obs_version"], m.get("forecast_mode", "perfect"))


def policies(regime: int) -> dict:
    pols = dict(BASELINES)
    for algo, (root, tag) in _selected_runs(regime).items():
        pols[f"{algo.lower()}_sel"] = f"{algo} (selected: {tag.split('_V3')[0]})"
    return pols


def default_policy(regime: int) -> str:
    sel = _selected_runs(regime)
    return "sac_sel" if "SAC" in sel else "dqn_sel" if "DQN" in sel else "b1"


def _resolve(regime: int, key: str):
    if key in BASELINES:
        return key, None
    algo = key.split("_")[0].upper()
    root, tag = _selected_runs(regime)[algo]
    return "model", _model_tuple(root, tag)


# ------------------------------------------------------------------------------------------------ rollouts
def _record(i, year, info, ghi, tamb, cum, k):
    t = datetime.datetime(2001, 1, 1) + datetime.timedelta(minutes=i * 60 + IST_MIN)       # 8760-h series: 29 Feb is absent, so use a 365-day calendar
    req, exe = int(info["action_requested"]), int(info["action_executed"])
    cum["del"] += info["E_del45_Wh"] / 1000.0
    cum["shield"] += (info["shield_water_s"] + info["shield_pcm_s"]) / 3600.0
    cum["safety"] += int(info["n_safety_violations"])
    return {"k": k, "ts": t.strftime("%d %b %H:%M"), "hod": int(info["hour_of_day"]), "ghi": round(float(ghi), 1), "tamb": round(float(tamb), 2),
            "demand_L": round(float(info["draw_kg"]), 2), "exe": exe, "overridden": bool(req != exe or info["shield_override"]),
            "tw": round(float(info["T_top_C"]), 2), "tpcm": round(float(info["T_pcm_C"]), 2), "melt": round(float(info["f_melt"]) * 100.0, 1),
            "pump": 1.0 if info["Q_collector_Wh"] > 0.0 else 0.0,
            "cum_del": round(cum["del"], 2), "cum_shield": round(cum["shield"], 2), "cum_safety": cum["safety"]}


@st.cache_data(show_spinner=False)
def _replay_cached(regime: int, key: str, model: tuple | None, year: int, start_h: int, n_h: int) -> list:
    if year not in weather_roles.TRAIN_YEARS + weather_roles.VAL_YEARS:
        raise PermissionError(f"{year} is not a train / validation year")
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=regime, year=year,
                 obs_version=(model[2] if model else "V2"), forecast_mode=(model[3] if model else "perfect"))
    pol = _policy(key, model)
    obs, _ = env.reset(year=year)
    pol.reset()
    if hasattr(pol, "attach_env"):
        pol.attach_env(env)
    cum, rows = {"del": 0.0, "shield": 0.0, "safety": 0}, []
    for i in range(min(start_h + n_h, 8760)):
        a = pol.act(obs)
        obs, _, done, _, info = env.step(a)
        if i >= start_h:
            rows.append(_record(i, year, info, env._ghi[i], env._tamb[i], cum, len(rows)))
        if done:
            break
    return rows


def replay(regime, key, year, start_h, n_h):
    kind, model = _resolve(regime, key)
    return _replay_cached(regime, kind, model, int(year), int(start_h), int(n_h))


# ------------------------------------------------------------------------------------------------ LIVE
def _fetch_open_meteo(lat, lon, past_days, forecast_days):
    url = ("https://api.open-meteo.com/v1/forecast" f"?latitude={lat}&longitude={lon}&hourly=shortwave_radiation,temperature_2m"
           f"&past_days={past_days}&forecast_days={forecast_days}&timezone=Asia%2FKolkata")
    with urllib.request.urlopen(url, timeout=12) as resp:
        h = json.loads(resp.read())["hourly"]
    ghi = [float(x) if x is not None else 0.0 for x in h["shortwave_radiation"]]
    tamb = [float(x) if x is not None else 28.0 for x in h["temperature_2m"]]
    return h["time"], ghi, tamb, datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def live(regime, keys, past_days, forecast_days):
    m = MEDOIDS[regime]
    times, ghi, tamb, fetched = _fetch_open_meteo(m["lat"], m["lon"], past_days, forecast_days)
    runs = {}
    for key in dict.fromkeys(keys):
        if key == "b2":
            raise ValueError("B2 needs the stored year's forecast and is not available in LIVE mode; pick another policy")
        kind, model = _resolve(regime, key)
        env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=regime, year=weather_roles.VAL_YEARS[0],
                     obs_version=(model[2] if model else "V2"), forecast_mode=(model[3] if model else "perfect"))
        pol = _policy(kind, model)
        env.reset(year=weather_roles.VAL_YEARS[0])
        pol.reset()
        if hasattr(pol, "attach_env"):
            pol.attach_env(env)
        cum, rows = {"del": 0.0, "shield": 0.0, "safety": 0}, []
        n = len(ghi)
        for k in range(n):
            hod = int(times[k][11:13])
            fc = tuple(ghi[k + j] if k + j < n else 0.0 for j in (1, 2, 3))     # the provider's own next three hours
            obs = env.set_weather_override(ghi[k], tamb[k], hod, fc)
            _, _, done, _, info = env.step(pol.act(obs))
            r = _record(0, 2021, info, ghi[k], tamb[k], cum, k)
            r["ts"] = times[k][8:10] + " " + times[k][11:16]
            rows.append(r)
            if done:
                break
        runs[key] = rows
    ctx = {"label": f"{m['place']} (regime {regime} medoid {m['point']})", "lat": m["lat"], "lon": m["lon"], "fetched": fetched,
           "now_idx": min(past_days * 24, len(ghi) - 1)}
    return runs, ctx


# ------------------------------------------------------------------------------------------------ deployment tab
@st.cache_data(show_spinner=False)
def deployment(regime: int):
    p = RES / f"deployment_metrics_r{regime}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    names = {"TWIN_water_only": "Water-only twin", "B0_fixed_charge": "B0 always-on", "B1_reactive_rule": "B1 reactive", "B2_forecast_rule": "B2 forecast"}
    ctr = {}
    for k, v in d["policies"].items():
        ctr[names.get(k, k)] = {"delivered_kWh": v["E_del45_kWh_yr"], "thermal_efficiency_pct": v["thermal_efficiency_pct"],
                                "demand_coverage_pct": v["demand_coverage_pct"], "shield_h": v["shield_h_yr"], "violations": v["violations"],
                                "backup_kWh": v.get("backup_heat_kWh_yr")}
    inf = d.get("inference", {})
    algo = d["run"]["algo"].upper()
    rows = [{"policy": pol, "fault": f, "delivered_kWh": m["E45_kWh"], "shield_h": m["shield_h"], "violations": m["violations"]}
            for pol, faults in d.get("sensor_faults", {}).items() for f, m in faults.items()]
    return {"controllers": ctr,
            "inference": {algo: {"ms": inf["ms_per_decision"], "params": inf["policy_parameters"], "kB": inf["fp32_weights_kB"]}} if inf else None,
            "note": inf.get("note", ""), "faults": pd.DataFrame(rows) if rows else None,
            "assumptions": f"Run {d['run']['tag']} · validation years {d['years']} · incident solar {d['incident_solar_kWh_yr']:.0f} kWh/yr on "
                           f"{d['collector_area_m2']} m² · {d.get('heater_note', '')}"}


ALGO_ROWS = [
    ("PPO", "On-policy policy gradient", "Collapsed to always-charge (B0) on regime 1 (3 seeds)"),
    ("DQN", "Value-based, ε-greedy", "Regime 1 headline run (seed 42, weight 1.7): 98.2% of B0 with 57% fewer guard hours, 98.0% / −61% on the one-shot "
                                    "locked test. A retrain does not reproduce it (seed-fragile); regimes 0 and 2 are B0-equivalent only"),
    ("SAC", "Off-policy, entropy-regularised", "Passes the annual 95% rule with a real shield cut in all three regimes: regime 0 95.2% / −96%, regime 1 "
                                              "95.7% / −86%, regime 2 96.4% / −71% (validation years; no held-out test, and it costs some 45 °C retention)"),
]
SELECTION_NOTE = ("Selection rule: zero hard-limit violations → annual delivery ≥ 95% of B0 → maximum shield reduction. This annual-only rule "
                  "matches Tamil Nadu and was adopted after the first locked test (post-hoc), so SAC results have no held-out confirmation.")

PORTAL = {
    "state": "Rajasthan", "banner": BANNER,
    "title": "Objective 3 — Climate-Adaptive PCM Controller Portal",
    "caption": "Rajasthan solar water heating (50 L tank, 1.5 m² collector, 45 °C target) · safety-shielded DRL · SOFTWARE-IN-THE-LOOP — "
               "thermal state is simulated, not measured. Field validation is Objective 4.",
    "regimes": {0: "0 — Udaipur (savE OM55)", 1: "1 — Nagaur (PureTemp 60)", 2: "2 — Jaipur (PureTemp 58)"},
    "policies": policies, "default_policy": default_policy, "always_on": ALWAYS_ON,
    "years": weather_roles.TRAIN_YEARS + weather_roles.VAL_YEARS, "default_year": 2021, "default_month": 5,
    "replay": replay, "live": live, "deployment": deployment,
    "guards": (72.0, 75.0, 45.0), "shield_unit": "hours",
    "pump_note": "Pump 'ran' = the collector delivered heat in that hour (collector gain > 0), so it counts whole hours.",
    "algo_rows": ALGO_ROWS, "selection_note": SELECTION_NOTE,
}

portal_ui.run_portal(PORTAL)
