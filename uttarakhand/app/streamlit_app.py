"""
app/streamlit_app.py -- Objective 3 Uttarakhand portal, same design as tamilnadu/scripts/streamlit_portal.py.

    streamlit run app/streamlit_app.py          (from the uttarakhand/ folder, with the .venv active)

    Software-in-the-loop simulation. Not field validation.

The user interface is app/portal_ui.py (shared with the other states); this file is the Uttarakhand adapter. The tank and PCM are
simulated by the unchanged O2 kernel through o3/live.py (`simulate` runs every controller and the water-only twin through one weather
frame with the same reset state); the portal reads the hourly logs it returns. Replay is limited to train / validate years (the O3 test
years and O2's used years are refused by o3/roles_o3.py). LIVE uses Open-Meteo at the regime's medoid point (o3/live.fetch_live).
The previous Uttarakhand demo is kept as app/classic_app.py.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))
from o3 import config as o3c          # noqa: E402
from o3 import live                   # noqa: E402
import portal_ui                      # noqa: E402

st.set_page_config(page_title="O3 Uttarakhand portal", layout="wide")

CFG = o3c.load()
ALWAYS_ON = "B0"
LABELS = {"B0": "B0 always-on pump", "B1": "B1 reactive rule", "B2": "B2 forecast rule", "PPO": "PPO (frozen M4 policy)"}


@st.cache_resource(show_spinner=False)
def _controllers(regime: int):
    return live.controllers(regime)


def policies(regime: int) -> dict:
    return {c.name: LABELS.get(c.name, c.name) for c in _controllers(regime)}


def default_policy(regime: int) -> str:
    return "PPO" if "PPO" in policies(regime) else "B1" if "B1" in policies(regime) else ALWAYS_ON


# ------------------------------------------------------------------------------------------------ records
def _records(df: pd.DataFrame, start_h: int, n_h: int, label_fmt: str = "%d %b %H:%M") -> list:
    """Hourly log of one controller -> portal records for the window [start_h, start_h + n_h); cumulative columns count inside it."""
    w = df.iloc[start_h:start_h + n_h].reset_index(drop=True)
    shield_h = (w["shield_water_s"] + w["shield_pcm_s"]) / 3600.0
    cum_del, cum_shield = (w["Edel_Wh"] / 1000.0).cumsum(), shield_h.cumsum()
    cum_safety = (w["over_limit_s"] / 3600.0).cumsum()
    out = []
    for i, r in w.iterrows():
        enabled = int(r["action"]) == 1
        out.append({"k": i, "ts": pd.Timestamp(r["local_time"]).strftime(label_fmt), "hod": int(pd.Timestamp(r["local_time"]).hour),
                    "ghi": round(float(r["GHI_Wm2"]), 1), "tamb": round(float(r["T_amb_C"]), 2), "demand_L": round(float(r["draw_kg"]), 2),
                    "exe": 0 if enabled else 2, "overridden": bool(shield_h.iloc[i] > 0 or r.get("guard_hit", False)),
                    "tw": round(float(r["T_top"]), 2), "tpcm": round(float(r["T_pcm"]), 2), "melt": round(float(r["f_melt"]) * 100.0, 1),
                    "pump": 1.0 if r["Q_coll_Wh"] > 0 else 0.0,
                    "cum_del": round(float(cum_del.iloc[i]), 2), "cum_shield": round(float(cum_shield.iloc[i]), 2),
                    "cum_safety": round(float(cum_safety.iloc[i]), 2)})
    return out


SCENARIO = CFG["reset_scenarios"][0]


@st.cache_data(show_spinner="Simulating the year...")
def _year_runs(regime: int, year: int, scenario: str) -> dict:
    frame = live.replay_frame(regime, year)                       # raises for any year that is not train / validate
    return live.simulate(regime, frame, _controllers(regime), scenario)


def replay(regime, key, year, start_h, n_h):
    runs = _year_runs(regime, int(year), SCENARIO)
    return _records(runs[key], int(start_h), int(n_h))


# ------------------------------------------------------------------------------------------------ LIVE
def live_run(regime, keys, past_days, forecast_days):
    past = max(1, int(past_days)) * 24
    fr, rec = live.fetch_live(regime, past_days=max(1, int(past_days)))
    if not rec["ok"]:
        raise RuntimeError(f"live fetch failed ({rec.get('error')})")
    fr = fr.iloc[-past:].reset_index(drop=True)
    fr, _ = live.extend_stale(fr)
    runs = live.simulate(regime, fr, _controllers(regime), SCENARIO, nowcast=True)
    lat, lon = live.MEDOID_COORDS[regime]
    ctx = {"label": live.REGIME_LABELS[regime], "lat": lat, "lon": lon, "fetched": rec.get("request_utc", ""), "now_idx": len(fr) - 1}
    return {k: _records(runs[k], 0, len(fr)) for k in dict.fromkeys(keys)}, ctx


# ------------------------------------------------------------------------------------------------ deployment tab
@st.cache_data(show_spinner=False)
def deployment(regime: int):
    p = o3c.RESULTS / f"m10_deployment_r{regime}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ctr = {k: {"delivered_kWh": v["E_del_annual_kWh"], "thermal_efficiency_pct": 100 * v["thermal_efficiency"],
               "demand_coverage_pct": 100 * v["demand_coverage"], "shield_h": v["overheat_h"], "violations": int(v["hours_over_limit"] > 0)}
           for k, v in d["controllers"].items()}
    fp = d.get("ppo_footprint")
    out = {"controllers": ctr,
           "inference": {"PPO": {"ms": fp["latency_ms_median"], "params": fp["n_parameters"], "kB": fp["weights_fp32_bytes"] / 1000.0}} if fp else None,
           "note": (fp or {}).get("note", ""),
           "assumptions": f"Validation years {d['years']} · collector {d['collector_area_m2']} m² · heater efficiency assumed {d['eta_heater_assumed']} · "
                          "'shield hours' here are the hours the overheat shield acted (overheat_h in the M10 table)."}
    fcsv = o3c.RESULTS / f"m10_sensor_faults_r{regime}.csv"
    if fcsv.exists():
        f = pd.read_csv(fcsv)
        g = f.groupby(["controller", "scenario"]).agg(delivered_kWh=("E_del_annual_kWh", "mean"), shield_h=("overheat_h", "mean"),
                                                      violations=("hours_over_limit", lambda s: int((s > 0).sum()))).reset_index()
        out["faults"] = g.rename(columns={"controller": "policy", "scenario": "fault"})
    return out


ALGO_ROWS = [
    ("PPO", "On-policy policy gradient (frozen after milestone M4)", "The controller this state runs; compared against B0, the tuned reactive rule B1 and the forecast rule B2"),
]
SELECTION_NOTE = ("Uttarakhand's controller is the frozen PPO from milestone M4 (no Gen-2 DQN/SAC study was run here). Compare it with B0 on shield hours "
                  "as well as delivery: a policy that only matches always-on delivery has not improved on it.")

PORTAL = {
    "state": "Uttarakhand", "banner": live.BANNER,
    "title": "Objective 3 — Climate-Adaptive PCM Controller Portal",
    "caption": "Uttarakhand solar water heating (1.5 m² collector, 50 °C target) · safety-shielded DRL bypass control · SOFTWARE-IN-THE-LOOP — "
               "thermal state is simulated, not measured. Field validation is Objective 4.",
    "regimes": dict(live.REGIME_LABELS),
    "policies": policies, "default_policy": default_policy, "always_on": ALWAYS_ON,
    "years": CFG["years"]["train"] + CFG["years"]["validate"], "default_year": CFG["years"]["validate"][0], "default_month": 5,
    "replay": replay, "live": live_run, "deployment": deployment,
    "guards": (72.0, 75.0, 50.0), "shield_unit": "hours",
    "pump_note": "Pump 'ran' = the collector delivered heat in that hour (collector gain > 0), so it counts whole hours. "
                 "The Uttarakhand action is binary (enable / bypass): there is no separate 'idle' action.",
    "algo_rows": ALGO_ROWS, "selection_note": SELECTION_NOTE,
}

portal_ui.run_portal(PORTAL)
