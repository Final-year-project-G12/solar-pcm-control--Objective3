"""
Phase O3-7 (visual) — Live software-in-the-loop DASHBOARD.

Runs the selected, frozen policy through the real Objective-2 physics for a
window of a validation year and emits a single self-contained HTML page that
*plays back* the run hour-by-hour: a CHARGING / SERVING / BYPASS status badge,
live KPI tiles (water & PCM temperature, melt fraction, irradiance, delivered
energy, shield fires), a temperature chart with the delivery target and the
75/65 degC hard limits + 72/62 degC guard bands drawn in, an irradiance chart,
and a colour-coded action ribbon. A moving "now" marker sweeps across as it
plays. No server required — open the .html in any browser and press Play.

SOFTWARE-IN-THE-LOOP ONLY: thermal state comes from the frozen simulator, not
sensors (contract section 9 / 15.9). Never uses the locked test years.

Run from the objective3 root:
    python scripts/make_live_dashboard.py --algo sac --regime 0 \
        --year 2021 --start-day 160 --days 10
"""

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parents[1]
import os


def _find_o2_root() -> pathlib.Path:
    """Locate the Objective-2 code (needs src/io_utils.py): $O2_ROOT first,
    then the usual sibling locations."""
    cands = [os.environ.get("O2_ROOT", ""),
             REPO_ROOT.parent / "objective2_design_optimization",
             REPO_ROOT.parents[2] / "new_obj" / "objective2_design_optimization",
             REPO_ROOT.parents[1] / "objective2_design_optimization"]
    for c in cands:
        if c and (pathlib.Path(c) / "src" / "io_utils.py").exists():
            return pathlib.Path(c)
    return REPO_ROOT.parent / "objective2_design_optimization"


_O2_ROOT  = _find_o2_root()
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"

# locked test years must never drive the demo
TEST_YEARS = {2011, 2012, 2013, 2014, 2015}

# validated, colourblind-safe palette (dataviz skill)
C_CHARGE = "#2a78d6"   # blue   — collector loop ON, charging the PCM/tank
C_IDLE   = "#1baf7a"   # aqua   — pump off, tank serving demand (discharge)
C_BYPASS = "#eb6834"   # orange — collector isolated to prevent overheat
C_WATER  = "#2a78d6"
C_PCM    = "#eb6834"
ACTION_COLOR = {0: C_CHARGE, 1: C_IDLE, 2: C_BYPASS}
ACTION_NAME  = {0: "CHARGING", 1: "SERVING (idle)", 2: "BYPASS"}
ACTION_SHORT = {0: "CHARGE", 1: "IDLE", 2: "BYPASS"}

_ALGO_DIR = {"ppo": "ppo", "dqn": "dqn", "recurrent": "recurrent_ppo",
             "qrdqn": "qrdqn", "maskable": "maskable_ppo", "sac": "sac"}


def model_variant_available(algo: str, regime_id: int, tag: str) -> bool:
    """True if a tagged (e.g. Generation-2) checkpoint exists for algo/regime."""
    base = RESULTS_DIR / _ALGO_DIR[algo] / f"r{regime_id}_{tag}"
    return (base / "best_model" / "best_model.zip").exists()


def _load_model(algo: str, regime_id: int, model_tag: str | None = None):
    suffix = f"_{model_tag}" if model_tag else ""
    base = RESULTS_DIR / _ALGO_DIR[algo] / f"r{regime_id}{suffix}"
    path = base / "best_model" / "best_model"
    if not path.with_suffix(".zip").exists():
        path = base / f"{_ALGO_DIR[algo]}_final"
    p = str(path)
    if algo == "ppo":
        from stable_baselines3 import PPO; return PPO.load(p)
    if algo == "dqn":
        from stable_baselines3 import DQN; return DQN.load(p)
    if algo == "sac":
        from stable_baselines3 import SAC; return SAC.load(p)
    if algo == "recurrent":
        from sb3_contrib import RecurrentPPO; return RecurrentPPO.load(p)
    if algo == "qrdqn":
        from sb3_contrib import QRDQN; return QRDQN.load(p)
    if algo == "maskable":
        from sb3_contrib import MaskablePPO; return MaskablePPO.load(p)
    raise ValueError(algo)


def run_rollout(algo: str, regime_id: int, year: int,
                start_hour: int, n_hours: int,
                model_tag: str | None = None, obs_version: str = "V1",
                forecast_mode: str = "perfect",
                control_step_min: int = 60,
                force_charge: bool = False) -> list[dict]:
    """Step the policy through the env; log each hour in the window.

    model_tag selects a tagged checkpoint (e.g. a Generation-2 or "smartfc"
    run from scripts/train_one.py); obs_version/forecast_mode/control_step_min
    must match how that checkpoint was trained (V2 obs_version pairs with
    every GEN2 tag and forecast_mode="perfect"; V3 + forecast_mode="smart"
    pairs with the "smartfc" tag; use control_step_min=15 for a
    "subhourly15..." tag, else 60).
    """
    from o3.environment.gym_wrapper import PCMGymEnv, PCMContinuousGymEnv
    continuous = (algo == "sac")
    EnvCls = PCMContinuousGymEnv if continuous else PCMGymEnv
    env = EnvCls(CONTRACT_PATH, regime_id=regime_id, fixed_year=year,
                obs_version=obs_version, forecast_mode=forecast_mode,
                control_step_min=control_step_min)
    model = None if force_charge else _load_model(algo, regime_id, model_tag=model_tag)

    obs, _ = env.reset()
    inner  = env.inner
    weather = inner._weather
    demand  = inner._demand_model

    recs = []
    cum_del = cum_unmet = 0.0
    cum_shield = cum_safety = 0
    lstm_state = None
    done = False
    step = 0
    end_hour = start_hour + n_hours
    while not done and step < end_hour:
        if force_charge:                       # always-on baseline (B0)
            action = np.array([-1.0], dtype=np.float32) if continuous else 0
        elif algo == "recurrent":
            action, lstm_state = model.predict(
                obs, state=lstm_state, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        hour_idx = int(info["hour_idx"])
        if step >= start_hour:
            # cumulative KPIs count only within the displayed window
            cum_del    += info["E_delivered_Wh"] / 1000.0
            cum_unmet  += info["E_unmet_Wh"] / 1000.0
            cum_shield += int(info["n_shield_water"]) + int(info["n_shield_pcm"])
            cum_safety += int(info["n_safety_violations"])
            row = weather.iloc[hour_idx]
            ghi = float(row["GHI_Wm2"])
            tamb = float(row["T_amb_C"])
            hod = int(info["hour_of_day"])
            draw = float(demand.draw_mass_kg_for_hour(hod))
            req = int(info["action_requested"])
            exe = int(info["action_executed"])
            ts = (datetime.datetime(year, 1, 1) +
                  datetime.timedelta(hours=hour_idx)).strftime("%d %b %H:%M")
            recs.append({
                "k":        len(recs),
                "hour_idx": hour_idx,
                "day":      hour_idx // 24,
                "hod":      hod,
                "ts":       ts,
                "ghi":      round(ghi, 1),
                "tamb":     round(tamb, 2),
                "demand_L": round(draw, 2),
                "req":      req,
                "exe":      exe,
                "overridden": bool(req != exe or info["shield_override"]),
                "tw":       round(float(info["T_w_C"]), 2),
                "tpcm":     round(float(info["T_pcm_C"]), 2),
                "melt":     round(float(info["f_melt"]) * 100.0, 1),
                "e_step":   round(info["E_delivered_Wh"] / 1000.0, 3),
                "unmet_step": round(info["E_unmet_Wh"] / 1000.0, 3),
                "pump":     round(float(info["pump_on_frac"]), 2),
                "shield_step": int(info["n_shield_water"]) + int(info["n_shield_pcm"]),
                "cum_del":  round(cum_del, 1),
                "cum_unmet": round(cum_unmet, 1),
                "cum_shield": cum_shield,
                "cum_safety": cum_safety,
            })
        step += 1
    return recs


# --------------------------------------------------------------- LIVE --------
# representative coordinates per Tamil Nadu regime medoid (lat, lon, label)
REGIME_COORDS = {
    0: (9.93, 78.12, "Madurai area (regime-0 medoid)"),
    1: (11.41, 76.70, "Nilgiris-adjacent (regime-1 medoid)"),
    2: (13.08, 80.27, "Chennai coast (regime-2 medoid)"),
}


def fetch_open_meteo(lat: float, lon: float,
                     past_days: int, forecast_days: int):
    """Fetch hourly GHI + ambient temperature from the free Open-Meteo API.

    Returns (DataFrame[GHI_Wm2, T_amb_C, local_hour], iso_timestamps, fetched_utc)
    or raises on failure (caller handles fallback).
    """
    import urllib.request
    import pandas as pd
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,temperature_2m"
        f"&past_days={past_days}&forecast_days={forecast_days}"
        "&timezone=Asia%2FKolkata"
    )
    with urllib.request.urlopen(url, timeout=12) as resp:
        data = json.loads(resp.read())
    h = data["hourly"]
    times = h["time"]
    ghi   = [float(x) if x is not None else 0.0 for x in h["shortwave_radiation"]]
    tamb  = [float(x) if x is not None else 28.0 for x in h["temperature_2m"]]
    local_hour = [int(t[11:13]) for t in times]
    df = pd.DataFrame({"GHI_Wm2": ghi, "T_amb_C": tamb, "local_hour": local_hour})
    fetched = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return df, times, fetched


def run_rollout_live(algo: str, regime_id: int, wx_df, iso_times: list[str],
                     model_tag: str | None = None, obs_version: str = "V1",
                     forecast_mode: str = "perfect",
                     control_step_min: int = 60,
                     force_charge: bool = False) -> list[dict]:
    """Run the policy through the frozen physics on *live* fetched weather.

    NOTE: Open-Meteo's feed has no clear-sky-irradiance column, so
    forecast_mode="smart" (smart persistence on the clear-sky index) is not
    supported here regardless of what is passed — a model trained with
    forecast_mode="smart" must not be selected for LIVE mode until an
    analytic clear-sky model (pvlib, from lat/lon) is ported; see
    CORRECTIONS_OBJECTIVE3.md "Still open" #3. This function will silently
    fall back the forecast fields to 0 (same as forecast_mode="none") if a
    smart-forecast model is run here, since there is no clear-sky column to
    compute Kc from.
    """
    from o3.environment.gym_wrapper import PCMGymEnv, PCMContinuousGymEnv
    continuous = (algo == "sac")
    EnvCls = PCMContinuousGymEnv if continuous else PCMGymEnv
    live_forecast_mode = "none" if forecast_mode == "smart" else forecast_mode
    env = EnvCls(CONTRACT_PATH, regime_id=regime_id, fixed_year=2021,
                obs_version=obs_version, forecast_mode=live_forecast_mode,
                control_step_min=control_step_min)
    model = None if force_charge else _load_model(algo, regime_id, model_tag=model_tag)

    obs, _ = env.reset()                 # sets initial tank/PCM state
    inner  = env.inner
    inner._weather = wx_df.reset_index(drop=True)   # inject live weather
    inner._state["tick_idx"] = 0
    inner._state["hour_idx"] = 0
    obs = inner._get_obs()
    demand = inner._demand_model

    recs = []
    cum_del = cum_unmet = 0.0
    cum_shield = cum_safety = 0
    lstm_state = None
    n = len(wx_df)
    done = False
    while not done and len(recs) < n:
        if force_charge:                       # always-on baseline (B0)
            action = np.array([-1.0], dtype=np.float32) if continuous else 0
        elif algo == "recurrent":
            action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cum_del    += info["E_delivered_Wh"] / 1000.0
        cum_unmet  += info["E_unmet_Wh"] / 1000.0
        cum_shield += int(info["n_shield_water"]) + int(info["n_shield_pcm"])
        cum_safety += int(info["n_safety_violations"])
        k = len(recs)
        t = iso_times[k] if k < len(iso_times) else ""
        hod = int(info["hour_of_day"])
        recs.append({
            "k": k, "hour_idx": k, "day": k // 24, "hod": hod,
            "ts": (t[8:10] + " " + t[11:16]) if t else f"h{k}",
            "ghi": round(float(wx_df.iloc[k]["GHI_Wm2"]), 1),
            "tamb": round(float(wx_df.iloc[k]["T_amb_C"]), 2),
            "demand_L": round(float(demand.draw_mass_kg_for_hour(hod)), 2),
            "req": int(info["action_requested"]), "exe": int(info["action_executed"]),
            "overridden": bool(info["action_requested"] != info["action_executed"]
                               or info["shield_override"]),
            "tw": round(float(info["T_w_C"]), 2),
            "tpcm": round(float(info["T_pcm_C"]), 2),
            "melt": round(float(info["f_melt"]) * 100.0, 1),
            "e_step": round(info["E_delivered_Wh"] / 1000.0, 3),
            "unmet_step": round(info["E_unmet_Wh"] / 1000.0, 3),
            "pump": round(float(info["pump_on_frac"]), 2),
            "shield_step": int(info["n_shield_water"]) + int(info["n_shield_pcm"]),
            "cum_del": round(cum_del, 1), "cum_unmet": round(cum_unmet, 1),
            "cum_shield": cum_shield, "cum_safety": cum_safety,
        })
    return recs


# ---------------------------------------------------------------- HTML -------
def build_html(recs: list[dict], meta: dict) -> str:
    data_json = json.dumps(recs, separators=(",", ":"))
    meta_json = json.dumps(meta, separators=(",", ":"))
    # action ribbon segments (one coloured cell per hour)
    return _HTML_TEMPLATE.replace("__DATA__", data_json)\
                         .replace("__META__", meta_json)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>O3 Controller Playback</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
  :root{--bg:#f6f7f5;--surface:#ffffff;--ink:#0b0b0b;--muted:#5c5b57;
        --line:rgba(11,11,11,.12);--collect:#2a78d6;--wait:#b4b7bd;--hold:#1baf7a;--bypass:#eb6834;
        --good:#0f7b4d;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:16px;}
  h1{font-size:20px;margin:0 0 2px}
  h2{font-size:14px;margin:0 0 8px}
  .sub{color:var(--muted);font-size:13px;margin:0 0 8px}
  .sil{display:inline-block;background:#fff3e0;color:#8a4b00;border:1px solid #f0c38a;
       border-radius:6px;padding:3px 9px;font-size:12px;font-weight:600}
  .wrap{max-width:1040px;margin:0 auto}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
  .score{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
  .sc{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
  .sc .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .sc .big{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;margin:3px 0}
  .sc .ctx{font-size:12px;color:var(--muted);line-height:1.35}
  .good{color:var(--good)}
  .verdict{font-size:14px;line-height:1.5;margin:0 0 12px}
  .controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  button{font:inherit;font-weight:600;border:none;border-radius:8px;padding:9px 18px;background:var(--collect);color:#fff;cursor:pointer}
  button.sec{background:#e9e9e6;color:var(--ink)}
  select{font:inherit}
  input[type=range]{flex:1;min-width:180px;accent-color:var(--collect)}
  .clock{font-variant-numeric:tabular-nums;font-weight:600;font-size:14px;color:var(--muted)}
  .now{display:flex;align-items:stretch;gap:14px;flex-wrap:wrap;margin-top:14px}
  .lamp{font-size:19px;font-weight:800;color:#fff;border-radius:10px;padding:12px 18px;
        min-width:250px;display:flex;align-items:center;justify-content:center;text-align:center;transition:background .15s}
  .why{flex:1;min-width:240px;font-size:14px;line-height:1.45;align-self:center}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:10px;margin-top:12px}
  .tile{border:1px solid var(--line);border-radius:10px;padding:8px 10px}
  .tile .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .tile .val{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}
  .tile .u{font-size:12px;color:var(--muted);font-weight:500}
  .ribbon{display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid var(--line);cursor:pointer;margin-top:6px}
  .cell{flex:1 0 0}
  .cursor{height:34px;width:2px;background:var(--ink);position:relative;margin-top:-30px;transition:left .12s linear;pointer-events:none}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-top:8px;flex-wrap:wrap}
  .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
  details{font-size:13px;line-height:1.55;color:var(--ink)}
  summary{cursor:pointer;font-weight:700;font-size:14px}
  details p{margin:8px 0}
  @media (prefers-color-scheme:dark){:root:not([data-theme=light]){
     --bg:#15161a;--surface:#1e2026;--ink:#f2f2ef;--muted:#a3a39d;--line:rgba(242,242,239,.14);--good:#4cc794;}
     .sil{background:#3a2a10;color:#f0c38a;border-color:#6b4a1a} button.sec{background:#2b2e36;color:var(--ink)}}
</style></head>
<body><div class="wrap">
  <h1>Solar water heater — controller playback</h1>
  <p class="sub" id="subtitle"></p>
  <span class="sil">Simulation: temperatures come from the validated simulator, not field sensors</span>

  <div class="card">
    <h2>Did the controller do its job over these <span id="nhrs"></span> hours?</h2>
    <p class="verdict" id="verdict"></p>
    <div class="score" id="score"></div>
  </div>

  <div class="card">
    <div class="controls">
      <button id="play">▶ Play</button>
      <button id="reset" class="sec">⟲ Reset</button>
      <label class="clock">speed
        <select id="speed"><option value="240">0.5×</option><option value="120" selected>1×</option>
          <option value="60">2×</option><option value="24">5×</option></select></label>
      <input type="range" id="scrub" min="0" value="0">
      <span class="clock" id="clock">—</span>
    </div>
    <div class="now">
      <div class="lamp" id="lamp">—</div>
      <div class="why" id="why"></div>
    </div>
    <div class="tiles" id="tiles"></div>
  </div>

  <div class="card"><div id="tempChart" style="height:290px"></div>
    <div class="ribbon" id="ribbon"></div>
    <div class="cursor" id="cursor"></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--collect)"></span>Collecting — pump running, solar heat going into tank + PCM</span>
      <span><span class="sw" style="background:var(--wait)"></span>Standby — pump off (no useful sun / no net gain)</span>
      <span><span class="sw" style="background:var(--hold)"></span>Held off — controller paused the pump in sun</span>
      <span><span class="sw" style="background:var(--bypass)"></span>Bypass — collector isolated to avoid overheating</span>
      <span>hatched = safety shield forced it</span>
    </div>
  </div>

  <div class="card"><div id="dayChart" style="height:230px"></div></div>
  <div class="card"><div id="ghiChart" style="height:180px"></div></div>

  <div class="card"><details>
    <summary>How to read this page</summary>
    <p><b>The controller picks one of three commands every hour:</b> CHARGE (allow the pump), IDLE (pump off) or BYPASS (isolate the collector).
       CHARGE only means "pump allowed". The simulator itself switches the pump off whenever the collector would lose heat — all night and in weak sun — so a CHARGE command at night changes nothing.
       That is why this page colours each hour by what the <b>pump physically did</b>, not just by the command: otherwise most of the strip would look like "always charge".</p>
    <p><b>Why so few Held-off / Bypass hours?</b> Hot water is only at risk of overheating for a few midday hours (water ≥ 72 °C or PCM ≥ 62 °C). Outside that, running the pump whenever the sun gives net gain is already the right thing to do, so the controller only steps in for a few hours per day. IDLE and BYPASS both switch the pump off; BYPASS marks the cases that protect against overheating.</p>
    <p><b>What "efficient" means here.</b> Both the controller and an always-on pump collect nearly all available heat. The gain is avoiding overheat guard hits (wasted heat dumped by the safety shield and thermal stress on tank and PCM) while the same hot water is delivered. The scorecard compares both on the identical weather window.</p>
    <p><b>Playback</b> = pressing ▶ steps through the window one simulated hour at a time; drag the slider or click the colour strip to jump to any hour.</p>
  </details></div>

<script>
const DATA = __DATA__;
const META = __META__;
const N = DATA.length;
const COL = {collect:"#2a78d6", wait:"#b4b7bd", hold:"#1baf7a", bypass:"#eb6834"};
const LBL = {collect:"COLLECTING SOLAR HEAT", wait:"STANDBY — PUMP OFF", hold:"HELD OFF (in sun)", bypass:"BYPASS — OVERHEAT PROTECTION"};
const SUN = 50;   // W/m2 threshold for "useful sun"
const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme:dark)").matches;
const INK = isDark?"#f2f2ef":"#0b0b0b", MUT = isDark?"#a3a39d":"#5c5b57",
      GRID = isDark?"rgba(242,242,239,.10)":"rgba(11,11,11,.08)", PAPER="rgba(0,0,0,0)";

function mode(d){
  if(d.exe===2 && d.overridden) return "bypass";
  if(d.ghi<SUN && d.pump<0.5) return "wait";
  if(d.exe===2) return "bypass";
  if(d.exe===1) return "hold";
  return d.pump>=0.5 ? "collect" : "wait";
}
function reason(d){
  const m=mode(d), tw=d.tw.toFixed(0), g=Math.round(d.ghi);
  if(m==="collect") return "Sun is "+g+" W/m² and the collector is hotter than the tank ("+tw+" °C), so the pump runs and stores the heat.";
  if(m==="bypass" && d.overridden) return "Water/PCM reached the guard band ("+tw+" °C water). The safety shield isolated the collector"+(d.req!==2?" even though the controller asked for "+["CHARGE","IDLE","BYPASS"][d.req]:"")+".";
  if(m==="bypass") return "Controller isolated the collector before the guard band: tank "+tw+" °C and sun "+g+" W/m² — heading for overheating.";
  if(m==="hold") return "Sun is available ("+g+" W/m²) but the controller paused the pump to keep temperatures (water "+tw+" °C, PCM "+d.tpcm.toFixed(0)+" °C) below the limits.";
  return d.ghi<SUN ? "No useful sun ("+g+" W/m²). The pump is off by itself; the stored heat is serving the "+d.demand_L+" L draw."
                   : "Sun is "+g+" W/m² but the collector cannot add net heat to a "+tw+" °C tank, so the pump stays off.";
}

// ---------- header + scorecard
document.getElementById("subtitle").textContent =
  (META.mode==="live" ? "LIVE weather ("+META.location+", fetched "+META.fetched+")" : "Historical replay, year "+META.year)
  +" · "+META.algo_name+" controller · regime "+META.regime+" · "+META.pcm;
document.getElementById("nhrs").textContent = N;

const sum=(f)=>DATA.reduce((a,d)=>a+f(d),0);
const cnt={collect:0,wait:0,hold:0,bypass:0}; DATA.forEach(d=>cnt[mode(d)]++);
const sunHrs=DATA.filter(d=>d.ghi>=SUN).length;
const pumpHrs=sum(d=>d.pump), pumpSun=DATA.filter(d=>d.ghi>=SUN).reduce((a,d)=>a+d.pump,0);
const del=DATA[N-1].cum_del, unmet=DATA[N-1].cum_unmet, metPct=100*del/(del+unmet);
const shield=DATA[N-1].cum_shield, safety=DATA[N-1].cum_safety;
const B=META.base, bMet=100*B.del/(B.del+B.unmet);
const shieldCut = B.shield>0 ? 100*(B.shield-shield)/B.shield : 0;
const delDiff = 100*(del-B.del)/B.del;

document.getElementById("verdict").innerHTML =
  "Compared with an <b>always-on pump</b> on the same weather, this controller delivered "+
  (Math.abs(delDiff)<0.05?"the same":"<b>"+Math.abs(delDiff).toFixed(1)+"% "+(delDiff<0?"less":"more")+"</b>")+
  " hot water and triggered the overheat safety shield <b>"+shield+"</b> times instead of <b>"+B.shield+"</b>"+
  (B.shield>0?" (<b class='good'>"+shieldCut.toFixed(0)+"% fewer</b>)":"")+". "+
  "Hard safety limits were breached <b>"+safety+"</b> times.";

const SC=[
 ["Hot-water energy delivered", del.toFixed(0)+" kWh", metPct.toFixed(1)+"% of demand met · always-on: "+B.del.toFixed(0)+" kWh ("+bMet.toFixed(1)+"%)"],
 ["Overheat guard hits", String(shield), "always-on pump: "+B.shield+" · each hit = wasted heat + thermal stress"],
 ["Pump actually ran", pumpHrs.toFixed(0)+" h", "of "+N+" h · "+pumpSun.toFixed(0)+" h of the "+sunHrs+" sunny hours"],
 ["Controller stepped in", (cnt.hold+cnt.bypass)+" h", cnt.hold+" h held off · "+cnt.bypass+" h bypass ("+DATA.filter(d=>d.overridden).length+" forced by shield) — only needed at hot midday"],
 ["Hard-limit breaches", String(safety), "75 °C water / 65 °C PCM limits — target is 0"],
];
const scEl=document.getElementById("score");
SC.forEach(([l,b,c])=>{const e=document.createElement("div");e.className="sc";
  e.innerHTML='<div class="lab">'+l+'</div><div class="big">'+b+'</div><div class="ctx">'+c+'</div>';scEl.appendChild(e);});

// ---------- temperature chart
const xs=DATA.map(d=>d.k), tw=DATA.map(d=>d.tw), tp=DATA.map(d=>d.tpcm), ghi=DATA.map(d=>d.ghi);
const hl=(y,col,dash)=>({type:"line",x0:0,x1:N-1,y0:y,y1:y,xref:"x",yref:"y",line:{color:col,width:1,dash:dash}});
const ann=(y,t,col)=>({x:N-1,y:y,xref:"x",yref:"y",text:t,showarrow:false,xanchor:"right",yanchor:"bottom",font:{size:10,color:col}});
const nowLine=()=>({type:"line",x0:0,x1:0,y0:0,y1:1,xref:"x",yref:"paper",line:{color:INK,width:2}});
const dayTicks=[], dayText=[]; let last=-1;
DATA.forEach(d=>{ if(d.day!==last){dayTicks.push(d.k);dayText.push("day "+d.day);last=d.day;} });
Plotly.newPlot("tempChart",[
  {x:xs,y:tw,name:"Tank water",mode:"lines",line:{color:"#2a78d6",width:2}},
  {x:xs,y:tp,name:"PCM",mode:"lines",line:{color:"#eb6834",width:2}}],
 {margin:{l:44,r:10,t:62,b:30},paper_bgcolor:PAPER,plot_bgcolor:PAPER,
  title:{text:"Tank water and PCM temperature (°C) — stay under the guard lines",font:{size:13,color:INK},x:0},
  font:{color:MUT,size:11},legend:{orientation:"h",y:1.0,yanchor:"bottom",x:0},
  xaxis:{showgrid:false,color:MUT,tickvals:dayTicks,ticktext:dayText},
  yaxis:{gridcolor:GRID,color:MUT,zeroline:false},
  shapes:[hl(45,"#1baf7a","dot"),hl(72,"#eb6834","dash"),hl(75,"#b01d0f","solid"),nowLine()],
  annotations:[ann(45,"45° delivery target",MUT),ann(72,"72° water guard","#eb6834"),ann(75,"75° water limit","#b01d0f")]},
 {displayModeBar:false,responsive:true});
const T_NOW=3;

// ---------- hours-per-day stacked bars
const days=[...new Set(DATA.map(d=>d.day))];
const perDay=(m)=>days.map(dy=>DATA.filter(d=>d.day===dy&&mode(d)===m).length);
Plotly.newPlot("dayChart",["collect","hold","bypass","wait"].map(m=>({
    type:"bar",name:{collect:"Collecting",hold:"Held off",bypass:"Bypass",wait:"Standby"}[m],
    x:days.map(x=>"day "+x),y:perDay(m),marker:{color:COL[m]}})),
 {barmode:"stack",margin:{l:40,r:10,t:62,b:30},paper_bgcolor:PAPER,plot_bgcolor:PAPER,
  title:{text:"What the pump did each day (hours)",font:{size:13,color:INK},x:0},
  font:{color:MUT,size:11},legend:{orientation:"h",y:1.0,yanchor:"bottom",x:0},
  xaxis:{color:MUT},yaxis:{gridcolor:GRID,color:MUT,zeroline:false}},
 {displayModeBar:false,responsive:true});

Plotly.newPlot("ghiChart",[{x:xs,y:ghi,mode:"lines",fill:"tozeroy",line:{color:"#e0a21a",width:1.5},fillcolor:"rgba(224,162,26,.18)"}],
 {margin:{l:44,r:10,t:30,b:30},paper_bgcolor:PAPER,plot_bgcolor:PAPER,
  title:{text:"Sunshine available (GHI, W/m²)",font:{size:13,color:INK},x:0},font:{color:MUT,size:11},showlegend:false,
  xaxis:{showgrid:false,color:MUT,tickvals:dayTicks,ticktext:dayText},
  yaxis:{gridcolor:GRID,color:MUT,zeroline:false,rangemode:"tozero"},shapes:[nowLine()]},
 {displayModeBar:false,responsive:true});

// ---------- ribbon
const ribbon=document.getElementById("ribbon");
DATA.forEach(d=>{const c=document.createElement("div");c.className="cell";c.style.background=COL[mode(d)];
  if(d.overridden)c.style.backgroundImage="repeating-linear-gradient(45deg,rgba(255,255,255,.55) 0 3px,transparent 3px 6px)";
  c.title=(d.ts||("day "+d.day))+" — "+LBL[mode(d)];ribbon.appendChild(c);});

// ---------- now tiles
const TILES=[["ghi","Sun","W/m²"],["tamb","Outside","°C"],["tw","Tank water","°C"],["tpcm","PCM","°C"],
  ["melt","PCM melted","%"],["demand_L","Hot water drawn","L"],["cum_del","Delivered so far","kWh"],["cum_shield","Guard hits so far",""]];
const tilesEl=document.getElementById("tiles");
TILES.forEach(([k,l,u])=>{const t=document.createElement("div");t.className="tile";
  t.innerHTML='<div class="lab">'+l+'</div><div class="val"><span id="t_'+k+'">—</span>'+(u?' <span class="u">'+u+'</span>':'')+'</div>';
  tilesEl.appendChild(t);});

// ---------- playback
let k=0,timer=null,stepMs=120;
const lamp=document.getElementById("lamp"),why=document.getElementById("why"),clock=document.getElementById("clock"),
      scrub=document.getElementById("scrub"),cursor=document.getElementById("cursor");
scrub.max=N-1;
function render(i){
  const d=DATA[i],m=mode(d);
  lamp.textContent=LBL[m];lamp.style.background=COL[m];
  lamp.style.color = m==="wait" ? "#1a1a1a":"#fff";
  why.textContent=reason(d);
  clock.textContent=d.ts||("day "+d.day+" "+String(d.hod).padStart(2,"0")+":00");
  scrub.value=i;
  for(const [key] of TILES)document.getElementById("t_"+key).textContent=d[key];
  Plotly.relayout("tempChart",{["shapes["+T_NOW+"].x0"]:i,["shapes["+T_NOW+"].x1"]:i});
  Plotly.relayout("ghiChart",{"shapes[0].x0":i,"shapes[0].x1":i});
  cursor.style.left=((i+0.5)/N*ribbon.clientWidth)+"px";
}
function play(){if(timer)return;document.getElementById("play").textContent="❚❚ Pause";
  timer=setInterval(()=>{k++;if(k>=N){k=N-1;stop();return;}render(k);},stepMs);}
function stop(){clearInterval(timer);timer=null;document.getElementById("play").textContent="▶ Play";}
document.getElementById("play").onclick=()=>{timer?stop():play();};
document.getElementById("reset").onclick=()=>{stop();k=0;render(0);};
document.getElementById("speed").onchange=(e)=>{stepMs=+e.target.value;if(timer){stop();play();}};
scrub.oninput=(e)=>{stop();k=+e.target.value;render(k);};
ribbon.onclick=(e)=>{stop();const r=ribbon.getBoundingClientRect();
  k=Math.min(N-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*N)));render(k);};
render(0);
</script>
</div></body></html>
"""


def _baseline_kpis(base: list[dict]) -> dict:
    """KPIs of the always-charge (B0) run on the identical window."""
    last = base[-1]
    return {"del": last["cum_del"], "unmet": last["cum_unmet"],
            "shield": last["cum_shield"], "safety": last["cum_safety"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="sac",
                    choices=list(_ALGO_DIR.keys()))
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--start-day", type=int, default=160,
                    help="day-of-year the display window starts (warm-started)")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--live", action="store_true",
                    help="drive the policy with present-day Open-Meteo weather")
    ap.add_argument("--past-days", type=int, default=2)
    ap.add_argument("--forecast-days", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.year in TEST_YEARS:
        print(f"[ERROR] {args.year} is a locked test year; demo refuses it.")
        sys.exit(1)

    import json as _json
    contract = _json.load(open(CONTRACT_PATH))
    design = {d["regime_id"]: d for d in contract["static_design_per_regime"]}
    pcm = design.get(args.regime, {}).get("selected_pcm", "PCM")
    name = {"ppo":"PPO","dqn":"DQN","recurrent":"RecurrentPPO","qrdqn":"QRDQN",
            "maskable":"MaskablePPO","sac":"SAC"}[args.algo]

    if args.live:
        lat, lon, label = REGIME_COORDS.get(args.regime, REGIME_COORDS[0])
        print(f"[live-dash] LIVE — fetching Open-Meteo for {label} "
              f"({lat},{lon}) past={args.past_days}d forecast={args.forecast_days}d ...")
        try:
            wx_df, iso_times, fetched = fetch_open_meteo(
                lat, lon, args.past_days, args.forecast_days)
        except Exception as e:
            print(f"[ERROR] live weather fetch failed ({e}). "
                  f"Check network; falling back to historical replay.")
            args.live = False
        else:
            print(f"[live-dash] fetched {len(wx_df)} hours — running policy "
                  f"through real physics ...")
            recs = run_rollout_live(args.algo, args.regime, wx_df, iso_times)
            base = run_rollout_live(args.algo, args.regime, wx_df, iso_times,
                                    force_charge=True)
            meta = {"algo": args.algo, "algo_name": name, "regime": args.regime,
                    "mode": "live", "location": label, "fetched": fetched,
                    "year": "live", "days": round(len(recs)/24, 1), "pcm": pcm,
                    "n": len(recs), "base": _baseline_kpis(base)}

    if not args.live:
        start_hour = args.start_day * 24
        n_hours    = args.days * 24
        print(f"[live-dash] REPLAY — {args.algo} r{args.regime} {args.year} "
              f"window day {args.start_day}..{args.start_day+args.days} "
              f"({n_hours} h) — running policy through real physics ...")
        recs = run_rollout(args.algo, args.regime, args.year, start_hour, n_hours)
        base = run_rollout(args.algo, args.regime, args.year, start_hour, n_hours,
                           force_charge=True)
        meta = {"algo": args.algo, "algo_name": name, "regime": args.regime,
                "mode": "replay", "year": args.year, "days": args.days,
                "pcm": pcm, "n": len(recs), "base": _baseline_kpis(base)}

    if not recs:
        print("[ERROR] no steps logged.")
        sys.exit(1)

    # action mix summary (text, for the console)
    from collections import Counter
    mix = Counter(ACTION_SHORT[r["exe"]] for r in recs)
    overr = sum(1 for r in recs if r["overridden"])
    print(f"[live-dash] actions: {dict(mix)}  | shield-overrides: {overr}  "
          f"| delivered {recs[-1]['cum_del']} kWh | shield fires {recs[-1]['cum_shield']}")

    tag = "live" if meta.get("mode") == "live" else str(args.year)
    out = pathlib.Path(args.out) if args.out else \
        RESULTS_DIR / "plots" / "live_demo" / \
        f"live_dashboard_{args.algo}_r{args.regime}_{tag}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(recs, meta), encoding="utf-8")
    print(f"[live-dash] wrote -> {out}")
    print("[live-dash] open it in a browser and press Play.")


if __name__ == "__main__":
    main()
