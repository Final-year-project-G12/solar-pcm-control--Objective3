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

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
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
                control_step_min: int = 60) -> list[dict]:
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
    model = _load_model(algo, regime_id, model_tag=model_tag)

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
        if algo == "recurrent":
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
                     control_step_min: int = 60) -> list[dict]:
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
    model = _load_model(algo, regime_id, model_tag=model_tag)

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
        if algo == "recurrent":
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
<title>O3 Live Controller Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
  :root{--bg:#f6f7f5;--surface:#ffffff;--ink:#0b0b0b;--muted:#5c5b57;
        --line:rgba(11,11,11,.12);--charge:#2a78d6;--idle:#1baf7a;--bypass:#eb6834;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:16px;}
  h1{font-size:20px;margin:0 0 2px}
  .sub{color:var(--muted);font-size:13px;margin:0 0 10px}
  .sil{display:inline-block;background:#fff3e0;color:#8a4b00;border:1px solid #f0c38a;
       border-radius:6px;padding:3px 9px;font-size:12px;font-weight:600;margin-bottom:12px}
  .wrap{max-width:1040px;margin:0 auto}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
        padding:14px;margin:12px 0;}
  .controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  button{font:inherit;font-weight:600;border:none;border-radius:8px;padding:9px 18px;
         background:var(--charge);color:#fff;cursor:pointer}
  button.sec{background:#e9e9e6;color:var(--ink)}
  input[type=range]{flex:1;min-width:180px;accent-color:var(--charge)}
  .clock{font-variant-numeric:tabular-nums;font-weight:600;font-size:14px;color:var(--muted)}
  .badge{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  .lamp{font-size:22px;font-weight:800;letter-spacing:.3px;color:#fff;border-radius:10px;
        padding:12px 22px;min-width:230px;text-align:center;transition:background .15s}
  .flag{font-size:12px;font-weight:700;color:#8a4b00;background:#fff3e0;
        border:1px solid #f0c38a;border-radius:6px;padding:6px 10px;display:none}
  .flag.on{display:inline-block}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:10px}
  .tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
  .tile .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .tile .val{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:3px}
  .tile .u{font-size:12px;color:var(--muted);font-weight:500}
  .ribbonwrap{margin-top:6px}
  .ribbon{display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid var(--line);cursor:pointer}
  .cell{flex:1 0 0;position:relative}
  .cursor{height:34px;width:2px;background:#0b0b0b;position:relative;margin-top:-30px;
          transition:left .12s linear;pointer-events:none}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-top:8px;flex-wrap:wrap}
  .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
  @media (prefers-color-scheme:dark){:root:not([data-theme=light]){
     --bg:#15161a;--surface:#1e2026;--ink:#f2f2ef;--muted:#a3a39d;--line:rgba(242,242,239,.14);}}
</style></head>
<body><div class="wrap">
  <h1>Objective 3 — Live Controller Dashboard</h1>
  <p class="sub" id="subtitle"></p>
  <span class="sil">SOFTWARE-IN-THE-LOOP — thermal state from the frozen simulator, not field sensors</span>

  <div class="card">
    <div class="controls">
      <button id="play">▶ Play</button>
      <button id="reset" class="sec">⟲ Reset</button>
      <label class="clock">speed
        <select id="speed">
          <option value="240">0.5×</option>
          <option value="120" selected>1×</option>
          <option value="60">2×</option>
          <option value="24">5×</option>
        </select>
      </label>
      <input type="range" id="scrub" min="0" value="0">
      <span class="clock" id="clock">—</span>
    </div>
    <div class="badge" style="margin-top:14px">
      <div class="lamp" id="lamp">—</div>
      <div class="flag" id="flag">⚠ SHIELD OVERRODE POLICY</div>
      <div class="clock" id="intent"></div>
    </div>
  </div>

  <div class="card tiles" id="tiles"></div>

  <div class="card"><div id="tempChart" style="height:300px"></div>
    <div class="ribbonwrap">
      <div class="ribbon" id="ribbon"></div>
      <div class="cursor" id="cursor"></div>
      <div class="legend">
        <span><span class="sw" style="background:var(--charge)"></span>CHARGE — loop on, heating PCM</span>
        <span><span class="sw" style="background:var(--idle)"></span>IDLE/SERVE — pump off, serving demand</span>
        <span><span class="sw" style="background:var(--bypass)"></span>BYPASS — collector isolated</span>
        <span>▦ hatched = shield overrode the policy</span>
      </div>
    </div>
  </div>

  <div class="card"><div id="ghiChart" style="height:210px"></div></div>

<script>
const DATA = __DATA__;
const META = __META__;
const AC = {0:"#2a78d6",1:"#1baf7a",2:"#eb6834"};
const ANAME = {0:"CHARGING",1:"SERVING (idle)",2:"BYPASS"};
const ASHORT = {0:"CHARGE",1:"IDLE",2:"BYPASS"};
const N = DATA.length;
const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme:dark)").matches;
const INK = isDark? "#f2f2ef":"#0b0b0b", MUT = isDark? "#a3a39d":"#5c5b57",
      GRID = isDark? "rgba(242,242,239,.10)":"rgba(11,11,11,.08)",
      PAPER = "rgba(0,0,0,0)";

if(META.mode==="live"){
  document.getElementById("subtitle").innerHTML =
    "<b>LIVE</b> — "+META.algo_name+" policy · regime "+META.regime+" · "+
    META.location+" · present weather from Open-Meteo (fetched "+META.fetched+
    ") · "+N+" hours · "+META.pcm;
}else{
  document.getElementById("subtitle").textContent =
    "REPLAY — "+META.algo_name+" policy · regime "+META.regime+" · historical year "+
    META.year+" · "+N+" hours ("+META.days+" days) · "+META.pcm;
}

// ---- time label (prefers real timestamp; falls back to day/hour)
function clockLabel(d){
  if(d.ts) return d.ts;
  return "day "+d.day+"  "+String(d.hod).padStart(2,"0")+":00"; }

const xs = DATA.map(d=>d.k);
const tw = DATA.map(d=>d.tw), tp = DATA.map(d=>d.tpcm), ghi = DATA.map(d=>d.ghi);

// ---- temperature chart with guard/limit reference lines
const hl = (y,txt,col,dash)=>({type:"line",x0:0,x1:N-1,y0:y,y1:y,xref:"x",yref:"y",
  line:{color:col,width:1,dash:dash}});
const ann = (y,txt,col)=>({x:N-1,y:y,xref:"x",yref:"y",text:txt,showarrow:false,
  xanchor:"right",yanchor:"bottom",font:{size:10,color:col}});
let nowLine = {type:"line",x0:0,x1:0,y0:0,y1:1,xref:"x",yref:"paper",
  line:{color:INK,width:2}};

Plotly.newPlot("tempChart",[
  {x:xs,y:tw,name:"water T",mode:"lines",line:{color:"#2a78d6",width:2}},
  {x:xs,y:tp,name:"PCM T",mode:"lines",line:{color:"#eb6834",width:2}},
],{
  margin:{l:48,r:12,t:28,b:30},paper_bgcolor:PAPER,plot_bgcolor:PAPER,
  title:{text:"Tank water & PCM temperature (°C)",font:{size:13,color:INK},x:0},
  font:{color:MUT,size:11},showlegend:true,
  legend:{orientation:"h",y:1.14,x:0,font:{color:MUT}},
  xaxis:{showgrid:false,color:MUT,title:{text:"hour in window",font:{size:10}}},
  yaxis:{gridcolor:GRID,color:MUT,zeroline:false},
  shapes:[hl(45,"","#1baf7a","dot"),hl(72,"","#eb6834","dash"),hl(75,"","#b01d0f","solid"),
          hl(62,"","#eb6834","dash"),hl(65,"","#b01d0f","solid"),nowLine],
  annotations:[ann(45,"deliver 45°",MUT),ann(75,"water hard 75°","#b01d0f"),
               ann(72,"water shield 72°","#eb6834"),ann(65,"PCM hard 65°","#b01d0f"),
               ann(62,"PCM shield 62°","#eb6834")],
},{displayModeBar:false,responsive:true});

Plotly.newPlot("ghiChart",[
  {x:xs,y:ghi,name:"GHI",mode:"lines",fill:"tozeroy",
   line:{color:"#e0a21a",width:1.5},fillcolor:"rgba(224,162,26,.18)"},
],{
  margin:{l:48,r:12,t:28,b:30},paper_bgcolor:PAPER,plot_bgcolor:PAPER,
  title:{text:"Solar irradiance GHI (W/m²)",font:{size:13,color:INK},x:0},
  font:{color:MUT,size:11},showlegend:false,
  xaxis:{showgrid:false,color:MUT,title:{text:"hour in window",font:{size:10}}},
  yaxis:{gridcolor:GRID,color:MUT,zeroline:false,rangemode:"tozero"},
  shapes:[Object.assign({},nowLine)],
},{displayModeBar:false,responsive:true});

// ---- action ribbon (one cell per hour; hatch if overridden)
const ribbon = document.getElementById("ribbon");
DATA.forEach(d=>{
  const c=document.createElement("div"); c.className="cell";
  c.style.background=AC[d.exe];
  if(d.overridden){
    c.style.backgroundImage="repeating-linear-gradient(45deg,"+
      "rgba(255,255,255,.55) 0 3px,transparent 3px 6px)";
  }
  c.title="day "+d.day+" "+String(d.hod).padStart(2,"0")+":00 — "+ANAME[d.exe];
  ribbon.appendChild(c);
});

// ---- KPI tiles
const TILES=[
  ["ghi","GHI","W/m²"],["tamb","Ambient","°C"],["tw","Water","°C"],
  ["tpcm","PCM","°C"],["melt","Melt","%"],["demand_L","Draw now","L"],
  ["cum_del","Delivered","kWh"],["cum_shield","Shield fires",""],
];
const tilesEl=document.getElementById("tiles");
TILES.forEach(([k,lab,u])=>{
  const t=document.createElement("div");t.className="tile";
  t.innerHTML='<div class="lab">'+lab+'</div><div class="val">'+
              '<span class="num" id="t_'+k+'">—</span>'+
              (u?' <span class="u">'+u+'</span>':'')+'</div>';
  tilesEl.appendChild(t);
});

// ---- playback state
let k=0, timer=null, stepMs=120;
const lamp=document.getElementById("lamp"), flag=document.getElementById("flag"),
      intent=document.getElementById("intent"), clock=document.getElementById("clock"),
      scrub=document.getElementById("scrub"), cursor=document.getElementById("cursor"),
      ribbonBox=ribbon.getBoundingClientRect;
scrub.max=N-1;

function render(i){
  const d=DATA[i];
  lamp.textContent=ANAME[d.exe]; lamp.style.background=AC[d.exe];
  flag.classList.toggle("on", d.overridden);
  intent.textContent = d.overridden ? ("policy asked: "+ASHORT[d.req]) : "";
  clock.textContent=clockLabel(d);
  scrub.value=i;
  for(const [key] of TILES){
    document.getElementById("t_"+key).textContent = d[key];
  }
  // move now-line on both charts
  const nl={ "shapes[5].x0":i,"shapes[5].x1":i };
  Plotly.relayout("tempChart",nl);
  Plotly.relayout("ghiChart",{ "shapes[0].x0":i,"shapes[0].x1":i });
  // move ribbon cursor
  const frac=(i+0.5)/N;
  cursor.style.left=(frac*ribbon.clientWidth)+"px";
}

function play(){ if(timer) return;
  document.getElementById("play").textContent="❚❚ Pause";
  timer=setInterval(()=>{ k++; if(k>=N){k=N-1;stop();return;} render(k); }, stepMs); }
function stop(){ clearInterval(timer); timer=null;
  document.getElementById("play").textContent="▶ Play"; }

document.getElementById("play").onclick=()=>{ timer?stop():play(); };
document.getElementById("reset").onclick=()=>{ stop(); k=0; render(0); };
document.getElementById("speed").onchange=(e)=>{ stepMs=+e.target.value;
  if(timer){stop();play();} };
scrub.oninput=(e)=>{ stop(); k=+e.target.value; render(k); };
ribbon.onclick=(e)=>{ stop(); const r=ribbon.getBoundingClientRect();
  k=Math.min(N-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*N))); render(k); };

render(0);
</script>
</div></body></html>
"""


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
    pcm = design.get(args.regime, {}).get("pcm", "PCM")
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
            meta = {"algo": args.algo, "algo_name": name, "regime": args.regime,
                    "mode": "live", "location": label, "fetched": fetched,
                    "year": "live", "days": round(len(recs)/24, 1), "pcm": pcm,
                    "n": len(recs)}

    if not args.live:
        start_hour = args.start_day * 24
        n_hours    = args.days * 24
        print(f"[live-dash] REPLAY — {args.algo} r{args.regime} {args.year} "
              f"window day {args.start_day}..{args.start_day+args.days} "
              f"({n_hours} h) — running policy through real physics ...")
        recs = run_rollout(args.algo, args.regime, args.year, start_hour, n_hours)
        meta = {"algo": args.algo, "algo_name": name, "regime": args.regime,
                "mode": "replay", "year": args.year, "days": args.days,
                "pcm": pcm, "n": len(recs)}

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
