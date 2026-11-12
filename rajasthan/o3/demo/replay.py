"""
o3/demo/replay.py — run policies through the SAME environment step and return an hourly log (used by the Streamlit app).

Replay and live share one code path: PCMEnv.step(). `run_replay` steps a stored allowed year from 1 January (the tank
state needs its history) and returns every hour; the app then shows the chosen date range. Test years are refused by
weather_roles unless the O3 lock is held, so the app can never show them.
"""

import numpy as np
import pandas as pd

import config_O2
from o3.environment.pcm_env import PCMEnv, WINDOW_HOURS
from o3.evaluation.rollout import make_baseline, ModelPolicy, ALGO_KIND, load_model
from o3.utils.models import find_model

ACTION_NAMES = {0: "CHARGE", 1: "IDLE", 2: "BYPASS"}
POLICY_LABELS = {"twin": "Water-only twin", "b0": "B0 always on", "b1": "B1 reactive rule", "b2": "B2 forecast rule"}


def available_models(regime_id: int) -> dict:
    """{label: path} of trained agents found under results/."""
    out = {}
    for algo in ("ppo", "maskable", "recurrent", "dqn", "qrdqn", "sac"):
        p = find_model(algo, regime_id)
        if p is not None:
            out[f"{algo.upper()} (trained)"] = (algo, str(p), "V2")
    out.update(_gen2_models(regime_id))
    return out


def _gen2_models(regime_id: int) -> dict:
    """Generation-2 runs (results/gen2/<tag>) at hourly control: {label: (algo, path, obs_version)}. The plan-18.1 selected run
    of each algorithm is listed as 'selected'; every other finished run is listed by its tag."""
    import json
    from o3.agents.gen2 import gen2_dir, run_model_path
    g = gen2_dir()
    out = {}
    if not g.exists():
        return out
    selected = {}
    for f in g.glob("gen2_selected_*.json"):
        try:
            selected.update({d["tag"]: k for k, d in json.loads(f.read_text()).items()})
        except Exception:
            pass
    for d in sorted(g.iterdir()):
        mp = d / "run_meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        if m["regime_id"] != regime_id or m["control_step_min"] != 60:
            continue
        label = f"{m['algo'].upper()} Gen-2 {'SELECTED ' if m['tag'] in selected else ''}[{m['tag']}]"
        out[label] = (m["algo"], str(run_model_path(d)), m["obs_version"])
    return out


def _policy(key: str, model: tuple | None):
    if key in ("twin", "b0", "b1", "b2"):
        return make_baseline("b0" if key == "twin" else key)
    algo, path = model[0], model[1]
    return ModelPolicy(load_model(algo, path), ALGO_KIND[algo])


def run_replay(regime_id: int, year: int, key: str, n_hours: int, model: tuple | None = None) -> pd.DataFrame:
    """Hourly log for the first n_hours of `year` (key: twin | b0 | b1 | b2 | 'model' with model=(algo, path))."""
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=regime_id, year=year, water_only=(key == "twin"),
                 obs_version=(model[2] if (model is not None and len(model) > 2) else "V2"))
    pol = _policy(key, model)
    obs, _ = env.reset(year=year)
    pol.reset()
    if hasattr(pol, "attach_env"):
        pol.attach_env(env)
    rows = []
    for i in range(min(n_hours, 8760)):
        a = pol.act(obs)
        obs, r, done, _, info = env.step(a)
        rows.append({"hour": i, "day": i // 24, "hour_of_day": info["hour_of_day"], "action": ACTION_NAMES[a],
                     "T_top_C": info["T_top_C"], "T_bottom_C": info["T_bottom_C"], "T_pcm_C": info["T_pcm_C"],
                     "f_melt": info["f_melt"], "draw_kg": info["draw_kg"], "E_del45_Wh": info["E_del45_Wh"],
                     "shield_water_s": info["shield_water_s"], "shield_pcm_s": info["shield_pcm_s"],
                     "over_limit_s": info["over_limit_s"], "shield_active": bool(info["shield_override"]),
                     "ghi_Wm2": float(env._ghi[i]), "T_amb_C": float(env._tamb[i])})
        if done:
            break
    df = pd.DataFrame(rows)
    df["cum_E_del45_kWh"] = df["E_del45_Wh"].cumsum() / 1000.0
    df["cum_over_limit_h"] = df["over_limit_s"].cumsum() / 3600.0
    df["in_window"] = df["hour_of_day"].isin(WINDOW_HOURS)
    return df
