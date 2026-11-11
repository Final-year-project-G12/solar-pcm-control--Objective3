"""
o3/live.py
==========
Software-in-the-loop demo back end (plan section 9 / 15.9, milestone M7). NOT field validation: the tank and PCM
state are simulated by the unchanged O2 kernel; only the weather is real (recorded or live).

One code path for every mode. A "session" is just an hourly weather frame with O2's columns
(timestamp_utc, GHI_Wm2, T_amb_C, local_hour, local_date). It is pushed through the SAME UKPCMEnv / HourlyStepper /
controllers used in training and evaluation (FrameEnv only replaces where the weather comes from), so:

  replay  weather = an O2 point-year file of an ALLOWED role (train / validate, via roles_o3.load_weather);
          the O3 test years 2006-2010 and O2's used years 2024-2025 are refused by roles_o3 itself
  live    weather = Open-Meteo hourly forecast API (free, no key; plan Q5 choice, same provider as the Tamil Nadu
          demo) at the regime's medoid grid point (plan Q6: the user picks one of the four medoids). Only hours up
          to "now" are used; the raw JSON is stored unmodified; missing / stale hours are filled by persistence of
          the last valid value and flagged. A saved live log re-runs exactly through `simulate`.

Caveats shown in the app: Open-Meteo `shortwave_radiation` is the mean over the PRECEDING hour on a ~11 km model
grid, not NASA POWER's product, so a live run is not comparable to training-weather statistics; T_pcm, f_melt and the
stored / recoverable energies are simulator internals (a real tank would need an estimator).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from o3 import config as o3c
from o3 import o2_bridge as ob
from o3 import roles_o3
from o3.controllers import B0AlwaysOn
from o3.env import UKPCMEnv
from o3.evaluate import METRIC_WEIGHTS, episode_metrics, run_episode

BANNER = "Software-in-the-loop simulation. Not field validation."
IST = timezone(timedelta(hours=5, minutes=30))
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
STALE_AFTER_H = 2            # O3 choice: newest observed hour older than this -> persistence fallback
DEMO_LOG_DIR = o3c.RESULTS / "demo_logs"

# medoid coordinates: O1 climate_signature_uttarakhand.csv (same source as scripts/m0_fetch_test_weather.py)
MEDOID_COORDS = {0: (29.375, 78.875), 1: (29.625, 79.375), 2: (30.375, 78.375), 3: (30.375, 79.375)}
REGIME_LABELS = {0: "r0 - warm, low elevation (UKP_0014)", 1: "r1 - mid-hills, largest population (UKP_0029)",
                 2: "r2 - higher hills, cloudiest warm regime (UKP_0025)",
                 3: "r3 - cold high-altitude, water-only (UKP_0041)"}
FRAME_COLS = ["timestamp_utc", "GHI_Wm2", "T_amb_C", "local_hour", "local_date"]


class FrameEnv(UKPCMEnv):
    """UKPCMEnv whose weather is a given hourly frame instead of a role-checked file (everything else identical)."""

    def __init__(self, frame: pd.DataFrame, **kw):
        self._frame = frame.reset_index(drop=True)
        super().__init__(years=[0], **kw)

    def _weather(self, year):
        return self._frame


# ---------------------------------------------------------------------------------------------------- weather
def replay_frame(regime: int, year: int) -> pd.DataFrame:
    """Recorded weather from an allowed role only (roles_o3 refuses test and O2-used years)."""
    role = roles_o3.role_of(year)
    if role not in ("train", "validate"):
        raise PermissionError(f"{year} is a '{role}' year: the demo may only replay train / validate years")
    return roles_o3.load_weather(regime, year, role)


def _to_frame(utc_index, ghi, tamb) -> pd.DataFrame:
    ts = pd.DatetimeIndex(utc_index).tz_convert("UTC")
    loc = ts.tz_convert(IST)
    return pd.DataFrame({"timestamp_utc": ts, "GHI_Wm2": np.maximum(np.asarray(ghi, float), 0.0),
                         "T_amb_C": np.asarray(tamb, float), "local_hour": loc.hour,
                         "local_date": loc.strftime("%Y-%m-%d")})


def fetch_live(regime: int, past_days: int = 3, now_utc: datetime | None = None, timeout_s: float = 10.0,
               session=None) -> tuple[pd.DataFrame, dict]:
    """Hourly GHI and air temperature for the regime's medoid, from `past_days` ago up to the last completed hour.

    Returns (frame with flag columns, fetch record). Never raises on a network failure: the record says what happened
    and the frame is empty, so the caller can fall back to persistence."""
    import requests
    lat, lon = MEDOID_COORDS[regime]
    params = {"latitude": lat, "longitude": lon, "hourly": "shortwave_radiation,temperature_2m",
              "past_days": int(past_days), "forecast_days": 1, "timezone": "GMT"}
    now_utc = now_utc or datetime.now(timezone.utc)
    rec = {"provider": "open-meteo", "url": OPEN_METEO_URL, "params": params, "request_utc": now_utc.isoformat(),
           "timezone_requested": "GMT", "ok": False}
    t0 = time.perf_counter()
    try:
        r = (session or requests).get(OPEN_METEO_URL, params=params, timeout=timeout_s)
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        rec["http_status"] = r.status_code
        r.raise_for_status()
        raw = r.json()
    except Exception as e:                                                         # noqa: BLE001
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        rec["error"] = repr(e)
        return pd.DataFrame(columns=FRAME_COLS + ["missing", "stale_fill"]), rec
    rec["raw_json"] = raw                                                          # stored unmodified
    h = raw.get("hourly", {})
    # Open-Meteo stamps the END of each averaging hour; O2/NASA POWER stamp the hour START -> shift back one hour
    ends = pd.to_datetime(h.get("time", []), utc=True)
    starts = ends - pd.Timedelta(hours=1)
    keep = ends <= pd.Timestamp(now_utc).floor("h")                                # completed hours only
    ghi = pd.Series(h.get("shortwave_radiation", []), dtype=float)[keep].to_numpy()
    tam = pd.Series(h.get("temperature_2m", []), dtype=float)[keep].to_numpy()
    fr = _to_frame(starts[keep], np.nan_to_num(ghi, nan=0.0), tam)
    fr["GHI_Wm2"] = ghi                                                            # keep NaN for the fill step
    fr["missing"] = np.isnan(ghi) | np.isnan(tam)
    fr = persistence_fill(fr)
    rec.update({"ok": True, "n_hours": int(len(fr)), "n_missing": int(fr["missing"].sum()),
                "last_hour_start_utc": fr["timestamp_utc"].iloc[-1].isoformat() if len(fr) else None,
                "utc_offset_seconds": raw.get("utc_offset_seconds"),
                "grid_lat": raw.get("latitude"), "grid_lon": raw.get("longitude"),
                "grid_elevation_m": raw.get("elevation")})
    return fr, rec


def persistence_fill(fr: pd.DataFrame) -> pd.DataFrame:
    """Missing GHI / T_amb -> last valid value (first rows with no history: GHI 0, T_amb = first valid)."""
    fr = fr.copy()
    if "stale_fill" not in fr:
        fr["stale_fill"] = False
    for c in ("GHI_Wm2", "T_amb_C"):
        fr[c] = fr[c].ffill()
    fr["GHI_Wm2"] = fr["GHI_Wm2"].fillna(0.0)
    fr["T_amb_C"] = fr["T_amb_C"].bfill().fillna(15.0)
    return fr


def extend_stale(fr: pd.DataFrame, now_utc: datetime | None = None) -> tuple[pd.DataFrame, int]:
    """If the newest observed hour is older than STALE_AFTER_H, append persistence hours up to the last completed
    hour (flagged stale_fill). Returns (frame, number of hours appended)."""
    now_h = pd.Timestamp(now_utc or datetime.now(timezone.utc)).floor("h")
    if fr.empty:
        return fr, 0
    last = fr["timestamp_utc"].iloc[-1]
    target = now_h - pd.Timedelta(hours=1)                                         # start of last completed hour
    gap = int((target - last) / pd.Timedelta(hours=1))
    if gap < STALE_AFTER_H:
        return fr, 0
    idx = pd.date_range(last + pd.Timedelta(hours=1), periods=gap, freq="h", tz="UTC")
    add = _to_frame(idx, np.full(gap, fr["GHI_Wm2"].iloc[-1]), np.full(gap, fr["T_amb_C"].iloc[-1]))
    add["missing"], add["stale_fill"] = True, True
    return pd.concat([fr, add], ignore_index=True), gap


def save_live_log(regime: int, frame: pd.DataFrame, fetches: list[dict]) -> Path:
    """Weather actually simulated + every fetch record (raw JSON included) -> results/demo_logs/."""
    DEMO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frame.to_csv(DEMO_LOG_DIR / f"live_r{regime}_{tag}_weather.csv", index=False)
    p = DEMO_LOG_DIR / f"live_r{regime}_{tag}_fetches.json"
    p.write_text(json.dumps(fetches, indent=1, default=str))
    return p


def load_weather_log(path_or_buf) -> pd.DataFrame:
    """Re-read a saved weather log (from save_live_log or the app's download) for an exact re-run."""
    fr = pd.read_csv(path_or_buf, parse_dates=["timestamp_utc"], float_precision="round_trip")   # bit-exact re-run
    fr["timestamp_utc"] = pd.to_datetime(fr["timestamp_utc"], utc=True)
    fr["local_date"] = fr["local_date"].astype(str)
    return fr


# ---------------------------------------------------------------------------------------------------- simulation
def controllers(regime: int) -> list:
    """B0, tuned B1 and (PCM regimes with a frozen M4 policy) the primary PPO seed."""
    from o3 import frozen
    out = [B0AlwaysOn()]
    try:
        out.append(frozen.b1(regime))
    except FileNotFoundError:
        pass
    if regime in o3c.load()["regimes"]["pcm"]:
        try:
            out.append(frozen.primary_policy(regime))
        except (FileNotFoundError, ob.ContractError):
            pass
    return out


def simulate(regime: int, frame: pd.DataFrame, ctrls=None, scenario: str = "fully_solid") -> dict:
    """Run every controller and the water-only twin through `frame` with the same reset state and O2's deterministic
    demand. Returns {name: hourly DataFrame} (the twin under 'water_only_twin')."""
    ctrls = ctrls if ctrls is not None else controllers(regime)
    out = {}
    for c in ctrls:
        env = FrameEnv(frame, regime=regime, purpose="validate", scenarios=[scenario], normalise=c.needs_norm,
                       weights=METRIC_WEIGHTS)
        df, _ = run_episode(c, regime, 0, scenario, "validate", env=env)
        out[c.name] = _stamp(df, frame)
    env = FrameEnv(frame, regime=regime, purpose="validate", scenarios=[scenario], normalise=False,
                   water_only=True, weights=METRIC_WEIGHTS)
    df, _ = run_episode(B0AlwaysOn(), regime, 0, scenario, "validate", env=env)
    out["water_only_twin"] = _stamp(df, frame)
    return out


def _stamp(df: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_utc"] = frame["timestamp_utc"].to_numpy()
    df["local_time"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_convert(IST).dt.tz_localize(None).to_numpy()
    for c in ("missing", "stale_fill"):
        if c in frame:
            df[c] = frame[c].to_numpy()
    return df


def metrics_table(runs: dict, target_C: float = 50.0) -> pd.DataFrame:
    rows = []
    twin = episode_metrics(runs["water_only_twin"], target_C)
    for name, df in runs.items():
        m = episode_metrics(df, target_C)
        m["g_window_kWh"] = m["E_del_window_kWh"] - twin["E_del_window_kWh"]
        rows.append({"controller": name, **m})
    cols = ["controller", "E_del_window_kWh", "g_window_kWh", "E_del_annual_kWh", "serve_frac_window", "overheat_h",
            "shield_water_h", "shield_pcm_h", "hours_over_limit", "guard_hours", "t_water_max_C", "t_pcm_max_C",
            "bypass_hours", "bypass_switches", "residual_pct"]
    return pd.DataFrame(rows)[cols]
