"""
o3/environment/forecast.py — the irradiance look-ahead the V3 observation carries (plan v2, section 12.2).

Three modes (PCMEnv(forecast_mode=...)):
  perfect  the stored year's own GHI for the next 1/2/3 hours. What every result before 2026-11-13 used; the default, so
           nothing existing changes. It is a perfect forecast and therefore optimistic.
  smart    smart persistence on the clear-sky index:   GHI_hat(t+h) = Kc(t) * GHI_clear(t+h),   Kc(t) = GHI(t) / GHI_clear(t).
           The standard "moderate" reference in solar forecasting: it keeps the deterministic sun path and assumes the
           current cloudiness persists. No training, no extra data, no assumed noise size.
  none     zeros (the look-ahead fields carry no information) - the ablation.

Causality: Kc at hour t uses GHI up to and including hour t only (the current-hour value the observation already shows),
never an hour after t. GHI_clear is solar geometry plus climatology, known in advance by construction.
"""
from __future__ import annotations

import numpy as np

FORECAST_MODES = ("perfect", "smart", "none")
CLEAR_MIN_WM2 = 20.0        # below this clear-sky value Kc is undefined (night / twilight): hold the last daytime value
KC_MAX = 1.2                # cloud-enhancement ceiling
KC_INIT = 1.0               # Kc before the first daytime hour of an episode (a clear day; affects only the first few hours)


def check_mode(mode: str) -> str:
    if mode not in FORECAST_MODES:
        raise ValueError(f"forecast_mode must be one of {list(FORECAST_MODES)}, got {mode!r}")
    return mode


def clear_sky_index(ghi: np.ndarray, clear: np.ndarray) -> np.ndarray:
    """Causal clear-sky index per hour: Kc[t] = clip(GHI[t] / GHI_clear[t], 0, KC_MAX) when GHI_clear[t] > CLEAR_MIN_WM2,
    otherwise the last daytime value (KC_INIT before the first one). Kc[t] depends only on hours <= t."""
    ghi = np.asarray(ghi, dtype=float)
    clear = np.asarray(clear, dtype=float)
    day = clear > CLEAR_MIN_WM2
    raw = np.clip(ghi / np.where(day, clear, 1.0), 0.0, KC_MAX)
    last = np.maximum.accumulate(np.where(day, np.arange(len(ghi)), -1))
    return np.where(last >= 0, raw[np.maximum(last, 0)], KC_INIT)


def smart_forecast(kc: np.ndarray, clear: np.ndarray, hour: int, k: int) -> float:
    """Forecast made at hour index `hour` for `hour + k` (0 past the end of the series)."""
    j = hour + k
    return float(kc[hour] * clear[j]) if j < len(clear) else 0.0


def analytic_clear_sky(lat: float, lon: float, times_utc) -> np.ndarray:
    """Closed-form clear-sky GHI (W/m2) from position and time (pvlib Ineichen). For live / embedded use, where no
    clear-sky dataset column exists. Measured on the validation years it is rRMSE 17.5 % off the stored NASA column
    (see plan v2 12.2.1), so the offline study uses the stored column; this is the fallback."""
    import pandas as pd
    from pvlib.location import Location
    idx = pd.DatetimeIndex(pd.to_datetime(times_utc, utc=True))
    return Location(float(lat), float(lon), tz="UTC").get_clearsky(idx, model="ineichen")["ghi"].to_numpy(dtype=float)
