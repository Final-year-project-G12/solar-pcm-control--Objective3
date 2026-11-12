"""
o3/environment/obs.py — observation layouts and a decoder shared by baselines, masks and the demo.

Two layouts (both normalised with FIXED, declared scale constants — no data-dependent statistics,
so nothing from any weather year leaks into the observation scaling):

  "V2" (default, 13 dims) — the 13 fields of the contract's dynamic_state_schema / plan 3.2:
      0 T_water_top      (T-30)/40        7  hour_of_day      /24
      1 T_water_bottom   (T-30)/40        8  draw_mass_kg_now /20
      2 T_water_mean     (T-30)/40        9  T_mains          (T-20)/20
      3 T_pcm            (T-30)/40       10  store_energy_above_mains_kWh /3
      4 f_melt           [0,1]           11  recoverable_energy_above_45C_kWh /1.5
      5 GHI              /1000           12  minutes_since_last_draw /600
      6 T_amb            (T-15)/25

  "V1" (11 dims) — the Tamil Nadu layout, kept so TN-style baselines/tests can be compared:
      0 GHI/1000, 1 (T_amb-15)/25, 2 (T_top-30)/40, 3 (T_pcm-30)/40, 4 f_melt, 5 draw/20,
      6 (T_mains-20)/20, 7 hour/24, 8 day-of-year proxy, 9 prev_action/2, 10 (T_max_water - T_top)/10

In both layouts 'T_water' means the TOP node (the delivery node, where the shield looks).
"""

import numpy as np

OBS_DIMS = {"V1": 11, "V2": 13, "V3": 17}
DEFAULT_VERSION = "V2"

V2_FIELDS = ["T_water_top_C", "T_water_bottom_C", "T_water_mean_C", "T_pcm_C", "f_melt", "GHI_Wm2", "T_amb_C",
             "hour_of_day", "draw_mass_kg_now", "T_mains_C", "store_energy_above_mains_kWh",
             "recoverable_energy_above_45C_kWh", "minutes_since_last_draw"]


def decode(obs: np.ndarray) -> dict:
    """Physical values from a normalised observation (layout inferred from its length)."""
    o = np.asarray(obs, dtype=float).reshape(-1)
    if o.shape[0] in (13, 17):          # V3 = V2 + [GHI +1h, +2h, +3h (/1000), next-hour draw (/20)]
        return {"T_water_C": o[0] * 40.0 + 30.0, "T_bottom_C": o[1] * 40.0 + 30.0, "T_mean_C": o[2] * 40.0 + 30.0,
                "T_pcm_C": o[3] * 40.0 + 30.0, "f_melt": o[4], "GHI_Wm2": o[5] * 1000.0, "T_amb_C": o[6] * 25.0 + 15.0,
                "hour": int(round(o[7] * 24.0)) % 24, "draw_kg": o[8] * 20.0, "T_mains_C": o[9] * 20.0 + 20.0,
                "E_stored_kWh": o[10] * 3.0, "E_rec45_kWh": o[11] * 1.5, "min_since_draw": o[12] * 600.0}
    if o.shape[0] == 11:
        return {"T_water_C": o[2] * 40.0 + 30.0, "T_bottom_C": float("nan"), "T_mean_C": float("nan"),
                "T_pcm_C": o[3] * 40.0 + 30.0, "f_melt": o[4], "GHI_Wm2": o[0] * 1000.0, "T_amb_C": o[1] * 25.0 + 15.0,
                "hour": int(round(o[7] * 24.0)) % 24, "draw_kg": o[5] * 20.0, "T_mains_C": o[6] * 20.0 + 20.0,
                "E_stored_kWh": float("nan"), "E_rec45_kWh": float("nan"), "min_since_draw": float("nan")}
    raise ValueError(f"unexpected observation length {o.shape[0]}")


def build(version: str, d: dict) -> np.ndarray:
    """Normalised observation from physical values. `d` holds the keys used below."""
    if version == "V3":
        return np.concatenate([build("V2", d), np.array([d["GHI_p1"] / 1000.0, d["GHI_p2"] / 1000.0, d["GHI_p3"] / 1000.0,
                                                         d["draw_p1"] / 20.0], dtype=np.float32)])
    if version == "V2":
        return np.array([
            (d["T_top"] - 30.0) / 40.0, (d["T_bottom"] - 30.0) / 40.0, (d["T_mean"] - 30.0) / 40.0,
            (d["T_pcm"] - 30.0) / 40.0, d["f_melt"], d["GHI"] / 1000.0, (d["T_amb"] - 15.0) / 25.0,
            d["hour"] / 24.0, d["draw"] / 20.0, (d["T_mains"] - 20.0) / 20.0,
            d["E_stored_kWh"] / 3.0, d["E_rec45_kWh"] / 1.5, d["min_since_draw"] / 600.0,
        ], dtype=np.float32)
    if version == "V1":
        return np.array([
            d["GHI"] / 1000.0, (d["T_amb"] - 15.0) / 25.0, (d["T_top"] - 30.0) / 40.0, (d["T_pcm"] - 30.0) / 40.0,
            d["f_melt"], d["draw"] / 20.0, (d["T_mains"] - 20.0) / 20.0, d["hour"] / 24.0,
            (d["hour_idx"] % 8760) / 8760.0, d["prev_action"] / 2.0, (d["max_water_C"] - d["T_top"]) / 10.0,
        ], dtype=np.float32)
    raise ValueError(f"obs_version must be one of {list(OBS_DIMS)}, got {version!r}")
