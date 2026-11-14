"""
o3/env_ext.py
=============
Post-lock RESEARCH EXTENSIONS of the Uttarakhand environment (Tamil Nadu plan section 19.1 items 2 and 5). The frozen
prototype environment (o3/env.py) is not changed; with obs_version="V1" and domain_randomize=False this class is
identical to UKPCMEnv (tests/test_env_ext.py checks it bit for bit).

  obs_version "V2"   appends 4 fields to the 15 V1 fields: GHI 1 h, 2 h and 3 h ahead (perfect forecast from the
                     episode's own weather, 0 past the end of the year) and next-hour hot-water draw (kg). They are
                     normalised with the frozen statistics of the matching V1 field (GHI_Wm2 / draw_mass_kg_now), so
                     the V1 part of the observation is unchanged.
  domain_randomize   per episode, held fixed for the episode, applied to the PHYSICS inputs (not only to what the
                     policy sees): GHI x U[0.90, 1.10], ambient temperature + U[-2, 2] C, hot-water demand x
                     U[0.85, 1.15] (Tamil Nadu 19.1 item 5 ranges, O3 choice for Uttarakhand). Training only;
                     evaluation always runs nominal.
  forecast_mode      (V2 only) what the three GHI look-ahead fields contain:
                       "perfect"  the episode's own future GHI (default; bit-identical to the original V2)
                       "smart"    smart persistence on the clear-sky index, a forecast a real controller can make:
                                  GHI_hat(t+h) = Kc(t) x GHI_cs(t+h), Kc(t) = GHI(t) / GHI_cs(t) clipped to [0, 1.2],
                                  Kc held at its last daytime value while GHI_cs(t) <= 20 W/m2 (1.0 before the first
                                  daytime hour of the episode). GHI_cs = the weather file's GHI_clearsky_Wm2 column
                                  (NASA POWER, same radiation scheme as GHI). Uses no hour after t except GHI_cs, which
                                  is solar geometry and known in advance (Inman et al. 2013).
                       "none"     zeros (the forecast-removal ablation; same observation length)
                     The next-hour draw field stays the scheduled value in every mode (O2's fixed draw profile).
Sub-hourly control (19.1 item 1) is NOT offered: the unchanged O2 kernel consumes whole-hour weather rows, so a
15-min decision step would require editing O2's physics (forbidden by the plan, section 0 / 3.0).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from o3.env import OBS_FIELDS, UKPCMEnv
from o3.stepper import HourlyStepper

V2_EXTRA = ["GHI_ahead_1h_Wm2", "GHI_ahead_2h_Wm2", "GHI_ahead_3h_Wm2", "draw_next_hour_kg"]
DR_RANGES = {"ghi_mult": (0.90, 1.10), "tamb_bias_C": (-2.0, 2.0), "demand_mult": (0.85, 1.15)}
FORECAST_MODES = ("perfect", "smart", "none")
KC_CLIP = (0.0, 1.2)
CS_DAY_WM2 = 20.0


def smart_persistence_kc(ghi: np.ndarray, ghi_cs: np.ndarray) -> np.ndarray:
    """Causal clear-sky index per hour: Kc(t) uses GHI(t) and GHI_cs(t) only; held through night hours."""
    kc = np.empty(len(ghi), dtype=float)
    last = 1.0
    for t in range(len(ghi)):
        if ghi_cs[t] > CS_DAY_WM2:
            last = min(max(ghi[t] / ghi_cs[t], KC_CLIP[0]), KC_CLIP[1])
        kc[t] = last
    return kc


class UKPCMEnvExt(UKPCMEnv):
    def __init__(self, *args, obs_version: str = "V1", domain_randomize: bool = False,
                 forecast_mode: str = "perfect", **kw):
        if obs_version not in ("V1", "V2"):
            raise ValueError(obs_version)
        if forecast_mode not in FORECAST_MODES:
            raise ValueError(forecast_mode)
        self.obs_version, self.domain_randomize = obs_version, bool(domain_randomize)
        self.forecast_mode = forecast_mode
        self._kc = self._cs = None
        super().__init__(*args, **kw)
        if obs_version == "V2":
            n = len(OBS_FIELDS) + len(V2_EXTRA)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
            if self.normalise:
                gi, di = OBS_FIELDS.index("GHI_Wm2"), OBS_FIELDS.index("draw_mass_kg_now")
                self._mu = np.concatenate([self._mu, self._mu[[gi, gi, gi, di]]]).astype(np.float32)
                self._sd = np.concatenate([self._sd, self._sd[[gi, gi, gi, di]]]).astype(np.float32)
        self.dr = {"ghi_mult": 1.0, "tamb_bias_C": 0.0, "demand_mult": 1.0}

    def reset(self, *, seed=None, options=None):
        self._kc = self._cs = None              # never reuse the previous episode's forecast arrays
        obs, info = super().reset(seed=seed, options=options)
        if self.domain_randomize:
            self.dr = {k: float(self.np_random.uniform(*r)) for k, r in DR_RANGES.items()}
            w = self._weather(self.year).copy()
            w["GHI_Wm2"] = w["GHI_Wm2"] * self.dr["ghi_mult"]
            w["T_amb_C"] = w["T_amb_C"] + self.dr["tamb_bias_C"]
            self.stepper = HourlyStepper(self.case, w)          # collector coefficients rebuilt from the new weather
            self.demand.days = self.demand.days * self.dr["demand_mult"]
            obs = self._obs()
        if self.obs_version == "V2" and self.forecast_mode == "smart":
            self._cs = self._weather(self.year)["GHI_clearsky_Wm2"].to_numpy(dtype=float)
            if not np.all(np.isfinite(self._cs)):
                raise ValueError(f"GHI_clearsky_Wm2 has missing values (regime {self.regime}, year {self.year})")
            self._kc = smart_persistence_kc(self.stepper.ghi, self._cs)
            obs = self._obs()
        info["domain_randomisation"] = dict(self.dr)
        info["forecast_mode"] = self.forecast_mode
        return obs, info

    def forecast_ghi(self, t: int, h: int) -> float:
        """GHI forecast for hour t+h made at decision hour t (0 past the end of the year)."""
        st = self.stepper
        if t + h >= st.n_hours or self.forecast_mode == "none":
            return 0.0
        if self.forecast_mode == "perfect":
            return float(st.ghi[t + h])
        if self._kc is None:                    # only inside reset(), before the arrays exist; that obs is rebuilt
            return 0.0
        return float(self._kc[t] * self._cs[t + h])

    def _raw_obs(self):
        raw = super()._raw_obs()
        if self.obs_version == "V1":
            return raw
        st = self.stepper
        t = min(st.t, st.n_hours - 1)
        ahead = [self.forecast_ghi(t, h) for h in (1, 2, 3)]
        tn = t + 1
        draw_next = (float(self.demand.for_day(int(self._day[tn]))[int(st.hod[tn])]) if tn < st.n_hours else 0.0)
        return np.concatenate([raw, np.array(ahead + [draw_next], dtype=np.float32)])


def bin_action(a) -> int:
    """Continuous action in [-1, 1] -> 0 = bypass (a <= 0), 1 = enable (a > 0)."""
    return int(float(np.asarray(a).reshape(-1)[0]) > 0.0)


class ContinuousActionWrapper(gym.ActionWrapper):
    """1-D continuous action view for SAC (as in Tamil Nadu's PCMContinuousGymEnv), binned to the two actions."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def action(self, a):
        return bin_action(a)
