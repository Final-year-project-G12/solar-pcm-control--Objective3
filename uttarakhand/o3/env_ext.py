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
Sub-hourly control (19.1 item 1) is NOT offered: the unchanged O2 kernel consumes whole-hour weather rows, so a
15-min decision step would require editing O2's physics (forbidden by the plan, section 0 / 3.0).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from o3.env import OBS_FIELDS, UKPCMEnv
from o3.stepper import HourlyStepper

V2_EXTRA = ["GHI_ahead_1h_Wm2", "GHI_ahead_2h_Wm2", "GHI_ahead_3h_Wm2", "draw_next_hour_kg"]
DR_RANGES = {"ghi_mult": (0.90, 1.10), "tamb_bias_C": (-2.0, 2.0), "demand_mult": (0.85, 1.15)}


class UKPCMEnvExt(UKPCMEnv):
    def __init__(self, *args, obs_version: str = "V1", domain_randomize: bool = False, **kw):
        if obs_version not in ("V1", "V2"):
            raise ValueError(obs_version)
        self.obs_version, self.domain_randomize = obs_version, bool(domain_randomize)
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
        obs, info = super().reset(seed=seed, options=options)
        if self.domain_randomize:
            self.dr = {k: float(self.np_random.uniform(*r)) for k, r in DR_RANGES.items()}
            w = self._weather(self.year).copy()
            w["GHI_Wm2"] = w["GHI_Wm2"] * self.dr["ghi_mult"]
            w["T_amb_C"] = w["T_amb_C"] + self.dr["tamb_bias_C"]
            self.stepper = HourlyStepper(self.case, w)          # collector coefficients rebuilt from the new weather
            self.demand.days = self.demand.days * self.dr["demand_mult"]
            obs = self._obs()
        info["domain_randomisation"] = dict(self.dr)
        return obs, info

    def _raw_obs(self):
        raw = super()._raw_obs()
        if self.obs_version == "V1":
            return raw
        st = self.stepper
        t = min(st.t, st.n_hours - 1)
        ahead = [float(st.ghi[t + h]) if t + h < st.n_hours else 0.0 for h in (1, 2, 3)]
        tn = t + 1
        draw_next = (float(self.demand.for_day(int(self._day[tn]))[int(st.hod[tn])]) if tn < st.n_hours else 0.0)
        return np.concatenate([raw, np.array(ahead + [draw_next], dtype=np.float32)])
