"""
o3/stepper.py
=============
Hour-by-hour stepping of the UNCHANGED O2 multinode kernel.

Each step calls the compiled `_kernel` on a one-hour slice and carries the end-of-hour node water
temperatures `T_nodes` and node PCM enthalpies `h_nodes` into the next call -- the kernel's only state.
(For water-only nodes the kernel resets the PCM temperature to the water temperature, so nothing else
needs carrying.) Stepping 8760 times is therefore bit-identical to one full-year call (test T1).

Actions (plan section 3.3, Discrete(2)):
    0 = BYPASS  -- collector loop off for the hour (equivalent to GHI = 0 in that hour: circ = a = b = 0)
    1 = ENABLE  -- leave the real collector coefficients; the kernel's own logic (irradiance cutoff,
                   a - b*T_bottom >= 0 re-solve, water shield at top node 72 C, PCM shield at 62 C)
                   still decides whether the loop actually runs. The policy cannot override the shield.
"""

from __future__ import annotations

import numpy as np

from o3 import o2_bridge as ob

BYPASS, ENABLE = 0, 1
J_PER_KWH = 3.6e6

# column index of the kernel's hourly record (tank_model_multinode._kernel, `hr`)
HR = {"T_top": 0, "T_mean": 1, "T_pcm": 2, "f_melt": 3, "Q_coll_Wh": 4, "Q_load_Wh": 5, "Q_pcm_Wh": 6,
      "Q_loss_Wh": 7, "Q_unmet_Wh": 8, "draw_kg": 9, "vol_ge_target_kg": 10, "Edel_Wh": 11, "T_top_min": 12,
      "T_top_max": 13, "T_pcm_max": 14, "f_min": 15, "f_max": 16, "E_stored_kWh": 17, "E_rec_kWh": 18,
      "T_bottom": 19, "Q_load_net_Wh": 20}


class HourlyStepper:
    """Holds one episode's weather arrays and the carried kernel state."""

    def __init__(self, case: ob.KernelCase, weather):
        self.case = case
        self.weather = weather
        self.n_hours = len(weather)
        self.tamb = weather["T_amb_C"].to_numpy(dtype=float)
        self.ghi = weather["GHI_Wm2"].to_numpy(dtype=float)
        self.hod = ob.hour_of_day_array(weather)
        self.a, self.b, self.circ = ob.collector_arrays(weather, case.system_config)
        self._k = ob.kernel()
        self._zero1 = np.zeros(1)
        self.reset()

    def reset(self, T0=None, h0=None):
        self.t = 0
        self.T = (self.case.T0 if T0 is None else np.asarray(T0, dtype=float)).copy()
        self.h = (self.case.h0 if h0 is None else np.asarray(h0, dtype=float)).copy()
        self.residual_J = 0.0
        self.collector_J = 0.0

    @property
    def done(self) -> bool:
        return self.t >= self.n_hours

    def step(self, action: int, demand24: np.ndarray | None = None) -> dict:
        """Advance one hour. Returns the kernel's hourly record plus shield / limit seconds."""
        if self.done:
            raise RuntimeError("episode finished")
        t = self.t
        on = 1.0 if int(action) == ENABLE else 0.0
        sl = slice(t, t + 1)
        a = self.a[sl] * on
        b = self.b[sl] * on
        circ = self.circ[sl] * on
        d24 = self.case.demand24 if demand24 is None else demand24
        hr, node_T, node_h, sc = self._k(self._zero1, self.tamb[sl], self.hod[sl], a, b, circ, d24,
                                         self.case.m_w, self.case.n_caps, self.case.loss_ua, self.case.P,
                                         self.T, self.h, 1.0)
        self.T = node_T[0].copy()
        self.h = node_h[0].copy()
        # energy closure over the hour: collector + E_initial = net load + loss + E_final (telescopes over steps)
        self.residual_J += (sc[0] + sc[7]) - (sc[2] + sc[3] + sc[8])
        self.collector_J += sc[0]
        self.t += 1
        row = hr[0]
        out = {k: float(row[i]) for k, i in HR.items()}
        out.update({
            "hour_index": t, "hour_of_day": int(self.hod[t]), "GHI_Wm2": float(self.ghi[t]),
            "T_amb_C": float(self.tamb[t]), "action": int(action),
            "loop_available": bool(self.circ[t] > 0.5),          # irradiance above the collector cutoff
            "shield_water_s": float(sc[26]), "shield_pcm_s": float(sc[27]), "over_limit_s": float(sc[28]),
            "n_shield_water": int(sc[13]), "n_shield_pcm": int(sc[14]),
            "max_water_C": float(sc[9]), "max_pcm_C": float(sc[10]),
            "collector_J": float(sc[0]),
        })
        return out

    def residual_pct(self) -> float:
        return abs(self.residual_J) / max(abs(self.collector_J), 1.0) * 100.0
