"""
o3/env.py
=========
Gymnasium environment around the unchanged O2 Uttarakhand multinode kernel (plan sections 3, 6, 15.3-15.4).

  action_space       Discrete(2): 0 = bypass collector loop, 1 = enable it (the kernel's shield still applies)
  observation_space  Box(15): the 13 contract fields (hour_of_day as sin/cos) + previous action,
                     standardised with train-year statistics (results/normalisation_r{k}.json)
  reward             r_t = w1*E_del50_t/E_norm - w2*overheat_h_t - w3*safety_violation_t - w4*bypass_switch_t
  episode            one simulated year (8760 h), terminated after the last hour, never truncated

The PCM fields (T_pcm, f_melt, stored and recoverable energy) are simulator internals: a real tank would only
have them through an estimator (contract: measurable=false). The current-hour GHI is the hour's mean, a mild
look-ahead; the live demo must use a nowcast.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from o3 import config as o3c
from o3 import o2_bridge as ob
from o3 import roles_o3
from o3.demand import EpisodeDemand
from o3.stepper import BYPASS, ENABLE, HourlyStepper

OBS_FIELDS = ["T_water_top_C", "T_water_bottom_C", "T_water_mean_C", "T_pcm_C", "f_melt", "GHI_Wm2", "T_amb_C",
              "hour_sin", "hour_cos", "draw_mass_kg_now", "T_mains_C", "store_energy_above_mains_kWh",
              "recoverable_energy_above_target_kWh", "minutes_since_last_draw", "prev_action"]
WINDOW_HOURS = set(range(18, 24)) | set(range(0, 9))      # O2 tes_metrics.WINDOW_HOURS, 18:00-09:00 local
GUARD_WATER_C, GUARD_PCM_C = 72.0, 62.0                    # contract shield guard band


def reward_weights(regime: int, w2: float, w4: float) -> dict:
    """w1 = 1; w3 by the dominance rule (larger than the largest possible one-hour normalised delivery)."""
    cfg = o3c.load()
    norm = o3c.load_normalisation(regime)
    return {"w1": float(cfg["reward"]["w1"]), "w2": float(w2), "w4": float(w4),
            "w3": float(cfg["reward"]["w3_dominance_factor"] * norm["max_hour_delivery_normalised"]),
            "E_norm_Wh": float(norm["E_norm_Wh"])}


def initial_record(case: ob.KernelCase) -> dict:
    """The observation fields at the reset state, with the same definitions as the kernel's end-of-hour record
    (mass-weighted PCM temperature and melt fraction; stored / recoverable energy above mains / target)."""
    P, T0, h0, m_w = case.P, case.T0, case.h0, case.m_w
    hp = (P[8], P[9], P[10], P[11], P[12], P[13])          # Ts, Tl, L, cp_s, cp_l, wall cp (kernel order)
    cp = P[0]
    tm = float((T0 * m_w).sum() / m_w.sum())
    e_st = float((m_w * cp * (T0 - case.mains_C)).sum())
    e_rec = float((m_w * cp * np.maximum(T0 - case.target_C, 0.0)).sum())
    tp, f = tm, 0.0
    has = case.n_caps > 0
    if has.any():
        m_node = case.n_caps * P[23]
        h_m, h_t = ob._tmm._h_of_T(case.mains_C, *hp), ob._tmm._h_of_T(case.target_C, *hp)
        tf = np.array([ob._tmm._T_f_of_h(h0[i], *hp) for i in np.where(has)[0]])
        wts = m_node[has] / m_node[has].sum()
        tp, f = float(wts @ tf[:, 0]), float(wts @ tf[:, 1])
        e_st += float((m_node[has] * (h0[has] - h_m)).sum())
        e_rec += float((m_node[has] * np.maximum(h0[has] - h_t, 0.0)).sum())
    return {"T_top": float(T0[-1]), "T_bottom": float(T0[0]), "T_mean": tm, "T_pcm": tp, "f_melt": f,
            "draw_kg": 0.0, "E_stored_kWh": e_st / 3.6e6, "E_rec_kWh": e_rec / 3.6e6}


class UKPCMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, regime: int = 1, purpose: str = "train", years=None, scenarios=None,
                 demand_mode: str = "deterministic", w2: float = 0.0, w4: float = 0.0,
                 water_only: bool = False, normalise: bool = True, weights: dict | None = None):
        super().__init__()
        ob.verify_contract()
        self.cfg = o3c.load()
        self.regime = int(regime)
        self.purpose = purpose
        self.years = list(years) if years is not None else roles_o3.years_for(purpose)
        self.scenarios = list(scenarios or self.cfg["reset_scenarios"])
        self.demand_mode = demand_mode
        self.water_only = water_only
        self.weights = weights or reward_weights(self.regime, w2, w4)
        self.normalise = normalise
        if normalise:
            n = o3c.load_normalisation(self.regime)
            self._mu = np.asarray(n["obs_mean"], dtype=np.float32)
            self._sd = np.asarray(n["obs_std"], dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_FIELDS),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._weather_cache: dict = {}
        self._case_cache: dict = {}
        self.stepper: HourlyStepper | None = None

    # ------------------------------------------------------------------ helpers
    def _weather(self, year):
        if year not in self._weather_cache:
            self._weather_cache[year] = roles_o3.load_weather(self.regime, year, self.purpose)
        return self._weather_cache[year]

    def _case(self, scenario, w):
        if scenario not in self._case_cache:
            self._case_cache[scenario] = ob.build_kernel_case(self.regime, w.iloc[:1], scenario, self.water_only)
        return self._case_cache[scenario]

    def _raw_obs(self) -> np.ndarray:
        st = self.stepper
        t = min(st.t, st.n_hours - 1)
        last = self._last
        hod = int(st.hod[t])
        ang = 2.0 * np.pi * hod / 24.0
        return np.array([
            last["T_top"], last["T_bottom"], last["T_mean"], last["T_pcm"], last["f_melt"],
            st.ghi[t], st.tamb[t], np.sin(ang), np.cos(ang), last["draw_kg"], self.case.mains_C,
            last["E_stored_kWh"], last["E_rec_kWh"], self._min_since_draw, float(self._prev_action),
        ], dtype=np.float32)

    def _obs(self):
        raw = self._raw_obs()
        return (raw - self._mu) / self._sd if self.normalise else raw

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        year = options.get("year", self.years[int(self.np_random.integers(len(self.years)))])
        scenario = options.get("scenario", self.scenarios[int(self.np_random.integers(len(self.scenarios)))])
        w = self._weather(year)
        self.case = self._case(scenario, w)
        self.stepper = HourlyStepper(self.case, w)
        self.year, self.scenario = year, scenario
        self._day = np.unique(w["local_date"].astype(str).to_numpy(), return_inverse=True)[1]
        drng = np.random.default_rng(int(self.np_random.integers(2**31)))
        self.demand = EpisodeDemand(int(self._day.max()) + 1, self.demand_mode, drng, self.cfg["demand"]["stochastic"])
        self._last = initial_record(self.case)
        self._min_since_draw = float(self.cfg["observation"]["minutes_since_draw_cap"])
        self._prev_action = ENABLE
        return self._obs(), {"year": year, "scenario": scenario, "regime": self.regime}

    def step(self, action):
        st = self.stepper
        t = st.t
        action = int(action)
        rec = st.step(action, self.demand.for_day(int(self._day[t])))
        w = self.weights
        overheat_h = (rec["shield_water_s"] + rec["shield_pcm_s"]) / 3600.0
        guard = rec["T_top_max"] >= GUARD_WATER_C or (not self.water_only and self.case.n_caps.sum() > 0
                                                        and rec["T_pcm_max"] >= GUARD_PCM_C)
        switch = int(action == BYPASS and self._prev_action != BYPASS)
        e_del_n = rec["Edel_Wh"] / w["E_norm_Wh"]
        reward = w["w1"] * e_del_n - w["w2"] * overheat_h - w["w3"] * float(guard) - w["w4"] * switch
        if rec["draw_kg"] >= self.cfg["observation"]["draw_threshold_kg"]:
            self._min_since_draw = 0.0
        else:
            self._min_since_draw = min(self._min_since_draw + 60.0, float(self.cfg["observation"]["minutes_since_draw_cap"]))
        self._last = rec
        self._prev_action = action
        terminated = st.done
        info = {**rec, "overheat_h": overheat_h, "guard_hit": bool(guard), "bypass_switch": switch,
                "in_window": rec["hour_of_day"] in WINDOW_HOURS, "reward": reward,
                "r_delivery": w["w1"] * e_del_n, "r_overheat": -w["w2"] * overheat_h,
                "r_safety": -w["w3"] * float(guard), "r_switch": -w["w4"] * switch}
        if terminated:
            info["episode_residual_pct"] = st.residual_pct()
        return self._obs(), float(reward), terminated, False, info
