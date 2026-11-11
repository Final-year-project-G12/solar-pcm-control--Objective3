"""
Phase O3-1 — step-based PCM solar water heating environment (Rajasthan).

Wraps the FROZEN O2 multi-node kernel (src/simulation/tank_model_multinode.py) into a
Gymnasium-style step(action) -> (obs, reward, terminated, truncated, info) interface, one
decision per hour (plan 3.0). The physics is not rewritten and not re-implemented: every hour is
one call of the O2 numba kernel through o3/environment/o2_adapter.py, with the previous
end-of-hour node temperatures and enthalpies carried forward (plan 3.1).

Differences from the Tamil Nadu environment this file mirrors (details in docs/CORRECTIONS.md):
  * TN's environment stepped a hand-inlined SINGLE-NODE tank although its contract says multi-node;
    this one steps O2's released multi-node (N=10) kernel, and tests/test_environment.py::test_T1
    proves the stepped year equals one run_year_multinode call.
  * T_w in the state dict / info is the TOP node (the delivery node the shield looks at).
  * Observation default is the contract's 13 fields ("V2"); TN's 11-dim layout stays as "V1".

Action space A1 (discrete):
    0 = CHARGE  — collector loop enabled (kernel circulates when GHI is above the cutoff)
    1 = IDLE    — loop off, PCM exchanges heat with the water passively
    2 = BYPASS  — loop off, collector explicitly bypassed
  In the O2 kernel IDLE and BYPASS are physically identical (loop off); they differ only in the
  switching penalty and in what the policy reports. This is plan 0: "the only controllable thing is
  whether the collector loop circulates".

The O2 shield (top node >= 72 C, PCM >= 62 C) runs INSIDE the kernel and can override a CHARGE.

Reward R2 (shield weight re-derived for Rajasthan, see W_SHIELD_BY_REGIME; other weights are the TN values,
unswept — plan 6 / Q2 sweep is a validation-year job):
    r = w_energy*E_del45_Wh - w_unmet*E_unmet_Wh - w_safety*n_safety - w_switching*switch - w_shield*n_shield
"""

import hashlib
import json
import pathlib
import numpy as np

import config_O2  # noqa: F401
from o3.utils.contract_loader import assert_contract_valid
from o3.utils import weather_roles
from o3.environment import obs as _obs
from o3.environment.o2_adapter import build_runtime, OneHourKernel, HR, SC

J_PER_KWH = 3.6e6

# action constants
CHARGE = 0
IDLE = 1
BYPASS = 2
N_ACTIONS = 3

# O2's night window (src/metrics/tes_metrics.py WINDOW_HOURS): 18:00-09:00 local
WINDOW_HOURS = frozenset(list(range(18, 24)) + list(range(0, 9)))

# reward weights (R2 defaults carried from the TN reference; tune on VALIDATION years only)
W_ENERGY = 1.0
W_UNMET = 2.0
W_SAFETY = 50.0
W_SWITCHING = 0.1
# Shield weight (per guard-band activation) — R2. 0.0 recovers R1 (delivery only).
# NOT the TN value (10): TN's was tuned against TN's single-node counters. The multi-node kernel counts a PCM
# shield per NODE per sub-step, so Rajasthan's B0 racks up 1.3e5-3.3e5 activations a year (TN ~5e4). With 10 the
# shield term is ~5x the delivery term and B0 (the best policy on the plan's metric, +136 kWh window gain) scores far
# below the water-only twin. Rajasthan value = TN's METHOD applied to Rajasthan data: 1.19 x the break-even weight
# w* at which B0 and B1 tie, with w* measured on the TRAINING years 2016-2020 only (TN: 10 vs break-even 8.4):
#     regime 0: w* = 3.26   regime 1: w* = 1.43   regime 2: w* = 1.45          (reproduce: phase2 --calibrate-shield)
# Provisional O3 choice: plan Q2 says the shield weight is swept on validation years (grid incl. 0) before it is final.
W_SHIELD_BY_REGIME = {0: 3.9, 1: 1.7, 2: 1.7}


def default_w_shield(regime_id: int) -> float:
    return W_SHIELD_BY_REGIME[int(regime_id)]


# ---- reward mode "Q2": the contract / plan-section-6 form with the protocol the student accepted (plan Q2) ----
#   r_t = w1 * E_del45_t / E_norm  -  w2 * overheat_h_t  -  w3 * safety_violation_t  -  w4 * bypass_switch_t
#   w1 = 1; E_norm = the water-only twin's mean hourly E_del45 over 2016-2020 (so the twin earns ~1 per hour);
#   overheat_h_t = shield-active time in the hour (h); safety_violation_t = 1 if the guard band was reached in the hour
#   (top node >= 72 C or PCM >= 62 C); bypass_switch_t = 1 when the action changes INTO bypass;
#   w3 is NOT tuned: dominance rule = 1.01 x the largest possible one-hour normalised delivery;
#   only w2 and w4 are swept (grid incl. 0, validation years). Defaults 0 / 0 = delivery + dominance term only.
# E_norm and w3 come from contracts/reward_normalisation_rajasthan.json (python scripts/phase2_baselines.py --reward-norm).
REWARD_NORM_FILE = config_O2.CONTRACTS_DIR / "reward_normalisation_rajasthan.json"


def load_reward_norm(regime_id: int) -> dict:
    import json
    if not REWARD_NORM_FILE.exists():
        raise FileNotFoundError(f"{REWARD_NORM_FILE} missing: run `python scripts/phase2_baselines.py --reward-norm` first")
    return json.loads(REWARD_NORM_FILE.read_text())["regimes"][str(int(regime_id))]

# stochastic demand (plan 3.4): ASSUMED ranges, no measured Indian draw-off data was found (plan Q3).
STOCH_VOLUME_SD = 0.20       # day-to-day volume factor ~ LogNormal(mean 1, sd 0.20), clipped to [0.5, 1.5]
STOCH_SHIFT_SD_H = 1.0       # day-to-day timing shift ~ Normal(0, 1 h)


class PCMEnv:
    """Single-regime PCM solar water heating RL environment.

    Parameters
    ----------
    contract_path : path to obj3_environment_contract_rajasthan_v3.json
    regime_id     : 0, 1 or 2 (Rajasthan has 3 regimes)
    year          : weather year for the episode (train 2016-2020, validate 2021-2023)
    obs_version   : "V2" (13 contract fields, default) or "V1" (TN's 11 dims)
    reward_weights: dict with any of energy, unmet, safety, switching, shield
    water_only    : build the plain-tank twin (same tank, flow, shield, weather, zero capsules)
    demand_mode   : "deterministic" (O2's 300 L/day curve every day) or "stochastic" (plan 3.4)
    """

    metadata = {"render_modes": []}

    def __init__(self, contract_path: str | pathlib.Path, regime_id: int = 0, year: int = 2016,
                 obs_version: str = "V2", reward_weights: dict | None = None, water_only: bool = False,
                 demand_mode: str = "deterministic", demand_volume_multiplier: float = 1.0,
                 demand_timing_shift_h: float = 0.0, reward_mode: str = "R2"):
        self.contract = assert_contract_valid(contract_path)
        self.regime_id = regime_id
        self.year = year
        self.obs_version = obs_version
        self.water_only = water_only
        self.demand_mode = demand_mode
        if obs_version not in _obs.OBS_DIMS:
            raise ValueError(f"obs_version must be one of {list(_obs.OBS_DIMS)}")
        if demand_mode not in ("deterministic", "stochastic"):
            raise ValueError("demand_mode must be 'deterministic' or 'stochastic'")

        rw = reward_weights or {}
        self.w_energy = rw.get("energy", W_ENERGY)
        self.w_unmet = rw.get("unmet", W_UNMET)
        self.w_safety = rw.get("safety", W_SAFETY)
        self.w_switching = rw.get("switching", W_SWITCHING)
        self.w_shield = rw.get("shield", default_w_shield(regime_id))
        self.reward_mode = rw.get("mode", reward_mode)
        if self.reward_mode not in ("R2", "Q2"):
            raise ValueError("reward_mode must be 'R2' or 'Q2'")
        if self.reward_mode == "Q2":
            nrm = load_reward_norm(regime_id)
            self._e_norm, self.w3 = nrm["E_norm_Wh_per_hour"], nrm["w3_dominance"]
            self.w2, self.w4 = rw.get("w2", 0.0), rw.get("w4", 0.0)

        self.run_config = self._build_run_config(contract_path, demand_volume_multiplier, demand_timing_shift_h)
        self.config_sha = hashlib.sha256(json.dumps(self.run_config, sort_keys=True).encode()).hexdigest()

        self._rt = build_runtime(self.contract, regime_id, water_only=water_only,
                                 volume_multiplier=demand_volume_multiplier, timing_shift_hours=demand_timing_shift_h)
        self._k = OneHourKernel(self._rt)
        self.system_config = self._rt.cfg
        self.mains_temp_C = self._rt.mains_C
        self.pcm_props = self._rt.pcm_props
        self._max_water_C = self._rt.cfg["safety"]["max_water_temp_C"]
        self._load_weather()

        self.obs_dim = _obs.OBS_DIMS[obs_version]
        self.n_actions = N_ACTIONS
        self._state: dict | None = None
        self._daily_demand: np.ndarray | None = None
        self._log: dict | None = None
        self._override: tuple | None = None       # (ghi, tamb, hour_of_day) injected by the live demo

    # ------------------------------------------------------------------
    # Gymnasium-style interface
    # ------------------------------------------------------------------

    def reset(self, year: int | None = None, initial_pcm_state: str = "solid",
              seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Start a new episode.  Returns (observation, info)."""
        if year is not None:
            self.year = year
            self._load_weather()
        T, h = self._k.initial_state(initial_pcm_state)
        n = self._n_hours
        if self.demand_mode == "stochastic":
            self._daily_demand = self._draw_daily_demand(seed, n)
        else:
            self._daily_demand = None
        d = self._k.derived(T, h)
        self._state = {
            "hour_idx": 0, "T_nodes": T, "h_nodes": h,
            "T_w": d["T_top"], "T_bottom": d["T_bottom"], "T_w_mean": d["T_mean"],
            "T_pcm": d["T_pcm"], "f_melt": d["f_melt"], "h_pcm": float(np.mean(h)),
            "E_stored_kWh": d["E_stored_kWh"], "E_rec45_kWh": d["E_rec45_kWh"],
            "prev_action": CHARGE, "hours_since_draw": 24.0,
            "cumulative_delivered_Wh": 0.0, "cumulative_unmet_Wh": 0.0,
            "n_shield_water": 0, "n_shield_pcm": 0, "n_safety_violations": 0, "action_switches": 0,
            "E_init_J": float(self._energy_J(T, h)),
        }
        self._log = {k: np.zeros(n) for k in (
            "E_del45", "E_load", "E_unmet", "vol_ge", "draw", "Q_coll", "shield_w_s", "shield_p_s", "over_s",
            "T_top_max", "T_pcm_max", "hod", "action", "executed")}
        self._coll_J = 0.0
        self._resid_J = 0.0
        return self._get_obs(), {"year": self.year, "regime_id": self.regime_id, "initial_pcm_state": initial_pcm_state}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run one hour of the simulator with the given action."""
        assert self._state is not None, "Call reset() before step()."
        assert action in (CHARGE, IDLE, BYPASS), f"Invalid action {action}"
        s = self._state
        i = s["hour_idx"]
        if i >= self._n_hours:
            raise RuntimeError("Episode already ended. Call reset().")

        if self._override is not None:               # live / demo mode: weather supplied from outside
            ghi, tamb, hod = self._override
        else:
            ghi, tamb, hod = float(self._ghi[i]), float(self._tamb[i]), int(self._hod[i])
        d24 = None if self._daily_demand is None else self._daily_demand[i // 24]
        hr, sc, T_new, h_new = self._k.hour(s["T_nodes"], s["h_nodes"], ghi, tamb, hod, action == CHARGE, d24)

        e_del45 = hr[HR["E_del45"]]
        e_load = hr[HR["Q_load"]]
        e_unmet = hr[HR["Q_unmet"]]
        n_sw, n_sp, n_saf = int(sc[SC["shield_w_n"]]), int(sc[SC["shield_p_n"]]), int(sc[SC["safety"]])
        shield_fired = (n_sw + n_sp) > 0
        executed = BYPASS if (action == CHARGE and n_sw > 0 and hr[HR["Q_coll"]] <= 0.0) else action

        switched = int(action != s["prev_action"])
        # end-of-hour state comes straight from the kernel's own hourly output (identical to derived(), but
        # without a second Python pass over the nodes: profiling showed that pass was a third of the step time)
        d = {"T_top": hr[HR["T_top"]], "T_bottom": hr[HR["T_bottom"]], "T_mean": hr[HR["T_mean"]],
             "T_pcm": hr[HR["T_pcm"]], "f_melt": hr[HR["f_melt"]],
             "E_stored_kWh": hr[HR["E_stored"]], "E_rec45_kWh": hr[HR["E_rec45"]]}
        draw = float(hr[HR["draw"]])
        s.update({"T_nodes": T_new, "h_nodes": h_new, "T_w": d["T_top"], "T_bottom": d["T_bottom"],
                  "T_w_mean": d["T_mean"], "T_pcm": d["T_pcm"], "f_melt": d["f_melt"], "h_pcm": float(np.mean(h_new)),
                  "E_stored_kWh": d["E_stored_kWh"], "E_rec45_kWh": d["E_rec45_kWh"],
                  "prev_action": action, "hour_idx": i + 1,
                  "hours_since_draw": 0.0 if draw > 1e-3 else min(s["hours_since_draw"] + 1.0, 24.0)})
        s["cumulative_delivered_Wh"] += e_del45
        s["cumulative_unmet_Wh"] += e_unmet
        s["n_shield_water"] += n_sw
        s["n_shield_pcm"] += n_sp
        s["n_safety_violations"] += n_saf
        s["action_switches"] += switched
        self._coll_J += sc[SC["E_coll"]]
        self._resid_J += (sc[SC["E_coll"]] + sc[SC["E_init"]]) - (sc[SC["E_load_net"]] + sc[SC["E_loss"]] + sc[SC["E_fin"]])

        lg = self._log
        lg["E_del45"][i], lg["E_load"][i], lg["E_unmet"][i] = e_del45, e_load, e_unmet
        lg["vol_ge"][i], lg["draw"][i], lg["Q_coll"][i] = hr[HR["vol_ge"]], draw, hr[HR["Q_coll"]]
        lg["shield_w_s"][i], lg["shield_p_s"][i], lg["over_s"][i] = sc[SC["shield_w_s"]], sc[SC["shield_p_s"]], sc[SC["over_s"]]
        lg["T_top_max"][i], lg["T_pcm_max"][i] = hr[HR["T_max"]], hr[HR["T_pcm_max"]]
        lg["hod"][i], lg["action"][i], lg["executed"][i] = hod, action, executed

        n_shield = n_sw + n_sp
        if self.reward_mode == "Q2":
            sh_h = (sc[SC["shield_w_s"]] + sc[SC["shield_p_s"]]) / 3600.0
            into_bypass = int(action == BYPASS and (i == 0 or lg["action"][i - 1] != BYPASS))
            reward = e_del45 / self._e_norm - self.w2 * sh_h - self.w3 * float(shield_fired) - self.w4 * into_bypass
        else:
            reward = (self.w_energy * e_del45 - self.w_unmet * e_unmet - self.w_safety * n_saf
                      - self.w_switching * switched - self.w_shield * n_shield)
        terminated = (i + 1) >= self._n_hours
        info = {
            "hour_idx": i, "hour_of_day": hod,
            "T_w_C": s["T_w"], "T_top_C": s["T_w"], "T_bottom_C": s["T_bottom"], "T_w_mean_C": s["T_w_mean"],
            "T_pcm_C": s["T_pcm"], "f_melt": s["f_melt"],
            "E_delivered_Wh": e_del45,           # delivered up to the 45 C target (contract E_del45)
            "E_del45_Wh": e_del45, "E_load_Wh": e_load, "E_unmet_Wh": e_unmet,
            "Q_collector_Wh": float(hr[HR["Q_coll"]]), "draw_kg": draw,
            "shield_override": shield_fired,
            "n_shield_water": n_sw, "n_shield_pcm": n_sp, "n_safety_violations": n_saf,
            "shield_water_s": float(sc[SC["shield_w_s"]]), "shield_pcm_s": float(sc[SC["shield_p_s"]]),
            "over_limit_s": float(sc[SC["over_s"]]),
            "E_stored_kWh": s["E_stored_kWh"], "E_rec45_kWh": s["E_rec45_kWh"],
            "action_requested": action, "action_executed": executed,
        }
        return self._get_obs(), float(reward), bool(terminated), False, info

    def close(self) -> None:
        pass

    def set_weather_override(self, ghi: float, tamb: float, hour_of_day: int) -> np.ndarray:
        """Demo mode only: use this hour's weather instead of the stored year, and return the observation
        for it. Pass None to go back to the stored series. Never used by training or evaluation."""
        self._override = None if ghi is None else (float(max(ghi, 0.0)), float(tamb), int(hour_of_day) % 24)
        return self._get_obs()

    # ------------------------------------------------------------------
    # Episode metrics (O2's definitions where the hourly log allows)
    # ------------------------------------------------------------------

    def episode_summary(self) -> dict:
        """Totals for the finished (or partial) episode, from the hourly log."""
        s, lg = self._state, self._log
        n = s["hour_idx"]
        win = np.isin(lg["hod"][:n].astype(int), list(WINDOW_HOURS))
        draw_w = lg["draw"][:n][win].sum()
        return {
            "n_steps": n,
            "E_del45_annual_kWh": lg["E_del45"][:n].sum() / 1000.0,
            "E_del45_window_kWh": lg["E_del45"][:n][win].sum() / 1000.0,
            "E_load_kWh": lg["E_load"][:n].sum() / 1000.0,
            "E_unmet_kWh": lg["E_unmet"][:n].sum() / 1000.0,
            "E_collector_kWh": self._coll_J / J_PER_KWH,
            "serve_frac_45_window_insitu": float(lg["vol_ge"][:n][win].sum() / draw_w) if draw_w > 0 else float("nan"),
            "shield_water_h": lg["shield_w_s"][:n].sum() / 3600.0,
            "shield_pcm_h": lg["shield_p_s"][:n].sum() / 3600.0,
            "hours_over_limit": lg["over_s"][:n].sum() / 3600.0,
            "t_top_max_C": float(lg["T_top_max"][:n].max()) if n else float("nan"),
            "t_pcm_max_C": float(lg["T_pcm_max"][:n].max()) if (n and not self._rt.water_only) else float("nan"),
            "n_safety_violations": s["n_safety_violations"],
            "action_switches": s["action_switches"],
            "bypass_switches": int(np.sum((lg["action"][1:n] == BYPASS) & (lg["action"][:n - 1] != BYPASS))) if n > 1 else 0,
            "residual_pct_of_collector": abs(self._resid_J) / max(abs(self._coll_J), 1.0) * 100.0,
            "final_T_w_C": s["T_w"], "final_T_pcm_C": s["T_pcm"], "final_f_melt": s["f_melt"],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_run_config(self, contract_path, vol_mult: float, shift_h: float) -> dict:
        """Plan 15.4: reward weights, demand-model parameters, reset-scenario mix, year roles, obs layout and the
        contract hash in one dict; its SHA-256 (`config_sha`) is stamped on every result row."""
        from o3.utils.contract_loader import contract_sha256
        reward = {"mode": self.reward_mode, "energy": self.w_energy, "unmet": self.w_unmet, "safety": self.w_safety,
                  "switching": self.w_switching, "shield": self.w_shield}
        if self.reward_mode == "Q2":
            reward = {"mode": "Q2", "w1": 1.0, "E_norm_Wh_per_hour": self._e_norm, "w2": self.w2, "w3_dominance": self.w3, "w4": self.w4}
        return {
            "state": "rajasthan", "regime_id": int(self.regime_id), "contract_sha256": contract_sha256(contract_path),
            "obs_version": self.obs_version, "n_actions": N_ACTIONS, "water_only": bool(self.water_only), "reward": reward,
            "demand": {"mode": self.demand_mode, "volume_multiplier": vol_mult, "timing_shift_h": shift_h,
                       "stochastic_volume_sd": STOCH_VOLUME_SD, "stochastic_shift_sd_h": STOCH_SHIFT_SD_H},
            "reset_scenarios": ["solid", "partial", "liquid"],
            "year_roles": {"train": weather_roles.TRAIN_YEARS, "validate": weather_roles.VAL_YEARS,
                           "test": weather_roles.TEST_YEARS, "used_o2": weather_roles.USED_O2_YEARS},
        }

    def forecast_ghi(self, k: int) -> float:
        """GHI of the hour k steps ahead of the current one (perfect-forecast baselines; 0 past the end)."""
        j = self._state["hour_idx"] + k
        return float(self._ghi[j]) if j < self._n_hours else 0.0

    def _energy_J(self, T, h) -> float:
        rt = self._rt
        e = float(np.sum(rt.m_w * rt.cp_w * T))
        if rt.has_pcm:
            e += float(np.sum(rt.m_pcm_node * h))
        return e

    def _get_obs(self) -> np.ndarray:
        s = self._state
        i = s["hour_idx"]
        if self._override is not None:
            ghi, tamb, hod = self._override
            d24 = self._rt.demand24 if self._daily_demand is None else self._daily_demand[min(i // 24, len(self._daily_demand) - 1)]
            draw = float(d24[hod])
        elif i < self._n_hours:
            ghi, tamb, hod = float(self._ghi[i]), float(self._tamb[i]), int(self._hod[i])
            d24 = self._rt.demand24 if self._daily_demand is None else self._daily_demand[i // 24]
            draw = float(d24[hod])
        else:
            ghi = tamb = draw = 0.0
            hod = 0
        return _obs.build(self.obs_version, {
            "T_top": s["T_w"], "T_bottom": s["T_bottom"], "T_mean": s["T_w_mean"], "T_pcm": s["T_pcm"],
            "f_melt": s["f_melt"], "GHI": ghi, "T_amb": tamb, "hour": hod, "draw": draw,
            "T_mains": self.mains_temp_C, "E_stored_kWh": s["E_stored_kWh"], "E_rec45_kWh": s["E_rec45_kWh"],
            "min_since_draw": s["hours_since_draw"] * 60.0, "hour_idx": i, "prev_action": s["prev_action"],
            "max_water_C": self._max_water_C,
        })

    def _load_weather(self) -> None:
        w = weather_roles.load_weather(self.regime_id, self.year)
        self._ghi, self._tamb, self._hod = w["GHI_Wm2"], w["T_amb_C"], w["local_hour"]
        self._n_hours = len(self._ghi)

    def _draw_daily_demand(self, seed: int | None, n_hours: int) -> np.ndarray:
        """One 24-vector per simulated day: volume factor and timing shift drawn per day (plan 3.4)."""
        from src.io_utils import load_demand_profile
        from src.simulation.demand_profile import load_demand_model
        rng = np.random.default_rng(seed)
        df = load_demand_profile("rajasthan")
        n_days = n_hours // 24
        sig = np.sqrt(np.log(1.0 + STOCH_VOLUME_SD ** 2))
        fac = np.clip(rng.lognormal(-0.5 * sig ** 2, sig, n_days), 0.5, 1.5)
        shift = rng.normal(0.0, STOCH_SHIFT_SD_H, n_days)
        out = np.zeros((n_days, 24))
        for d in range(n_days):
            m = load_demand_model(df, float(fac[d]), float(shift[d]))
            out[d] = [m.draw_mass_kg_for_hour(h) for h in range(24)]
        return out
