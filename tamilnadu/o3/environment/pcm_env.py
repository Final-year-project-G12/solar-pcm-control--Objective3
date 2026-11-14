"""
Phase O3-1 — step-based PCM solar water heating environment.

Wraps the frozen O2 simulator (src/simulation/tank_model.py) into a
Gymnasium-compatible step(action) -> (obs, reward, done, info) interface.

DESIGN RULE: this file must NEVER rewrite the physics equations.
It only orchestrates one hour at a time by inlining the per-hour body
from run_year(), modified to accept an action from the RL policy.

Action space A1 (discrete, implemented here):
    0 = CHARGE   — circulate pump, solar -> tank + PCM
    1 = IDLE     — pump off, PCM discharges passively to water
    2 = BYPASS   — pump off, explicitly bypass the collector

The existing safety shield (bypass_water_C=72, bypass_pcm_C=62) still
runs INSIDE this environment and can override the policy's action.
When the shield fires, info["shield_override"] is set to True.

Observation vector V1 (11 dims, all normalised):
    [GHI, T_amb, T_water, T_pcm, f_melt,
     demand_volume_L, T_mains, hour/24, day/365, prev_action/2,
     temp_margin_water]

Observation vector V2 (15 dims — V1 + a forecast tail, APPENDED so every V1
index is unchanged and the B1/B2/B4 baselines and action-masking code, which
all index obs[0..7] positionally, keep working unmodified):
    V1[0..10] + [GHI_fwd_h+1, GHI_fwd_h+2, GHI_fwd_h+3, demand_fwd_h+1]
This gives the policy the same 2-3h-ahead look-ahead the B2 rule baseline uses,
so anticipation no longer has to be learned purely from the current state.

Control step: the decision period is configurable (`control_step_min`,
default 60 = unchanged, hourly). The shield already runs every ~150s physics
sub-step; an hourly ACTION cannot react to overshoot discovered mid-hour. At
control_step_min=15 the policy gets 4 decisions per weather-hour (weather/
demand are still hourly-resolution and held constant across the sub-periods;
only the control cadence changes), so it can switch to BYPASS/IDLE before a
full hour of unchecked charging accumulates.

Domain randomisation (`domain_randomize=True`, training only): at each
reset(), sample a GHI multiplier, an ambient-temperature bias, and a demand
multiplier, held fixed for the whole episode, and apply them to the physics
inputs (not just the observation) so the dynamics themselves vary run to run.
Evaluation/locked-test/robustness envs keep this off (nominal physics); the
existing phase-5 robustness suite already stress-tests *trained* policies with
perturbations — this instead perturbs *training* so the policy has seen
variability before evaluation ever does.

Reward R1:
    r = w_E * E_delivered_Wh - w_U * E_unmet_Wh - w_S * safety_penalty
"""

import pathlib
import sys
import numpy as np
from dataclasses import replace as _dc_replace

# --- O2 root (objective2_design_optimization) must be on sys.path
# conftest.py adds it; this guard covers direct script execution.
_O2_ROOT = pathlib.Path(__file__).resolve().parents[3] / "objective2_design_optimization"
if str(_O2_ROOT) not in sys.path:
    sys.path.insert(0, str(_O2_ROOT))

from src.io_utils import (
    load_system_config, load_design_bounds, get_regime, get_pcm_properties,
    load_demand_profile, drop_leap_day,
)
from config import BASE_DIR
from src.simulation.capsule_enthalpy import (
    PCMThermalProps, pcm_props_from_record,
    enthalpy_of_temperature, temperature_and_liquid_fraction_of_enthalpy,
)
from src.simulation.collector_model import collector_linear_coeffs
from src.simulation.heat_transfer import ua_eff_total_w_k
from src.simulation.hydraulic_model import pressure_drop_and_pump_power
from src.simulation.capsule_mass import capsule_masses, wall_cp_per_kg_pcm
from src.simulation.demand_profile import load_demand_model
from src.design.geometry import compute_geometry
from src.design.schema import DesignVector
from src.design.constraints import check_design

# O3 utils
from o3.utils.contract_loader import assert_contract_valid, get_regime_design

J_PER_KWH = 3.6e6

# action constants
CHARGE  = 0
IDLE    = 1
BYPASS  = 2
N_ACTIONS = 3

# reward weights (R2 defaults — tune on validation, never on test set)
#
# R1 (delivery-only) made every DRL agent collapse to the B0 "always charge"
# policy: with the safety shield preventing all hard-limit breaches,
# n_safety_violations is ~always 0, so nothing penalised the 50k/yr guard-band
# (shield) activations and maximal charging was trivially optimal. R2 adds
# W_SHIELD per shield activation so the agent has a reason to learn the
# anticipatory control that is Objective 3's contribution. Set shield weight to
# 0.0 via reward_weights to recover the original R1 reward.
W_ENERGY    = 1.0
W_UNMET     = 2.0
W_SAFETY    = 50.0
W_SWITCHING = 0.1
W_SHIELD    = 10.0   # per guard-band (shield) activation — R2
# Empirically the best setting. At 10, DQN learns the anticipatory policy
# (B0-level delivery, ~26% fewer shields, beats all baselines). Raising it to 20
# did NOT help: it destabilised DQN's value learning and the agent collapsed back
# to always-charge, even though the shield-reducing policy still scores higher
# there. So the effective lever for this problem is exploration/seed robustness,
# not penalty magnitude. Tune on validation, never on the test set.


class PCMEnv:
    """Single-regime PCM solar water heating RL environment.

    Parameters
    ----------
    contract_path : path to obj3_environment_contract_tamilnadu_v3.json
    regime_id     : 0, 1 or 2 (Tamil Nadu has 3 regimes)
    year          : weather year for the episode (use 2016-2020 for training)
    obs_version   : "V1" (minimal), more will follow
    """

    # Gymnasium-compatible metadata
    metadata = {"render_modes": []}

    def __init__(
        self,
        contract_path: str | pathlib.Path,
        regime_id: int = 0,
        year: int = 2016,
        obs_version: str = "V1",
        forecast_mode: str = "perfect",
        reward_weights: dict | None = None,
        control_step_min: int = 60,
        domain_randomize: bool = False,
        domain_rand_ranges: dict | None = None,
    ):
        self.contract = assert_contract_valid(contract_path)
        self.regime_id = regime_id
        self.year = year
        self.obs_version = obs_version
        assert forecast_mode in ("perfect", "smart", "none"), forecast_mode
        self.forecast_mode = forecast_mode

        rw = reward_weights or {}
        self.w_energy    = rw.get("energy",    W_ENERGY)
        self.w_unmet     = rw.get("unmet",     W_UNMET)
        self.w_safety    = rw.get("safety",    W_SAFETY)
        self.w_switching = rw.get("switching", W_SWITCHING)
        self.w_shield    = rw.get("shield",    W_SHIELD)

        # sub-hourly control: N ticks per weather-hour (60 % control_step_min
        # must be 0; default 60 -> N=1, bit-for-bit the original hourly env)
        assert 60 % control_step_min == 0, "control_step_min must divide 60"
        self.control_step_min = control_step_min
        self._ticks_per_hour   = 60 // control_step_min

        # domain randomisation (training-time only; see module docstring)
        self.domain_randomize = domain_randomize
        dr = domain_rand_ranges or {}
        self._dr_ranges = {
            "ghi_mult":    dr.get("ghi_mult",    (0.90, 1.10)),
            "amb_bias_C":  dr.get("amb_bias_C",  (-2.0, 2.0)),
            "demand_mult": dr.get("demand_mult", (0.85, 1.15)),
        }
        self._dr_rng = np.random.default_rng()
        self._dr = {"ghi_mult": 1.0, "amb_bias_C": 0.0, "demand_mult": 1.0}

        self._setup_from_contract()
        self._load_weather()

        # observation / action space sizes (V2 and V3 share the same 15-dim
        # shape: V1 + 3h GHI forecast + next-hour demand; V3 is V2 driven by
        # forecast_mode="smart" instead of "perfect" — see forecast_ghi())
        self.obs_dim    = 15 if obs_version in ("V2", "V3") else 11
        self.n_actions  = N_ACTIONS

        # normalisation stats (training-set statistics — placeholders, computed
        # from 2016-2020 data before the first training run)
        self._obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
        self._obs_std  = np.ones(self.obs_dim,  dtype=np.float32)

        self._state: dict | None = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, year: int | None = None, initial_pcm_state: str = "solid",
              seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Start a new episode.  Returns (observation, info)."""
        if year is not None:
            self.year = year
            self._load_weather()

        cfg = self.system_config
        solver = cfg["solver"]

        T_w = solver["initial_water_temp_C"]
        if initial_pcm_state == "solid":
            T_pcm_init = self.pcm_props.Ts_C - 5.0
        elif initial_pcm_state == "liquid":
            T_pcm_init = self.pcm_props.Tl_C + 5.0
        else:
            T_pcm_init = self.pcm_props.Tm_C

        h_pcm = enthalpy_of_temperature(T_pcm_init, self.pcm_props)
        T_pcm, f_melt = temperature_and_liquid_fraction_of_enthalpy(h_pcm, self.pcm_props)

        if self.domain_randomize:
            if seed is not None:
                self._dr_rng = np.random.default_rng(seed)
            lo, hi = self._dr_ranges["ghi_mult"];    ghi_m = self._dr_rng.uniform(lo, hi)
            lo, hi = self._dr_ranges["amb_bias_C"];  amb_b = self._dr_rng.uniform(lo, hi)
            lo, hi = self._dr_ranges["demand_mult"]; dem_m = self._dr_rng.uniform(lo, hi)
            self._dr = {"ghi_mult": ghi_m, "amb_bias_C": amb_b, "demand_mult": dem_m}
        else:
            self._dr = {"ghi_mult": 1.0, "amb_bias_C": 0.0, "demand_mult": 1.0}

        self._state = {
            "tick_idx": 0,
            "hour_idx": 0,
            "T_w":     T_w,
            "T_pcm":   T_pcm,
            "h_pcm":   h_pcm,
            "f_melt":  f_melt,
            "prev_action": CHARGE,
            "cumulative_delivered_Wh": 0.0,
            "cumulative_unmet_Wh":     0.0,
            "n_shield_water": 0,
            "n_shield_pcm":   0,
            "n_safety_violations": 0,
            "action_switches": 0,
        }
        obs = self._get_obs()
        return obs, {"year": self.year, "regime_id": self.regime_id}

    def _n_ticks(self) -> int:
        return len(self._weather) * self._ticks_per_hour

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run one control period (default = one hour; see control_step_min).

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self._state is not None, "Call reset() before step()."
        assert action in (CHARGE, IDLE, BYPASS), f"Invalid action {action}"

        s = self._state
        t = s["tick_idx"]
        i = t // self._ticks_per_hour              # hour index into weather

        if i >= len(self._weather):
            raise RuntimeError("Episode already ended. Call reset().")

        row       = self._weather.iloc[i]
        I_t       = float(row["GHI_Wm2"])  * self._dr["ghi_mult"]
        T_amb     = float(row["T_amb_C"])  + self._dr["amb_bias_C"]
        hour_of_day = int(row.get("local_hour", i % 24))

        draw_mass_kg_hr = (self._demand_model.draw_mass_kg_for_hour(hour_of_day)
                          * self._dr["demand_mult"])
        draw_mass_kg = draw_mass_kg_hr / self._ticks_per_hour
        period_s = 3600.0 / self._ticks_per_hour

        result = self._run_period(
            I_t, T_amb, draw_mass_kg, period_s,
            s["T_w"], s["T_pcm"], s["h_pcm"], s["f_melt"],
            action,
        )

        # update state
        action_switched = int(action != s["prev_action"])
        s["T_w"]     = result["T_w"]
        s["T_pcm"]   = result["T_pcm"]
        s["h_pcm"]   = result["h_pcm"]
        s["f_melt"]  = result["f_melt"]
        s["prev_action"] = action
        s["tick_idx"]    = t + 1
        s["hour_idx"]    = (t + 1) // self._ticks_per_hour
        s["cumulative_delivered_Wh"] += result["E_delivered_Wh"]
        s["cumulative_unmet_Wh"]     += result["E_unmet_Wh"]
        s["n_shield_water"] += result["n_shield_water"]
        s["n_shield_pcm"]   += result["n_shield_pcm"]
        s["n_safety_violations"] += result["n_safety_violations"]
        s["action_switches"] += action_switched

        # reward R2 — delivery, unmet, hard-limit, switching, and shield cost
        n_shield = result["n_shield_water"] + result["n_shield_pcm"]
        reward = (
            self.w_energy    *  result["E_delivered_Wh"]
            - self.w_unmet   *  result["E_unmet_Wh"]
            - self.w_safety  *  result["n_safety_violations"]
            - self.w_switching * action_switched
            - self.w_shield  *  n_shield
        )

        terminated = (t + 1) >= self._n_ticks()
        truncated  = False

        obs  = self._get_obs()
        info = {
            "hour_idx":          i,
            "tick_idx":          t,
            "hour_of_day":       hour_of_day,
            "T_w_C":             result["T_w"],
            "T_pcm_C":           result["T_pcm"],
            "f_melt":            result["f_melt"],
            "E_delivered_Wh":    result["E_delivered_Wh"],
            "E_unmet_Wh":        result["E_unmet_Wh"],
            "shield_override":   result["shield_override"],
            "n_shield_water":    result["n_shield_water"],
            "n_shield_pcm":      result["n_shield_pcm"],
            "n_safety_violations": result["n_safety_violations"],
            "action_requested":  action,
            "action_executed":   result["action_executed"],
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Physics: one-hour stepper (extracted from run_year inner loop)
    # ------------------------------------------------------------------

    def _run_period(
        self,
        I_t: float, T_amb: float, draw_mass_kg: float, period_s: float,
        T_w: float, T_pcm: float, h_pcm: float, f_melt: float,
        action: int,
    ) -> dict:
        """Run one control period's worth of sub-steps (default period = one
        hour, i.e. control_step_min=60). Preserves the O2 physics exactly —
        only the OUTER period length changes; the inner ~150s/37.5s physics
        sub-stepping logic below is untouched."""
        cfg     = self.system_config
        solver  = cfg["solver"]
        cp_w    = cfg["water"]["cp_J_kgK"]
        u_tank  = cfg["tank"]["U_tank_W_m2K"]
        a_tank  = self.design.tank_surface_area_m2
        mains   = self.mains_temp_C
        deliv_T = cfg["delivery"]["target_temp_C"]
        max_wC  = cfg["safety"]["max_water_temp_C"]
        max_pC  = cfg["safety"]["max_pcm_temp_C"]

        shield_cfg     = cfg.get("safety_shield", {})
        shield_enabled = bool(shield_cfg.get("enabled", True))
        bypass_wC      = shield_cfg.get("bypass_water_C", max_wC - 3.0)
        bypass_pC      = shield_cfg.get("bypass_pcm_C",  max_pC - 3.0)

        dt_total_s         = solver["timestep_s"]
        n_substeps_per_hr  = max(1, int(round(period_s / dt_total_s)))
        n_sub_in_band      = solver["substeps_in_melt_band"]
        band_margin        = solver["melt_band_margin_K"]
        STIFFNESS_FACTOR   = 0.5
        MAX_SUBSTEPS       = 60

        draw_per_substep = draw_mass_kg / n_substeps_per_hr

        # action -> circulating flag before the shield may override it
        policy_circulate = (action == CHARGE)

        hour_E_del_Wh = hour_E_unmet_Wh = 0.0
        n_shield_w = n_shield_p = n_safety = 0
        shield_fired_this_hour = False
        action_executed = action   # may change if shield overrides

        for _sub_outer in range(n_substeps_per_hr):
            in_band = (self.pcm_props.Ts_C - band_margin <= T_pcm <= self.pcm_props.Tl_C + band_margin)
            n_sub = n_sub_in_band if in_band else 1

            ua_probe = ua_eff_total_w_k(
                self.design.n_capsule, self.design.capsule_diameter_m,
                self.design.capsule_area_m2, self._superficial_v,
                self.pcm_props, f_melt, cfg,
            )
            if ua_probe > 0:
                cp_min = min(self.pcm_props.cp_solid_J_kgK, self.pcm_props.cp_liquid_J_kgK) + self.pcm_props.wall_cp_J_kgK
                tau = self._m_pcm * cp_min / ua_probe
                n_sub_stiff = max(1, int(np.ceil(dt_total_s / (STIFFNESS_FACTOR * tau))))
                n_sub = min(max(n_sub, n_sub_stiff), MAX_SUBSTEPS)

            dt_sub = dt_total_s / n_sub
            draw_this_sub = draw_per_substep

            for _ in range(n_sub):
                draw = draw_this_sub / n_sub
                m_draw_rate = draw / dt_sub if dt_sub > 0 else 0.0

                a_coll, b_coll, circulating = collector_linear_coeffs(I_t, T_amb, cfg)

                # policy decision (can be overridden below by shield)
                if not policy_circulate:
                    circulating = False
                    if action == BYPASS:
                        action_executed = BYPASS

                # safety shield — water side
                if shield_enabled and T_w >= bypass_wC:
                    circulating = False
                    n_shield_w += 1
                    shield_fired_this_hour = True
                    action_executed = BYPASS

                # safety shield — PCM charge side
                ua_eff = ua_eff_total_w_k(
                    self.design.n_capsule, self.design.capsule_diameter_m,
                    self.design.capsule_area_m2, self._superficial_v,
                    self.pcm_props, f_melt, cfg,
                )
                if shield_enabled and T_pcm >= bypass_pC and T_w > T_pcm:
                    ua_eff = 0.0
                    n_shield_p += 1
                    shield_fired_this_hour = True

                base_coeff = (self._m_w * cp_w / dt_sub) + m_draw_rate * cp_w + ua_eff + u_tank * a_tank
                base_rhs   = (self._m_w * cp_w * T_w / dt_sub
                               + m_draw_rate * cp_w * mains
                               + ua_eff * T_pcm
                               + u_tank * a_tank * T_amb)

                if circulating:
                    coeff = base_coeff + b_coll
                    rhs   = base_rhs   + a_coll
                    T_w_trial = rhs / coeff if coeff > 0 else T_w
                    if a_coll - b_coll * T_w_trial < 0:
                        circulating = False
                    else:
                        T_w_new = T_w_trial
                if not circulating:
                    T_w_new = base_rhs / base_coeff if base_coeff > 0 else T_w

                q_unmet = (m_draw_rate * cp_w * max(deliv_T - T_w_new, 0.0)
                           if m_draw_rate > 0 else 0.0)
                q_load  = max(m_draw_rate * cp_w * (T_w_new - mains), 0.0)
                q_pcm   = ua_eff * (T_w_new - T_pcm)

                dh      = q_pcm * dt_sub / self._m_pcm if self._m_pcm > 0 else 0.0
                h_pcm_n = h_pcm + dh
                T_pcm_n, f_melt_n = temperature_and_liquid_fraction_of_enthalpy(h_pcm_n, self.pcm_props)
                if not (-50.0 <= T_pcm_n <= 500.0):
                    T_pcm_n = float(np.clip(T_pcm_n, -50.0, 500.0))
                    h_pcm_n = enthalpy_of_temperature(T_pcm_n, self.pcm_props)

                if T_w_new > max_wC or T_pcm_n > max_pC:
                    n_safety += 1

                hour_E_del_Wh   += q_load   * dt_sub / 3600.0
                hour_E_unmet_Wh += q_unmet  * dt_sub / 3600.0

                T_w, T_pcm, h_pcm, f_melt = T_w_new, T_pcm_n, h_pcm_n, f_melt_n

        return {
            "T_w":               T_w,
            "T_pcm":             T_pcm,
            "h_pcm":             h_pcm,
            "f_melt":            f_melt,
            "E_delivered_Wh":    hour_E_del_Wh,
            "E_unmet_Wh":        hour_E_unmet_Wh,
            "n_shield_water":    n_shield_w,
            "n_shield_pcm":      n_shield_p,
            "n_safety_violations": n_safety,
            "shield_override":   shield_fired_this_hour,
            "action_executed":   action_executed,
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        s   = self._state
        t   = s["tick_idx"]
        i   = t // self._ticks_per_hour
        cfg = self.system_config

        if i < len(self._weather):
            row   = self._weather.iloc[i]
            I_t   = float(row["GHI_Wm2"]) * self._dr["ghi_mult"]
            T_amb = float(row["T_amb_C"]) + self._dr["amb_bias_C"]
            hod   = int(row.get("local_hour", i % 24))
        else:
            I_t = T_amb = 0.0
            hod = 0

        demand_L = (self._demand_model.draw_mass_kg_for_hour(hod)
                   * self._dr["demand_mult"])                       # kg (≈ L)
        mains    = self.mains_temp_C
        max_wC   = cfg["safety"]["max_water_temp_C"]
        margin   = max_wC - s["T_w"]

        v1 = [
            I_t / 1000.0,                    # normalised GHI (0-1 for typical 1000 W/m2)
            (T_amb - 15.0) / 25.0,           # ambient temp centred ~25°C
            (s["T_w"]   - 30.0) / 40.0,     # water temp
            (s["T_pcm"] - 30.0) / 40.0,     # PCM temp
            s["f_melt"],                     # melt fraction 0-1
            demand_L / 20.0,                 # typical peak ~20 L/hr
            (mains - 20.0) / 20.0,           # mains temp
            hod / 24.0,                      # hour of day
            (i % 8760) / 8760.0,             # day of year proxy
            s["prev_action"] / (N_ACTIONS - 1.0),  # previous action
            margin / 10.0,                   # water temp safety margin
        ]
        if self.obs_version in ("V2", "V3"):
            # Forecast tail — next 1/2/3 hours of GHI + next hour's demand.
            # Driven by self.forecast_mode via forecast_ghi(k):
            #   "perfect" (default, V2's original behaviour) — true future
            #     value, zero error. Kept only as a baseline/ablation point —
            #     a real controller never has this.
            #   "smart"   — smart persistence on the clear-sky index (the
            #     realistic forecast; see forecast_ghi() docstring).
            #   "none"    — zeros, equivalent to dropping the fields.
            # obs_version "V2" and "V3" are the SAME 15-dim shape; "V3" is
            # just a naming convention for "V2 + forecast_mode='smart'" runs.
            ghi_fwd = [self.forecast_ghi(dh) / 1000.0 for dh in (1, 2, 3)]
            # next-hour demand forecast stays perfect by design (O2's fixed
            # draw-off schedule is known in advance even by a real
            # installation; see plan §12.2.4 — a modelling limitation, not a
            # bug, and domain_randomize's demand_mult already means the
            # realised draw is not perfectly predictable during training)
            hod_next = int(self._weather.iloc[min(i + 1, len(self._weather) - 1)]
                          .get("local_hour", (i + 1) % 24))
            demand_fwd = (self._demand_model.draw_mass_kg_for_hour(hod_next)
                         * self._dr["demand_mult"] / 20.0)
            return np.array(v1 + ghi_fwd + [demand_fwd], dtype=np.float32)

        return np.array(v1, dtype=np.float32)

    def forecast_ghi(self, k: int) -> float:
        """Forecast GHI (W/m2) for k hours ahead of the current tick.

        forecast_mode="perfect": the stored year's true future value (what a
        controller is NOT able to know in reality — an upper-bound ablation
        point, not a deployable forecast).

        forecast_mode="smart": smart persistence on the clear-sky index —
        GHI_hat(t+k) = Kc(t) * GHI_clear(t+k), where Kc(t) = GHI(t)/GHI_clear(t)
        is today's current cloud-attenuation ratio, clipped to [0, 1.2] and
        held at its last daytime value when GHI_clear(t) <= 20 W/m2 (so the
        forecast stays defined overnight). This keeps the deterministic,
        genuinely-known-in-advance diurnal/seasonal curve (GHI_clear) and
        treats only cloud variability as the forecast error — measured at
        9.9%/21.0%/36.1% rRMSE for h=1/2/3 on this project's own weather data.
        No training, no assumed noise parameter.

        forecast_mode="none": zero (equivalent to dropping the forecast
        fields — an ablation floor).
        """
        s = self._state
        i = s["tick_idx"] // self._ticks_per_hour
        j = i + k
        if j >= len(self._weather):
            return 0.0

        if self.forecast_mode == "none":
            return 0.0

        if self.forecast_mode == "perfect":
            return float(self._weather.iloc[j]["GHI_Wm2"]) * self._dr["ghi_mult"]

        # "smart" — clear-sky-index persistence
        ghi_now    = float(self._weather.iloc[i]["GHI_Wm2"]) * self._dr["ghi_mult"] \
                     if i < len(self._weather) else 0.0
        clear_now  = float(self._weather.iloc[i]["GHI_clearsky_Wm2"]) \
                     if i < len(self._weather) else 0.0
        clear_fwd  = float(self._weather.iloc[j]["GHI_clearsky_Wm2"])

        if clear_now > 20.0:
            kc = max(0.0, min(1.2, ghi_now / clear_now))
            s["_last_daytime_kc"] = kc
        else:
            kc = s.get("_last_daytime_kc", 1.0)

        return kc * clear_fwd

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_from_contract(self) -> None:
        self.system_config = load_system_config()
        rd = get_regime_design(self.contract, self.regime_id)

        self.mains_temp_C = rd["T_mains_est_C"]
        pcm_name = rd["selected_pcm"]

        pcm_record = get_pcm_properties("tamilnadu", pcm_name)
        half_w = self.system_config["pcm_integration"]["melting_half_width_K"]
        self.pcm_props: PCMThermalProps = pcm_props_from_record(pcm_record, half_w)

        # Build a DesignVector from the contract's selected design
        # capsule_outer_diameter_mm -> capsule_diameter_m (outer diameter for geometry)
        cap_diam_m = rd.get("capsule_outer_diameter_mm", 81.6) / 1000.0
        design_vec = DesignVector(
            capsule_diameter_m  = cap_diam_m,
            n_capsule           = rd["n_capsules"],
            flow_rate_kg_s      = rd["nominal_flow_kg_s"],
            capsule_arrangement = rd["arrangement"],
        )
        geom = check_design(design_vec, self.system_config, load_design_bounds())
        if not geom["valid"]:
            raise RuntimeError(f"Contract design failed geometry check: {geom['reason']}")

        from src.simulation.tank_model import DesignRuntime
        wall_m = self.system_config["pcm_integration"]["capsule_wall_thickness_m"]
        masses = capsule_masses(geom["n_capsule"], geom["capsule_diameter_m"], wall_m,
                                self.pcm_props.density_kg_m3)
        self._m_pcm = masses["m_pcm_kg"]
        self._m_w   = max(geom["tank_volume_m3"] - masses["v_outer_total_m3"], 1e-6) * 1000.0
        self.pcm_props = _dc_replace(self.pcm_props,
                                      wall_cp_J_kgK=wall_cp_per_kg_pcm(masses))

        self.design = DesignRuntime(
            n_capsule             = geom["n_capsule"],
            capsule_diameter_m    = geom["capsule_diameter_m"],
            capsule_area_m2       = geom["capsule_surface_area_m2"],
            capsule_volume_m3     = geom["capsule_volume_m3"],
            void_fraction         = geom["void_fraction_used_for_hydraulics"],
            cross_section_area_m2 = geom["tank_cross_section_area_m2"],
            bed_length_m          = geom.get("stack_height_m") or geom["tank_height_m"],
            flow_rate_kg_s        = geom["flow_rate_kg_s"],
            tank_volume_m3        = geom["tank_volume_m3"],
            tank_surface_area_m2  = geom["tank_surface_area_m2"],
            capsule_wall_thickness_m = wall_m,
        )

        self._superficial_v = (
            design_vec.flow_rate_kg_s
            / 1000.0
            / geom["tank_cross_section_area_m2"]
        ) if geom["tank_cross_section_area_m2"] > 0 else 0.0

    def _load_weather(self) -> None:
        # FIXED 2026-11-13: the previous implementation called
        # load_hourly_weather(), which always returns ONE fixed file
        # (weather_regime_tamilnadu_cluster{id}_hourly.csv, 2016 only, no
        # year parameter). When the requested self.year had no matching
        # rows in that file, the old code silently fell back to using the
        # whole (2016) file regardless of what year was asked for -- so
        # every "different year" (train 2016-2020, validate 2021-2023,
        # locked test 2011-2015) was actually identical 2016 data. This
        # loads the correct, year-specific, UTC-timestamp-verified file
        # directly, and raises loudly instead of silently substituting
        # wrong data if a requested year is missing.
        import pandas as pd

        regime = get_regime("tamilnadu", self.regime_id)
        default_path = BASE_DIR / regime["weather_hourly"]
        year_path = default_path.with_name(
            default_path.stem + f"_{self.year}" + default_path.suffix)
        if not year_path.exists():
            raise FileNotFoundError(
                f"No per-year weather file for regime {self.regime_id}, "
                f"year {self.year}: expected {year_path}. Refusing to "
                f"silently substitute a different year's data."
            )
        weather_year = pd.read_csv(year_path, parse_dates=["timestamp_utc"])
        if "year" in weather_year.columns:
            bad = weather_year.loc[weather_year["year"] != self.year, "year"]
            assert bad.empty, (
                f"{year_path} contains rows for year(s) {bad.unique().tolist()} "
                f"but is supposed to be {self.year}-only.")
        self._weather = drop_leap_day(weather_year).reset_index(drop=True)

        demand_df = load_demand_profile("tamilnadu")
        self._demand_model = load_demand_model(demand_df, volume_multiplier=1.0, timing_shift_hours=0.0)
