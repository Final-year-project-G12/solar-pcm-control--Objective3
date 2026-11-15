"""
src/simulation/tank_model.py
===============================
Phase 3 / D2.3 — the grey-box enthalpy simulator core. Combines the
collector, capsule-enthalpy, heat-transfer, hydraulic and demand submodels
into one backward-Euler timestepping loop over an hourly weather series.

SOLVER (documented choice — "backward_euler_lagged_pcm"):
Water-side energy balance

    m_w*cp_w*(Tw_new-Tw_old)/dt = Q_collector(Tw_new) - Q_load(Tw_new)
                                    - Q_pcm(Tw_new, Tpcm_OLD) - Q_loss(Tw_new)

is linear in Tw_new once the PCM temperature is evaluated at its OLD
(start-of-substep) value — every other term (collector, load, tank loss)
is itself linear in Tw_new, so this is solved in closed form each
sub-step (no Newton iteration needed) while still being fully implicit in
the water node, which is the stiffest part of the system (large UA, small
water mass). The PCM temperature is then updated explicitly using the
just-solved Tw_new (semi-implicit / IMEX coupling). This is stable for the
timesteps used here (5 min nominal, sub-stepped near the melt band) and is
the practical backward-Euler variant referenced in the framework docs for
this project (Barqawi 2025, Sec 4c: "backward Euler + adaptive
sub-stepping ... stable for phase change").

SIGN CONVENTION (must never silently flip — framework doc Sec 4.4):
  Q_pcm > 0  =>  heat flowing INTO the PCM (charging).
  Q_pcm < 0  =>  heat flowing OUT of the PCM into the water (discharging).

AMBIENT TANK LOSS IS NON-NEGOTIABLE (Bug-Fix 1): Q_loss = U_tank*A_tank*
(Tw-Tamb) is evaluated every single sub-step, with no code path that skips
it. Do not add one.

RULE-BASED SAFETY SHIELD (O2 Fix 2, 2026-09-13; RECONSTRUCTED 2026-09-13
after an unrelated git pull discarded the uncommitted original — see
docs/09_LIMITATIONS_AND_KNOWN_DIVERGENCES.md §6/§7 for the full history).
Previously the simulator only *recorded* temperature violations
(n_safety_violations) without ever preventing them, meaning the
"deployable design" Objective 2 handed to Objective 3 was unsafe ~45-57%
of the time at baseline (Phase 8 Monte Carlo). This implements, inside
run_year() itself, the rule-based shield already specified for Objective 3
in results/obj3_environment_contract_rajasthan.json ("safety_shield":
bypass at 72 C water / 62 C PCM, 3 C guard band below the 75/65 C hard
limits). Controlled by system_config["safety_shield"] (keys: enabled,
bypass_water_C, bypass_pcm_C) -- ADOPTED AS THE PIPELINE DEFAULT
(enabled: true) in system_config_shared.yaml as of 2026-09-13, so every
Phase 5-8 result reflects it; system_config_overrides={"safety_shield":
{"enabled": False}} remains available as a diagnostic-only escape hatch.

Two-part mechanism, each substep:
  1. WATER-SIDE (bypass): if T_water >= bypass_water_C, force
     circulating=False before that substep's collector solve -- stops the
     pump from pushing more collector-heated water into the tank. This is
     exactly IS 12976:2023 Sec 8.2's stated method for overheat protection
     ("stop circulation in the collector loop until the storage
     temperature decreases") -- see docs/09 §7.
  2. PCM-SIDE (charge-only block): if PCM is present and T_pcm >=
     bypass_pcm_C, block CHARGING ONLY -- ua_eff is zeroed only while the
     water-to-PCM heat flow direction would be charging (T_w > T_pcm,
     i.e. Q_pcm > 0 per the sign convention above), never while it would
     be discharging (T_pcm > T_w). Zeroing ua_eff unconditionally would
     also block the safe, beneficial PCM-to-water discharge that cools an
     over-temperature PCM back down -- only the charging direction is the
     actually dangerous one this shield exists to stop. Uses T_w/T_pcm as
     they stand at the START of the substep (pre-solve) to decide
     direction, consistent with the rest of this substep's use of
     start-of-substep state for the linear solve.

MODELING SIMPLIFICATION (documented, not hidden — same convention as the
other simplifications above): "bypass" here means the pump is commanded
off (circulating=False), not a literal 3-way diverter valve routing flow
away from the collector — a reasonable domestic-SWH proxy for the
mechanism IS 12976 Sec 8.2 describes, not a component-level valve model.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.simulation.capsule_enthalpy import (
    PCMThermalProps, enthalpy_of_temperature, temperature_and_liquid_fraction_of_enthalpy,
)
from src.simulation.collector_model import collector_linear_coeffs
from src.simulation.heat_transfer import ua_eff_total_w_k
from src.simulation.hydraulic_model import pressure_drop_and_pump_power
from src.simulation.energy_balance import EnergyAccumulator
from src.simulation.demand_profile import DemandModel
from src.simulation.capsule_mass import capsule_masses, wall_cp_per_kg_pcm
from dataclasses import replace as _dc_replace


J_PER_KWH_ = 3.6e6


def _demand_hour_of_day(row, i: int) -> int:
    """Local clock hour (0-23) used to index the demand profile.

    The demand profile is defined in local clock hours (07:00 / 19:00 peaks,
    build_demand_profile.py), so it must be indexed by the weather file's
    `local_hour` column, never by the UTC hour of `timestamp_utc` (2026-10-26
    fix: the UTC index put the "evening" draw at ~00:30 local). A frame with
    a UTC timestamp but no `local_hour` is refused rather than silently
    falling back to UTC. A frame with neither (synthetic gate inputs) keeps
    the legacy row-index fallback.
    """
    if "local_hour" in row:
        return int(row["local_hour"])
    if "timestamp_utc" in row:
        raise ValueError("weather frame has timestamp_utc but no local_hour; "
                         "refusing to index the local-time demand profile by UTC hour")
    return i % 24


@dataclass
class DesignRuntime:
    """Constant-for-the-run quantities derived once from the geometry record
    (src/design/geometry.compute_geometry output) — avoids recomputing
    geometry inside the hot timestep loop."""
    n_capsule: int
    capsule_diameter_m: float
    capsule_area_m2: float
    capsule_volume_m3: float
    void_fraction: float
    cross_section_area_m2: float
    bed_length_m: float
    flow_rate_kg_s: float
    tank_volume_m3: float
    tank_surface_area_m2: float
    capsule_wall_thickness_m: float = None   # None -> system_config pcm_integration.capsule_wall_thickness_m


@dataclass
class SimulationResult:
    hourly: pd.DataFrame
    energy: dict
    max_water_temp_C: float
    max_pcm_temp_C: float
    n_safety_violations: int
    final_f_melt: float
    complete_melt_cycles: int
    initial_f_melt: float = 0.0
    n_clipped_steps: int = 0
    n_shield_water_activations: int = 0
    n_shield_pcm_activations: int = 0
    summary: dict = field(default_factory=dict)   # always-on per-run outputs (no hourly recording needed)
    context: dict = field(default_factory=dict)   # masses / effective PCM props, for src/metrics


def run_year(weather_hourly: pd.DataFrame, demand: DemandModel, mains_temp_C: float,
             design: DesignRuntime, pcm_props: PCMThermalProps, system_config: dict,
             record_hourly: bool = True, initial_state: dict = None,
             stop_when_Tw_below_C: float = None) -> SimulationResult:
    """Runs the simulator across every hour in weather_hourly (repeats the
    24-hour demand curve every day). Returns per-hour summaries + the
    annual energy accumulator.

    initial_state: optional {"T_w_C", "T_pcm_C"} overriding the cold-start
    defaults (used by the WP1 night scenarios, which restart from a state taken
    out of an annual run). stop_when_Tw_below_C: end the run after the first
    hour whose end-of-hour water temperature is below this value (used by the
    no-draw hold test)."""
    solver = system_config["solver"]
    dt_total_s = solver["timestep_s"]
    n_substeps_per_hour = int(round(3600 / dt_total_s))
    n_sub_in_band = solver["substeps_in_melt_band"]
    band_margin = solver["melt_band_margin_K"]

    water_cfg = system_config["water"]
    cp_w = water_cfg["cp_J_kgK"]
    rho_w = water_cfg["density_kg_m3"]

    u_tank = system_config["tank"]["U_tank_W_m2K"]
    a_tank = design.tank_surface_area_m2

    delivery_target_C = system_config["delivery"]["target_temp_C"]
    max_water_C = system_config["safety"]["max_water_temp_C"]
    max_pcm_C = system_config["safety"]["max_pcm_temp_C"]

    # O2 Fix 2 — rule-based safety shield (see module docstring). Default
    # thresholds derive from the frozen safety maxima above with the same
    # 3.0 C guard band documented in obj3_environment_contract_rajasthan.json,
    # so no edit to the frozen system_config_shared.yaml is needed to use it.
    shield_cfg = system_config.get("safety_shield", {})
    shield_enabled = bool(shield_cfg.get("enabled", False))
    bypass_water_C = shield_cfg.get("bypass_water_C", max_water_C - 3.0)
    bypass_pcm_C = shield_cfg.get("bypass_pcm_C", max_pcm_C - 3.0)
    n_shield_water = 0
    n_shield_pcm = 0

    has_pcm = design.n_capsule > 0
    # G2: PCM fills the INNER sphere only; the wall's heat capacity rides in the
    # PCM node (wall_cp); displaced water uses the OUTER capsule volume.
    if has_pcm:
        wall_m = (design.capsule_wall_thickness_m if design.capsule_wall_thickness_m is not None
                  else system_config["pcm_integration"]["capsule_wall_thickness_m"])
        masses = capsule_masses(design.n_capsule, design.capsule_diameter_m, wall_m, pcm_props.density_kg_m3)
        m_pcm_total_kg = masses["m_pcm_kg"]
        v_displaced_m3 = masses["v_outer_total_m3"]
        pcm_props = _dc_replace(pcm_props, wall_cp_J_kgK=wall_cp_per_kg_pcm(masses))
    else:
        m_pcm_total_kg, v_displaced_m3 = 0.0, 0.0
    m_w = max(design.tank_volume_m3 - v_displaced_m3, 1e-6) * rho_w

    superficial_velocity = (design.flow_rate_kg_s / rho_w / design.cross_section_area_m2
                             if design.cross_section_area_m2 > 0 else 0.0)

    # ---- initial state ---------------------------------------------------
    T_w = initial_state["T_w_C"] if initial_state else solver["initial_water_temp_C"]
    if has_pcm:
        init_state = solver["initial_pcm_state"]
        if initial_state and "T_pcm_C" in initial_state:
            T_pcm_init = initial_state["T_pcm_C"]
        elif init_state == "solid":
            T_pcm_init = pcm_props.Ts_C - 5.0
        elif init_state == "liquid":
            T_pcm_init = pcm_props.Tl_C + 5.0
        else:
            T_pcm_init = pcm_props.Tm_C
        h_pcm = enthalpy_of_temperature(T_pcm_init, pcm_props)
        T_pcm, f_melt = temperature_and_liquid_fraction_of_enthalpy(h_pcm, pcm_props)
    else:
        h_pcm, T_pcm, f_melt = 0.0, T_w, 0.0

    acc = EnergyAccumulator()
    acc.E_initial_J = m_w * cp_w * T_w + (m_pcm_total_kg * h_pcm if has_pcm else 0.0)
    initial_f_melt = f_melt

    max_water_temp, max_pcm_temp = T_w, T_pcm
    n_safety = 0
    n_clipped_steps = 0
    last_f_melt, complete_cycles = f_melt, 0
    was_fully_liquid = f_melt >= 0.999

    cp_min_J_kgK = (min(pcm_props.cp_solid_J_kgK, pcm_props.cp_liquid_J_kgK) + pcm_props.wall_cp_J_kgK) if has_pcm else None
    STIFFNESS_SAFETY_FACTOR = 0.5   # require dt_sub <= 0.5 * (PCM thermal time constant)
    MAX_SUBSTEPS = 60               # hard cap: below this, cases are flagged via n_clipped_steps

    rows = []
    n_hours = len(weather_hourly)
    # G2: always-on per-run outputs, so DOE/optimiser runs (record_hourly=False)
    # carry the same summary fields as the recorded ones. Definitions match the
    # hourly frame: end-of-hour state.
    sum_f_melt, min_f_melt, max_f_melt = 0.0, 1.0, 0.0
    n_delivery_hours = 0
    sum_abs_df = 0.0                      # sum |change in liquid fraction| over every sub-step
    tot_draw_kg = tot_vol_ge_target_kg = tot_edel_target_J = 0.0
    # stored energy relative to the mains temperature (WP1 definition A)
    h_ref_mains = enthalpy_of_temperature(mains_temp_C, pcm_props) if has_pcm else 0.0
    def _e_stored_J(Tw, h):
        return m_w * cp_w * (Tw - mains_temp_C) + (m_pcm_total_kg * (h - h_ref_mains) if has_pcm else 0.0)
    e_stored_initial_J = _e_stored_J(T_w, h_pcm)

    n_done = 0
    for i in range(n_hours):
        n_done = i + 1
        row = weather_hourly.iloc[i]
        I_t = float(row["GHI_Wm2"])
        T_amb = float(row["T_amb_C"])
        hour_of_day = _demand_hour_of_day(row, i)

        draw_mass_hour_kg = demand.draw_mass_kg_for_hour(hour_of_day)
        draw_mass_per_substep_kg = draw_mass_hour_kg / n_substeps_per_hour

        hour_q_collector = hour_q_load = hour_q_pcm = hour_q_loss = 0.0
        hour_q_pump = hour_q_unmet = 0.0
        hour_draw = hour_vol_ge_target = hour_edel_target = 0.0
        hour_tw_min, hour_tw_max, hour_tpcm_max = float("inf"), float("-inf"), float("-inf")
        hour_f_min, hour_f_max = float("inf"), float("-inf")

        for sub in range(n_substeps_per_hour):
            in_band = has_pcm and (pcm_props.Ts_C - band_margin <= T_pcm <= pcm_props.Tl_C + band_margin)
            n_sub = n_sub_in_band if in_band else 1

            if has_pcm:
                # Adaptive stiffness check: the PCM node is coupled to the
                # water node via a one-substep-lagged explicit term, so if
                # the PCM's own thermal time constant (m*cp/UA) is shorter
                # than ~2x the candidate sub-step, that lag makes the scheme
                # oscillate and diverge (this is exactly what a very-high
                # conductivity capsule triggered before this fix). Estimate
                # UA at the current state and shrink dt_sub until it is
                # safely smaller than the time constant.
                ua_probe = ua_eff_total_w_k(design.n_capsule, design.capsule_diameter_m,
                                            design.capsule_area_m2, superficial_velocity,
                                            pcm_props, f_melt, system_config)
                if ua_probe > 0:
                    tau_pcm_s = m_pcm_total_kg * cp_min_J_kgK / ua_probe
                    n_sub_stiff = max(1, int(np.ceil(dt_total_s / (STIFFNESS_SAFETY_FACTOR * tau_pcm_s))))
                    n_sub = min(max(n_sub, n_sub_stiff), MAX_SUBSTEPS)

            dt_sub = dt_total_s / n_sub
            draw_mass_this_substep_total = draw_mass_per_substep_kg

            for _ in range(n_sub):
                draw_mass = draw_mass_this_substep_total / n_sub
                m_draw_rate = draw_mass / dt_sub if dt_sub > 0 else 0.0

                a_coll, b_coll, circulating = collector_linear_coeffs(I_t, T_amb, system_config)

                # O2 Fix 2 — water-side shield: stop the pump (no further
                # solar charging) once T_w reaches the bypass threshold.
                # Uses T_w (start-of-substep, pre-solve) per the module
                # docstring's "safety shield" note.
                if shield_enabled and T_w >= bypass_water_C:
                    circulating = False
                    n_shield_water += 1

                if has_pcm:
                    ua_eff = ua_eff_total_w_k(design.n_capsule, design.capsule_diameter_m,
                                               design.capsule_area_m2, superficial_velocity,
                                               pcm_props, f_melt, system_config)
                    # O2 Fix 2 — PCM-side shield: charge-only. Block further
                    # charging (T_w > T_pcm, i.e. Q_pcm > 0) once T_pcm hits
                    # the bypass threshold, but keep ua_eff active when the
                    # flow would be discharging (T_pcm > T_w) — that
                    # direction cools the PCM and is safe. See module
                    # docstring "PCM-SIDE (charge-only block)".
                    if shield_enabled and T_pcm >= bypass_pcm_C and T_w > T_pcm:
                        ua_eff = 0.0
                        n_shield_pcm += 1
                else:
                    ua_eff = 0.0

                base_coeff = (m_w * cp_w / dt_sub) + m_draw_rate * cp_w + ua_eff + u_tank * a_tank
                base_rhs = (m_w * cp_w * T_w / dt_sub + m_draw_rate * cp_w * mains_temp_C
                            + ua_eff * T_pcm + u_tank * a_tank * T_amb)

                # Differential-controller re-solve: a real SWH pump stops
                # circulating the instant the collector can no longer add net
                # heat (Tw at/above the collector's stagnation temperature),
                # otherwise the implicit solve would let the collector run in
                # reverse (extracting heat) while the accounting below still
                # clips logged Q_collector to >=0 -- silently breaking energy
                # conservation (this is exactly what produced a ~1.6% Gate-1
                # residual before this fix). Solving once assuming circulation,
                # then re-solving with the collector off if that would imply
                # reverse flow, keeps the log and the physics consistent.
                if circulating:
                    coeff = base_coeff + b_coll
                    rhs = base_rhs + a_coll
                    T_w_trial = rhs / coeff if coeff > 0 else T_w
                    if a_coll - b_coll * T_w_trial < 0:
                        circulating = False
                    else:
                        T_w_new = T_w_trial
                if not circulating:
                    coeff = base_coeff
                    rhs = base_rhs
                    T_w_new = rhs / coeff if coeff > 0 else T_w

                q_collector = max(a_coll - b_coll * T_w_new, 0.0) if circulating else 0.0
                q_load = max(m_draw_rate * cp_w * (T_w_new - mains_temp_C), 0.0)
                q_unmet = (m_draw_rate * cp_w * max(delivery_target_C - T_w_new, 0.0)
                           if m_draw_rate > 0 else 0.0)
                q_pcm = ua_eff * (T_w_new - T_pcm) if has_pcm else 0.0
                q_loss = u_tank * a_tank * (T_w_new - T_amb)   # ALWAYS active — Bug-Fix 1

                hyd = pressure_drop_and_pump_power(
                    design.flow_rate_kg_s, design.capsule_diameter_m, design.void_fraction,
                    design.cross_section_area_m2, design.bed_length_m, system_config,
                ) if circulating else {"pump_power_w": 0.0}
                q_pump = hyd["pump_power_w"]

                if has_pcm:
                    dh = q_pcm * dt_sub / m_pcm_total_kg
                    h_pcm_new = h_pcm + dh
                    T_pcm_new, f_melt_new = temperature_and_liquid_fraction_of_enthalpy(h_pcm_new, pcm_props)
                    # Numerical backstop: even with adaptive sub-stepping, an
                    # extreme/unphysical property combination should never be
                    # allowed to silently propagate NaN/inf through the rest
                    # of the run. Clip to a generous physical range and count
                    # it as a failed step (surfaced in the Gate-1/energy
                    # report) rather than letting the case look falsely clean.
                    if not (-50.0 <= T_pcm_new <= 500.0):
                        T_pcm_new = min(max(T_pcm_new, -50.0), 500.0)
                        h_pcm_new = enthalpy_of_temperature(T_pcm_new, pcm_props)
                        n_clipped_steps += 1
                else:
                    h_pcm_new, T_pcm_new, f_melt_new = 0.0, T_w_new, 0.0

                acc.add_step(q_collector, q_load, q_pcm, q_loss, q_pump, q_unmet, dt_sub)
                hour_q_collector += q_collector * dt_sub
                hour_q_load += q_load * dt_sub
                hour_q_pcm += q_pcm * dt_sub
                hour_q_loss += q_loss * dt_sub
                hour_q_pump += q_pump * dt_sub
                hour_q_unmet += q_unmet * dt_sub

                if T_w_new > max_water_C or (has_pcm and T_pcm_new > max_pcm_C):
                    n_safety += 1

                if draw_mass > 0:
                    hour_draw += draw_mass
                    hour_vol_ge_target += draw_mass if T_w_new >= delivery_target_C else 0.0
                    hour_edel_target += draw_mass * cp_w * max(min(T_w_new, delivery_target_C) - mains_temp_C, 0.0)
                sum_abs_df += abs(f_melt_new - f_melt)
                T_w, T_pcm, h_pcm, f_melt = T_w_new, T_pcm_new, h_pcm_new, f_melt_new
                hour_tw_min, hour_tw_max = min(hour_tw_min, T_w), max(hour_tw_max, T_w)
                hour_tpcm_max = max(hour_tpcm_max, T_pcm)
                hour_f_min, hour_f_max = min(hour_f_min, f_melt), max(hour_f_max, f_melt)
                max_water_temp = max(max_water_temp, T_w)
                max_pcm_temp = max(max_pcm_temp, T_pcm)

        if has_pcm:
            if f_melt >= 0.999 and not was_fully_liquid:
                was_fully_liquid = True
            if f_melt <= 0.001 and was_fully_liquid:
                complete_cycles += 1
                was_fully_liquid = False

        tot_draw_kg += hour_draw
        tot_vol_ge_target_kg += hour_vol_ge_target
        tot_edel_target_J += hour_edel_target
        sum_f_melt += f_melt
        min_f_melt, max_f_melt = min(min_f_melt, f_melt), max(max_f_melt, f_melt)
        n_delivery_hours += int(T_w >= delivery_target_C)

        if record_hourly:
            rows.append({
                "hour_index": i, "hour_of_day": hour_of_day, "I_t_Wm2": I_t, "T_amb_C": T_amb,
                "T_w_C": T_w, "T_pcm_C": T_pcm, "f_melt": f_melt,
                "Q_collector_Wh": hour_q_collector / 3600, "Q_load_Wh": hour_q_load / 3600,
                "Q_pcm_Wh": hour_q_pcm / 3600, "Q_loss_Wh": hour_q_loss / 3600,
                "Q_pump_Wh": hour_q_pump / 3600, "Q_unmet_Wh": hour_q_unmet / 3600,
                "draw_kg": hour_draw, "vol_ge_target_kg": hour_vol_ge_target,
                "Edel_target_Wh": hour_edel_target / 3600,
                "T_w_min_C": hour_tw_min, "T_w_max_C": hour_tw_max, "T_pcm_max_C": hour_tpcm_max,
                "f_melt_min": hour_f_min, "f_melt_max": hour_f_max,
            })

        if stop_when_Tw_below_C is not None and T_w < stop_when_Tw_below_C:
            break

    e_stored_final_J = _e_stored_J(T_w, h_pcm)
    acc.E_final_J = m_w * cp_w * T_w + (m_pcm_total_kg * h_pcm if has_pcm else 0.0)
    acc.n_failed_steps = n_clipped_steps

    hourly_df = pd.DataFrame(rows) if record_hourly else pd.DataFrame()
    energy = acc.residual_report()

    return SimulationResult(
        hourly=hourly_df, energy=energy,
        max_water_temp_C=max_water_temp, max_pcm_temp_C=max_pcm_temp,
        n_safety_violations=n_safety, final_f_melt=f_melt,
        complete_melt_cycles=complete_cycles,
        initial_f_melt=initial_f_melt, n_clipped_steps=n_clipped_steps,
        n_shield_water_activations=n_shield_water, n_shield_pcm_activations=n_shield_pcm,
        summary={
            "mean_f_melt": sum_f_melt / n_done if n_done else 0.0,
            "min_f_melt": min_f_melt, "max_f_melt": max_f_melt,
            "delivery_temp_hours": n_delivery_hours,
            "m_pcm_kg": m_pcm_total_kg,
            "m_wall_kg": masses["m_wall_kg"] if has_pcm else 0.0,
            "m_water_kg": m_w,
            "sum_abs_df": sum_abs_df,
            "E_del_target_kWh": tot_edel_target_J / J_PER_KWH_,
            "draw_kg_total": tot_draw_kg, "vol_ge_target_kg": tot_vol_ge_target_kg,
            "E_stored_initial_kWh": e_stored_initial_J / J_PER_KWH_,
            "E_stored_final_kWh": e_stored_final_J / J_PER_KWH_,
        },
        context={"m_w_kg": m_w, "m_pcm_kg": m_pcm_total_kg, "pcm_props": pcm_props if has_pcm else None,
                 "cp_w": cp_w, "mains_temp_C": mains_temp_C, "target_C": delivery_target_C,
                 "has_pcm": has_pcm, "dt_s": dt_total_s,
                 "ua_tank_W_K": u_tank * a_tank},
    )
