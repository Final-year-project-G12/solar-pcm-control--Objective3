"""
src/simulation/heat_transfer.py
==================================
Phase 3 / D2.3 — effective conductance UA_eff between the tank water node
and one capsule group, per Sec 4.5 of the Objective 2 workflow doc:

  1/UA_eff = 1/(h_w*A_w) + R_wall + R_pcm_eff

  h_w      : water-side convection, Wakao & Kaguei (1982) packed-bed
             correlation Nu = 2 + 1.1*Re^0.6*Pr^(1/3) — STANDARD correlation,
             not derived from scratch (framework doc requirement).
  R_wall   : capsule aluminium shell conduction resistance (thin-wall).
  R_pcm_eff: lumped internal PCM conduction resistance, r/(5*k_eff) — the
             common approximation for a sphere with an internally
             distributed thermal load; k_eff includes the documented
             liquid-natural-convection enhancement factor once f_melt>=0.5
             (capsule_enthalpy.effective_conductivity_W_mK).

All capsules in the (single) thermal group are identical and see the same
bulk water temperature (lumped-tank assumption) -> N_capsule identical
resistances IN PARALLEL:  UA_eff_total = N_capsule / R_per_capsule.

Which terms are measured / correlated / assumed (framework doc Sec 4.5):
  - h_w: CORRELATED (Wakao-Kaguei, standard packed-bed practice).
  - R_wall: CORRELATED from manufacturer wall thickness + aluminium
    conductivity (both frozen in system_config_shared.yaml).
  - R_pcm_eff: ASSUMED lumped 1/5 sphere-conduction approximation with an
    ASSUMED liquid-convection enhancement factor (no CFD/experimental
    validation performed in Objective 2 — documented limitation).
"""

import math

from src.simulation.capsule_enthalpy import PCMThermalProps, effective_conductivity_W_mK


def water_side_htc_w_m2k(superficial_velocity_m_s: float, capsule_diameter_m: float,
                          system_config: dict) -> float:
    """Wakao-Kaguei packed-bed Nusselt correlation -> h_w (W/m^2.K)."""
    water = system_config["water"]
    rho = water["density_kg_m3"]
    k_w = water["conductivity_W_mK"]
    pr = water["prandtl"]

    mu = _water_dynamic_viscosity_pa_s()
    re = rho * superficial_velocity_m_s * capsule_diameter_m / mu if mu > 0 else 0.0
    re = max(re, 1e-6)

    nu = 2.0 + 1.1 * re ** 0.6 * pr ** (1.0 / 3.0)
    h_w = nu * k_w / capsule_diameter_m
    return h_w


def _water_dynamic_viscosity_pa_s(t_c: float = 40.0) -> float:
    a, b, c = 2.414e-5, 247.8, 140.0
    t_k = t_c + 273.15
    return a * 10 ** (b / (t_k - c))


def ua_eff_total_w_k(n_capsule: int, capsule_diameter_m: float, capsule_area_m2: float,
                      superficial_velocity_m_s: float, pcm_props: PCMThermalProps,
                      f_melt: float, system_config: dict) -> float:
    """Total (all capsules, parallel) effective conductance in W/K."""
    if n_capsule <= 0:
        return 0.0

    integ = system_config["pcm_integration"]
    wall_thickness = integ["capsule_wall_thickness_m"]
    wall_k = integ["capsule_wall_conductivity_W_mK"]
    enhancement = integ["liquid_convection_enhancement"]
    # OFF-BY-DEFAULT closure option (O2 CFD recalibration, 2026-10-29): c in R_pcm = r/(c k_eff). Absent from the frozen config -> 5.0,
    # the value used by every result so far. Set only through system_config_overrides (pcm_integration.conduction_coefficient_c).
    c_cond = float(integ.get("conduction_coefficient_c", 5.0))

    h_w = water_side_htc_w_m2k(superficial_velocity_m_s, capsule_diameter_m, system_config)
    r_conv = 1.0 / (h_w * capsule_area_m2) if h_w > 0 else float("inf")

    r_wall = wall_thickness / (wall_k * capsule_area_m2)

    k_eff = effective_conductivity_W_mK(pcm_props, f_melt, enhancement)
    # O2 G7 (decision D3, thickness = inner radius): the PCM conduction path is the INNER sphere,
    # radius R_in = D/2 - t_wall, with the lumped resistance referred to the inner surface. Before this the
    # outer radius and outer area were used (an 8 % overstatement of the path at R_in = 10 mm).
    r_in = capsule_diameter_m / 2.0 - wall_thickness
    a_in = 4.0 * math.pi * r_in ** 2
    r_pcm = r_in / (c_cond * k_eff * a_in) if k_eff > 0 else float("inf")

    r_total_per_capsule = r_conv + r_wall + r_pcm
    ua_per_capsule = 1.0 / r_total_per_capsule if r_total_per_capsule > 0 else 0.0
    return n_capsule * ua_per_capsule
