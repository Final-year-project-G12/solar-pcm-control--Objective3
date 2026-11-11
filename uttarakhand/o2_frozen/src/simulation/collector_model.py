"""
src/simulation/collector_model.py
====================================
Phase 3 / D2.3 — flat-plate collector submodel (Hottel-Whillier-Bliss form):

  Q_useful = A_c * [ F_R*(tau*alpha) * I_t  -  F_R*U_L * (T_in - T_amb) ]

Direct-tank, single-node system: the tank water temperature IS the
collector inlet temperature (no separate collector-loop node in this
40-hr MVP — documented simplification, noted as future work for a
two-node collector-tank model).

Zero/low irradiance -> zero heat, always (Gate 2 requirement): below
`min_irradiance_cutoff_Wm2` we assume the circulation pump is off, so
there is no collector-to-tank coupling at all (not even a loss term) —
this mirrors how a real differential controller stops circulation at
night to avoid reverse (tank-to-collector) heat loss.
"""


def heat_removal_factor(flow_kg_s: float, system_config: dict) -> float:
    """F_R(m) = m*cp*(1 - exp(-A*U_L*F'/(m*cp))) / (A*U_L)  (Hottel-Whillier form, Eldokaishi 2022 Eq. 2)."""
    import math
    c, fd = system_config["collector"], system_config["collector"]["flow_dependent_fr"]
    m_cp = max(float(flow_kg_s), 1e-9) * system_config["water"]["cp_J_kgK"]
    a_ul = c["area_m2"] * fd["UL_for_flow_factor_W_m2K"]
    return m_cp * (1.0 - math.exp(-a_ul * fd["efficiency_factor_Fprime"] / m_cp)) / a_ul


def apply_flow_dependent_fr(system_config: dict, flow_kg_s: float) -> dict:
    """Returns system_config unchanged when collector.flow_dependent_fr.enabled is false (the default). When enabled, returns a COPY whose
    fr_tau_alpha and fr_ul_W_m2K are scaled by F_R(flow)/F_R(reference_flow), so flow acts on collection as well as on the PCM film coefficient."""
    fd = system_config["collector"].get("flow_dependent_fr") or {}
    if not fd.get("enabled", False):
        return system_config
    import copy
    scale = heat_removal_factor(flow_kg_s, system_config) / heat_removal_factor(fd["reference_flow_kg_s"], system_config)
    out = copy.deepcopy(system_config)
    out["collector"]["fr_tau_alpha"] = system_config["collector"]["fr_tau_alpha"] * scale
    out["collector"]["fr_ul_W_m2K"] = system_config["collector"]["fr_ul_W_m2K"] * scale
    out["collector"]["flow_dependent_fr_scale"] = scale
    return out


def collector_output_w(I_t_Wm2: float, T_in_C: float, T_amb_C: float, system_config: dict):
    """Returns (Q_useful_W, is_circulating: bool)."""
    cfg = system_config["collector"]
    cutoff = cfg["min_irradiance_cutoff_Wm2"]
    if I_t_Wm2 <= cutoff:
        return 0.0, False

    area = cfg["area_m2"]
    fr_tau_alpha = cfg["fr_tau_alpha"]
    fr_ul = cfg["fr_ul_W_m2K"]

    q = area * (fr_tau_alpha * I_t_Wm2 - fr_ul * (T_in_C - T_amb_C))
    q = max(q, 0.0)   # collector will not run in reverse (no heat extraction from tank)
    return q, True


def collector_linear_coeffs(I_t_Wm2: float, T_amb_C: float, system_config: dict):
    """For the implicit water-node solve in tank_model.py we need
    Q_useful expressed as (a - b*T_w_new) so it can be folded into one
    linear equation for T_w_new. Returns (a_W, b_W_per_K, is_circulating).

        Q_useful = a - b*T_w_new
        a = A_c*(FR_tau_alpha*I_t + FR_UL*T_amb)      [W]
        b = A_c*FR_UL                                  [W/K]

    Clipping Q_useful>=0 is enforced by the caller after solving, consistent
    with collector_output_w() above.
    """
    cfg = system_config["collector"]
    cutoff = cfg["min_irradiance_cutoff_Wm2"]
    if I_t_Wm2 <= cutoff:
        return 0.0, 0.0, False

    area = cfg["area_m2"]
    fr_tau_alpha = cfg["fr_tau_alpha"]
    fr_ul = cfg["fr_ul_W_m2K"]

    a = area * (fr_tau_alpha * I_t_Wm2 + fr_ul * T_amb_C)
    b = area * fr_ul
    return a, b, True
