"""
src/simulation/capsule_mass.py
=================================
O2 G2 (2026-10-26) — PCM and capsule-wall masses for a shell of outer
diameter D and wall thickness t_w.

The PCM fills only the INNER sphere, diameter D - 2*t_w; the aluminium wall is
the remaining shell. The old code multiplied the OUTER sphere volume by the PCM
density, overstating PCM mass by 1-((D-2t)/D)^3 (11.5 % at D=40 mm, 5.9 % at
80 mm for t=0.8 mm) and ignoring the wall's thermal mass. Water displacement
still uses the OUTER volume (the whole capsule occupies tank volume).

Aluminium density/specific heat are standard handbook values (2700 kg/m3,
900 J/kg.K) and are NOT in the frozen system_config_shared.yaml, which holds only
the wall material name, thickness and conductivity.
"""

import math

AL_DENSITY_KG_M3 = 2700.0
AL_CP_J_KGK = 900.0


def _sphere_volume(d: float) -> float:
    return math.pi / 6.0 * d ** 3


def capsule_masses(n_capsule: int, d_outer_m: float, wall_m: float, pcm_density_kg_m3: float,
                   wall_density_kg_m3: float = AL_DENSITY_KG_M3) -> dict:
    d_inner = d_outer_m - 2.0 * wall_m
    if d_inner <= 0:
        raise ValueError(f"capsule wall {wall_m} m leaves no PCM volume in a {d_outer_m} m sphere")
    v_out, v_in = _sphere_volume(d_outer_m), _sphere_volume(d_inner)
    return {
        "v_outer_total_m3": n_capsule * v_out,
        "v_pcm_total_m3": n_capsule * v_in,
        "m_pcm_kg": n_capsule * v_in * pcm_density_kg_m3,
        "m_wall_kg": n_capsule * (v_out - v_in) * wall_density_kg_m3,
    }


def wall_cp_per_kg_pcm(masses: dict, wall_cp_J_kgK: float = AL_CP_J_KGK) -> float:
    """Wall heat capacity expressed per kg of PCM (PCMThermalProps.wall_cp_J_kgK)."""
    return masses["m_wall_kg"] * wall_cp_J_kgK / masses["m_pcm_kg"] if masses["m_pcm_kg"] > 0 else 0.0
