"""
src/simulation/hydraulic_model.py
====================================
Phase 3 / D2.3 — thin runtime wrapper around src/design/geometry.py's Ergun
pressure-drop calculation, used inside the timestep loop to report pump
power separately from thermal energy (framework doc Sec 4.6):

  P_pump = dP * V_dot / eta_pump,   integrated over time -> E_pump.

Kept separate from geometry.py because geometry.py is called once per
design (Phase 2, before any simulation) while this module is called once
PER TIMESTEP by tank_model.py during Phase 3 simulation — same physics,
different call site, so this module is a deliberately thin re-export
rather than a duplicate implementation.
"""

from src.design.geometry import compute_hydraulics


def pump_power_w(mdot_kg_s: float, capsule_diameter_m: float, void_fraction: float,
                  cross_section_area_m2: float, bed_length_m: float,
                  system_config: dict) -> float:
    h = compute_hydraulics(mdot_kg_s, capsule_diameter_m, void_fraction,
                            cross_section_area_m2, bed_length_m, system_config)
    return h["pump_power_w"]


def pressure_drop_and_pump_power(mdot_kg_s: float, capsule_diameter_m: float, void_fraction: float,
                                  cross_section_area_m2: float, bed_length_m: float,
                                  system_config: dict) -> dict:
    return compute_hydraulics(mdot_kg_s, capsule_diameter_m, void_fraction,
                               cross_section_area_m2, bed_length_m, system_config)
