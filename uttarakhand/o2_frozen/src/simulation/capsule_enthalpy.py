"""
src/simulation/capsule_enthalpy.py
=====================================
Phase 3 / D2.3 — PCM enthalpy model with clipped liquid fraction.

h(T) =  Cp_solid*(T-Tref)                                   T  < Ts
        h_s + f*L                                            Ts <= T <= Tl
        h_s + L + Cp_liquid*(T-Tl)                            T  > Tl

  f = clip((h - h_s) / L, 0, 1)

The PCM database (data/objective1/pcm_database_tamilnadu.csv) reports a
single melting point Tm_C, not a measured (solidus, liquidus) interval,
so we treat melting as a narrow band Tm +/- melting_half_width_K
(system_config_shared.yaml, pcm_integration.melting_half_width_K) — a
documented simplification, not a measured property.

CAPSULE WALL (O2 G2, 2026-10-26): the aluminium shell shares the PCM
node's temperature, so its sensible heat is carried in the same node as an
extra specific heat per kg of PCM, `wall_cp_J_kgK` = m_wall*c_wall/m_pcm
(0.0 = no wall, the legacy behaviour). The enthalpy is then
h(T) + wall_cp*(T-Tref) everywhere, which stays exactly invertible: the
mushy zone just gains wall_cp*(Tl-Ts) of extra enthalpy per kg of PCM.

Reference temperature Tref = 0 C throughout (arbitrary; only enthalpy
DIFFERENCES matter for the energy balance, so the choice of Tref cancels
out of every residual check).
"""

from dataclasses import dataclass


TREF_C = 0.0


@dataclass(frozen=True)
class PCMThermalProps:
    Tm_C: float
    latent_heat_J_kg: float          # converted from kJ/kg at the call site
    cp_solid_J_kgK: float
    cp_liquid_J_kgK: float
    conductivity_W_mK: float
    density_kg_m3: float             # solid-basis, used for fixed capsule mass
    melting_half_width_K: float = 1.0
    wall_cp_J_kgK: float = 0.0       # capsule-wall heat capacity per kg PCM (see module docstring)

    @property
    def Ts_C(self):
        return self.Tm_C - self.melting_half_width_K

    @property
    def Tl_C(self):
        return self.Tm_C + self.melting_half_width_K

    @property
    def h_s_J_kg(self):
        """Enthalpy at the solidus temperature (start of melting)."""
        return (self.cp_solid_J_kgK + self.wall_cp_J_kgK) * (self.Ts_C - TREF_C)

    @property
    def melt_span_enthalpy_J_kg(self):
        """Enthalpy absorbed across the whole melting band: latent heat plus
        the wall's sensible heat over (Tl - Ts)."""
        return self.latent_heat_J_kg + self.wall_cp_J_kgK * (self.Tl_C - self.Ts_C)


def pcm_props_from_record(record: dict, melting_half_width_K: float = 1.0) -> PCMThermalProps:
    """Builds PCMThermalProps from one row of pcm_database_<state>.csv.
    Cp_solid/Cp_liquid are reported in kJ/kg.K in the database; some
    literature-only rows only report Cp_avg (not split solid/liquid) — if so
    both branches use the same value (documented, not measured separately)."""
    cp_liq = record.get("Cp_liquid_kJ_kgK")
    cp_sol = record.get("Cp_solid_kJ_kgK")
    if cp_liq is None or (isinstance(cp_liq, float) and cp_liq != cp_liq):  # NaN check
        cp_liq = record.get("Cp_avg_kJ_kgK", 2.0)
    if cp_sol is None or (isinstance(cp_sol, float) and cp_sol != cp_sol):
        cp_sol = record.get("Cp_avg_kJ_kgK", 2.0)

    density = record.get("density_solid_kg_m3")
    if density is None or (isinstance(density, float) and density != density):
        density = record.get("density_liquid_kg_m3", 800.0)

    return PCMThermalProps(
        Tm_C=float(record["Tm_C"]),
        latent_heat_J_kg=float(record["latent_heat_kJ_kg"]) * 1000.0,
        cp_solid_J_kgK=float(cp_sol) * 1000.0,
        cp_liquid_J_kgK=float(cp_liq) * 1000.0,
        conductivity_W_mK=float(record["TC_W_mK"]),
        density_kg_m3=float(density),
        melting_half_width_K=melting_half_width_K,
    )


def enthalpy_of_temperature(T_C: float, props: PCMThermalProps) -> float:
    """h(T) in J/kg PCM (wall included via wall_cp), piecewise per the module docstring."""
    if T_C < props.Ts_C:
        return (props.cp_solid_J_kgK + props.wall_cp_J_kgK) * (T_C - TREF_C)
    elif T_C <= props.Tl_C:
        f = (T_C - props.Ts_C) / (props.Tl_C - props.Ts_C)
        return props.h_s_J_kg + f * props.melt_span_enthalpy_J_kg
    else:
        return (props.h_s_J_kg + props.melt_span_enthalpy_J_kg
                + (props.cp_liquid_J_kgK + props.wall_cp_J_kgK) * (T_C - props.Tl_C))


def temperature_and_liquid_fraction_of_enthalpy(h_J_kg: float, props: PCMThermalProps):
    """Inverse of enthalpy_of_temperature(): returns (T_C, f_melt in [0,1])."""
    h_s = props.h_s_J_kg
    span = props.melt_span_enthalpy_J_kg
    h_l = h_s + span

    if h_J_kg < h_s:
        T_C = TREF_C + h_J_kg / (props.cp_solid_J_kgK + props.wall_cp_J_kgK)
        f = 0.0
    elif h_J_kg <= h_l:
        f_raw = (h_J_kg - h_s) / span if span > 0 else 1.0
        f = min(max(f_raw, 0.0), 1.0)
        T_C = props.Ts_C + f * (props.Tl_C - props.Ts_C)
    else:
        T_C = props.Tl_C + (h_J_kg - h_l) / (props.cp_liquid_J_kgK + props.wall_cp_J_kgK)
        f = 1.0
    return T_C, f


def effective_conductivity_W_mK(props: PCMThermalProps, f_melt: float, enhancement_factor: float) -> float:
    """Once >=50% liquid, apply a documented natural-convection enhancement
    multiplier to the reported (single-value) conductivity — the database
    does not separately report solid vs. liquid conductivity."""
    return props.conductivity_W_mK * (enhancement_factor if f_melt >= 0.5 else 1.0)
