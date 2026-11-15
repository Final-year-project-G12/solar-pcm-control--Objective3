"""
src/design/geometry.py
========================
Geometry & constraint engine.

Given a design vector (capsule diameter, capsule count, flow rate,
arrangement), returns volume/area/spacing/pressure-drop and a valid/invalid
flag with reason. Universal code — identical for every state; only the
demand/weather/PCM inputs supplied elsewhere differ per state.

GEOMETRIC MODEL (documented simplification, appropriate for a 40-hr MVP):
  - Tank is a vertical cylinder. Its diameter/height are derived from the
    frozen tank volume assuming height = 2 x diameter (typical domestic SWH
    proportion; see system_config_shared.yaml).
  - Capsules are spheres packed in horizontal layers stacked up the tank
    height. The in-layer packing pattern is one of three arrangements
    (restored as a searched variable 2026-09-17, see
    docs/00_MASTER_CHANGE_PLAN.md / docs/02_PROMPT_PHASE2_GEOMETRY.md):
      * staggered     - 2D hexagonal packing of circles (the original,
                        pre-2026-09-17 sole model).
      * single-layer  - simple square-grid packing, no row offset. Lower
                        packing density than staggered by construction
                        (larger footprint per capsule at the same pitch).
      * radial        - concentric rings of capsules around the tank's
                        vertical axis, ring count/capacity set by
                        tank_diameter_m and diameter_m.
    Each arrangement produces its own per-layer capsule count and its own
    "packing void fraction" (the porosity of the lattice itself, distinct
    from pcm_volume_fraction, which stays a pure mass-balance quantity
    n_capsule*V_capsule/V_tank, unaffected by how capsules are arranged).
    The packing void fraction feeds the passage-blocked check and the Ergun
    hydraulics below, which is why arrangement changes pressure_drop_pa and
    pump_power_w without any change to the simulator's physics code.
  - Pressure drop uses the Ergun equation for flow through a packed bed of
    spheres (Ergun, 1952) — a standard correlation, not derived from
    scratch, as required by the framework doc [1, Sec 3.2].

REASON CODES (constraints.py maps these to reject/accept):
  overlap           - capsule diameter too large for even one to fit in the
                       tank cross-section at the required spacing.
  volume_exceeded   - N_capsule*V_capsule exceeds the allocated PCM volume
                       fraction cap (pcm_volume_fraction.max of configs/design_space.yaml, 30 % of V_tank).
  passage_blocked   - the capsule stack does not fit within the tank height,
                       or the resulting packing void fraction is below the
                       minimum free-flow passage fraction.
  pressure_drop_limit - estimated pressure drop exceeds the system's max
                       pressure limit.
  flow_out_of_range - flow rate outside [flow_min, flow_max] (extra reason
                       code beyond the framework doc's base four, documented
                       here for clarity).

No new reason codes were added when arrangement was restored — the same six
codes above are just now computed from whichever packing model the
arrangement dispatches to (see pack_capsules()).

Determinism: pure functions of (design vector, frozen configs) -> same
inputs always produce identical outputs.
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.design.schema import DesignVector
from src.io_utils import load_system_config, load_design_bounds

WATER_DENSITY_KG_M3 = 1000.0


# ─────────────────────────────────────────────────────────────────────────
# Sphere primitives
# ─────────────────────────────────────────────────────────────────────────

def sphere_volume_m3(diameter_m: float) -> float:
    r = diameter_m / 2.0
    return (4.0 / 3.0) * math.pi * r ** 3


def sphere_surface_area_m2(diameter_m: float) -> float:
    r = diameter_m / 2.0
    return 4.0 * math.pi * r ** 2


# ─────────────────────────────────────────────────────────────────────────
# Tank envelope (derived once from the frozen tank volume)
# ─────────────────────────────────────────────────────────────────────────

def tank_dimensions_m(system_config: dict):
    """Vertical cylinder: V = (pi/4) D^2 H, H = ratio * D  ->  D = (4V/(pi*ratio))^(1/3)."""
    v_tank_m3 = system_config["tank"]["volume_L"] / 1000.0
    ratio = system_config["tank"]["height_to_diameter_ratio"]
    d_tank = (4.0 * v_tank_m3 / (math.pi * ratio)) ** (1.0 / 3.0)
    h_tank = ratio * d_tank
    cross_section_area_m2 = math.pi / 4.0 * d_tank ** 2
    tank_surface_area_m2 = math.pi * d_tank * h_tank + 2 * cross_section_area_m2  # side + 2 caps
    return {
        "tank_diameter_m": d_tank,
        "tank_height_m": h_tank,
        "tank_volume_m3": v_tank_m3,
        "tank_cross_section_area_m2": cross_section_area_m2,
        "tank_surface_area_m2": tank_surface_area_m2,
    }


# ─────────────────────────────────────────────────────────────────────────
# Arrangement-branched packing models
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PackingResult:
    """Common interface every arrangement's packing model returns. Fields
    are exactly what the rest of the pipeline already consumed before
    arrangement was restored (capsules_per_layer, n_layers, void_fraction,
    stack_height_m) — only what computes them differs per arrangement."""
    capsules_per_layer: int
    n_layers: int
    stack_height_m: float
    void_fraction: float
    arrangement: str
    per_layer_footprint_area_m2: Optional[float] = None  # bookkeeping only


def pack_staggered(diameter_m: float, tank_diameter_m: float, tank_height_m: float,
                    n_capsule: int, spacing_min_m: float) -> PackingResult:
    """2D hexagonal ("staggered") packing of circles in the tank's circular
    cross-section. Original, pre-2026-09-17 sole packing model — math kept
    exactly as validated by the pre-existing boundary self-test."""
    cross_section_area_m2 = math.pi / 4.0 * tank_diameter_m ** 2
    pitch = diameter_m + spacing_min_m
    footprint_area = (math.sqrt(3.0) / 2.0) * pitch ** 2   # hex-lattice footprint per sphere
    per_layer = max(0, math.floor(cross_section_area_m2 / footprint_area)) if footprint_area > 0 else 0
    return _finish_layered_packing("staggered", diameter_m, tank_height_m, n_capsule,
                                    spacing_min_m, per_layer, footprint_area)


def pack_single_layer(diameter_m: float, tank_diameter_m: float, tank_height_m: float,
                       n_capsule: int, spacing_min_m: float) -> PackingResult:
    """Simple planar/square-grid spacing within each horizontal layer (no
    row offset). Footprint area per sphere in a square lattice = pitch^2 —
    strictly larger than staggered's hex footprint at the same pitch, so
    this arrangement packs fewer capsules per layer by construction, not by
    an artificial penalty."""
    cross_section_area_m2 = math.pi / 4.0 * tank_diameter_m ** 2
    pitch = diameter_m + spacing_min_m
    footprint_area = pitch ** 2   # square-lattice footprint per sphere
    per_layer = max(0, math.floor(cross_section_area_m2 / footprint_area)) if footprint_area > 0 else 0
    return _finish_layered_packing("single-layer", diameter_m, tank_height_m, n_capsule,
                                    spacing_min_m, per_layer, footprint_area)


def pack_radial(diameter_m: float, tank_diameter_m: float, tank_height_m: float,
                 n_capsule: int, spacing_min_m: float) -> PackingResult:
    """Concentric rings of capsules around the tank's vertical axis. Ring
    radial pitch = diameter + clearance; ring 0 is a single capsule at the
    centerline; ring k>0 holds floor(2*pi*r_k / ring_pitch) capsules. Rings
    are stacked by tank_height_m the same way the other two arrangements
    stack layers."""
    tank_radius_m = tank_diameter_m / 2.0
    ring_pitch = diameter_m + spacing_min_m
    usable_radius_m = tank_radius_m - diameter_m / 2.0
    if ring_pitch <= 0 or usable_radius_m < 0:
        per_layer = 0
    else:
        n_rings = math.floor(usable_radius_m / ring_pitch) + 1
        per_layer = 0
        for k in range(n_rings):
            if k == 0:
                per_layer += 1   # single capsule on the centerline
            else:
                r_k = k * ring_pitch
                per_layer += math.floor(2.0 * math.pi * r_k / ring_pitch)
    # Rings span the full tank cross-section by construction (unlike the
    # unit-cell lattices above), so the reference footprint area for the
    # packing void fraction is the full disk, not a per-capsule tile.
    cross_section_area_m2 = math.pi / 4.0 * tank_diameter_m ** 2
    return _finish_layered_packing("radial", diameter_m, tank_height_m, n_capsule,
                                    spacing_min_m, per_layer, footprint_area=None,
                                    bed_footprint_area_override_m2=cross_section_area_m2)


def _finish_layered_packing(arrangement: str, diameter_m: float, tank_height_m: float,
                             n_capsule: int, spacing_min_m: float, per_layer: int,
                             footprint_area: Optional[float],
                             bed_footprint_area_override_m2: Optional[float] = None) -> PackingResult:
    """Shared layer-stacking + packing-void-fraction math for all three
    arrangements, once each has computed its own per_layer capsule count."""
    layer_pitch_m = diameter_m + spacing_min_m
    if per_layer < 1:
        return PackingResult(capsules_per_layer=per_layer, n_layers=0, stack_height_m=0.0,
                              void_fraction=1.0, arrangement=arrangement,
                              per_layer_footprint_area_m2=footprint_area)

    n_layers = math.ceil(n_capsule / per_layer) if n_capsule > 0 else 0
    stack_height_m = n_layers * layer_pitch_m

    if n_capsule <= 0 or stack_height_m <= 0:
        void_fraction = 1.0   # empty tank / no-PCM baseline — fully void
    else:
        bed_footprint_area_m2 = (bed_footprint_area_override_m2 if bed_footprint_area_override_m2 is not None
                                  else per_layer * footprint_area)
        bed_volume_m3 = bed_footprint_area_m2 * stack_height_m
        solid_volume_m3 = n_capsule * sphere_volume_m3(diameter_m)
        void_fraction = 1.0 - solid_volume_m3 / bed_volume_m3 if bed_volume_m3 > 0 else 1.0

    return PackingResult(capsules_per_layer=per_layer, n_layers=n_layers, stack_height_m=stack_height_m,
                          void_fraction=void_fraction, arrangement=arrangement,
                          per_layer_footprint_area_m2=footprint_area)


def pack_capsules(arrangement: str, diameter_m: float, tank_diameter_m: float, tank_height_m: float,
                   n_capsule: int, spacing_min_m: float) -> PackingResult:
    """Dispatcher: routes to the packing model named by design.capsule_arrangement."""
    if arrangement == "single-layer":
        return pack_single_layer(diameter_m, tank_diameter_m, tank_height_m, n_capsule, spacing_min_m)
    elif arrangement == "staggered":
        return pack_staggered(diameter_m, tank_diameter_m, tank_height_m, n_capsule, spacing_min_m)
    elif arrangement == "radial":
        return pack_radial(diameter_m, tank_diameter_m, tank_height_m, n_capsule, spacing_min_m)
    else:
        raise ValueError(f"unknown arrangement: {arrangement}")


def compute_geometry(design: DesignVector, system_config: dict = None,
                      design_bounds: dict = None) -> dict:
    """Deterministic geometry + hydraulics for one design vector.
    Returns a dict always containing `valid` (bool) and `reason` (str or None).
    """
    system_config = system_config or load_system_config()
    design_bounds = design_bounds or load_design_bounds()

    d = design.capsule_diameter_m
    n = design.n_capsule
    mdot = design.flow_rate_kg_s
    arrangement = design.capsule_arrangement

    tank = tank_dimensions_m(system_config)
    v_tank = tank["tank_volume_m3"]

    v_capsule = sphere_volume_m3(d)
    a_capsule = sphere_surface_area_m2(d)
    thickness_m = d / 2.0   # max PCM conduction distance for a sphere = radius

    spacing_min = design_bounds["geometry"]["spacing_min_m"]
    passage_min_fraction = design_bounds["geometry"]["passage_min_fraction"]

    result = {
        "capsule_diameter_m": d,
        "n_capsule": n,
        "flow_rate_kg_s": mdot,
        "arrangement": arrangement,
        "pcm_thickness_m": thickness_m,
        "capsule_volume_m3": v_capsule,
        "capsule_surface_area_m2": a_capsule,
        **tank,
        "valid": True,
        "reason": None,
    }

    # ---- flow range check --------------------------------------------
    flow_bounds = design_bounds["flow_rate_kg_s"]
    if not (flow_bounds["min"] - 1e-12 <= mdot <= flow_bounds["max"] + 1e-12):
        result.update(valid=False, reason="flow_out_of_range")
        return result

    # ---- arrangement-branched packing / overlap check -------------------
    packing = pack_capsules(arrangement, d, tank["tank_diameter_m"], tank["tank_height_m"], n, spacing_min)
    per_layer = packing.capsules_per_layer
    result["capsules_per_layer"] = per_layer
    if per_layer < 1:
        result.update(valid=False, reason="overlap")
        return result

    n_layers = packing.n_layers
    stack_height_m = packing.stack_height_m
    result["n_layers"] = n_layers
    result["stack_height_m"] = stack_height_m

    # ---- PCM volume fraction (derived, mass-balance — arrangement-agnostic) --
    v_pcm_total = n * v_capsule
    pcm_volume_fraction = v_pcm_total / v_tank if v_tank > 0 else float("inf")
    result["pcm_volume_total_m3"] = v_pcm_total
    result["pcm_volume_fraction"] = pcm_volume_fraction

    vf_bounds = design_bounds["pcm_volume_fraction"]
    if pcm_volume_fraction > vf_bounds["max"] + 1e-9:
        result.update(valid=False, reason="volume_exceeded")
        return result

    # ---- passage / envelope check (packing void fraction — arrangement-aware) --
    void_fraction = packing.void_fraction
    result["void_fraction"] = void_fraction
    if stack_height_m > tank["tank_height_m"]:
        result.update(valid=False, reason="passage_blocked")
        return result
    if void_fraction < passage_min_fraction:
        result.update(valid=False, reason="passage_blocked")
        return result
    if pcm_volume_fraction < vf_bounds["min"] - 1e-9:
        # Hard rejection below the loading floor (pcm_volume_fraction.min of the bounds dict).
        result["below_min_pcm_fraction"] = True
        result.update(valid=False, reason="below_min_pcm_fraction")
        return result
    result["below_min_pcm_fraction"] = False

    # ---- hydraulics: Ergun equation over the packed capsule bed --------
    hydraulics = compute_hydraulics(
        mdot_kg_s=mdot,
        capsule_diameter_m=d,
        void_fraction=void_fraction,
        cross_section_area_m2=tank["tank_cross_section_area_m2"],
        bed_length_m=stack_height_m if stack_height_m > 0 else tank["tank_height_m"],
        system_config=system_config,
    )
    result.update(hydraulics)

    max_pressure_pa = system_config["safety"]["max_pressure_bar"] * 1e5
    if result["pressure_drop_pa"] > max_pressure_pa:
        result.update(valid=False, reason="pressure_drop_limit")
        return result

    return result


# ─────────────────────────────────────────────────────────────────────────
# Hydraulics — Ergun equation (packed bed of spheres) + pump power
# ─────────────────────────────────────────────────────────────────────────

def compute_hydraulics(mdot_kg_s: float, capsule_diameter_m: float, void_fraction: float,
                        cross_section_area_m2: float, bed_length_m: float,
                        system_config: dict) -> dict:
    rho = system_config["water"]["density_kg_m3"]
    mu = _water_dynamic_viscosity_pa_s()

    eps = min(max(void_fraction, 1e-3), 0.999)
    dp = capsule_diameter_m
    volumetric_flow_m3_s = mdot_kg_s / rho
    superficial_velocity_m_s = volumetric_flow_m3_s / cross_section_area_m2 if cross_section_area_m2 > 0 else 0.0

    re_particle = rho * superficial_velocity_m_s * dp / mu if mu > 0 else 0.0

    # Ergun (1952): dP/L = 150*(1-eps)^2/eps^3 * mu*u/dp^2 + 1.75*(1-eps)/eps^3 * rho*u^2/dp
    viscous_term = 150.0 * (1 - eps) ** 2 / eps ** 3 * mu * superficial_velocity_m_s / dp ** 2
    inertial_term = 1.75 * (1 - eps) / eps ** 3 * rho * superficial_velocity_m_s ** 2 / dp
    dp_dl_pa_per_m = viscous_term + inertial_term
    pressure_drop_pa = dp_dl_pa_per_m * bed_length_m

    hydraulic_diameter_m = (2.0 / 3.0) * dp * eps / (1 - eps) if eps < 1.0 else dp

    eta_pump = system_config["pump"]["efficiency"]
    pump_power_w = pressure_drop_pa * volumetric_flow_m3_s / eta_pump if eta_pump > 0 else 0.0

    return {
        "void_fraction_used_for_hydraulics": eps,
        "superficial_velocity_m_s": superficial_velocity_m_s,
        "reynolds_number_particle": re_particle,
        "hydraulic_diameter_m": hydraulic_diameter_m,
        "pressure_drop_pa": pressure_drop_pa,
        "pump_power_w": pump_power_w,
    }


def _water_dynamic_viscosity_pa_s(t_c: float = 40.0) -> float:
    """Simple correlation for water dynamic viscosity (Pa.s) near typical
    operating temperature; adequate for a design-space pressure-drop estimate
    (documented simplification — not a full property library)."""
    # Vogel-like fit, valid ~0-100 C, matches tabulated water viscosity to
    # within a few percent in the 20-60 C range used by this project.
    a, b, c = 2.414e-5, 247.8, 140.0
    t_k = t_c + 273.15
    return a * 10 ** (b / (t_k - c))
