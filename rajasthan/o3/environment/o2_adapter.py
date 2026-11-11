"""
o3/environment/o2_adapter.py — one-hour calls into the FROZEN Objective-2 multi-node kernel.

This module is the ONLY place where O3 touches O2 internals. It does not rewrite any physics:

  * `run_year_multinode` (src/simulation/tank_model_multinode.py) does Python-side set-up
    (node masses, loss coefficients, parameter vector P, demand/hour arrays) and then calls the
    numba function `_kernel` once for the whole year.
  * Here the same set-up is done ONCE per (regime, design) in `build_runtime()`, and `hour()`
    calls the same `_kernel` with a one-row weather slice and the previous end-of-hour node
    temperatures / enthalpies as the initial state (plan 3.1, "fallback: call the same compiled
    kernel function directly" — pandas per-call overhead is avoided; no O2 file is modified).
  * tests/test_environment.py::test_T1 proves the stepped year equals one `run_year_multinode`
    call (hourly outputs and totals) so any drift in this copy of the set-up is caught.

'Bypass' (plan 3.1) = the hour's collector arrays are zeroed, exactly what a GHI of 0 does in
`_collector_arrays` (circ = GHI > cutoff -> 0, a = b = 0). The water shield (loop off at top node
>= 72 C) and the PCM shield (exchange cut at PCM >= 62 C when the water is hotter) stay inside the
kernel; the policy cannot override them.
"""

from dataclasses import dataclass, replace as _dc_replace

import numpy as np

import config_O2  # noqa: F401  (ensures O2 path)
from src.io_utils import (
    load_system_config, get_regime, get_pcm_properties, load_demand_profile,
)
from src.design.constraints import check_design
from src.design.space import to_design, geometry_bounds
from src.simulation.capsule_enthalpy import (
    PCMThermalProps, pcm_props_from_record, enthalpy_of_temperature,
)
from src.simulation.capsule_mass import capsule_masses, wall_cp_per_kg_pcm
from src.simulation.cfd_closure import apply_closure
from src.simulation.collector_model import apply_flow_dependent_fr
from src.simulation.demand_profile import load_demand_model
from src.simulation.tank_model import DesignRuntime
from src.simulation import tank_model_multinode as mn
from src.simulation.heat_transfer import _water_dynamic_viscosity_pa_s as _mu

from o3.utils.contract_loader import get_regime_design

J_PER_KWH = 3.6e6
N_NODES = 10                 # configs/design_space.yaml model.n_nodes
ANCHOR = "center"            # configs/design_space.yaml model.stack_anchor

# hourly output columns of mn._kernel (hr[:, k])
HR = dict(T_top=0, T_mean=1, T_pcm=2, f_melt=3, Q_coll=4, Q_load=5, Q_pcm=6, Q_loss=7, Q_unmet=8, draw=9,
          vol_ge=10, E_del45=11, T_min=12, T_max=13, T_pcm_max=14, f_min=15, f_max=16, E_stored=17,
          E_rec45=18, T_bottom=19, Q_load_net=20)
# scalar counters of mn._kernel (sc[k])
SC = dict(E_coll=0, E_load=1, E_load_net=2, E_loss=3, E_unmet=4, E_init=7, E_fin=8, n_unphys=12,
          safety=11, shield_w_n=13, shield_p_n=14, shield_w_s=26, shield_p_s=27, over_s=28)


@dataclass
class Runtime:
    """Everything the kernel needs that does not change from hour to hour."""
    regime_id: int
    water_only: bool
    cfg: dict
    pcm_props: PCMThermalProps          # with wall_cp already merged (as run_year_multinode does)
    design: DesignRuntime
    layout: mn.MultiNodeLayout
    n_caps: np.ndarray
    m_w: np.ndarray
    m_cap: float
    m_pcm_node: np.ndarray
    loss_ua: np.ndarray
    P: np.ndarray
    mains_C: float
    target_C: float
    cp_w: float
    demand24: np.ndarray
    n_nodes: int
    has_pcm: bool
    h_mains: float
    h_target: float
    sel: dict                            # contract design block (for reporting)


def build_runtime(contract: dict, regime_id: int, water_only: bool = False,
                  volume_multiplier: float = 1.0, timing_shift_hours: float = 0.0) -> Runtime:
    """Replicates src/simulation/run_case.run_case (multinode branch) + run_year_multinode set-up."""
    cfg = apply_closure(load_system_config())
    rd = get_regime_design(contract, regime_id)
    dv = to_design(rd["capsule_inner_radius_m"], rd["pcm_volume_fraction"], rd["nominal_flow_kg_s"], rd["arrangement"])
    if dv.n_capsule != rd["n_capsules"]:
        raise RuntimeError(f"regime {regime_id}: O2 count rule gives N={dv.n_capsule}, contract says {rd['n_capsules']}")
    cfg = apply_flow_dependent_fr(cfg, dv.flow_rate_kg_s)
    geom = check_design(dv, cfg, geometry_bounds())
    if not geom["valid"]:
        raise RuntimeError(f"regime {regime_id}: contract design failed the O2 geometry check: {geom['reason']}")

    mains_C = get_regime("rajasthan", regime_id)["T_mains_est_C"]
    pcm_record = get_pcm_properties("rajasthan", rd["selected_pcm"])
    pcm_props = pcm_props_from_record(pcm_record, cfg["pcm_integration"]["melting_half_width_K"])
    wall_m = cfg["pcm_integration"]["capsule_wall_thickness_m"]
    n_eff = 0 if water_only else geom["n_capsule"]
    design = DesignRuntime(
        n_capsule=n_eff, capsule_diameter_m=geom["capsule_diameter_m"], capsule_area_m2=geom["capsule_surface_area_m2"],
        capsule_volume_m3=geom["capsule_volume_m3"], void_fraction=geom["void_fraction_used_for_hydraulics"],
        cross_section_area_m2=geom["tank_cross_section_area_m2"],
        bed_length_m=geom["stack_height_m"] if geom["stack_height_m"] > 0 else geom["tank_height_m"],
        flow_rate_kg_s=geom["flow_rate_kg_s"], tank_volume_m3=geom["tank_volume_m3"],
        tank_surface_area_m2=geom["tank_surface_area_m2"], capsule_wall_thickness_m=wall_m,
    )
    if water_only:      # plain-tank twin: same as run_case with pcm_name=None
        pcm_props = PCMThermalProps(Tm_C=57.0, latent_heat_J_kg=0.0, cp_solid_J_kgK=2000.0, cp_liquid_J_kgK=2000.0,
                                    conductivity_W_mK=0.2, density_kg_m3=800.0)
        geom_l = {**geom, "n_capsule": 0, "n_layers": 0}
    else:
        geom_l = geom
    layout = mn.layout_from_geometry(geom_l, N_NODES, ANCHOR)

    # ---- the set-up run_year_multinode does before calling _kernel ----
    N = layout.n_nodes
    solver = cfg["solver"]
    dt = float(solver["timestep_s"])
    water = cfg["water"]
    cp_w, rho_w = water["cp_J_kgK"], water["density_kg_m3"]
    integ = cfg["pcm_integration"]
    shield_cfg = cfg.get("safety_shield", {})
    max_w, max_p = cfg["safety"]["max_water_temp_C"], cfg["safety"]["max_pcm_temp_C"]
    target = cfg["delivery"]["target_temp_C"]
    n_caps = layout.capsules_per_node.astype(float)
    has_pcm = bool(n_caps.sum() > 0)
    v_node = design.tank_volume_m3 / N
    if has_pcm:
        masses = capsule_masses(int(n_caps.sum()), design.capsule_diameter_m, wall_m, pcm_props.density_kg_m3)
        m_cap = masses["m_pcm_kg"] / n_caps.sum()
        v_cap_out = np.pi / 6.0 * design.capsule_diameter_m ** 3
        props = _dc_replace(pcm_props, wall_cp_J_kgK=wall_cp_per_kg_pcm(masses))
    else:
        m_cap, v_cap_out, props = 0.0, 0.0, pcm_props
    m_w = np.maximum(v_node - n_caps * v_cap_out, 1e-9) * rho_w
    Dt, Ht = layout.tank_diameter_m, layout.tank_height_m
    A_cross = np.pi / 4.0 * Dt ** 2
    dz = Ht / N
    side = np.pi * Dt * Ht / N
    loss_ua = np.full(N, cfg["tank"]["U_tank_W_m2K"] * side)
    loss_ua[0] += cfg["tank"]["U_tank_W_m2K"] * A_cross
    loss_ua[N - 1] += cfg["tank"]["U_tank_W_m2K"] * A_cross
    kcond = water["conductivity_W_mK"] * A_cross / dz
    P = np.array([
        cp_w, rho_w, mains_C, target, design.flow_rate_kg_s, dt, kcond, A_cross,
        props.Ts_C, props.Tl_C, props.latent_heat_J_kg, props.cp_solid_J_kgK, props.cp_liquid_J_kgK,
        props.wall_cp_J_kgK, props.conductivity_W_mK, integ["liquid_convection_enhancement"],
        water["conductivity_W_mK"], water["prandtl"], _mu(),
        integ["capsule_wall_thickness_m"], integ["capsule_wall_conductivity_W_mK"],
        design.capsule_diameter_m, design.capsule_area_m2, m_cap,
        solver["melt_band_margin_K"], 0.5, solver["substeps_in_melt_band"], 60,
        1.0 if shield_cfg.get("enabled", False) else 0.0,
        shield_cfg.get("bypass_water_C", max_w - 3.0), shield_cfg.get("bypass_pcm_C", max_p - 3.0),
        max_w, max_p, -1e9, float(integ.get("conduction_coefficient_c", 5.0)),
    ], dtype=float)
    P = np.concatenate([P, np.zeros(40 - len(P))])

    demand = load_demand_model(load_demand_profile("rajasthan"), volume_multiplier, timing_shift_hours)
    demand24 = np.array([demand.draw_mass_kg_for_hour(h) for h in range(24)], dtype=float)
    h_mains = float(mn._h_of_T(mains_C, props.Ts_C, props.Tl_C, props.latent_heat_J_kg, props.cp_solid_J_kgK,
                               props.cp_liquid_J_kgK, props.wall_cp_J_kgK))
    h_target = float(mn._h_of_T(target, props.Ts_C, props.Tl_C, props.latent_heat_J_kg, props.cp_solid_J_kgK,
                                props.cp_liquid_J_kgK, props.wall_cp_J_kgK))
    return Runtime(regime_id=regime_id, water_only=water_only, cfg=cfg, pcm_props=props, design=design, layout=layout,
                   n_caps=n_caps, m_w=m_w, m_cap=m_cap, m_pcm_node=n_caps * m_cap, loss_ua=loss_ua, P=P,
                   mains_C=mains_C, target_C=target, cp_w=cp_w, demand24=demand24, n_nodes=N, has_pcm=has_pcm,
                   h_mains=h_mains, h_target=h_target, sel=rd)


class OneHourKernel:
    """Calls mn._kernel for a single hour. Re-uses small input arrays to keep the per-call cost low."""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self._tamb = np.zeros(1)
        self._hod = np.zeros(1, dtype=np.int64)
        self._a = np.zeros(1)
        self._b = np.zeros(1)
        self._c = np.zeros(1)
        self._dummy = np.zeros(1)
        c = rt.cfg["collector"]
        self._area, self._fta, self._ful, self._cut = c["area_m2"], c["fr_tau_alpha"], c["fr_ul_W_m2K"], c["min_irradiance_cutoff_Wm2"]

    # -- initial state ----------------------------------------------------
    def initial_state(self, scenario: str, T_water_C: float | None = None):
        """(T_nodes, h_nodes) for the contract reset scenarios (solid | mid/partial | liquid)."""
        rt = self.rt
        T0 = np.full(rt.n_nodes, rt.cfg["solver"]["initial_water_temp_C"] if T_water_C is None else T_water_C)
        h0 = np.zeros(rt.n_nodes)
        if rt.has_pcm:
            p = rt.pcm_props
            if scenario == "solid":
                tp0 = p.Ts_C - 5.0
            elif scenario == "liquid":
                tp0 = p.Tl_C + 5.0
            elif scenario in ("mid", "partial"):
                tp0 = p.Tm_C
            else:
                raise ValueError(f"unknown reset scenario {scenario!r}")
            h0[:] = enthalpy_of_temperature(tp0, p)
        return T0, h0

    # -- derived state (same formulas as the kernel's end-of-hour block) ----
    def derived(self, T: np.ndarray, h: np.ndarray) -> dict:
        rt, p = self.rt, self.rt.pcm_props
        w = rt.m_w
        e_rel = float(np.sum(w * rt.cp_w * (T - rt.mains_C)))
        e_rec = float(np.sum(w * rt.cp_w * np.maximum(T - rt.target_C, 0.0)))
        T_pcm, f_melt = float(np.mean(T)), 0.0
        if rt.has_pcm:
            msk = rt.n_caps > 0
            mp = rt.m_pcm_node[msk]
            e_rel += float(np.sum(mp * (h[msk] - rt.h_mains)))
            e_rec += float(np.sum(mp * np.maximum(h[msk] - rt.h_target, 0.0)))
            Tp = np.zeros(msk.sum())
            fm = np.zeros(msk.sum())
            for j, hh in enumerate(h[msk]):
                Tp[j], fm[j] = mn._T_f_of_h(hh, p.Ts_C, p.Tl_C, p.latent_heat_J_kg, p.cp_solid_J_kgK, p.cp_liquid_J_kgK, p.wall_cp_J_kgK)
            T_pcm = float(np.sum(Tp * mp) / mp.sum())
            f_melt = float(np.sum(fm * mp) / mp.sum())
        return {"T_top": float(T[-1]), "T_bottom": float(T[0]), "T_mean": float(np.sum(T * w) / w.sum()),
                "T_pcm": T_pcm, "f_melt": f_melt, "E_stored_kWh": e_rel / J_PER_KWH, "E_rec45_kWh": e_rec / J_PER_KWH}

    # -- one hour ----------------------------------------------------------
    def hour(self, T: np.ndarray, h: np.ndarray, ghi: float, tamb: float, hod: int, circulate: bool,
             demand24: np.ndarray | None = None):
        """Advance one hour. `circulate=False` is a bypass (collector arrays zeroed, as GHI = 0 does).

        Returns (hr_row[21], sc[30], T_new, h_new). T and h are not modified.
        """
        rt = self.rt
        if circulate and ghi > self._cut:
            self._a[0] = self._area * (self._fta * ghi + self._ful * tamb)
            self._b[0] = self._area * self._ful
            self._c[0] = 1.0
        else:
            self._a[0] = 0.0
            self._b[0] = 0.0
            self._c[0] = 0.0
        self._tamb[0] = tamb
        self._hod[0] = hod
        hr, node_T, node_h, sc = mn._kernel(self._dummy, self._tamb, self._hod, self._a, self._b, self._c,
                                            rt.demand24 if demand24 is None else demand24,
                                            rt.m_w, rt.n_caps, rt.loss_ua, rt.P, T, h, 1.0)
        return hr[0], sc, node_T[0].copy(), node_h[0].copy()
