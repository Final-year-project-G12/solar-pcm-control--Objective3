"""
src/simulation/tank_model_multinode.py
=========================================
O2 G4 / plan WP2 Level 2 (decision D4, 2026-10-26) — 1-D stratified
multi-node tank. A NEW module: src/simulation/tank_model.py (single node) is kept
unchanged as the regression reference.

Physics (node 0 = bottom, node N-1 = top; N equal-height nodes):
  * Draw: mains enter the bottom node, hot water leaves the top node (plug flow,
    first-order implicit upwind advection between nodes).
  * Collector loop (when circulating): water leaves the bottom node at the design
    flow rate, is heated by Q = a - b*T_bottom (same linear collector as the
    single-node model, so F_R is flow-independent) and re-enters at the node
    whose temperature is the highest one not exceeding the loop outlet
    temperature (neutral-buoyancy "stratified inlet"). The interface flows
    follow from mass conservation; the flow rate therefore sets the loop
    delta-T and the inlet node -- it acts on stratification, not on Q.
  * Axial conduction k_w*A/dz between neighbours; ambient loss u*A_i per node
    (side wall split evenly, end caps on the bottom and top nodes).
  * Buoyancy: after each solve, any inversion (lower node hotter than the one
    above) is removed by mass-weighted pairwise mixing (energy conserving).
  * PCM capsules sit in nodes according to the packing: layers of `capsules_per_layer`
    stacked at the layer pitch, the stack anchored at the tank bottom/centre/top
    (`anchor`, default "centre" -- a neutral choice, to be ablated in G11). A node's
    capsules exchange heat with that node's water through the same UA_eff as the
    single-node model (Wakao-Kaguei h_w + wall + r/(5k) PCM resistance), but the
    water-side velocity is the node's actual through-flow, not the design flow.
    The PCM+wall enthalpy model is identical to capsule_enthalpy.py (re-implemented
    in numba; equality is unit-tested).
  * Shield, differential-controller re-solve, adaptive melt-band/stiffness
    sub-stepping: as in the single-node model, evaluated on the top node (water)
    and per node (PCM).

Outputs have the same shape as tank_model.run_year (SimulationResult), plus node-
resolved hourly E_stored/E_rec,45 and node states in hourly.attrs for the WP1
night scenarios. Pump energy is not modelled (flagged meaningless, G2).

Simplifications (documented, not hidden): first-order upwind adds numerical
diffusion, so the node count N is a modelling parameter (compared against the
project CFD draw case in tests/test_multinode_tank.py); no PCM supercooling or
volume change; the capsule closure is still the lumped r/(5k) one; no CFD-derived
resistance correction is applied.
"""

from dataclasses import dataclass, replace as _dc_replace

import numpy as np
import pandas as pd
from numba import njit

from src.simulation.capsule_enthalpy import enthalpy_of_temperature, temperature_and_liquid_fraction_of_enthalpy
from src.simulation.capsule_mass import capsule_masses, wall_cp_per_kg_pcm
from src.simulation.heat_transfer import _water_dynamic_viscosity_pa_s as _mu   # viscosity at 40 C
from src.simulation.tank_model import SimulationResult, _demand_hour_of_day

J_PER_KWH = 3.6e6
DEFAULT_NODES = 10


# ------------------------------------------------------------------ layout
@dataclass
class MultiNodeLayout:
    n_nodes: int
    capsules_per_node: np.ndarray      # float, capsules in each node (bottom -> top)
    tank_height_m: float
    tank_diameter_m: float
    anchor: str = "center"


def layout_from_geometry(geom: dict, n_nodes: int = DEFAULT_NODES, anchor: str = "center") -> MultiNodeLayout:
    """Distribute capsules over nodes from the packing record (layers of equal pitch)."""
    H = geom["tank_height_m"]
    dz = H / n_nodes
    counts = np.zeros(n_nodes)
    n = int(geom["n_capsule"])
    nl = int(geom.get("n_layers", 0) or 0)
    if n > 0 and nl > 0:
        per, stack = int(geom["capsules_per_layer"]), float(geom["stack_height_m"])
        pitch = stack / nl
        z0 = {"bottom": 0.0, "top": H - stack, "center": (H - stack) / 2.0}[anchor]
        remaining = n
        for j in range(nl):
            c = min(per, remaining)
            remaining -= c
            node = min(max(int((z0 + (j + 0.5) * pitch) / dz), 0), n_nodes - 1)
            counts[node] += c
        assert remaining == 0
    return MultiNodeLayout(n_nodes, counts, H, geom["tank_diameter_m"], anchor)


# ------------------------------------------------------------------ numba kernels
@njit(cache=True)
def _h_of_T(T, Ts, Tl, L, cps, cpl, x):
    hs = (cps + x) * Ts
    span = L + x * (Tl - Ts)
    if T < Ts:
        return (cps + x) * T
    if T <= Tl:
        return hs + span * (T - Ts) / (Tl - Ts)
    return hs + span + (cpl + x) * (T - Tl)


@njit(cache=True)
def _T_f_of_h(h, Ts, Tl, L, cps, cpl, x):
    hs = (cps + x) * Ts
    span = L + x * (Tl - Ts)
    hl = hs + span
    if h < hs:
        return h / (cps + x), 0.0
    if h <= hl:
        f = (h - hs) / span if span > 0 else 1.0
        f = min(max(f, 0.0), 1.0)
        return Ts + f * (Tl - Ts), f
    return Tl + (h - hl) / (cpl + x), 1.0


@njit(cache=True)
def _ua_node(n, d, area, v, f, rho, kw, pr, mu, wall_t, wall_k, k_pcm, enh, c_cond=5.0):
    if n <= 0:
        return 0.0
    re = max(rho * v * d / mu, 1e-6)
    nu = 2.0 + 1.1 * re ** 0.6 * pr ** (1.0 / 3.0)
    hw = nu * kw / d
    k_eff = k_pcm * (enh if f >= 0.5 else 1.0)
    r_in = d / 2.0 - wall_t                      # PCM conduction path = inner radius (decision D3)
    a_in = 4.0 * np.pi * r_in * r_in
    r_tot = 1.0 / (hw * area) + wall_t / (wall_k * area) + r_in / (c_cond * k_eff * a_in)
    return n / r_tot


@njit(cache=True)
def _solve_step(T, Tp, UA, m_w, loss_ua, tamb, N, cp, mains, mdot_d, mdot_c, circ, k_ret, a, b, kcond, dts):
    A = np.zeros((N, N))
    r = np.zeros(N)
    ext_in = np.zeros(N)
    ext_out = np.zeros(N)
    ext_in[0] += mdot_d
    ext_out[N - 1] += mdot_d
    if circ:
        ext_out[0] += mdot_c
        ext_in[k_ret] += mdot_c
    F = np.zeros(max(N - 1, 1))
    run = 0.0
    for i in range(N - 1):
        run += ext_in[i] - ext_out[i]
        F[i] = run
    for i in range(N):
        A[i, i] += m_w[i] * cp / dts + loss_ua[i] + UA[i] + ext_out[i] * cp
        r[i] += m_w[i] * cp / dts * T[i] + loss_ua[i] * tamb + UA[i] * Tp[i]
        if i == 0:
            r[0] += mdot_d * cp * mains
        if i > 0:
            Fb = F[i - 1]
            if Fb > 0:
                A[i, i - 1] -= Fb * cp
            else:
                A[i, i] -= Fb * cp
            A[i, i] += kcond
            A[i, i - 1] -= kcond
        if i < N - 1:
            Fa = F[i]
            if Fa > 0:
                A[i, i] += Fa * cp
            else:
                A[i, i + 1] += Fa * cp
            A[i, i] += kcond
            A[i, i + 1] -= kcond
    if circ:
        A[k_ret, 0] -= (mdot_c * cp - b)
        r[k_ret] += a
    return np.linalg.solve(A, r)


@njit(cache=True)
def _mix_inversions(T, m_w, N):
    """Remove every inversion (lower node hotter than the one above) by pooling the
    offending adjacent nodes into mass-weighted uniform blocks (pool-adjacent-violators);
    exactly energy conserving and leaves a non-decreasing bottom->top profile."""
    start = np.zeros(N, dtype=np.int64)
    mass = np.zeros(N)
    en = np.zeros(N)
    nb = 0
    for j in range(N):
        start[nb] = j
        mass[nb] = m_w[j]
        en[nb] = m_w[j] * T[j]
        nb += 1
        while nb > 1 and en[nb - 2] / mass[nb - 2] > en[nb - 1] / mass[nb - 1] + 1e-12:
            mass[nb - 2] += mass[nb - 1]
            en[nb - 2] += en[nb - 1]
            nb -= 1
    for k in range(nb):
        end = start[k + 1] if k + 1 < nb else N
        tm = en[k] / mass[k]
        for j in range(start[k], end):
            T[j] = tm


@njit(cache=True)
def _kernel(ghi_unused, tamb, hod, coll_a, coll_b, coll_circ, demand24, m_w, n_caps, loss_ua,
            P, T0, h0, use_h0):
    n_hours = tamb.shape[0]
    N = m_w.shape[0]
    cp, rho_w, mains, target = P[0], P[1], P[2], P[3]
    mdot_c, dt, kcond, A_cross = P[4], P[5], P[6], P[7]
    Ts, Tl, L, cps, cpl, x = P[8], P[9], P[10], P[11], P[12], P[13]
    k_pcm, enh, kw, pr, mu = P[14], P[15], P[16], P[17], P[18]
    wall_t, wall_k, d, area_cap, m_cap = P[19], P[20], P[21], P[22], P[23]
    margin, stiff_sf, n_band, max_sub = P[24], P[25], int(P[26]), int(P[27])
    shield, byp_w, byp_p, max_w, max_p, stop_thr = P[28], P[29], P[30], P[31], P[32], P[33]
    c_cond = P[34]                  # conduction coefficient c of R_pcm = r/(c k_eff); 5.0 unless the off-by-default option is used

    n_caps_tot = 0.0
    for i in range(N):
        n_caps_tot += n_caps[i]
    has_pcm = n_caps_tot > 0
    m_pcm_node = n_caps * m_cap
    m_pcm_tot = n_caps_tot * m_cap
    cp_min_eff = min(cps, cpl) + x

    T = T0.copy()
    h = np.zeros(N)
    Tp = np.zeros(N)
    f = np.zeros(N)
    for i in range(N):
        if n_caps[i] > 0:
            h[i] = h0[i]
            Tp[i], f[i] = _T_f_of_h(h[i], Ts, Tl, L, cps, cpl, x)
        else:
            Tp[i] = T[i]
    h_mains = _h_of_T(mains, Ts, Tl, L, cps, cpl, x)
    h_target = _h_of_T(target, Ts, Tl, L, cps, cpl, x)

    hr = np.zeros((n_hours, 21))
    node_T = np.zeros((n_hours, N))
    node_h = np.zeros((n_hours, N))
    sc = np.zeros(30)     # [26] shield-water s, [27] shield-PCM s (any node), [28] s over a limit
    m_w_tot = 0.0
    for i in range(N):
        m_w_tot += m_w[i]

    e_init = 0.0
    e_init_rel = 0.0
    for i in range(N):
        e_init += m_w[i] * cp * T[i] + m_pcm_node[i] * h[i]
        e_init_rel += m_w[i] * cp * (T[i] - mains) + m_pcm_node[i] * (h[i] - h_mains)
    sc[7] = e_init
    sc[24] = e_init_rel

    max_water = T.max()
    max_pcm = -1e9
    for i in range(N):
        if n_caps[i] > 0 and Tp[i] > max_pcm:
            max_pcm = Tp[i]
    nsteps = int(round(3600.0 / dt))
    n_done = 0
    was_liquid = False
    cycles = 0.0
    fm = 0.0
    if has_pcm:
        for i in range(N):
            fm += f[i] * m_pcm_node[i] / m_pcm_tot
    was_liquid = fm >= 0.999

    for ih in range(n_hours):
        n_done = ih + 1
        tam = tamb[ih]
        a_c, b_c = coll_a[ih], coll_b[ih]
        circ_allowed = coll_circ[ih] > 0.5
        hour_mass = demand24[hod[ih]]
        mdot_d = hour_mass / 3600.0          # constant rate over the hour
        step_draw = hour_mass / nsteps
        hq_c = hq_l = hq_p = hq_loss = hq_un = hq_ln = 0.0
        h_draw = h_vol = h_edel = 0.0
        t_top_min, t_top_max, t_p_max = 1e9, -1e9, -1e9
        f_min, f_max = 1e9, -1e9

        for st in range(nsteps):
            in_band = False
            if has_pcm:
                for i in range(N):
                    if n_caps[i] > 0 and Ts - margin <= Tp[i] <= Tl + margin:
                        in_band = True
            n_sub = n_band if in_band else 1
            if has_pcm:
                v_max = (mdot_c + mdot_d) / (rho_w * A_cross)
                for i in range(N):
                    if n_caps[i] > 0:
                        ua_p = _ua_node(n_caps[i], d, area_cap, v_max, f[i], rho_w, kw, pr, mu,
                                        wall_t, wall_k, k_pcm, enh, c_cond)
                        if ua_p > 0:
                            tau = m_pcm_node[i] * cp_min_eff / ua_p
                            ns = int(np.ceil(dt / (stiff_sf * tau)))
                            if ns > n_sub:
                                n_sub = ns
                if n_sub > max_sub:
                    n_sub = max_sub
            dts = dt / n_sub

            for sb in range(n_sub):
                circ = circ_allowed
                if shield > 0.5 and T[N - 1] >= byp_w:
                    circ = False
                    sc[13] += 1
                    sc[26] += dts
                k_ret = 0
                if circ:
                    t_out = T[0] + max(a_c - b_c * T[0], 0.0) / (mdot_c * cp)
                    k_ret = 0
                    for j in range(N - 1, -1, -1):
                        if T[j] <= t_out:
                            k_ret = j
                            break
                # per-node through-flow -> velocity -> UA
                UA = np.zeros(N)
                pcm_shield_now = False
                if has_pcm:
                    for i in range(N):
                        if n_caps[i] > 0:
                            thr = mdot_d + (mdot_c if circ else 0.0)
                            v = thr / (rho_w * A_cross)
                            ua = _ua_node(n_caps[i], d, area_cap, v, f[i], rho_w, kw, pr, mu,
                                          wall_t, wall_k, k_pcm, enh, c_cond)
                            if shield > 0.5 and Tp[i] >= byp_p and T[i] > Tp[i]:
                                ua = 0.0
                                sc[14] += 1
                                pcm_shield_now = True
                            UA[i] = ua
                if pcm_shield_now:
                    sc[27] += dts
                Tn = _solve_step(T, Tp, UA, m_w, loss_ua, tam, N, cp, mains, mdot_d, mdot_c, circ,
                                 k_ret, a_c, b_c, kcond, dts)
                if circ and (a_c - b_c * Tn[0]) < 0.0:
                    circ = False
                    Tn = _solve_step(T, Tp, UA, m_w, loss_ua, tam, N, cp, mains, mdot_d, mdot_c, False,
                                     0, a_c, b_c, kcond, dts)
                q_coll = max(a_c - b_c * Tn[0], 0.0) if circ else 0.0
                t_top = Tn[N - 1]
                q_load_net = mdot_d * cp * (t_top - mains)
                q_load = max(q_load_net, 0.0)
                q_un = mdot_d * cp * max(target - t_top, 0.0) if mdot_d > 0 else 0.0
                q_loss = 0.0
                for i in range(N):
                    q_loss += loss_ua[i] * (Tn[i] - tam)
                q_pcm_sum = 0.0
                f_prev_mean = 0.0
                f_new_mean = 0.0
                if has_pcm:
                    for i in range(N):
                        if n_caps[i] > 0:
                            q = UA[i] * (Tn[i] - Tp[i])
                            q_pcm_sum += q
                            hn = h[i] + q * dts / m_pcm_node[i]
                            tpn, fn = _T_f_of_h(hn, Ts, Tl, L, cps, cpl, x)
                            if tpn < -50.0 or tpn > 500.0:
                                tpn = min(max(tpn, -50.0), 500.0)
                                hn = _h_of_T(tpn, Ts, Tl, L, cps, cpl, x)
                                sc[12] += 1
                            w = m_pcm_node[i] / m_pcm_tot
                            sc[15] += w * abs(fn - f[i])
                            h[i], Tp[i], f[i] = hn, tpn, fn
                            if q >= 0:
                                sc[5] += q * dts
                            else:
                                sc[6] += -q * dts
                _mix_inversions(Tn, m_w, N)
                for i in range(N):
                    T[i] = Tn[i]
                    if n_caps[i] <= 0:
                        Tp[i] = T[i]
                # accounting
                sc[0] += q_coll * dts
                sc[1] += q_load * dts
                sc[2] += q_load_net * dts
                sc[3] += q_loss * dts
                sc[4] += q_un * dts
                hq_c += q_coll * dts
                hq_l += q_load * dts
                hq_ln += q_load_net * dts
                hq_p += q_pcm_sum * dts
                hq_loss += q_loss * dts
                hq_un += q_un * dts
                dm = step_draw / n_sub
                if dm > 0:
                    h_draw += dm
                    if t_top >= target:
                        h_vol += dm
                    h_edel += dm * cp * max(min(t_top, target) - mains, 0.0)
                t_top_min = min(t_top_min, t_top)
                t_top_max = max(t_top_max, t_top)
                bad = False
                for i in range(N):
                    max_water = max(max_water, T[i])
                    if T[i] > max_w:
                        bad = True
                    if n_caps[i] > 0:
                        max_pcm = max(max_pcm, Tp[i])
                        t_p_max = max(t_p_max, Tp[i])
                        if Tp[i] > max_p:
                            bad = True
                if bad:
                    sc[11] += 1
                    sc[28] += dts
                if has_pcm:
                    fm = 0.0
                    for i in range(N):
                        fm += f[i] * m_pcm_node[i] / m_pcm_tot
                    f_min = min(f_min, fm)
                    f_max = max(f_max, fm)

        # ---- end of hour
        t_mean = 0.0
        e_rel = 0.0
        e_rec = 0.0
        tp_mean = 0.0
        fm = 0.0
        for i in range(N):
            t_mean += T[i] * m_w[i] / m_w_tot
            e_rel += m_w[i] * cp * (T[i] - mains)
            e_rec += m_w[i] * cp * max(T[i] - target, 0.0)
            if n_caps[i] > 0:
                e_rel += m_pcm_node[i] * (h[i] - h_mains)
                e_rec += m_pcm_node[i] * max(h[i] - h_target, 0.0)
                tp_mean += Tp[i] * m_pcm_node[i] / m_pcm_tot
                fm += f[i] * m_pcm_node[i] / m_pcm_tot
            node_T[ih, i] = T[i]
            node_h[ih, i] = h[i]
        if not has_pcm:
            tp_mean = t_mean
            f_min, f_max = 0.0, 0.0
        hr[ih, 0] = T[N - 1]
        hr[ih, 1] = t_mean
        hr[ih, 2] = tp_mean
        hr[ih, 3] = fm
        hr[ih, 4] = hq_c / 3600.0
        hr[ih, 5] = hq_l / 3600.0
        hr[ih, 6] = hq_p / 3600.0
        hr[ih, 7] = hq_loss / 3600.0
        hr[ih, 8] = hq_un / 3600.0
        hr[ih, 9] = h_draw
        hr[ih, 10] = h_vol
        hr[ih, 11] = h_edel / 3600.0
        hr[ih, 12] = t_top_min
        hr[ih, 13] = t_top_max
        hr[ih, 14] = t_p_max if has_pcm else t_mean
        hr[ih, 15] = f_min
        hr[ih, 16] = f_max
        hr[ih, 17] = e_rel / 3.6e6
        hr[ih, 18] = e_rec / 3.6e6
        hr[ih, 19] = T[0]
        hr[ih, 20] = hq_ln / 3600.0
        sc[16] += fm
        sc[17] += 1.0 if T[N - 1] >= target else 0.0
        sc[18] += h_draw
        sc[19] += h_vol
        sc[20] += h_edel
        if has_pcm:
            if fm >= 0.999 and not was_liquid:
                was_liquid = True
            if fm <= 0.001 and was_liquid:
                cycles += 1.0
                was_liquid = False
            sc[21] = min(sc[21], fm) if ih > 0 else fm
            sc[22] = max(sc[22], fm)
        if stop_thr > -1e8 and T[N - 1] < stop_thr:
            break

    e_fin = 0.0
    e_fin_rel = 0.0
    for i in range(N):
        e_fin += m_w[i] * cp * T[i] + m_pcm_node[i] * h[i]
        e_fin_rel += m_w[i] * cp * (T[i] - mains) + m_pcm_node[i] * (h[i] - h_mains)
    sc[8] = e_fin
    sc[9] = max_water
    sc[10] = max_pcm
    sc[23] = n_done
    sc[25] = cycles
    return hr, node_T, node_h, sc


# ------------------------------------------------------------------ python wrapper
def _collector_arrays(weather: pd.DataFrame, system_config: dict):
    c = system_config["collector"]
    I = weather["GHI_Wm2"].to_numpy(dtype=float)
    T = weather["T_amb_C"].to_numpy(dtype=float)
    circ = (I > c["min_irradiance_cutoff_Wm2"]).astype(float)
    a = circ * c["area_m2"] * (c["fr_tau_alpha"] * I + c["fr_ul_W_m2K"] * T)
    b = circ * c["area_m2"] * c["fr_ul_W_m2K"]
    return a, b, circ


def run_year_multinode(weather_hourly, demand, mains_temp_C, design, pcm_props, system_config,
                       record_hourly=True, initial_state=None, stop_when_Tw_below_C=None, layout=None):
    """Same contract as tank_model.run_year; `layout` (MultiNodeLayout) required."""
    if layout is None:
        raise ValueError("run_year_multinode needs a MultiNodeLayout (see layout_from_geometry)")
    N = layout.n_nodes
    solver = system_config["solver"]
    dt = float(solver["timestep_s"])
    water = system_config["water"]
    cp_w, rho_w = water["cp_J_kgK"], water["density_kg_m3"]
    integ = system_config["pcm_integration"]
    shield_cfg = system_config.get("safety_shield", {})
    max_w = system_config["safety"]["max_water_temp_C"]
    max_p = system_config["safety"]["max_pcm_temp_C"]
    target = system_config["delivery"]["target_temp_C"]

    n_caps = layout.capsules_per_node.astype(float)
    has_pcm = n_caps.sum() > 0
    v_node = design.tank_volume_m3 / N
    if has_pcm:
        wall_m = (design.capsule_wall_thickness_m if design.capsule_wall_thickness_m is not None
                  else integ["capsule_wall_thickness_m"])
        masses = capsule_masses(int(n_caps.sum()), design.capsule_diameter_m, wall_m, pcm_props.density_kg_m3)
        m_cap = masses["m_pcm_kg"] / n_caps.sum()
        v_cap_out = np.pi / 6.0 * design.capsule_diameter_m ** 3
        props = _dc_replace(pcm_props, wall_cp_J_kgK=wall_cp_per_kg_pcm(masses))
    else:
        masses, m_cap, v_cap_out, props = None, 0.0, 0.0, pcm_props
    m_w = np.maximum(v_node - n_caps * v_cap_out, 1e-9) * rho_w

    Dt, Ht = layout.tank_diameter_m, layout.tank_height_m
    A_cross = np.pi / 4.0 * Dt ** 2
    dz = Ht / N
    side = np.pi * Dt * Ht / N
    loss_ua = np.full(N, system_config["tank"]["U_tank_W_m2K"] * side)
    loss_ua[0] += system_config["tank"]["U_tank_W_m2K"] * A_cross
    loss_ua[N - 1] += system_config["tank"]["U_tank_W_m2K"] * A_cross
    kcond = water["conductivity_W_mK"] * A_cross / dz

    a, b, circ = _collector_arrays(weather_hourly, system_config)
    hod = np.array([_demand_hour_of_day(weather_hourly.iloc[i], i) for i in range(len(weather_hourly))],
                   dtype=np.int64) if len(weather_hourly) else np.zeros(0, dtype=np.int64)
    demand24 = np.array([demand.draw_mass_kg_for_hour(h) for h in range(24)], dtype=float)

    # ---- initial state
    if initial_state and "T_nodes" in initial_state:
        T0 = np.asarray(initial_state["T_nodes"], dtype=float).copy()
    else:
        T0 = np.full(N, initial_state["T_w_C"] if initial_state else solver["initial_water_temp_C"], dtype=float)
    h0 = np.zeros(N)
    if has_pcm:
        if initial_state and "h_nodes" in initial_state:
            h0 = np.asarray(initial_state["h_nodes"], dtype=float).copy()
        else:
            if initial_state and "T_pcm_C" in initial_state:
                tp0 = initial_state["T_pcm_C"]
            elif solver["initial_pcm_state"] == "solid":
                tp0 = props.Ts_C - 5.0
            elif solver["initial_pcm_state"] == "liquid":
                tp0 = props.Tl_C + 5.0
            else:
                tp0 = props.Tm_C
            h0[:] = enthalpy_of_temperature(tp0, props)

    P = np.array([
        cp_w, rho_w, mains_temp_C, target, design.flow_rate_kg_s, dt, kcond, A_cross,
        props.Ts_C, props.Tl_C, props.latent_heat_J_kg, props.cp_solid_J_kgK, props.cp_liquid_J_kgK,
        props.wall_cp_J_kgK, props.conductivity_W_mK, integ["liquid_convection_enhancement"],
        water["conductivity_W_mK"], water["prandtl"], _mu(),
        integ["capsule_wall_thickness_m"], integ["capsule_wall_conductivity_W_mK"],
        design.capsule_diameter_m, design.capsule_area_m2, m_cap,
        solver["melt_band_margin_K"], 0.5, solver["substeps_in_melt_band"], 60,
        1.0 if shield_cfg.get("enabled", False) else 0.0,
        shield_cfg.get("bypass_water_C", max_w - 3.0), shield_cfg.get("bypass_pcm_C", max_p - 3.0),
        max_w, max_p, -1e9 if stop_when_Tw_below_C is None else float(stop_when_Tw_below_C),
        float(integ.get("conduction_coefficient_c", 5.0)),
    ], dtype=float)
    P = np.concatenate([P, np.zeros(40 - len(P))])
    hr, node_T, node_h, sc = _kernel(np.zeros(1), weather_hourly["T_amb_C"].to_numpy(dtype=float), hod,
                                     a, b, circ, demand24, m_w, n_caps, loss_ua, P, T0, h0, 1.0)
    n_done = int(sc[23])
    hr, node_T, node_h = hr[:n_done], node_T[:n_done], node_h[:n_done]

    e_init, e_fin = sc[7], sc[8]
    denom = max(abs(sc[0]), 1.0)
    residual = (sc[0] + e_init) - (sc[2] + sc[3] + e_fin)
    energy = {
        "E_collector_kWh": sc[0] / J_PER_KWH, "E_load_kWh": sc[1] / J_PER_KWH, "E_loss_kWh": sc[3] / J_PER_KWH,
        "E_pump_kWh": 0.0, "E_unmet_kWh": sc[4] / J_PER_KWH, "E_charge_kWh": sc[5] / J_PER_KWH,
        "E_discharge_kWh": sc[6] / J_PER_KWH, "E_initial_kWh": e_init / J_PER_KWH, "E_final_kWh": e_fin / J_PER_KWH,
        "residual_J": residual, "residual_pct_of_collector": abs(residual) / denom * 100.0,
        "n_steps": 0, "n_failed_steps": int(sc[12]),
    }
    hourly = pd.DataFrame()
    if record_hourly:
        hod_rec = hod[:n_done]
        hourly = pd.DataFrame({
            "hour_index": np.arange(n_done), "hour_of_day": hod_rec,
            "I_t_Wm2": weather_hourly["GHI_Wm2"].to_numpy()[:n_done],
            "T_amb_C": weather_hourly["T_amb_C"].to_numpy()[:n_done],
            "T_w_C": hr[:, 0], "T_w_mean_C": hr[:, 1], "T_pcm_C": hr[:, 2], "f_melt": hr[:, 3],
            "Q_collector_Wh": hr[:, 4], "Q_load_Wh": hr[:, 5], "Q_pcm_Wh": hr[:, 6], "Q_loss_Wh": hr[:, 7],
            "Q_pump_Wh": 0.0, "Q_unmet_Wh": hr[:, 8],
            "draw_kg": hr[:, 9], "vol_ge_target_kg": hr[:, 10], "Edel_target_Wh": hr[:, 11],
            "T_w_min_C": hr[:, 12], "T_w_max_C": hr[:, 13], "T_pcm_max_C": hr[:, 14],
            "f_melt_min": hr[:, 15], "f_melt_max": hr[:, 16],
            "E_stored_kWh": hr[:, 17], "E_rec45_kWh": hr[:, 18], "T_bottom_C": hr[:, 19],
            "Q_load_net_Wh": hr[:, 20],
        })
        hourly.attrs["node_T"], hourly.attrs["node_h"] = node_T, node_h

    summary = {
        "mean_f_melt": sc[16] / n_done if n_done else 0.0, "min_f_melt": sc[21], "max_f_melt": sc[22],
        "delivery_temp_hours": int(sc[17]), "m_pcm_kg": masses["m_pcm_kg"] if has_pcm else 0.0,
        "m_wall_kg": masses["m_wall_kg"] if has_pcm else 0.0, "m_water_kg": float(m_w.sum()),
        "sum_abs_df": sc[15], "E_del_target_kWh": sc[20] / J_PER_KWH, "draw_kg_total": sc[18],
        "vol_ge_target_kg": sc[19], "E_stored_initial_kWh": sc[24] / J_PER_KWH,
        "shield_water_h": sc[26] / 3600.0, "shield_pcm_h": sc[27] / 3600.0, "hours_over_limit": sc[28] / 3600.0,
        "E_stored_final_kWh": float(hr[-1, 17]) if n_done else 0.0,
    }
    context = {"m_w_kg": float(m_w.sum()), "m_pcm_kg": summary["m_pcm_kg"], "pcm_props": props if has_pcm else None,
               "cp_w": cp_w, "mains_temp_C": mains_temp_C, "target_C": target, "has_pcm": has_pcm, "dt_s": dt,
               "ua_tank_W_K": float(loss_ua.sum()), "n_nodes": N, "node_resolved": True}
    return SimulationResult(
        hourly=hourly, energy=energy, max_water_temp_C=sc[9], max_pcm_temp_C=sc[10] if has_pcm else sc[9],
        n_safety_violations=int(sc[11]), final_f_melt=float(hr[-1, 3]) if n_done else 0.0,
        complete_melt_cycles=int(sc[25]), initial_f_melt=0.0, n_clipped_steps=int(sc[12]),
        n_shield_water_activations=int(sc[13]), n_shield_pcm_activations=int(sc[14]),
        summary=summary, context=context,
    )
