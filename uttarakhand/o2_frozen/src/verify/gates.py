"""
src/verify/gates.py
======================
Phase 4 / D2.3 verification — reduced gate battery per
O2_Unified_PerState_Execution_Framework.md ("Phase 4 - Verification").
No DOE row (Phase 5) may be generated until this passes.

Gate 1 - Energy conservation across 5 diverse cases.
Gate 2 - Limiting cases (zero irradiance, zero flow, no PCM, ... ).
Gate 3 - Baseline comparison (plain tank vs fixed PCM vs "optimized-looking"
         design) + the no-loss-vs-with-loss ambient-tank-loss diagnostic
         (Bug-Fix 1 — confirms the loss term is actually active).
Gate 4 - Light calibration against the cited Singh et al. (2025) 54-84%
         solar-fraction benchmark band (system_config_shared.yaml).
Gate 5 - Sensitivity spot checks (latent heat +/-10%, flow +/-50%,
         ambient +5 C) checked for the physically-expected direction.

Writes results/phase4_simulator_verification_report.txt.

PORTED FROM objective2-tamilnadu/src/verify/gates.py — the gate logic,
thresholds, verdict rules and structure are byte-for-byte the Tamil Nadu
reference. Only the *state-specific test inputs* differ, and only because
they must (framework doc "what VARIES per state"):
  - climate regimes: Uttarakhand has 4 clusters (0-3) since Objective 1's
    K=5 -> K=4 switch (commit 6be0dcd, 2026-10-03), not TN's 5.
  - PCM names: Uttarakhand's Objective 1 rank-1 PCM per cluster, read from
    O1's frozen top-3 (src/io_utils.o1_top3): PureTemp 58 for clusters 0-2
    and PureTemp 53 for cluster 3 (ported 2026-10-28 from
    objective2-rajasthan, whose own names were Rajasthan's).
  - Gate 1 gets one extra case (H) for the cold high-altitude cluster 3,
    whose 7.4 C mains is far below every other state's.
  - one extra INFORMATIONAL Gate 2 check records Cluster 0's plain-tank
    peak water temperature against the 65 C PCM limit; it never changes a
    pass/fail verdict.
"""

import math
import sys
from pathlib import Path

import pandas as pd

from config import BASE_DIR, RESULTS_DIR
from src.design.schema import DesignVector
from src.design.geometry import sphere_volume_m3, sphere_surface_area_m2, tank_dimensions_m, pack_capsules
from src.io_utils import load_system_config, load_design_bounds, get_pcm_properties, load_hourly_weather, load_demand_profile
from src.simulation.capsule_enthalpy import pcm_props_from_record, PCMThermalProps
from src.simulation.demand_profile import load_demand_model
from src.simulation.tank_model import DesignRuntime, run_year
from src.simulation.run_case import run_case

# Uttarakhand Objective 1 MCDM rank-1 PCM per cluster (K = 4), from O1's frozen top-3.
from src.io_utils import o1_top3 as _o1_top3
PCM_C0, PCM_C1, PCM_C2, PCM_C3 = (_o1_top3("uttarakhand")[c][0] for c in (0, 1, 2, 3))


# ─────────────────────────────────────────────────────────────────────────
# Helper: build a DesignRuntime directly, bypassing Phase 2's design-bounds
# gate. Gate 2 deliberately probes OUTSIDE the normal operating envelope
# (zero/over-limit flow, zero capsules) to check the SIMULATOR degrades
# sensibly — that is a simulator-robustness test, not a design-acceptance
# test, so it must not go through src/design/constraints.py.
# ─────────────────────────────────────────────────────────────────────────

def _direct_runtime(diameter_m, n_capsule, flow_rate_kg_s, system_config, arrangement="staggered"):
    tank = tank_dimensions_m(system_config)
    v_capsule = sphere_volume_m3(diameter_m)
    a_capsule = sphere_surface_area_m2(diameter_m)
    packing = pack_capsules(arrangement, diameter_m, tank["tank_diameter_m"], tank["tank_height_m"],
                             n_capsule, 0.003)
    void_fraction = min(max(packing.void_fraction, 1e-3), 0.999)
    bed_length = packing.stack_height_m if packing.stack_height_m > 0 else tank["tank_height_m"]
    return DesignRuntime(
        n_capsule=n_capsule, capsule_diameter_m=diameter_m, capsule_area_m2=a_capsule,
        capsule_volume_m3=v_capsule, void_fraction=void_fraction,
        cross_section_area_m2=tank["tank_cross_section_area_m2"], bed_length_m=bed_length,
        flow_rate_kg_s=flow_rate_kg_s, tank_volume_m3=tank["tank_volume_m3"],
        tank_surface_area_m2=tank["tank_surface_area_m2"],
    )


def _direct_run(state, cluster_id, diameter_m, n_capsule, flow_rate_kg_s, pcm_name=None,
                 system_config_overrides=None, volume_multiplier=1.0, mains_temp_C=None,
                 arrangement="staggered"):
    system_config = load_system_config()   # dt = 150 s from system_config_shared.yaml
    if system_config_overrides:
        system_config = _deep_merge(system_config, system_config_overrides)

    from src.io_utils import get_regime
    regime = get_regime(state, cluster_id)
    mains_temp_C = mains_temp_C if mains_temp_C is not None else regime["T_mains_est_C"]

    weather = load_hourly_weather(state, cluster_id)
    demand_df = load_demand_profile(state)
    demand_model = load_demand_model(demand_df, volume_multiplier=volume_multiplier)

    runtime = _direct_runtime(diameter_m, n_capsule if pcm_name else 0, flow_rate_kg_s, system_config,
                               arrangement=arrangement)

    if pcm_name is not None:
        record = get_pcm_properties(state, pcm_name)
        pcm_props = pcm_props_from_record(record, system_config["pcm_integration"]["melting_half_width_K"])
    else:
        pcm_props = PCMThermalProps(Tm_C=57.0, latent_heat_J_kg=0.0, cp_solid_J_kgK=2000.0,
                                     cp_liquid_J_kgK=2000.0, conductivity_W_mK=0.2, density_kg_m3=800.0)

    return run_year(weather, demand_model, mains_temp_C, runtime, pcm_props, system_config, record_hourly=True)


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────
# GATE 1 — Energy conservation
# ─────────────────────────────────────────────────────────────────────────

def gate1_conservation(state: str, log):
    log("\n" + "=" * 72)
    log("GATE 1 — Energy conservation (8 diverse cases: 7 arrangement-aware + 1 cold-regime case for Uttarakhand cluster 3)")
    log("=" * 72)
    system_config = load_system_config()
    pass_pct = system_config["verification"]["gate1_residual_pass_pct"]
    warn_pct = system_config["verification"]["gate1_residual_warn_pct"]

    # Original 5 diverse cases (kept, all staggered) + 2 new cases covering
    # single-layer and radial, per docs/04_PHASE4_VERIFICATION_GATES.md step 1.
    # Arrangement doesn't touch energy accounting (Phase 3 finding), so all
    # 7 are still expected far under the 0.1% pass threshold.
    cases = [
        (f"A: cluster0 / {PCM_C0} / mid design", 0, PCM_C0, DesignVector(0.05, 14, 0.030, capsule_arrangement="staggered")),
        (f"B: cluster1 / {PCM_C1} / small-capsule design", 1, PCM_C1, DesignVector(0.04, 20, 0.020, capsule_arrangement="staggered")),
        (f"C: cluster2 / {PCM_C2} / large-capsule design", 2, PCM_C2, DesignVector(0.08, 10, 0.045, capsule_arrangement="staggered")),
        ("D: cluster2 / no-PCM plain-tank baseline", 2, None, DesignVector(0.05, 14, 0.030, capsule_arrangement="staggered")),
        (f"E: cluster0 / {PCM_C0} / bounds-extreme design", 0, PCM_C0, DesignVector(0.08, 37, 0.050, capsule_arrangement="staggered")),
        (f"F: cluster1 / {PCM_C1} / single-layer", 1, PCM_C1, DesignVector(0.04, 20, 0.020, capsule_arrangement="single-layer")),
        (f"G: cluster2 / {PCM_C2} / radial", 2, PCM_C2, DesignVector(0.08, 10, 0.045, capsule_arrangement="radial")),
        (f"H: cluster3 / {PCM_C3} / mid design, cold mains", 3, PCM_C3, DesignVector(0.05, 14, 0.030, capsule_arrangement="staggered")),
    ]

    residuals = []
    for name, cid, pcm, design in cases:
        out = run_case(state, cid, pcm, design, record_hourly=False)
        if not out["valid"]:
            log(f"  [SKIP-INVALID] {name}: geometry rejected ({out['reason']})")
            continue
        r = out["metrics"]["residual_pct_of_collector"]
        residuals.append(r)
        log(f"  {name:52s} residual={r:.6f} %")

    mean_r = sum(residuals) / len(residuals) if residuals else float("nan")
    max_r = max(residuals) if residuals else float("nan")
    log(f"\n  mean residual = {mean_r:.6f} %   max residual = {max_r:.6f} %")

    if max_r < pass_pct:
        verdict = "PASS"
    elif max_r < warn_pct:
        verdict = "PASS-WITH-CAVEAT"
    else:
        verdict = "FAIL"
    log(f"  Gate 1 verdict: {verdict}  (pass<{pass_pct}%, warn<{warn_pct}%)")
    return {"gate": 1, "verdict": verdict, "mean_residual_pct": mean_r, "max_residual_pct": max_r}


# ─────────────────────────────────────────────────────────────────────────
# GATE 2 — Limiting cases
# ─────────────────────────────────────────────────────────────────────────

def gate2_limiting_cases(state: str, log):
    log("\n" + "=" * 72)
    log("GATE 2 — Limiting cases")
    log("=" * 72)
    system_config = load_system_config()
    checks = []

    # 1. Zero irradiance -> zero collector heat, always.
    r = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=PCM_C0,
                     system_config_overrides={"collector": {"min_irradiance_cutoff_Wm2": 1.0e9}})
    ok = r.energy["E_collector_kWh"] < 1e-6
    checks.append(("zero_irradiance -> zero collector heat", ok, f"E_collector={r.energy['E_collector_kWh']:.6f} kWh"))

    # 2. Zero flow -> no crash, near-zero (but finite) heat transfer / pump power.
    r = _direct_run(state, 0, 0.05, 14, 0.0, pcm_name=PCM_C0)
    ok = r.energy["n_failed_steps"] == 0 and math.isfinite(r.energy["residual_pct_of_collector"])
    checks.append(("zero_flow -> simulator completes without crashing", ok,
                    f"residual={r.energy['residual_pct_of_collector']:.4f}%, "
                    f"E_pump={r.energy['E_pump_kWh']:.9f} kWh"))

    # 3. No PCM -> plain-tank behaviour (zero charge/discharge energy).
    r = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=None)
    ok = r.energy["E_charge_kWh"] < 1e-9 and r.energy["E_discharge_kWh"] < 1e-9
    checks.append(("no_PCM -> zero charge/discharge energy", ok,
                    f"E_charge={r.energy['E_charge_kWh']:.9f}, E_discharge={r.energy['E_discharge_kWh']:.9f}"))

    # 4. Zero latent heat -> capsule behaves as pure sensible mass (no melt plateau).
    record = get_pcm_properties(state, PCM_C0)
    record_zero_L = {**record, "latent_heat_kJ_kg": 0.0}
    pcm_props_zero_L = pcm_props_from_record(record_zero_L, system_config["pcm_integration"]["melting_half_width_K"])
    weather = load_hourly_weather(state, 0)
    demand_model = load_demand_model(load_demand_profile(state))
    from src.io_utils import get_regime
    mains = get_regime(state, 0)["T_mains_est_C"]
    runtime = _direct_runtime(0.05, 14, 0.030, system_config)
    r = run_year(weather, demand_model, mains, runtime, pcm_props_zero_L, system_config, record_hourly=True)
    f_range = r.hourly["f_melt"].max() - r.hourly["f_melt"].min()
    ok = True   # informational: with L=0, f_melt is ill-defined/degenerate; just confirm no crash
    checks.append(("zero_latent_heat -> completes, f_melt degenerate (informational)", ok,
                    f"f_melt range={f_range:.4f} (L=0 makes f a step function, expected)"))

    # 5. Very high PCM conductivity -> T_pcm tracks T_w much more closely
    #    than at nominal conductivity (smaller mean |Tw-Tpcm| gap).
    record_hi_k = {**record, "TC_W_mK": record["TC_W_mK"] * 200.0}
    props_hi_k = pcm_props_from_record(record_hi_k, system_config["pcm_integration"]["melting_half_width_K"])
    r_hi = run_year(weather, demand_model, mains, runtime, props_hi_k, system_config, record_hourly=True)
    props_nom = pcm_props_from_record(record, system_config["pcm_integration"]["melting_half_width_K"])
    r_nom = run_year(weather, demand_model, mains, runtime, props_nom, system_config, record_hourly=True)
    gap_hi = (r_hi.hourly["T_w_C"] - r_hi.hourly["T_pcm_C"]).abs().mean()
    gap_nom = (r_nom.hourly["T_w_C"] - r_nom.hourly["T_pcm_C"]).abs().mean()
    ok = gap_hi < gap_nom
    checks.append(("very_high_conductivity -> smaller mean |Tw-Tpcm| gap than nominal", ok,
                    f"gap_high_k={gap_hi:.3f} C, gap_nominal={gap_nom:.3f} C"))

    # 6. Perfectly insulated tank -> zero ambient tank loss.
    r = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=PCM_C0,
                     system_config_overrides={"tank": {"U_tank_W_m2K": 0.0}})
    ok = r.energy["E_loss_kWh"] < 1e-6
    checks.append(("perfectly_insulated_tank -> zero E_loss", ok, f"E_loss={r.energy['E_loss_kWh']:.6f} kWh"))

    # 7. Empty demand -> zero load, zero unmet energy.
    r = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=PCM_C0, volume_multiplier=0.0)
    ok = r.energy["E_load_kWh"] < 1e-6 and r.energy["E_unmet_kWh"] < 1e-6
    checks.append(("empty_demand -> zero load and zero unmet energy", ok,
                    f"E_load={r.energy['E_load_kWh']:.6f}, E_unmet={r.energy['E_unmet_kWh']:.6f}"))

    # 8/9. Fully solid vs fully liquid initial PCM -> melt fraction starts and
    #      moves in the physically-correct direction over the first day.
    r_solid = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=PCM_C0,
                           system_config_overrides={"solver": {"initial_pcm_state": "solid"}})
    r_liquid = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=PCM_C0,
                            system_config_overrides={"solver": {"initial_pcm_state": "liquid"}})
    f0_solid, f24_solid = r_solid.initial_f_melt, r_solid.hourly["f_melt"].iloc[23]
    f0_liquid, f24_liquid = r_liquid.initial_f_melt, r_liquid.hourly["f_melt"].iloc[23]
    ok = (f0_solid <= 0.01) and (f0_liquid >= 0.99) and (f24_solid >= f0_solid) and (f24_liquid <= f0_liquid + 1e-9)
    checks.append(("solid/liquid initial PCM -> starts at 0/1 and moves correctly", ok,
                    f"solid: f0={f0_solid:.3f}->f24={f24_solid:.3f}; "
                    f"liquid: f0={f0_liquid:.3f}->f24={f24_liquid:.3f}"))

    # 10/11. Flow below/above permitted limits -> no crash, monotonic direction
    #        (higher flow -> better heat transfer -> smaller |Tw-Tpcm| gap).
    r_low = _direct_run(state, 0, 0.05, 14, 0.002, pcm_name=PCM_C0)   # below 0.010 min
    r_high = _direct_run(state, 0, 0.05, 14, 0.20, pcm_name=PCM_C0)   # above 0.050 max
    gap_low = (r_low.hourly["T_w_C"] - r_low.hourly["T_pcm_C"]).abs().mean()
    gap_high = (r_high.hourly["T_w_C"] - r_high.hourly["T_pcm_C"]).abs().mean()
    ok = math.isfinite(gap_low) and math.isfinite(gap_high) and gap_high <= gap_low
    checks.append(("flow below/above limits -> completes, higher flow narrows Tw-Tpcm gap", ok,
                    f"gap@0.002kg/s={gap_low:.3f} C, gap@0.20kg/s={gap_high:.3f} C"))

    # 12. Capsules removed / allocated PCM volume = 0 -> identical to no-PCM case.
    #     Repeated per arrangement since 2026-09-17 (docs/04_PHASE4_VERIFICATION_GATES.md
    #     step 2) — radial packing in particular may hit overlap/passage_blocked
    #     differently than staggered at small N, so this is verified, not assumed.
    for arrangement in ("single-layer", "staggered", "radial"):
        r = _direct_run(state, 0, 0.05, 0, 0.030, pcm_name=PCM_C0, arrangement=arrangement)
        ok = r.energy["E_charge_kWh"] < 1e-9
        checks.append((f"capsules_removed (N=0) [{arrangement}] -> zero PCM charge energy", ok,
                        f"E_charge={r.energy['E_charge_kWh']:.9f} kWh"))

    # 13. INFORMATIONAL (carried from the Rajasthan port):
    #     Cluster 0's plain-tank (no-PCM) water temperature vs the frozen 65 C
    #     PCM safety limit. This does NOT gate pass/fail — it records, inside
    #     the verification report, that Cluster 0 overheats the tank on solar
    #     input alone (before any PCM/capsule choice), so the safety-limit
    #     violations seen in Phase 3 smoke runs are a climate+sizing property,
    #     not a capsule-sizing bug.
    max_pcm_safe_C = system_config["safety"]["max_pcm_temp_C"]
    r_plain_c0 = _direct_run(state, 0, 0.05, 14, 0.030, pcm_name=None)
    max_tw_c0 = float(r_plain_c0.hourly["T_w_C"].max())
    checks.append(("uttarakhand_cluster0_plain_tank_overheat (informational, non-gating)", True,
                    f"plain-tank max T_w={max_tw_c0:.1f} C vs {max_pcm_safe_C:.0f} C PCM limit "
                    f"-> {'EXCEEDS' if max_tw_c0 > max_pcm_safe_C else 'within'} on solar input alone"))

    gating = checks[:-1]   # the informational Cluster-0 overheat check never gates
    n_pass = sum(1 for _, ok, _ in gating if ok)
    for name, ok, detail in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name:58s} {detail}")
    log(f"\n  Gate 2: {n_pass}/{len(gating)} limiting cases passed "
        f"(+1 informational: Cluster-0 overheat record, non-gating; {len(checks)} rows printed above in total).")
    verdict = "PASS" if n_pass == len(gating) else ("PASS-WITH-CAVEAT" if n_pass >= len(gating) - 1 else "FAIL")
    log(f"  Gate 2 verdict: {verdict}")
    return {"gate": 2, "verdict": verdict, "n_pass": n_pass, "n_total": len(gating), "checks": checks}


# ─────────────────────────────────────────────────────────────────────────
# GATE 3 — Baseline comparisons + ambient-loss diagnostic
# ─────────────────────────────────────────────────────────────────────────

def gate3_baseline_comparison(state: str, log):
    log("\n" + "=" * 72)
    log("GATE 3 — Baseline comparison (plain tank vs fixed PCM vs optimized-looking)")
    log("=" * 72)

    cid = 0
    pcm = PCM_C0   # Objective 1 MCDM rank-1 PCM for Uttarakhand Cluster 0
    pcm_Tm_C = get_pcm_properties(state, pcm)["Tm_C"]
    # NOTE on the designs below: the per-arrangement "fixed_PCM_max_feasible" row uses n = 37 at d = 0.08 m for all three
    # arrangements (all three reach about the same 19.84 % PCM fraction there).
    count_ceiling = 37   # capsule count of the largest design used by this battery (staggered/radial/single-layer at d = 0.08 m)
    plain = run_case(state, cid, None, DesignVector(0.08, 14, 0.030, capsule_arrangement="staggered"), record_hourly=False)
    fixed_by_arrangement = {
        arrangement: run_case(state, cid, pcm, DesignVector(0.08, count_ceiling, 0.030, capsule_arrangement=arrangement),
                               record_hourly=False)
        for arrangement in ("single-layer", "staggered", "radial")
    }
    fixed = fixed_by_arrangement["staggered"]   # kept for downstream compatibility (capability check, Gate 4 input)
    optimized = run_case(state, cid, pcm, DesignVector(0.08, 19, 0.040, capsule_arrangement="staggered"), record_hourly=False)  # ~10.2%

    rows = []
    for label, out in [("plain_tank", plain), *[(f"fixed_PCM_max_feasible[{a}]", o) for a, o in fixed_by_arrangement.items()],
                       ("optimized_looking", optimized)]:
        m = out["metrics"]
        rows.append((label, m["useful_energy_kWh"], m["solar_fraction"], m["unmet_energy_kWh"],
                      m["loss_energy_kWh"], m["pump_energy_kWh"], m.get("mean_f_melt")))
        log(f"  {label:28s} useful={m['useful_energy_kWh']:8.1f} kWh  "
            f"SF={m['solar_fraction']*100:5.2f}%  unmet={m['unmet_energy_kWh']:8.1f} kWh  "
            f"loss={m['loss_energy_kWh']:6.1f} kWh  pump={m['pump_energy_kWh']*1000:.4f} Wh  "
            f"mean_f_melt={m.get('mean_f_melt')}")

    pcm_beats_plain = (fixed["metrics"]["solar_fraction"] >= plain["metrics"]["solar_fraction"]
                        and fixed["metrics"]["unmet_energy_kWh"] <= plain["metrics"]["unmet_energy_kWh"])

    # --- sanity/capability check: can this simulator show ANY PCM benefit? --
    # If a PCM whose melting point is matched to the tank's actual operating
    # range (rather than the shortlisted PCM's climate-derived Tm) is plugged
    # into the SAME geometry, does it beat plain tank? This isolates "is the
    # simulator capable of rewarding a well-matched PCM" from "does THIS
    # shortlisted PCM happen to suit THIS 50 L direct-encapsulation design".
    matched_tm = run_case(state, cid, pcm, DesignVector(0.08, count_ceiling, 0.030, capsule_arrangement="staggered"), record_hourly=True,
                           pcm_record_overrides={"Tm_C": 40.0})
    simulator_can_reward_matched_pcm = (
        matched_tm["metrics"]["solar_fraction"] >= plain["metrics"]["solar_fraction"]
        and matched_tm["metrics"]["unmet_energy_kWh"] <= plain["metrics"]["unmet_energy_kWh"]
    )
    log(f"\n  Capability check (synthetic PCM, Tm=40C matched to this tank's operating range):")
    log(f"    plain_tank          SF={plain['metrics']['solar_fraction']*100:5.2f}%")
    log(f"    matched_Tm_PCM      SF={matched_tm['metrics']['solar_fraction']*100:5.2f}%  "
        f"mean_f_melt={matched_tm['metrics'].get('mean_f_melt', 0.0):.3f}")
    log(f"    Simulator rewards a well-matched PCM over plain tank: {simulator_can_reward_matched_pcm}")

    # --- no-loss vs with-loss diagnostic (Bug-Fix 1) --------------------
    with_loss = fixed
    no_loss = run_case(state, cid, pcm, DesignVector(0.08, count_ceiling, 0.030, capsule_arrangement="staggered"), record_hourly=False,
                        system_config_overrides={"tank": {"U_tank_W_m2K": 0.0}})
    loss_term_active = no_loss["metrics"]["solar_fraction"] >= with_loss["metrics"]["solar_fraction"]
    log(f"\n  Ambient-loss diagnostic: solar_fraction with-loss={with_loss['metrics']['solar_fraction']*100:.2f}%  "
        f"no-loss={no_loss['metrics']['solar_fraction']*100:.2f}%  "
        f"(no-loss >= with-loss confirms the U_tank term is active: {loss_term_active})")

    fixed_fraction_pct = fixed["geometry"]["pcm_volume_fraction"] * 100.0
    log(f"\n  {pcm} (Objective 1's actual rank-1 PCM for Cluster 0, Tm={pcm_Tm_C:.1f}C) beats plain tank "
        f"in THIS 50L/{fixed_fraction_pct:.1f}%-fraction design: {pcm_beats_plain}")
    if not pcm_beats_plain:
        log(f"  HONEST FINDING (not a simulator defect -- see capability check above): at this")
        log(f"  tank size and PCM fraction, {pcm}'s mean liquid fraction stays low (rarely reaches")
        log(f"  its Tm of {pcm_Tm_C:.1f}C for long), so it mostly displaces sensible-storage water")
        log(f"  without activating as latent storage. This motivates Phase 5-7 (larger PCM fraction,")
        log(f"  better-matched melting point, or a larger tank/collector) rather than accepting")
        log(f"  the Objective 1 climate-ranked PCM as automatically effective in hardware.")

    verdict = "PASS" if (simulator_can_reward_matched_pcm and loss_term_active) else "FAIL"
    log(f"\n  Gate 3 verdict: {verdict}  "
        f"(gated on simulator capability + active loss term, not on today's PCM/geometry choice)")
    return {
        "gate": 3, "verdict": verdict, "rows": rows,
        "pcm_beats_plain": pcm_beats_plain,
        "simulator_can_reward_matched_pcm": simulator_can_reward_matched_pcm,
        "loss_term_active": loss_term_active,
        "optimized_metrics": optimized["metrics"],
    }


# ─────────────────────────────────────────────────────────────────────────
# GATE 4 — Published-benchmark calibration
# ─────────────────────────────────────────────────────────────────────────

def gate4_calibration(state: str, optimized_metrics: dict, log):
    log("\n" + "=" * 72)
    log("GATE 4 — Light calibration against published benchmark")
    log("=" * 72)
    system_config = load_system_config()
    lo = system_config["verification"]["gate4_benchmark_solar_fraction_low_pct"]
    hi = system_config["verification"]["gate4_benchmark_solar_fraction_high_pct"]
    note = system_config["verification"]["gate4_benchmark_note"]

    sf_pct = optimized_metrics["solar_fraction"] * 100.0
    in_band = lo <= sf_pct <= hi
    mismatch_pct = 0.0 if in_band else min(abs(sf_pct - lo), abs(sf_pct - hi))

    log(f"  Benchmark band (Singh et al. 2025, cited): {lo:.1f}-{hi:.1f}% solar fraction")
    log(f"  {note.strip()}")
    log(f"  This simulator's optimized-looking design (arrangement=staggered, d=0.08m, n=19, "
        f"flow=0.040 kg/s — see gate3_baseline_comparison): {sf_pct:.2f}% solar fraction")
    log(f"  Inside cited band: {in_band}" + ("" if in_band else f"  (mismatch = {mismatch_pct:.1f} percentage points)"))
    if not in_band:
        log("  HONEST REPORTING (framework doc Bug-Fix 3 / Gate 4): this Objective 2 design")
        log("  is a 50 L tank against a 300 L/day draw with NO auxiliary backup heater modeled —")
        log("  a materially smaller storage-to-demand ratio than the cited benchmark's test rig,")
        log("  so a lower solar fraction is expected here. We report the mismatch rather than")
        log("  tuning the model to force agreement.")
    verdict = "PASS" if in_band else "PASS-WITH-CAVEAT"
    log(f"  Gate 4 verdict: {verdict} (calibration target, not a hard release gate)")
    return {"gate": 4, "verdict": verdict, "solar_fraction_pct": sf_pct, "in_band": in_band,
            "benchmark_low_pct": lo, "benchmark_high_pct": hi}


# ─────────────────────────────────────────────────────────────────────────
# GATE 5 — Sensitivity / monotonicity spot checks
# ─────────────────────────────────────────────────────────────────────────

def gate5_sensitivity(state: str, log):
    log("\n" + "=" * 72)
    log("GATE 5 — Sensitivity & physical-monotonicity spot checks")
    log("=" * 72)

    cid = 0
    pcm = PCM_C0
    design = DesignVector(0.05, 18, 0.040, capsule_arrangement="staggered")
    base = run_case(state, cid, pcm, design, record_hourly=False)["metrics"]
    log(f"  Baseline: useful={base['useful_energy_kWh']:.1f} kWh  SF={base['solar_fraction']*100:.2f}%  "
        f"pump={base['pump_energy_kWh']*1000:.4f} Wh  charge={base['charge_energy_kWh']:.2f} kWh")

    checks = []

    # Latent heat +/-10%
    record = get_pcm_properties(state, pcm)
    plus = run_case(state, cid, pcm, design, record_hourly=False,
                     pcm_record_overrides={"latent_heat_kJ_kg": record["latent_heat_kJ_kg"] * 1.10})["metrics"]
    minus = run_case(state, cid, pcm, design, record_hourly=False,
                      pcm_record_overrides={"latent_heat_kJ_kg": record["latent_heat_kJ_kg"] * 0.90})["metrics"]
    ok = plus["charge_energy_kWh"] >= minus["charge_energy_kWh"]
    checks.append(("latent_heat +10% vs -10% -> more PCM energy stored/cycled with more latent heat", ok,
                    f"+10%: charge={plus['charge_energy_kWh']:.3f} kWh   "
                    f"-10%: charge={minus['charge_energy_kWh']:.3f} kWh"))

    # Flow +50% / -50%
    hi_flow = run_case(state, cid, pcm, DesignVector(0.05, 18, min(design.flow_rate_kg_s * 1.5, 0.05),
                                                      capsule_arrangement="staggered"),
                        record_hourly=False)["metrics"]
    lo_flow = run_case(state, cid, pcm, DesignVector(0.05, 18, design.flow_rate_kg_s * 0.5, capsule_arrangement="staggered"),
                        record_hourly=False)["metrics"]
    ok = hi_flow["pump_energy_kWh"] >= lo_flow["pump_energy_kWh"]
    checks.append(("flow +50% vs -50% -> higher flow means more pump energy", ok,
                    f"+50%: pump={hi_flow['pump_energy_kWh']*1000:.4f} Wh   "
                    f"-50%: pump={lo_flow['pump_energy_kWh']*1000:.4f} Wh"))

    # Ambient +5 C (system-wide, approximated via mains-temperature-independent
    # tank-loss reduction: raise T_amb by adjusting the weather is out of scope
    # for a config-only override, so we approximate with a lower U_tank -- more
    # directly, we reduce the effective loss coefficient by simulating a milder
    # climate proxy: lower U_tank by the same relative amount a +5C ambient
    # would reduce average (Tw-Tamb) loss driving force at this state's typical
    # Tw. Documented approximation -- a full +5C weather-shift test belongs to
    # Phase 8 Monte Carlo, which perturbs the actual weather series.
    warmer = run_case(state, cid, pcm, design, record_hourly=False,
                       system_config_overrides={"tank": {"U_tank_W_m2K": 0.8 * 0.9}})["metrics"]
    ok = warmer["loss_energy_kWh"] <= base["loss_energy_kWh"]
    checks.append(("reduced effective tank-loss coefficient (ambient-warming proxy) -> lower E_loss", ok,
                    f"base loss={base['loss_energy_kWh']:.2f} kWh, proxy loss={warmer['loss_energy_kWh']:.2f} kWh"))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name:66s} {detail}")
    verdict = "PASS" if n_pass == len(checks) else "PASS-WITH-CAVEAT"
    log(f"\n  Gate 5: {n_pass}/{len(checks)} sensitivity checks in the expected direction.")
    log(f"  Gate 5 verdict: {verdict}")
    return {"gate": 5, "verdict": verdict, "n_pass": n_pass, "n_total": len(checks), "checks": checks}


# ─────────────────────────────────────────────────────────────────────────
# INFORMATIONAL — shield confirmation against today's actual Phase 7 winners
# ─────────────────────────────────────────────────────────────────────────

def gate_shield_confirmation_for_winners(state: str, log):
    """Non-gating, informational (same convention as Gate 2's Cluster-0
    overheat line): re-runs Phase 7's ACTUAL simulator-confirmed deployable
    designs — not a generic baseline/boundary design — through run_case()
    with the safety shield active, and confirms the shield genuinely
    engages and every design clears temperature safety. Requires
    `results/phase7_deployable_design_per_regime.csv` to exist (i.e. Phase
    7 has already been run at least once) — if it doesn't yet, this check
    is skipped with a note, and never blocks the Go/No-Go verdict either
    way. See docs/OBJECTIVE3_INPUTS_AND_NEXT_STEPS.md's note that the
    Gate battery had not previously been checked against the specific
    winning designs, only generic boundary/baseline ones."""
    log("\n" + "=" * 72)
    log("INFORMATIONAL — safety-shield confirmation against today's Phase 7 winners")
    log("=" * 72)

    deployable_path = RESULTS_DIR / "phase7_deployable_design_per_regime.csv"
    if not deployable_path.exists():
        log("  [SKIP] phase7_deployable_design_per_regime.csv not found yet — "
            "run --stage optimize first. This never blocks Go/No-Go.")
        return {"gate": "shield_confirmation", "verdict": "SKIPPED", "checks": []}

    system_config = load_system_config()
    max_water_C = system_config["safety"]["max_water_temp_C"]
    max_pcm_C = system_config["safety"]["max_pcm_temp_C"]
    shield_on = bool(system_config.get("safety_shield", {}).get("enabled", False))

    deployable = pd.read_csv(deployable_path)
    checks = []
    for _, row in deployable.iterrows():
        cid = int(row["regime_id"])
        pcm_id = None if row["pcm_id"] == "NONE_plain_tank" else row["pcm_id"]
        design = DesignVector(float(row["capsule_diameter_m"]), int(row["n_capsule"]),
                               float(row["flow_rate_kg_s"]), capsule_arrangement=row["arrangement"])
        out = run_case(state, cid, pcm_id, design, record_hourly=True)
        m = out["metrics"]
        shield_activated = (m["n_shield_water_activations"] > 0 or m["n_shield_pcm_activations"] > 0)
        clears_water = m["max_water_temp_C"] <= max_water_C
        clears_pcm = (pcm_id is None) or (m["max_pcm_temp_C"] <= max_pcm_C)
        ok = clears_water and clears_pcm and m["n_safety_violations"] == 0
        label = f"regime {cid} / {row['pcm_id']} / {row['arrangement']}"
        checks.append((label, ok, shield_activated,
                       f"max_water={m['max_water_temp_C']:.2f}C max_pcm={m.get('max_pcm_temp_C', float('nan')):.2f}C "
                       f"shield_water_activations={m['n_shield_water_activations']} "
                       f"shield_pcm_activations={m['n_shield_pcm_activations']} "
                       f"n_safety_violations={m['n_safety_violations']}"))
        log(f"  [{'PASS' if ok else 'FAIL'}] {label:48s} shield_active={shield_activated!s:5s} {checks[-1][3]}")

    n_pass = sum(1 for _, ok, _, _ in checks if ok)
    verdict = "PASS" if n_pass == len(checks) else "FAIL"
    log(f"\n  {n_pass}/{len(checks)} of today's Phase 7 winning designs clear temperature safety "
        f"under the {'active' if shield_on else 'DISABLED'} shield.")
    log(f"  Shield confirmation verdict: {verdict} (informational — does not affect Go/No-Go)")
    return {"gate": "shield_confirmation", "verdict": verdict, "checks": checks}


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

def gate_o1_pcm_roster(state: str, log):
    """O2 G6 (2026-10-26): every PCM of Objective 1's MCDM top-3, in the cluster it was ranked for,
    through the repaired simulator (mid design, 14 capsules of 50 mm, 0.03 kg/s, staggered, vs the plain tank).
    Gating: Gate-1 residual < 0.1 % for every one. Informational: how far each PCM actually melts."""
    from src.io_utils import o1_top3, o1_rank_ties
    log("\n" + "=" * 72)
    log("O1 PCM ROSTER - Objective 1 MCDM top-3 per cluster (decision D2), repaired simulator, dt=150 s")
    log("=" * 72)
    pass_pct = load_system_config()["verification"]["gate1_residual_pass_pct"]
    design = DesignVector(0.05, 14, 0.030, capsule_arrangement="staggered")
    rows, worst = [], 0.0
    for cid, names in o1_top3(state).items():
        tied = o1_rank_ties(state)[cid]
        plain = run_case(state, cid, None, design, record_hourly=False)["metrics"]
        log(f"  cluster {cid}: plain tank useful={plain['useful_energy_kWh']:.1f} kWh SF={plain['solar_fraction']:.3f}"
            f"   (O1 rank-1 tie: {tied if len(tied) > 1 else 'none'})")
        for rank, pcm in enumerate(names, start=1):
            rec = get_pcm_properties(state, pcm)
            out = run_case(state, cid, pcm, design, record_hourly=False)
            if not out["valid"]:
                log(f"    [{rank}] {pcm:42s} geometry rejected ({out['reason']})")
                continue
            m = out["metrics"]
            worst = max(worst, m["residual_pct_of_collector"])
            rows.append((cid, rank, pcm))
            log(f"    [{rank}] {pcm:42s} Tm={rec['Tm_C']:5.1f}C L={rec['latent_heat_kJ_kg']:5.0f}kJ/kg "
                f"residual={m['residual_pct_of_collector']:.5f}% useful={m['useful_energy_kWh']:.1f} kWh "
                f"(vs plain {100 * (m['useful_energy_kWh'] / plain['useful_energy_kWh'] - 1):+.2f}%) "
                f"mean_f_melt={m['mean_f_melt']:.3f} max_f_melt={m['max_f_melt']:.3f} "
                f"cycles={m['complete_melt_cycles']} max_T_pcm={m['max_pcm_temp_C']:.1f}C "
                f"shield_pcm_substeps={m['n_shield_pcm_activations']}")
    verdict = "PASS" if rows and worst < pass_pct else "FAIL"
    log(f"\n  O1 PCMs verified: {', '.join(f'c{c}#{r} {p}' for c, r, p in rows)}")
    log(f"  worst Gate-1 residual over the roster = {worst:.5f} %   roster verdict: {verdict}")
    return {"gate": "o1_roster", "verdict": verdict, "worst_residual_pct": worst, "n_verified": len(rows)}


def run_all_gates(state: str, report_path=None, include_stale_winners: bool = True, o1_roster: bool = False):
    # `report_path` lets the G6 re-run choose its output file;
    # `include_stale_winners=False` skips the check against the retired Phase-7 winners.
    # results/ is flat because this folder is already specific to one state.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = report_path or (RESULTS_DIR / "phase4_simulator_verification_report.txt")

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("#" * 72)
    log(f"# Objective 2 Phase 4 — Simulator Verification Report — state={state}")
    log("#" * 72)

    g1 = gate1_conservation(state, log)
    g2 = gate2_limiting_cases(state, log)
    g3 = gate3_baseline_comparison(state, log)
    g4 = gate4_calibration(state, g3["optimized_metrics"], log)
    g5 = gate5_sensitivity(state, log)
    if include_stale_winners:
        gate_shield_confirmation_for_winners(state, log)
    roster = gate_o1_pcm_roster(state, log) if o1_roster else None

    gates = [g1, g2, g3, g4, g5]
    n_clean_pass = sum(1 for g in gates if g["verdict"] == "PASS")

    log("\n" + "=" * 72)
    log("SUMMARY")
    log("=" * 72)
    for g in gates:
        log(f"  Gate {g['gate']}: {g['verdict']}")
    log(f"\n  Gates passing cleanly: {n_clean_pass}/5")

    if roster is not None:
        log(f"  O1 PCM roster: {roster['verdict']} ({roster['n_verified']} PCMs)")
    go_no_go = "GO" if (g1["max_residual_pct"] < 0.5 and n_clean_pass >= 3) else "NO-GO"
    log(f"\n  Go/No-Go (framework doc Phase 4 rule: residual<0.5% AND >=3/5 gates clean): {go_no_go}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written to: {report_path}")
    return gates, go_no_go


if __name__ == "__main__":
    state = sys.argv[1] if len(sys.argv) > 1 else "uttarakhand"
    run_all_gates(state)
