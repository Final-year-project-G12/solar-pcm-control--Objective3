"""
Phase O3-1 tests — PCMEnv step interface and the plan's equivalence tests T1-T8 (plan 15.5).

Gate: stepping the environment one hour at a time must equal ONE call of O2's run_year_multinode
(and run_case) on the same year — the plan asks for exact equality (relative difference <= 1e-9).
The Tamil Nadu gate was "within 5 %" against a different (single-node) physics.

Run with: pytest tests/test_environment.py -v
Regime 1 (Nagaur, plan 15.1 first prototype) is the default fixture.
"""

import copy

import numpy as np
import pytest

import config_O2

CONTRACT_PATH = config_O2.CONTRACT_FILE
REGIME = 1
YEAR = 2016


@pytest.fixture(scope="module")
def env():
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR)


@pytest.fixture(scope="module")
def o2_reference():
    """One O2 run_case call (multinode, explore year) with the hourly frame."""
    from src.simulation.run_case import run_case
    from src.design.space import to_design, geometry_bounds
    from src.weather import roles
    from o3.utils.contract_loader import load_contract
    from o3.utils.weather_roles import MEDOID_POINTS
    c = load_contract(CONTRACT_PATH)
    rd = c["static_design_per_regime"][REGIME]
    dv = to_design(rd["capsule_inner_radius_m"], rd["pcm_volume_fraction"], rd["nominal_flow_kg_s"], rd["arrangement"])
    w = roles.load_point_year("rajasthan", REGIME, MEDOID_POINTS[REGIME], YEAR, "explore")
    r = run_case("rajasthan", REGIME, rd["selected_pcm"], dv, design_bounds=geometry_bounds(), weather=w,
                 tank_model="multinode", n_nodes=10, stack_anchor="center", compute_tes_metrics=True, night_day_stride=0)
    return {"r": r, "weather": w, "dv": dv, "rd": rd}


def _run(env, schedule_fn, year=YEAR, init="solid", seed=None):
    env.reset(year=year, initial_pcm_state=init, seed=seed)
    done, T, Q, Ttop = False, [], [], []
    i = 0
    while not done:
        _, _, done, _, info = env.step(schedule_fn(i))
        T.append(info["T_pcm_C"]); Q.append(info["Q_collector_Wh"]); Ttop.append(info["T_top_C"])
        i += 1
    return np.array(Ttop), np.array(T), np.array(Q)


# ---------------------------------------------------------------- interface

def test_reset_returns_obs_shape(env):
    obs, info = env.reset(year=YEAR)
    assert obs.shape == (13,), f"Expected the 13 contract fields, got {obs.shape}"
    assert obs.dtype == np.float32


def test_v1_layout_still_available():
    from o3.environment.pcm_env import PCMEnv
    e = PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR, obs_version="V1")
    obs, _ = e.reset(year=YEAR)
    assert obs.shape == (11,)


def test_step_returns_valid_types(env):
    env.reset(year=YEAR)
    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2.shape == (13,)
    assert isinstance(reward, float) and isinstance(terminated, bool)
    for k in ("E_delivered_Wh", "shield_override", "n_shield_water", "n_shield_pcm", "n_safety_violations", "T_top_C"):
        assert k in info


def test_episode_runs_to_end(env):
    env.reset(year=YEAR)
    n, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(0)
        n += 1
    assert n == 8760, f"Unexpected episode length {n}"


def test_no_hard_safety_violations_b0(env):
    _run(env, lambda i: 0)
    assert env._state["n_safety_violations"] == 0
    assert env.episode_summary()["hours_over_limit"] == 0.0


def test_delivered_energy_positive(env):
    _run(env, lambda i: 0)
    assert env.episode_summary()["E_del45_annual_kWh"] > 100.0


def test_action_space_all_valid(env):
    from o3.environment.pcm_env import CHARGE, IDLE, BYPASS
    for action in (CHARGE, IDLE, BYPASS):
        env.reset(year=YEAR)
        assert env.step(action)[0] is not None


def test_obs_range_reasonable(env):
    env.reset(year=YEAR)
    done, out, total = False, 0, 0
    while not done:
        obs, _, done, _, _ = env.step(0)
        out += int(np.any(np.abs(obs) > 5.0))
        total += 1
    assert out / total < 0.02


def test_end_of_hour_state_equals_adapter_derived(env):
    """pcm_env reads the end-of-hour state from the kernel output; the adapter's independent formula must agree."""
    env.reset(year=YEAR)
    for i in range(48):
        env.step(0 if i % 3 else 1)
    s = env._state
    d = env._k.derived(s["T_nodes"], s["h_nodes"])
    assert abs(d["T_top"] - s["T_w"]) < 1e-9 and abs(d["T_pcm"] - s["T_pcm"]) < 1e-9
    assert abs(d["f_melt"] - s["f_melt"]) < 1e-9 and abs(d["E_stored_kWh"] - s["E_stored_kWh"]) < 1e-9
    assert abs(d["E_rec45_kWh"] - s["E_rec45_kWh"]) < 1e-9


# ---------------------------------------------------------------- T1  full-year equivalence

def test_T1_stepped_year_equals_run_case(env, o2_reference):
    """B0 stepped 8760 times == one O2 run_case(multinode) year: hourly T_top, T_pcm, Q_collector and totals."""
    Ttop, Tp, Qc = _run(env, lambda i: 0)
    h = o2_reference["r"]["hourly"]
    assert np.max(np.abs(Ttop - h["T_w_C"].to_numpy())) <= 1e-9
    assert np.max(np.abs(Tp - h["T_pcm_C"].to_numpy())) <= 1e-9
    assert np.max(np.abs(Qc - h["Q_collector_Wh"].to_numpy())) <= 1e-9
    s, m, t = env.episode_summary(), o2_reference["r"]["metrics"], o2_reference["r"]["tes"]
    assert abs(s["E_del45_window_kWh"] - t["E_del45_window_kWh"]) <= 1e-9 * t["E_del45_window_kWh"]
    assert abs(s["E_del45_annual_kWh"] - t["E_del45_annual_kWh"]) <= 1e-9 * t["E_del45_annual_kWh"]
    assert abs(s["E_load_kWh"] - m["useful_energy_kWh"]) <= 1e-9 * m["useful_energy_kWh"]
    assert abs(s["shield_water_h"] - m["shield_water_h"]) <= 1e-9 and abs(s["shield_pcm_h"] - m["shield_pcm_h"]) <= 1e-9


# ---------------------------------------------------------------- T2  split run

def test_T2_two_half_year_calls_equal_one_full_call(o2_reference):
    from src.simulation.tank_model_multinode import run_year_multinode
    from o3.environment.o2_adapter import build_runtime
    from o3.utils.contract_loader import load_contract
    from src.simulation.demand_profile import load_demand_model
    from src.io_utils import load_demand_profile
    rt = build_runtime(load_contract(CONTRACT_PATH), REGIME)
    w = o2_reference["weather"]
    from src.io_utils import drop_leap_day
    w = drop_leap_day(w)
    dem = load_demand_model(load_demand_profile("rajasthan"))
    full = run_year_multinode(w, dem, rt.mains_C, rt.design, rt.pcm_props, rt.cfg, layout=rt.layout)
    half = len(w) // 2
    a = run_year_multinode(w.iloc[:half].reset_index(drop=True), dem, rt.mains_C, rt.design, rt.pcm_props, rt.cfg, layout=rt.layout)
    st = {"T_nodes": a.hourly.attrs["node_T"][-1], "h_nodes": a.hourly.attrs["node_h"][-1]}
    b = run_year_multinode(w.iloc[half:].reset_index(drop=True), dem, rt.mains_C, rt.design, rt.pcm_props, rt.cfg,
                           layout=rt.layout, initial_state=st)
    joined = np.concatenate([a.hourly["T_w_C"].to_numpy(), b.hourly["T_w_C"].to_numpy()])
    assert np.max(np.abs(joined - full.hourly["T_w_C"].to_numpy())) <= 1e-9
    assert abs(a.energy["E_collector_kWh"] + b.energy["E_collector_kWh"] - full.energy["E_collector_kWh"]) <= 1e-9 * full.energy["E_collector_kWh"]


# ---------------------------------------------------------------- T3  bypass mask

def test_T3_bypass_schedule_equals_zeroed_ghi_year(env, o2_reference):
    from src.simulation.run_case import run_case
    from src.design.space import geometry_bounds
    rng = np.random.default_rng(1)
    bypass = rng.random(8760) < 0.4
    Ttop, Tp, Qc = _run(env, lambda i: 2 if bypass[i] else 0)
    w = o2_reference["weather"].copy()
    from src.io_utils import drop_leap_day
    w = drop_leap_day(w)
    w.loc[bypass, "GHI_Wm2"] = 0.0
    rd = o2_reference["rd"]
    r = run_case("rajasthan", REGIME, rd["selected_pcm"], o2_reference["dv"], design_bounds=geometry_bounds(), weather=w,
                 tank_model="multinode", n_nodes=10, stack_anchor="center", compute_tes_metrics=False)
    assert np.max(np.abs(Ttop - r["hourly"]["T_w_C"].to_numpy())) <= 1e-9
    assert np.max(np.abs(Tp - r["hourly"]["T_pcm_C"].to_numpy())) <= 1e-9


def test_idle_and_bypass_are_physically_identical(env):
    a = _run(env, lambda i: 1)
    b = _run(env, lambda i: 2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


# ---------------------------------------------------------------- T4  energy balance

@pytest.mark.parametrize("regime", [0, 1, 2])
def test_T4_energy_residual_below_gate1(regime):
    from o3.environment.pcm_env import PCMEnv
    e = PCMEnv(CONTRACT_PATH, regime_id=regime, year=YEAR)
    _run(e, lambda i: 0 if i % 5 else 1)
    assert e.episode_summary()["residual_pct_of_collector"] < 0.1          # O2 Gate 1 threshold


# ---------------------------------------------------------------- T5  determinism

def test_T5_same_seed_same_trajectory():
    from o3.environment.pcm_env import PCMEnv
    e1 = PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR, demand_mode="stochastic")
    e2 = PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR, demand_mode="stochastic")
    rng = np.random.default_rng(3)
    acts = rng.integers(0, 3, 400)
    a = _run_n(e1, acts, seed=7)
    b = _run_n(e2, acts, seed=7)
    c = _run_n(e2, acts, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)              # a different seed changes the stochastic demand


def _run_n(env, acts, seed):
    env.reset(year=YEAR, seed=seed)
    out = []
    for a in acts:
        obs, r, *_ = env.step(int(a))
        out.append(np.concatenate([obs, [r]]))
    return np.array(out)


# ---------------------------------------------------------------- T6  shield cannot be bypassed by the policy

def test_T6_shield_overrides_charge_when_top_node_hot(env):
    """Start with a tank above the 72 C guard band at midday and CHARGE: the kernel's shield must cut the loop."""
    from o3.environment.pcm_env import CHARGE
    env.reset(year=YEAR)
    # advance to a clear-sky hour with no draw: 2016-01-01 is hour 0; pick local hour 12 (index with local_hour == 12)
    i0 = int(np.flatnonzero(env._hod == 12)[100])
    env._state["hour_idx"] = i0
    n = env._rt.n_nodes
    env._state["T_nodes"] = np.full(n, 80.0)                      # tank far above the guard band
    obs, r, done, _, info = env.step(CHARGE)
    assert info["n_shield_water"] > 0 and info["shield_water_s"] > 0
    assert info["shield_override"] is True
    assert info["Q_collector_Wh"] < 1e-6, "loop must be off while the top node is above the guard band"
    assert info["action_executed"] != CHARGE


# ---------------------------------------------------------------- T7  guards

def test_T7_contract_hash_mismatch_makes_env_refuse(tmp_path):
    from o3.environment.pcm_env import PCMEnv
    c = copy.deepcopy(__import__("json").loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    c["provenance"]["g10_heldout_results.csv"] = "f" * 64
    bad = tmp_path / "bad_contract.json"
    bad.write_text(__import__("json").dumps(c), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Provenance mismatch"):
        PCMEnv(bad, regime_id=REGIME, year=YEAR)


def test_T7_test_year_refused_without_lock():
    from o3.utils import weather_roles
    weather_roles.release_for_tests()
    with pytest.raises(weather_roles.TestLockError):
        weather_roles.load_weather(REGIME, 2006)
    from o3.environment.pcm_env import PCMEnv
    with pytest.raises(weather_roles.TestLockError):
        PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=2008)


def test_T7_used_o2_years_never_usable():
    from o3.utils import weather_roles
    for y in (2011, 2015, 2024):          # O2's held-out and v1 test years
        with pytest.raises(PermissionError):
            weather_roles.load_weather(REGIME, y)


def test_T7_role_mismatch_refused():
    from o3.utils import weather_roles
    with pytest.raises(PermissionError):
        weather_roles.load_weather(REGIME, 2021, purpose="train")


def test_T7_lock_is_exclusive(tmp_path):
    from o3.utils import weather_roles
    lock = tmp_path / "lock"
    weather_roles.acquire_test_lock("run1", "fp", lock)
    try:
        with pytest.raises(weather_roles.TestLockError):
            weather_roles.acquire_test_lock("run2", "fp", lock)
        assert weather_roles.load_weather(REGIME, 2006)["GHI_Wm2"].shape == (8760,)      # allowed while held
    finally:
        weather_roles.release_for_tests()
    with pytest.raises(weather_roles.TestLockError):
        weather_roles.load_weather(REGIME, 2006)


# ---------------------------------------------------------------- T8  SB3 compatibility

def test_T8_check_env():
    from stable_baselines3.common.env_checker import check_env
    from o3.environment.gym_wrapper import PCMGymEnv
    check_env(PCMGymEnv(CONTRACT_PATH, regime_id=REGIME, fixed_year=YEAR), warn=True)


# ---------------------------------------------------------------- physics sanity

def test_water_only_twin_has_no_pcm_and_pcm_helps_at_night():
    from o3.environment.pcm_env import PCMEnv
    twin = PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR, water_only=True)
    _run(twin, lambda i: 0)
    pcm = PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR)
    _run(pcm, lambda i: 0)
    st, sp = twin.episode_summary(), pcm.episode_summary()
    assert np.isnan(st["t_pcm_max_C"])
    assert st["shield_pcm_h"] == 0.0
    assert sp["E_del45_window_kWh"] > st["E_del45_window_kWh"], "O2's released design must beat its water-only twin in the 18-09 window"
