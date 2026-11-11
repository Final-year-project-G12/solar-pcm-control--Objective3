"""
Phase O3-2 tests — B0, B1, B2, B4 and the water-only twin.

B1 must pass the O3 acceptance checklist before DRL training. Run with: pytest tests/test_baselines.py -v
Regime 1 (Nagaur), year 2016 (a training year).
"""

import numpy as np
import pytest

import config_O2

CONTRACT_PATH = config_O2.CONTRACT_FILE
REGIME, YEAR = 1, 2016


@pytest.fixture(scope="module")
def env():
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=YEAR)


@pytest.fixture(scope="module")
def b0_result(env):
    from o3.baselines.b0_fixed import run_episode
    return run_episode(env, year=YEAR)


@pytest.fixture(scope="module")
def b1_result(env):
    from o3.baselines.b1_reactive import run_episode
    return run_episode(env, year=YEAR)


@pytest.fixture(scope="module")
def b2_result(env):
    from o3.baselines.b2_forecast import run_episode
    return run_episode(env, year=YEAR)


@pytest.fixture(scope="module")
def twin_result():
    from o3.evaluation.rollout import evaluate_twin
    return evaluate_twin(REGIME, YEAR)


# --- B0 -----------------------------------------------------------------

def test_b0_name(b0_result):
    assert b0_result["policy"] == "B0_fixed_charge"


def test_b0_zero_safety_violations(b0_result):
    assert b0_result["n_safety_violations"] == 0
    assert b0_result["hours_over_limit"] == 0.0


def test_b0_delivered_energy_positive(b0_result):
    assert b0_result["total_delivered_kWh"] > 100.0


def test_b0_runs_full_year(b0_result):
    assert b0_result["n_steps"] == 8760


def test_b0_has_no_switches(b0_result):
    assert b0_result["action_switches"] == 0


def test_b0_beats_water_only_twin_in_night_window(b0_result, twin_result):
    """O2's released design vs its water-only twin (the plan's primary reference)."""
    assert b0_result["E_del45_window_kWh"] > twin_result["E_del45_window_kWh"]


# --- B1 -----------------------------------------------------------------

def test_b1_name(b1_result):
    assert b1_result["policy"] == "B1_reactive_rule"


def test_b1_zero_safety_violations(b1_result):
    assert b1_result["n_safety_violations"] == 0


def test_b1_delivered_energy_positive(b1_result):
    assert b1_result["total_delivered_kWh"] > 100.0


def test_b1_fewer_shield_activations_than_b0(b0_result, b1_result):
    b0s = b0_result["total_shield_water"] + b0_result["total_shield_pcm"]
    b1s = b1_result["total_shield_water"] + b1_result["total_shield_pcm"]
    assert b1s < b0s


def test_b1_runs_full_year(b1_result):
    assert b1_result["n_steps"] == 8760


# --- B2 -----------------------------------------------------------------

def test_b2_name(b2_result):
    assert b2_result["policy"] == "B2_forecast_rule"


def test_b2_zero_safety_violations(b2_result):
    assert b2_result["n_safety_violations"] == 0


def test_b2_delivered_energy_positive(b2_result):
    assert b2_result["total_delivered_kWh"] > 50.0


def test_b2_runs_full_year(b2_result):
    assert b2_result["n_steps"] == 8760


def test_b2_shield_not_worse_than_b0(b0_result, b2_result):
    b0s = b0_result["total_shield_water"] + b0_result["total_shield_pcm"]
    b2s = b2_result["total_shield_water"] + b2_result["total_shield_pcm"]
    assert b2s <= b0s * 1.20


# --- B4 / masks / twin ---------------------------------------------------

def test_b4_is_reproducible_and_safe(env):
    from o3.baselines.b4_random import run_episode, RandomSafePolicy
    a = run_episode(env, RandomSafePolicy(seed=3), year=YEAR)
    b = run_episode(env, RandomSafePolicy(seed=3), year=YEAR)
    assert a["total_delivered_kWh"] == b["total_delivered_kWh"] and a["n_safety_violations"] == 0


def test_action_mask_physical_thresholds():
    from o3.environment.gym_wrapper import action_mask_from_obs
    from o3.environment import obs as O
    base = dict(T_top=40.0, T_bottom=30.0, T_mean=35.0, T_pcm=40.0, f_melt=0.1, GHI=600.0, T_amb=30.0, hour=12, draw=0.0,
                T_mains=25.0, E_stored_kWh=0.5, E_rec45_kWh=0.0, min_since_draw=600.0)
    assert action_mask_from_obs(O.build("V2", base)).tolist() == [True, True, True]
    assert action_mask_from_obs(O.build("V2", {**base, "T_top": 69.0})).tolist() == [False, True, True]   # near the 72 C shield
    assert action_mask_from_obs(O.build("V2", {**base, "GHI": 0.0})).tolist() == [False, True, True]       # no sun


def test_twin_is_plain_tank(twin_result):
    assert twin_result["shield_pcm_h"] == 0.0 and twin_result["policy"] == "TWIN_water_only"


# --- SAC continuous-action wrapper --------------------------------------

def test_continuous_wrapper_binning():
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    from o3.environment.pcm_env import CHARGE, IDLE, BYPASS
    assert PCMContinuousGymEnv._to_discrete(np.array([-0.9])) == CHARGE
    assert PCMContinuousGymEnv._to_discrete(np.array([0.0])) == IDLE
    assert PCMContinuousGymEnv._to_discrete(np.array([0.9])) == BYPASS


def test_continuous_wrapper_action_space():
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=REGIME, fixed_year=2021)
    assert env.action_space.shape == (1,)
    env.reset()
    obs2, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs2.shape == (13,)


def test_print_comparison(b0_result, b1_result, twin_result, capsys):
    print(f"\n=== Baseline Comparison (Regime {REGIME}, Year {YEAR}) ===")
    for r in [twin_result, b0_result, b1_result]:
        print(f"{r['policy']:20s} | E45={r['total_delivered_kWh']:7.1f} kWh | window={r['E_del45_window_kWh']:6.1f} | "
              f"shield_w={r['total_shield_water']:6d} | shield_p={r['total_shield_pcm']:6d} | reward={r['total_reward']:10.1f}")
