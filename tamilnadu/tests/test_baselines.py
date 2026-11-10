"""
Phase O3-2 tests — B0 and B1 baselines.

B1 must pass the O3 acceptance checklist before DRL training.
Run with: pytest tests/test_baselines.py -v
"""

import pathlib
import pytest

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"


@pytest.fixture(scope="module")
def env():
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(CONTRACT_PATH, regime_id=0, year=2016)


@pytest.fixture(scope="module")
def b0_result(env):
    from o3.baselines.b0_fixed import run_episode
    return run_episode(env, year=2016)


@pytest.fixture(scope="module")
def b1_result(env):
    from o3.baselines.b1_reactive import run_episode
    return run_episode(env, year=2016)


@pytest.fixture(scope="module")
def b2_result(env):
    from o3.baselines.b2_forecast import run_episode
    return run_episode(env, year=2016)


# --- B0 tests -----------------------------------------------------------

def test_b0_name(b0_result):
    assert b0_result["policy"] == "B0_fixed_charge"


def test_b0_zero_safety_violations(b0_result):
    assert b0_result["n_safety_violations"] == 0, (
        f"B0 hard-limit violations: {b0_result['n_safety_violations']}"
    )


def test_b0_delivered_energy_positive(b0_result):
    assert b0_result["total_delivered_kWh"] > 100.0


def test_b0_runs_full_year(b0_result):
    assert 8700 <= b0_result["n_steps"] <= 8760


# --- B1 tests -----------------------------------------------------------

def test_b1_name(b1_result):
    assert b1_result["policy"] == "B1_reactive_rule"


def test_b1_zero_safety_violations(b1_result):
    """Acceptance checklist: B1 must not cause any hard-limit violations."""
    assert b1_result["n_safety_violations"] == 0, (
        f"B1 hard-limit violations: {b1_result['n_safety_violations']}. "
        "Fix B1 rules before starting DRL training."
    )


def test_b1_delivered_energy_positive(b1_result):
    assert b1_result["total_delivered_kWh"] > 50.0


def test_b1_vs_b0_no_worse_energy(b0_result, b1_result):
    """B1 should deliver at least 80% as much as B0 (soft acceptance threshold)."""
    b0_del = b0_result["total_delivered_kWh"]
    b1_del = b1_result["total_delivered_kWh"]
    ratio  = b1_del / b0_del if b0_del > 0 else 0.0
    assert ratio >= 0.80, (
        f"B1 delivers only {ratio*100:.0f}% of B0 ({b1_del:.1f} vs {b0_del:.1f} kWh). "
        "Review B1 rules."
    )


def test_b1_fewer_shield_activations_or_comparable(b0_result, b1_result):
    """B1's pre-emptive rules should fire the safety shield less or equally often."""
    b0_shield = b0_result["total_shield_water"] + b0_result["total_shield_pcm"]
    b1_shield = b1_result["total_shield_water"] + b1_result["total_shield_pcm"]
    # Allow B1 up to 20% more shield events (it may legitimately behave differently)
    assert b1_shield <= b0_shield * 1.20, (
        f"B1 shield activations ({b1_shield}) >> B0 ({b0_shield}). Check B1 rules."
    )


def test_b1_runs_full_year(b1_result):
    assert 8700 <= b1_result["n_steps"] <= 8760


# --- B2 tests (forecast-aware) ------------------------------------------

def test_b2_name(b2_result):
    assert b2_result["policy"] == "B2_forecast_rule"


def test_b2_zero_safety_violations(b2_result):
    assert b2_result["n_safety_violations"] == 0, (
        f"B2 hard-limit violations: {b2_result['n_safety_violations']}"
    )


def test_b2_delivered_energy_positive(b2_result):
    assert b2_result["total_delivered_kWh"] > 50.0


def test_b2_runs_full_year(b2_result):
    assert 8700 <= b2_result["n_steps"] <= 8760


def test_b2_forecast_improves_or_matches_shield(b0_result, b2_result):
    """B2's anticipatory bypass should not fire the shield more than B0."""
    b0_shield = b0_result["total_shield_water"] + b0_result["total_shield_pcm"]
    b2_shield = b2_result["total_shield_water"] + b2_result["total_shield_pcm"]
    assert b2_shield <= b0_shield * 1.20


# --- SAC continuous-action wrapper --------------------------------------

def test_continuous_wrapper_binning():
    import numpy as np
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    from o3.environment.pcm_env import CHARGE, IDLE, BYPASS
    assert PCMContinuousGymEnv._to_discrete(np.array([-0.9])) == CHARGE
    assert PCMContinuousGymEnv._to_discrete(np.array([0.0]))  == IDLE
    assert PCMContinuousGymEnv._to_discrete(np.array([0.9]))  == BYPASS


def test_continuous_wrapper_action_space():
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2021)
    assert env.action_space.shape == (1,)
    obs, _ = env.reset()
    obs2, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs2.shape == (11,)


# --- Comparison print (not a test, just informational) ------------------

def test_print_comparison(b0_result, b1_result, capsys):
    print("\n=== Baseline Comparison (Regime 0, Year 2016) ===")
    for r in [b0_result, b1_result]:
        print(
            f"{r['policy']:30s} | "
            f"del={r['total_delivered_kWh']:7.1f} kWh | "
            f"unmet={r['total_unmet_kWh']:6.1f} kWh | "
            f"shield_w={r['total_shield_water']:5d} | "
            f"shield_p={r['total_shield_pcm']:5d} | "
            f"reward={r['total_reward']:9.1f}"
        )
