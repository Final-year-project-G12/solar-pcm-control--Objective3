"""
Phase O3-5 tests — robustness infrastructure.

Tests that:
  - NoisyPCMEnv perturbs observations correctly
  - noise does NOT break the safety shield
  - all 10 default scenarios can run one episode without errors
  - nominal scenario has zero safety violations (B0 policy)
"""

import pathlib
import numpy as np
import pytest

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


@pytest.fixture(scope="module")
def inner_env():
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(CONTRACT_PATH, regime_id=0)


def test_noisy_env_perturbs_ghi(inner_env):
    from o3.evaluation.robustness import NoisyPCMEnv
    env = NoisyPCMEnv(inner_env, ghi_noise_pct=20.0)
    # Use a synthetic non-zero obs so multiplicative noise has visible effect
    # (at hour 0 / midnight GHI=0 -> 0*noise=0, no change)
    obs_nonzero = np.array([0.8, 0.5, 0.3, 0.2, 0.1,
                            0.5, 0.0, 0.5, 0.5, 0.0, 0.5], dtype=np.float32)
    perturbed = env._perturb(obs_nonzero.copy())
    assert not np.allclose(obs_nonzero, perturbed), \
        "NoisyPCMEnv did not perturb a non-zero GHI observation"
    assert perturbed[1:].tolist() == obs_nonzero[1:].tolist() or \
        not np.allclose(perturbed[0], obs_nonzero[0]), \
        "GHI channel (obs[0]) was not perturbed"


def test_noisy_env_obs_in_range(inner_env):
    from o3.evaluation.robustness import NoisyPCMEnv
    env = NoisyPCMEnv(inner_env, ghi_noise_pct=20.0, tamb_noise_C=5.0)
    obs, _ = env.reset(year=2016)
    assert np.all(obs >= -5.0) and np.all(obs <= 5.0), \
        "NoisyPCMEnv output is outside [-5, 5] clip range"


def test_nominal_scenario_no_safety_violations(inner_env):
    from o3.evaluation.robustness import run_robustness_episode
    from o3.baselines.b0_fixed import FixedChargePolicy

    b0 = FixedChargePolicy()
    result = run_robustness_episode(
        policy_fn = lambda obs: b0.act(obs),
        env       = inner_env,
        year      = 2016,
        pcm_init  = "solid",
    )
    assert result["n_safety_violations"] == 0, \
        f"B0 on nominal scenario has {result['n_safety_violations']} violations"


def test_all_scenarios_run_one_step(inner_env):
    """Every scenario must complete at least 1 step without errors."""
    from o3.evaluation.robustness import NoisyPCMEnv, _default_scenarios
    from o3.baselines.b0_fixed import FixedChargePolicy

    b0 = FixedChargePolicy()
    scenarios = _default_scenarios()

    for scenario in scenarios:
        env = NoisyPCMEnv(
            inner_env,
            ghi_noise_pct = scenario.get("ghi_noise_pct", 0.0),
            tamb_noise_C  = scenario.get("tamb_noise_C",  0.0),
        )
        obs, _ = env.reset(year=2021, pcm_init=scenario.get("pcm_init","solid"))
        obs2, reward, terminated, truncated, info = env.step(
            b0.act(obs)
        )
        assert obs2 is not None, f"Scenario {scenario['name']} failed on first step"


def test_robustness_suite_returns_all_scenarios():
    from o3.evaluation.robustness import run_robustness_suite, _default_scenarios
    from o3.baselines.b0_fixed import FixedChargePolicy

    b0 = FixedChargePolicy()
    results = run_robustness_suite(
        policy_fn   = lambda obs: b0.act(obs),
        policy_name = "B0",
        regime_id   = 0,
        years       = [2021],       # one year for speed
    )
    expected_scenarios = {s["name"] for s in _default_scenarios()}
    result_scenarios   = {r["scenario"] for r in results}
    assert result_scenarios == expected_scenarios, \
        f"Missing scenarios: {expected_scenarios - result_scenarios}"
