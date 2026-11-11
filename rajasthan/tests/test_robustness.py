"""
Phase O3-5 tests — robustness infrastructure.

  - NoisyPCMEnv perturbs observations correctly (including a NEGATIVE temperature bias; TN's did nothing)
  - noise does NOT break the safety shield
  - every default scenario runs and the suite returns all scenarios
  - the result does not depend on the number of worker processes
Run with: pytest tests/test_robustness.py -v
"""

import numpy as np
import pytest

import config_O2

CONTRACT_PATH = config_O2.CONTRACT_FILE
REGIME = 1


@pytest.fixture(scope="module")
def inner_env():
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(CONTRACT_PATH, regime_id=REGIME)


def _obs(**kw):
    from o3.environment import obs as O
    d = dict(T_top=40.0, T_bottom=30.0, T_mean=35.0, T_pcm=40.0, f_melt=0.1, GHI=600.0, T_amb=30.0, hour=12, draw=0.0,
             T_mains=25.0, E_stored_kWh=0.5, E_rec45_kWh=0.0, min_since_draw=600.0)
    d.update(kw)
    return O.build("V2", d)


def test_noisy_env_perturbs_ghi(inner_env):
    from o3.evaluation.robustness import NoisyPCMEnv
    env = NoisyPCMEnv(inner_env, ghi_noise_pct=20.0)
    o = _obs()
    p = env._perturb(o.copy())
    assert not np.isclose(p[5], o[5]), "GHI channel (V2 index 5) was not perturbed"
    assert np.array_equal(np.delete(p, 5), np.delete(o, 5))


def test_negative_temperature_bias_is_applied(inner_env):
    from o3.evaluation.robustness import NoisyPCMEnv
    o = _obs()
    up = NoisyPCMEnv(inner_env, tamb_noise_C=2.0)._perturb(o.copy())[6]
    dn = NoisyPCMEnv(inner_env, tamb_noise_C=-2.0)._perturb(o.copy())[6]
    assert up == pytest.approx(o[6] + 2.0 / 25.0, abs=1e-6) and dn == pytest.approx(o[6] - 2.0 / 25.0, abs=1e-6)


def test_noisy_env_obs_in_range(inner_env):
    from o3.evaluation.robustness import NoisyPCMEnv
    env = NoisyPCMEnv(inner_env, ghi_noise_pct=20.0, tamb_noise_C=5.0)
    obs, _ = env.reset(year=2016)
    assert np.all(obs >= -5.0) and np.all(obs <= 5.0)


def test_nominal_scenario_no_safety_violations(inner_env):
    from o3.evaluation.robustness import run_robustness_episode
    from o3.baselines.b0_fixed import FixedChargePolicy
    b0 = FixedChargePolicy()
    r = run_robustness_episode(lambda o: b0.act(o), inner_env, 2016, "solid")
    assert r["n_safety_violations"] == 0


def test_all_scenarios_run_one_step():
    from o3.evaluation.robustness import NoisyPCMEnv, _default_scenarios
    from o3.environment.pcm_env import PCMEnv
    for sc in _default_scenarios():
        inner = PCMEnv(CONTRACT_PATH, regime_id=REGIME, demand_mode=sc.get("demand_mode", "deterministic"),
                       demand_volume_multiplier=sc.get("demand_scale", 1.0))
        env = NoisyPCMEnv(inner, sc.get("ghi_noise_pct", 0.0), sc.get("tamb_noise_C", 0.0), sc.get("demand_scale", 1.0))
        obs, _ = env.reset(year=2021, pcm_init=sc.get("pcm_init", "solid"))
        assert env.step(0)[0] is not None, sc["name"]


def test_demand_scale_changes_the_physics():
    from o3.environment.pcm_env import PCMEnv
    a = PCMEnv(CONTRACT_PATH, regime_id=REGIME, demand_volume_multiplier=1.0)
    b = PCMEnv(CONTRACT_PATH, regime_id=REGIME, demand_volume_multiplier=1.5)
    for e in (a, b):
        e.reset(year=2021)
        for _ in range(240):
            e.step(0)
    assert b.episode_summary()["E_load_kWh"] > a.episode_summary()["E_load_kWh"]


def test_robustness_suite_returns_all_scenarios_and_is_jobs_independent():
    from o3.evaluation.robustness import run_robustness_suite, _default_scenarios
    sc = _default_scenarios()[:4] + _default_scenarios()[-1:]            # nominal, init x2, ghi5, stochastic demand
    kw = dict(policy_name="B0", regime_id=REGIME, years=[2021], scenarios=sc, policy_spec=("baseline", "b0"), verbose=False)
    ser = run_robustness_suite(n_jobs=1, **kw)
    par = run_robustness_suite(n_jobs=2, **kw)
    assert {r["scenario"] for r in ser} == {s["name"] for s in sc}
    for a, b in zip(ser, par):
        assert a["total_delivered_kWh"] == b["total_delivered_kWh"] and a["total_reward"] == b["total_reward"]
