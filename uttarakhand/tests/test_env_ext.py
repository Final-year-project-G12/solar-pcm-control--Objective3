"""Extension env: V1 / no randomisation is identical to the frozen env; V2 fields and domain randomisation behave."""
import numpy as np

from o3.env import UKPCMEnv
from o3.env_ext import UKPCMEnvExt


def _roll(env, n=200, seed=3):
    obs, _ = env.reset(seed=seed, options={"year": 2016, "scenario": "fully_solid"})
    rng = np.random.default_rng(0)
    out = [obs]
    for _ in range(n):
        o, r, *_ = env.step(int(rng.integers(2)))
        out.append(np.append(o, r))
    return out


def test_v1_identical_to_frozen_env():
    a = _roll(UKPCMEnv(1, "train", w2=0.0, w4=0.3))
    b = _roll(UKPCMEnvExt(1, "train", w2=0.0, w4=0.3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_v2_fields():
    env = UKPCMEnvExt(1, "train", obs_version="V2", normalise=False, w2=0.0, w4=0.3)
    obs, _ = env.reset(options={"year": 2016, "scenario": "fully_solid"})
    assert obs.shape == (19,)
    for _ in range(10):
        obs, *_ = env.step(1)
    t = env.stepper.t
    assert np.allclose(obs[15:18], env.stepper.ghi[t + 1:t + 4])
    assert np.isclose(obs[18], env.demand.for_day(int(env._day[t + 1]))[int(env.stepper.hod[t + 1])])
    assert np.array_equal(obs[:15], env._raw_obs()[:15])


def test_domain_randomisation_changes_physics_within_ranges():
    env = UKPCMEnvExt(1, "train", domain_randomize=True, w2=0.0, w4=0.3)
    env.reset(seed=1, options={"year": 2016, "scenario": "fully_solid"})
    d = env.dr
    assert 0.9 <= d["ghi_mult"] <= 1.1 and -2 <= d["tamb_bias_C"] <= 2 and 0.85 <= d["demand_mult"] <= 1.15
    w = env._weather(2016)
    assert np.allclose(env.stepper.ghi, w.GHI_Wm2.to_numpy() * d["ghi_mult"])
    assert np.allclose(env.stepper.tamb, w.T_amb_C.to_numpy() + d["tamb_bias_C"])
    assert not np.allclose(w.GHI_Wm2.to_numpy(), env.stepper.ghi)            # cached nominal weather untouched
