"""Forecast modes of the extension env (o3/env_ext.py): perfect is unchanged, smart persistence is causal and bounded,
none is zeros; SAC's continuous action view bins correctly."""
import numpy as np
import pytest

from o3.env_ext import (KC_CLIP, ContinuousActionWrapper, UKPCMEnvExt, bin_action,
                        smart_persistence_kc)

YEAR, SC = 2016, "fully_solid"


def _env(mode=None, **kw):
    extra = {} if mode is None else {"forecast_mode": mode}
    return UKPCMEnvExt(1, "train", obs_version="V2", normalise=False, w2=0.0, w4=0.3, **extra, **kw)


def _episode_obs(env, n=None, seed=3):
    obs, _ = env.reset(seed=seed, options={"year": YEAR, "scenario": SC})
    rng = np.random.default_rng(0)
    out = [obs.copy()]
    steps = n or env.stepper.n_hours
    for _ in range(steps):
        obs, r, done, *_ = env.step(int(rng.integers(2)))
        out.append(np.append(obs, r))
        if done:
            break
    return out


def test_default_is_perfect_and_bit_identical_full_episode():
    a = _episode_obs(_env())                 # no forecast_mode argument: the original V2 behaviour
    b = _episode_obs(_env("perfect"))
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_perfect_fields_are_the_stored_future_ghi():
    env = _env("perfect")
    env.reset(options={"year": YEAR, "scenario": SC})
    ghi = env.stepper.ghi
    for _ in range(500):
        obs, *_ = env.step(1)
        t = env.stepper.t
        exp = [ghi[t + h] if t + h < env.stepper.n_hours else 0.0 for h in (1, 2, 3)]
        assert np.allclose(obs[15:18], exp, rtol=0, atol=1e-4)


def test_none_mode_is_zeros_same_length():
    env = _env("none")
    obs, _ = env.reset(options={"year": YEAR, "scenario": SC})
    assert obs.shape == (19,)
    for _ in range(48):
        obs, *_ = env.step(1)
        assert np.all(obs[15:18] == 0.0)


def test_smart_formula():
    env = _env("smart")
    env.reset(options={"year": YEAR, "scenario": SC})
    ghi, cs = env.stepper.ghi, env._weather(YEAR)["GHI_clearsky_Wm2"].to_numpy()
    for _ in range(300):
        obs, *_ = env.step(1)
        t = env.stepper.t
        kc = env._kc[t]
        assert KC_CLIP[0] <= kc <= KC_CLIP[1]
        if cs[t] > 20:
            assert np.isclose(kc, np.clip(ghi[t] / cs[t], *KC_CLIP))
        assert np.allclose(obs[15:18], [kc * cs[t + h] for h in (1, 2, 3)], atol=1e-3)


def test_smart_is_causal():
    """Changing GHI after hour t leaves the observation at t unchanged; changing GHI at t changes it (daytime)."""
    probe = _env("smart")
    probe.reset(options={"year": YEAR, "scenario": SC})
    cs = probe._weather(YEAR)["GHI_clearsky_Wm2"].to_numpy()
    t = int(np.where(cs[200:] > 300)[0][0]) + 200           # a daytime decision hour

    def obs_at_t(perturb_hours):
        env = _env("smart")
        w = env._weather(YEAR).copy()
        w.loc[w.index[perturb_hours], "GHI_Wm2"] *= 0.3
        env._weather_cache[YEAR] = w
        env.reset(options={"year": YEAR, "scenario": SC})
        for _ in range(t):
            obs, *_ = env.step(1)
        assert env.stepper.t == t
        return obs

    base = obs_at_t([])
    assert np.array_equal(base, obs_at_t([t + 1, t + 2, t + 3]))
    assert not np.array_equal(base[15:18], obs_at_t([t])[15:18])


def test_kc_function_is_causal_and_bounded():
    rng = np.random.default_rng(1)
    cs = np.clip(800 * np.sin(np.linspace(0, 20 * np.pi, 480)), 0, None)
    ghi = cs * rng.uniform(0, 1.5, 480)
    kc = smart_persistence_kc(ghi, cs)
    assert np.all((kc >= KC_CLIP[0]) & (kc <= KC_CLIP[1]))
    g2 = ghi.copy()
    g2[300:] = 0.0
    assert np.array_equal(kc[:300], smart_persistence_kc(g2, cs)[:300])
    night = cs <= 20
    held = np.where(night[1:])[0] + 1
    assert np.all(kc[held] == kc[held - 1])                  # Kc is held through night hours


def test_smart_bounded_end_of_year_and_night():
    env = _env("smart")
    env.reset(options={"year": YEAR, "scenario": SC})
    vals = []
    done = False
    while not done:
        obs, _, done, *_ = env.step(1)
        vals.append(obs[15:18])
    v = np.array(vals)
    assert np.all(np.isfinite(v)) and np.all(v >= 0)
    assert np.all(v[-1] == 0.0)                              # past the end of the year


def test_reproducible_same_seed():
    a = _episode_obs(_env("smart", domain_randomize=True), n=400, seed=7)
    b = _episode_obs(_env("smart", domain_randomize=True), n=400, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_smart_differs_from_perfect_but_physics_identical():
    """The forecast only changes what the policy sees, never the plant: same actions -> same plant trajectory."""
    ep, es = _env("perfect"), _env("smart")
    for e in (ep, es):
        e.reset(options={"year": YEAR, "scenario": SC})
    diff = False
    for k in range(400):
        op, rp, *_ = ep.step(k % 2)
        os_, rs, *_ = es.step(k % 2)
        assert np.array_equal(op[:15], os_[:15]) and rp == rs
        diff |= not np.array_equal(op[15:18], os_[15:18])
    assert diff


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        _env("oracle")


@pytest.mark.parametrize("a,exp", [(-1.0, 0), (0.0, 0), (1e-6, 1), (1.0, 1), ([0.4], 1), (np.array([-0.2]), 0)])
def test_bin_action(a, exp):
    assert bin_action(a) == exp


def test_continuous_wrapper_matches_discrete():
    d = _env("smart")
    c = ContinuousActionWrapper(_env("smart"))
    d.reset(options={"year": YEAR, "scenario": SC})
    c.reset(options={"year": YEAR, "scenario": SC})
    rng = np.random.default_rng(2)
    for _ in range(200):
        u = rng.uniform(-1, 1)
        od, rd, *_ = d.step(bin_action(u))
        oc, rc, *_ = c.step(np.array([u], dtype=np.float32))
        assert np.array_equal(od, oc) and rd == rc
