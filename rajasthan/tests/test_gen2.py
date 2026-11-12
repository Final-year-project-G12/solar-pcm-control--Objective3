"""
Generation-2 environment (plan section 19): sub-hourly control, forecast observation (V3), training-time domain
randomisation. The hourly / V2 / no-randomisation defaults must stay bit-for-bit what they were.
"""
import numpy as np
import pytest

import config_O2

C = config_O2.CONTRACT_FILE
REGIME = 1


def _env(**kw):
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(C, regime_id=REGIME, year=2016, **kw)


def test_four_15min_steps_equal_one_hourly_step():
    """Same physics, different decision cadence: with the same action held for the hour the trajectories are identical."""
    h, q = _env(), _env(control_step_min=15)
    h.reset(year=2016); q.reset(year=2016)
    rng = np.random.default_rng(0)
    eh = eq = 0.0
    for _ in range(24 * 20):
        a = int(rng.integers(0, 3))
        ih = h.step(a)[4]
        for _k in range(4):
            iq = q.step(a)[4]
            eq += iq["E_del45_Wh"]
        eh += ih["E_del45_Wh"]
        assert abs(ih["T_top_C"] - iq["T_top_C"]) <= 1e-9 and abs(ih["T_pcm_C"] - iq["T_pcm_C"]) <= 1e-9
    assert eq == pytest.approx(eh, rel=1e-9)


def test_subhourly_episode_length_and_bad_cadence():
    q = _env(control_step_min=15)
    q.reset(year=2016)
    assert q._n_ticks == 4 * 8760
    with pytest.raises(ValueError):
        _env(control_step_min=7)
    with pytest.raises(ValueError):
        _env(control_step_min=45)            # 45 does not divide 60


def test_subhourly_action_can_differ_within_the_hour():
    a, b = _env(control_step_min=15), _env(control_step_min=15)
    a.reset(year=2016); b.reset(year=2016)
    i0 = int(np.flatnonzero(a._hod == 12)[50]) * 4
    a._state["hour_idx"] = b._state["hour_idx"] = i0
    qa = sum(a.step(0)[4]["Q_collector_Wh"] for _ in range(4))
    qb = sum(b.step(x)[4]["Q_collector_Wh"] for x in (0, 2, 2, 0))
    assert 0 < qb < qa


def test_v3_observation_has_forecast_fields():
    e = _env(obs_version="V3")
    o, _ = e.reset(year=2016)
    assert o.shape == (17,) and o.dtype == np.float32
    for _ in range(30):
        o = e.step(0)[0]
    i = e._state["hour_idx"]
    assert o[13] == pytest.approx(e._ghi[i + 1] / 1000.0, abs=1e-6) and o[15] == pytest.approx(e._ghi[i + 3] / 1000.0, abs=1e-6)
    v2 = _env()
    v2.reset(year=2016)
    for _ in range(30):
        o2 = v2.step(0)[0]
    assert np.array_equal(o[:13], o2)                      # the first 13 fields are exactly the V2 contract fields


def test_baselines_and_mask_accept_v3():
    from o3.baselines.b1_reactive import ReactiveRulePolicy
    from o3.environment.gym_wrapper import action_mask_from_obs
    e = _env(obs_version="V3")
    o, _ = e.reset(year=2016)
    assert ReactiveRulePolicy().act(o) in (0, 1, 2) and action_mask_from_obs(o).shape == (3,)


def test_domain_randomisation_changes_physics_is_seeded_and_off_by_default():
    d = _env(domain_randomize=True)
    _, i1 = d.reset(year=2016, seed=5)
    _, i2 = d.reset(year=2016, seed=5)
    _, i3 = d.reset(year=2016, seed=6)
    assert i1["domain_rand"] == i2["domain_rand"] != i3["domain_rand"]
    m, b, dm = i3["domain_rand"]
    assert 0.90 <= m <= 1.10 and -2.0 <= b <= 2.0 and 0.85 <= dm <= 1.15
    n = _env()
    assert n.reset(year=2016)[1]["domain_rand"] == (1.0, 0.0, 1.0)
    d.reset(year=2016, seed=6); n.reset(year=2016)
    ed = sum(d.step(0)[4]["E_load_Wh"] for _ in range(240))
    en = sum(n.step(0)[4]["E_load_Wh"] for _ in range(240))
    assert ed != en                                       # the dynamics, not just the observation, changed


def test_eval_env_never_randomises():
    from o3.agents import common as Cm
    ev = Cm.make_eval_vec(REGIME, obs_version="V3")
    try:
        assert ev.get_attr("inner")[0].domain_randomize is False
    finally:
        ev.close()


def test_train_one_run_and_score_it(tmp_path):
    from o3.agents.gen2 import train_run, evaluate_run, gen2_dir
    meta = train_run("dqn", REGIME, steps=600, seed=1, results_dir=str(tmp_path))
    assert meta["obs_version"] == "V3" and meta["domain_randomize"] is True
    r = evaluate_run(str(gen2_dir(str(tmp_path)) / meta["tag"]), 2021)
    assert r["n_steps"] == 8760 and r["n_safety_violations"] == 0 and r["tag"] == meta["tag"]


def test_sensor_fault_corruption_touches_only_the_named_channels():
    from scripts.phase9_deployment_metrics import corrupt
    o = np.linspace(0.1, 0.9, 17).astype(np.float32)
    rng = np.random.default_rng(0)
    assert corrupt(o, "nominal", rng, {}).tolist() == o.tolist()
    d = corrupt(o, "pyranometer_dead", rng, {})
    assert d[5] == 0.0 and np.array_equal(np.delete(d, 5), np.delete(o, 5))
    b = corrupt(o, "water_sensor_bias_p2C", rng, {})
    assert b[0] == pytest.approx(o[0] + 0.05) and np.array_equal(b[1:], o[1:])


def test_robustness_and_replay_accept_a_gen2_run(tmp_path):
    """A Gen-2 (V3) run goes through the robustness suite and the replay in its own observation layout."""
    import json
    from o3.agents.gen2 import train_run, gen2_dir, run_model_path
    from o3.evaluation.robustness import robustness_job
    from o3.demo.replay import run_replay
    meta = train_run("dqn", REGIME, steps=600, seed=1, results_dir=str(tmp_path))
    rd = gen2_dir(str(tmp_path)) / meta["tag"]
    m = json.loads((rd / "run_meta.json").read_text())
    spec = ("model", "dqn", str(run_model_path(rd)), dict(obs_version=m["obs_version"], control_step_min=m["control_step_min"]))
    r = robustness_job(spec, "t", REGIME, 2021, {"name": "ghi_noise_5pct", "ghi_noise_pct": 5})
    assert r["n_steps"] == 8760 and r["n_safety_violations"] == 0
    df = run_replay(REGIME, 2021, "model", 48, ("dqn", str(run_model_path(rd)), m["obs_version"]))
    assert len(df) == 48 and set(df.action) <= {"CHARGE", "IDLE", "BYPASS"}
