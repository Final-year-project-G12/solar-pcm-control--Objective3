"""
Phase O3-7 tests — software-in-the-loop demo.

The demo feeds weather frames into the SAME environment step as evaluation, so replaying a stored
validation year through the demo with B0 must reproduce the environment's own numbers (plan 15.10 M7:
"replay reproduces the numbers for allowed years").
"""

import pytest

import config_O2

CONTRACT_PATH = config_O2.CONTRACT_FILE
REGIME = 1


def test_replay_reproduces_environment_numbers(tmp_path):
    from o3.demo.live_adapter import LiveDemoRunner, DemoMode
    from o3.environment.pcm_env import PCMEnv
    n = 96
    runner = LiveDemoRunner(algo="b0", regime_id=REGIME, mode=DemoMode.REPLAY, replay_year=2021, log_dir=tmp_path)
    frames = runner.run(n_steps=n, verbose=False)
    env = PCMEnv(CONTRACT_PATH, regime_id=REGIME, year=2021)
    env.reset(year=2021)
    ref = [env.step(0)[4] for _ in range(n)]
    for f, r in zip(frames, ref):
        assert f["E_del45_Wh"] == r["E_del45_Wh"] and f["T_top_C"] == r["T_top_C"] and f["T_pcm_C"] == r["T_pcm_C"]


def test_stub_mode_runs_and_every_row_carries_the_banner(tmp_path):
    from o3.demo.live_adapter import LiveDemoRunner, DemoMode, BANNER
    frames = LiveDemoRunner(algo="b1", regime_id=REGIME, mode=DemoMode.STUB, log_dir=tmp_path).run(n_steps=24, verbose=False)
    assert len(frames) == 24 and all(f["banner"] == BANNER for f in frames)
    assert all(f["state_label"] == "simulator state (not measured)" for f in frames)
    assert list(tmp_path.glob("demo_log_*.csv"))


def test_demo_refuses_test_years_without_lock(tmp_path):
    from o3.demo.live_adapter import LiveDemoRunner, DemoMode
    from o3.utils import weather_roles
    weather_roles.release_for_tests()
    with pytest.raises(weather_roles.TestLockError):
        LiveDemoRunner(algo="b0", regime_id=REGIME, mode=DemoMode.REPLAY, replay_year=2008, log_dir=tmp_path)


def test_live_fetch_failure_falls_back_visibly(monkeypatch):
    import urllib.request
    from o3.demo import live_adapter as L

    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    fr = L.fetch_live_frame(REGIME, None)
    assert fr.quality_ok is False and "failed" in fr.note
