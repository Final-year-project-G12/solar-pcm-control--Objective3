"""Streamlit demo (plan 15.9): banner on every view, replay reproduces the environment, test years never offered."""
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_replay_view_renders_with_banner_and_no_test_years():
    at = st_testing.AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("Not field validation" in e.value for e in at.error)
    sel = [s for s in at.selectbox if s.label.startswith("Year")][0]
    assert not {int(o) for o in sel.options} & {2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2024, 2025}
    assert len(at.dataframe) >= 1


def test_live_view_has_banner_and_degrades_without_network(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    at = st_testing.AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[1].set_value("Live").run()
    assert not at.exception and any("Not field validation" in e.value for e in at.error)
    at.button[0].click().run()
    assert not at.exception and at.session_state.live_rows[0]["quality_ok"] is False


def test_replay_log_equals_environment_numbers():
    from o3.demo.replay import run_replay
    from o3.environment.pcm_env import PCMEnv
    import config_O2
    df = run_replay(1, 2021, "b0", 72)
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=1, year=2021)
    env.reset(year=2021)
    ref = [env.step(0)[4]["E_del45_Wh"] for _ in range(72)]
    assert list(df.E_del45_Wh) == ref
