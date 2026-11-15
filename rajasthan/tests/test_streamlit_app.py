"""Streamlit demos (plan 15.9): banner on every view, replay reproduces the environment, test years never offered.
The first three tests cover the original demo (app/classic_app.py); the portal tests cover app/streamlit_app.py (Tamil Nadu design)."""
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP_DIR = pathlib.Path(__file__).resolve().parents[1] / "app"
APP = str(APP_DIR / "classic_app.py")            # the original demo (kept); the portal is app/streamlit_app.py
PORTAL = str(APP_DIR / "streamlit_app.py")


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


FORBIDDEN_YEARS = {2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2024, 2025}


def test_portal_renders_with_banner_and_never_offers_test_years():
    at = st_testing.AppTest.from_file(PORTAL, default_timeout=180).run()
    assert not at.exception
    assert not any("Not field validation" in e.value for e in at.error)        # banner removed from the portal on request
    years = {int(o) for o in at.sidebar.selectbox(key="p_year").options}
    assert not years & FORBIDDEN_YEARS and years >= {2021, 2022, 2023}


def test_portal_replay_compares_against_the_always_on_pump():
    at = st_testing.AppTest.from_file(PORTAL, default_timeout=600).run()
    at.sidebar.button(key="p_run").click().run()
    assert not at.exception
    m = {x.label: x for x in at.metric}
    assert "Hot water delivered" in m and "vs always-on" in str(m["Hot water delivered"].delta)
    assert "Pump actually ran" in m and "Hard-limit breaches" in m and m["Hard-limit breaches"].value == "0"


def test_portal_replay_equals_the_environment_numbers():
    """The portal's B0 window must be the environment's own E_del45 (same stepping, only re-labelled)."""
    import sys
    sys.path.insert(0, str(APP_DIR))
    from o3.demo.replay import run_replay
    at = st_testing.AppTest.from_file(PORTAL, default_timeout=600).run()
    at.sidebar.selectbox(key="p_regime").select(1).run()
    at.sidebar.selectbox(key="p_policy_1").select("b0").run()
    at.sidebar.button(key="p_run").click().run()
    shown = float([x for x in at.metric if x.label == "Hot water delivered"][0].value.split()[0])
    df = run_replay(1, 2021, "b0", 24 * 151 + 24 * 7)                    # default window: 1 Jun 2021 (day 151), 7 days
    ref = df.E_del45_Wh.iloc[24 * 151:24 * (151 + 7)].sum() / 1000.0
    assert abs(shown - ref) < 0.06, (shown, ref)


def test_portal_live_degrades_without_network(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    at = st_testing.AppTest.from_file(PORTAL, default_timeout=180).run()
    at.sidebar.radio(key="p_mode").set_value("LIVE (Open-Meteo)").run()
    at.sidebar.button(key="p_run").click().run()
    assert not at.exception
    assert any("Run failed" in e.value for e in at.error)
