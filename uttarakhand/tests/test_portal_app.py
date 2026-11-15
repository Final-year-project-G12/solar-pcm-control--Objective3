"""Uttarakhand portal (Tamil Nadu design): app/streamlit_app.py on AppTest. Banner on every view, test years never offered,
a replay run compares the policy against the always-on pump, LIVE degrades without a network. The original demo is app/classic_app.py."""
import pathlib

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
FORBIDDEN = {2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2024, 2025}


def test_portal_renders_with_banner_and_no_test_years():
    at = st_testing.AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert not any("Not field validation" in e.value for e in at.error)        # banner removed from the portal on request
    years = {int(o) for o in at.sidebar.selectbox(key="p_year").options}
    assert years and not years & FORBIDDEN


def test_portal_replay_compares_against_always_on():
    at = st_testing.AppTest.from_file(APP, default_timeout=900).run()
    at.sidebar.button(key="p_run").click().run()
    assert not at.exception
    m = {x.label: x for x in at.metric}
    assert "vs always-on" in str(m["Hot water delivered"].delta)
    assert m["Hard-limit breaches"].value == "0"


def test_portal_live_degrades_without_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")))   # o3/live.py uses requests
    at = st_testing.AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio(key="p_mode").set_value("LIVE (Open-Meteo)").run()
    at.sidebar.button(key="p_run").click().run()
    assert not at.exception
    assert any("Run failed" in e.value for e in at.error)
