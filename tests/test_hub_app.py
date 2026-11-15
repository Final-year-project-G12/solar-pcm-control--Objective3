"""
Smoke test for the combined dashboard: every page loads without an exception, and each state runs on ITS OWN `o3` package
(both states define one; mixing them is the failure the hub exists to prevent). Run from the repo root:  pytest tests/test_hub_app.py
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "streamlit_app.py")


def _o3_origin() -> str:
    m = sys.modules.get("o3")
    return str(getattr(m, "__file__", "")).replace("\\", "/")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=300).run()


def test_home_loads(app):
    assert not app.exception
    assert any("Objective 3" in t.value for t in app.title)


def test_vendored_o2_tree_is_present():
    v = REPO / "deploy" / "vendor" / "objective2-rajasthan"
    for rel in ("config.py", "src", "configs", "data/weather/point_years/rajasthan", "results/obj3_environment_contract_rajasthan_v3.json"):
        assert (v / rel).exists(), f"vendored O2 file missing: {rel}"


def test_vendored_o2_files_match_the_contract_hashes():
    """The O3 contract hashes the O2 files it was built from. Git line-ending conversion changes those bytes (Windows CRLF ->
    Linux LF), which is why .gitattributes marks deploy/vendor as -text. This test names the file instead of failing deep in the app."""
    import hashlib
    sys.path.insert(0, str(REPO))
    from hub.vendor_fix import VENDOR, ensure_vendor_bytes, expected_hashes
    # whatever line endings git delivered, the repair must reach the contract hashes (a real content difference cannot be repaired)
    bad = ensure_vendor_bytes()
    assert not bad, f"vendored files differ from the contract hashes beyond line endings: {bad}"
    assert all(hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == h for rel, h in expected_hashes().items())


def test_line_ending_repair_fixes_lf_checkouts(tmp_path, monkeypatch):
    """Simulate what a Linux runner gets when git has normalised the vendored files to LF: the repair must restore them."""
    import hashlib
    import shutil
    sys.path.insert(0, str(REPO))
    import hub.vendor_fix as vf
    fake = tmp_path / "vendor"
    for rel in vf.expected_hashes():
        dst = fake / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((vf.VENDOR / rel).read_bytes().replace(b"\r\n", b"\n"))     # LF, like a normalised checkout
    monkeypatch.setattr(vf, "VENDOR", fake)
    assert vf.ensure_vendor_bytes() == []
    for rel, h in vf.expected_hashes().items():
        assert hashlib.sha256((fake / rel).read_bytes()).hexdigest() == h
    shutil.rmtree(fake)


def test_states_stay_isolated_and_switching_works(app):
    for page, state in [("hub/page_rajasthan.py", "rajasthan"), ("hub/page_uttarakhand.py", "uttarakhand"),
                        ("hub/page_rajasthan.py", "rajasthan"), ("hub/page_tamilnadu.py", None)]:
        app.switch_page(page).run()
        assert not app.exception, f"{page}: {[e.value for e in app.exception]}"
        assert app.title, f"{page}: no title rendered"
        if state:
            assert f"/{state}/o3/" in _o3_origin(), f"{page} ran on {_o3_origin()}"


def test_banner_on_every_hub_page(app):
    for page in ("hub/page_home.py", "hub/page_tamilnadu.py"):
        app.switch_page(page).run()
        assert any("Not field validation" in e.value for e in app.error)
