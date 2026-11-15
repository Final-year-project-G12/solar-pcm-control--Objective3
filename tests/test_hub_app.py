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
    import json
    v = REPO / "deploy" / "vendor" / "objective2-rajasthan"
    prov = json.loads((REPO / "rajasthan" / "contracts" / "obj3_environment_contract_rajasthan_v3.json").read_text())["provenance"]
    want = {f"results/{k}": h for k, h in prov.get("selection", {}).items()}
    want.update({f"results/{k}": prov[k] for k in ("g10_heldout_results.csv", "held_out_test.lock", "wp6_band.csv") if k in prov})
    want.update({k: prov[k] for k in ("configs/system_config_shared.yaml", "configs/design_space.yaml", "configs/states/rajasthan.yaml") if k in prov})
    bad = [rel for rel, h in want.items() if hashlib.sha256((v / rel).read_bytes()).hexdigest() != h]
    assert not bad, f"vendored files differ from the contract hashes (line endings? see .gitattributes): {bad}"


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
