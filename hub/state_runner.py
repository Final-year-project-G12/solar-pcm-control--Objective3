"""
hub/state_runner.py - run one state's own Streamlit app inside the combined app, without editing it.

Why this is needed: rajasthan/ and uttarakhand/ each have a package called `o3` (and Rajasthan also puts a vendored copy of the
Objective-2 tree, with its own `config` and `src`, on sys.path). Python keeps ONE `o3` in sys.modules, so importing both would
silently run one state on the other's code. Before a state's app runs, this module therefore
  1. takes a process-wide lock,
  2. if a different state ran last, removes every module that lives under any state folder from sys.modules and every state
     folder from sys.path,
  3. puts only this state's folders on sys.path, sets the working directory and the environment the state expects,
  4. executes the state's own app file with runpy, with st.set_page_config turned into a no-op (the hub already called it).

Cost: the lock serialises state pages across visitors (fine for a demo on a free single-core host), and switching state re-imports
that state's packages, so the first page view after a switch is slower.
"""
from __future__ import annotations

import os
import runpy
import sys
import threading
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
VENDOR_RJ = REPO / "deploy" / "vendor" / "objective2-rajasthan"

STATES = {
    "rajasthan": dict(label="Rajasthan", root=REPO / "rajasthan", app="app/streamlit_app.py",
                      paths=[REPO / "rajasthan"], env={"O2_RAJASTHAN_ROOT": str(VENDOR_RJ)}),
    "uttarakhand": dict(label="Uttarakhand", root=REPO / "uttarakhand", app="app/streamlit_app.py",
                        paths=[REPO / "uttarakhand"], env={}),
}
_OWNED_ROOTS = [REPO / "rajasthan", REPO / "uttarakhand", REPO / "tamilnadu", VENDOR_RJ]
_PLAIN_NAMES = {"o3", "config", "config_O2", "src", "app", "scripts", "conftest"}   # top-level names the states reuse

_LOCK = threading.RLock()
_ACTIVE: dict = {"state": None, "env": {}}


def _under_owned_root(path) -> bool:
    if not path:
        return False
    try:
        p = Path(path).resolve()
    except OSError:
        return False
    return p in _OWNED_ROOTS or any(r in p.parents for r in _OWNED_ROOTS)


def _purge_modules() -> int:
    """Forget every module that came from a state folder (or the vendored O2 tree), plus the reused top-level names."""
    doomed = [n for n, m in list(sys.modules.items())
              if _under_owned_root(getattr(m, "__file__", None)) or n.split(".")[0] in _PLAIN_NAMES]
    for n in doomed:
        sys.modules.pop(n, None)
    return len(doomed)


def _purge_paths() -> None:
    sys.path[:] = [p for p in sys.path if not _under_owned_root(p)]


def _activate(key: str) -> None:
    cfg = STATES[key]
    if _ACTIVE["state"] != key:
        _purge_modules()
        _purge_paths()
        for k in _ACTIVE["env"]:
            os.environ.pop(k, None)
        os.environ.update(cfg["env"])
        _ACTIVE.update(state=key, env=dict(cfg["env"]))
    for p in reversed(cfg["paths"]):                      # (re)assert this state's path entries on every run
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
    os.chdir(cfg["root"])


def _reset_widget_state(key: str) -> None:
    """Session keys written by one state's app (e.g. `regime`) must not leak into the other's."""
    if st.session_state.get("_hub_state") != key:
        for k in [k for k in st.session_state.keys() if not str(k).startswith("_hub")]:
            del st.session_state[k]
        st.session_state["_hub_state"] = key


def run_state_app(key: str) -> None:
    cfg = STATES[key]
    app = cfg["root"] / cfg["app"]
    if not app.exists():
        st.error(f"{cfg['label']} app not found at {app.relative_to(REPO)}")
        return
    if key == "rajasthan" and not VENDOR_RJ.exists():
        st.error("The vendored Objective-2 Rajasthan tree (deploy/vendor/objective2-rajasthan) is missing, "
                 "so Rajasthan cannot run here.")
        return
    if key == "rajasthan":
        from hub.vendor_fix import ensure_vendor_bytes      # git may have changed the line endings the contract hashes depend on
        still_bad = ensure_vendor_bytes()
        if still_bad:
            st.error("Vendored Objective-2 files do not match the Rajasthan contract (not a line-ending problem): " + ", ".join(still_bad))
            return
    _reset_widget_state(key)
    with _LOCK:
        _activate(key)
        original = st.set_page_config
        st.set_page_config = lambda *a, **k: None         # the hub already called it; a second call raises
        try:
            runpy.run_path(str(app), run_name="__main__")
        finally:
            st.set_page_config = original
