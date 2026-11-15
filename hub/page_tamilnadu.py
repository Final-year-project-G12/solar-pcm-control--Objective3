from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from hub.state_runner import run_state_app

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "tamilnadu" / "results"

run_state_app("tamilnadu")                      # Tamil Nadu's own portal: tamilnadu/scripts/streamlit_portal.py

plots = RES / "plots"
tags = sorted(p.name for p in plots.iterdir() if p.is_dir()) if plots.exists() else []
if tags:
    with st.expander("Saved result dashboards (static HTML from make_plots.py)"):
        tag = st.selectbox("Result set", tags, index=tags.index("R2") if "R2" in tags else 0, key="_hub_tn_tag")
        files = sorted(f.name for f in (plots / tag).glob("*.html"))
        pick = st.selectbox("Dashboard", files, index=files.index("dashboard.html") if "dashboard.html" in files else 0, key="_hub_tn_file")
        components.html((plots / tag / pick).read_text(encoding="utf-8"), height=900, scrolling=True)
