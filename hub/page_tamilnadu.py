from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from hub.banner import banner

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "tamilnadu" / "results"

banner()
st.title("Objective 3 - Tamil Nadu")
st.info("Tamil Nadu has no interactive simulator app in this repository; this page shows its generated result dashboards and the "
        "baseline comparison table. Rajasthan and Uttarakhand have the interactive replay.", icon=":material/info:")

csv = RES / "phase4_comparison_r0.csv"
if csv.exists():
    df = pd.read_csv(csv)
    cols = [c for c in ("total_delivered_kWh", "total_unmet_kWh", "n_safety_violations", "action_switches") if c in df]
    st.subheader("Policy comparison (regime 0, mean over validation years)")
    st.dataframe(df.groupby("policy")[cols].mean(numeric_only=True).round(1), width="stretch")

plots = RES / "plots"
tags = sorted(p.name for p in plots.iterdir() if p.is_dir()) if plots.exists() else []
if not tags:
    st.warning("No result dashboards found under tamilnadu/results/plots/.")
else:
    tag = st.selectbox("Result set", tags, index=tags.index("R2") if "R2" in tags else 0)
    files = sorted(f.name for f in (plots / tag).glob("*.html"))
    pick = st.selectbox("Dashboard", files, index=files.index("dashboard.html") if "dashboard.html" in files else 0)
    components.html((plots / tag / pick).read_text(encoding="utf-8"), height=900, scrolling=True)
