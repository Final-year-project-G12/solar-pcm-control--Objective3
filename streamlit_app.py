"""
Combined Objective-3 dashboard (Rajasthan, Uttarakhand, Tamil Nadu).   Run from this folder:  streamlit run streamlit_app.py

Each state keeps its own unmodified app and code; hub/state_runner.py runs one at a time with the right sys.path / working
directory and a clean module table (both states define a package called `o3`).
"""
import streamlit as st

st.set_page_config(page_title="O3 PCM controller - all states", layout="wide")

pages = [
    st.Page("hub/page_home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("hub/page_rajasthan.py", title="Rajasthan", icon=":material/wb_sunny:", url_path="rajasthan"),
    st.Page("hub/page_uttarakhand.py", title="Uttarakhand", icon=":material/landscape:", url_path="uttarakhand"),
    st.Page("hub/page_tamilnadu.py", title="Tamil Nadu", icon=":material/water_drop:", url_path="tamilnadu"),
]
st.navigation(pages).run()
