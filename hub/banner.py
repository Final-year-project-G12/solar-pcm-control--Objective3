import streamlit as st

BANNER = "Software-in-the-loop simulation. Not field validation."


def banner() -> None:
    st.error(f"**{BANNER}** All tank, PCM and melt-fraction values come from the frozen Objective-2 simulator, not from sensors.",
             icon=":material/warning:")
