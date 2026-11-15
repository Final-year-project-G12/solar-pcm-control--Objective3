import streamlit as st

from hub.banner import banner

banner()
st.title("Objective 3 - climate-adaptive PCM controller")
st.write("A DRL controller decides, hour by hour, whether the solar collector loop circulates (charges the PCM tank) or is "
         "bypassed. The tank, PCM, collector, safety limits and simulator are fixed by Objective 2 and are never edited here. "
         "Pick a state in the sidebar.")
st.table({
    "State": ["Rajasthan", "Uttarakhand", "Tamil Nadu"],
    "What the page offers": [
        "Interactive replay of baselines and trained DQN / SAC / PPO agents; live-weather mode",
        "Interactive replay of baselines and trained PPO; live-weather mode",
        "Static result dashboards and the baseline comparison table (no interactive app exists for this state)"],
})
st.caption("Test years are never offered; every replay uses train or validation weather. Each state runs its own unmodified code "
           "(see README.md, section 'Combined dashboard').")
