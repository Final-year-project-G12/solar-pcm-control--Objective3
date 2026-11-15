import streamlit as st

st.title("Objective 3 - climate-adaptive PCM controller")
st.write("A DRL controller decides, hour by hour, whether the solar collector loop circulates (charges the PCM tank) or is "
         "bypassed. The tank, PCM, collector, safety limits and simulator are fixed by Objective 2 and are never edited here. "
         "Pick a state in the sidebar.")
st.write("Every state now has the same portal (the Tamil Nadu design): a **Live Controller** tab that runs a policy through the frozen "
         "physics on replayed or live weather and shows what the pump physically did hour by hour against an always-on pump, with "
         "click-to-explain charts; and a **Deployment & ML Models** tab with efficiency, baseline comparison, inference cost and "
         "sensor-fault robustness.")
st.table({
    "State": ["Rajasthan", "Uttarakhand", "Tamil Nadu"],
    "Policies you can run": [
        "B0 / B1 / B2 baselines, selected DQN and SAC (3 regimes)",
        "B0 / B1 baselines, frozen PPO (4 regimes; regime 3 is water-only)",
        "SAC and DQN, with or without the realistic-forecast variant (3 regimes)"],
    "Deployment tab data": [
        "all three regimes (regime 1: the released DQN; regimes 0 and 2: the selected SAC)",
        "all four regimes (M10)",
        "all three regimes, three model sets"],
})
st.caption("Test years are never offered; every replay uses train or validation weather. Each state runs its own code, unmodified "
           "apart from its portal file (see README.md, section 'Combined dashboard').")
