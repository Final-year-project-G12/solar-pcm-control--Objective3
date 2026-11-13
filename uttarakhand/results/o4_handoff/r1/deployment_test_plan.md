# Objective 4 deployment test plan - Uttarakhand regime 1 (SIMULATION-DERIVED, NOT VALIDATED)

Policy `policy_r1.zip` (SHA-256 `2db55a0b20842a710afe1384565aa734a5ad90225a14905c5b56a37cb260725b`), observation layout and normalisation in `hardware_interface_spec.json`.
**No embedded deployment until step 6 passes.**

1. **Bench interlock test (no policy).** Wire the independent interlock: pump OFF whenever the top-of-tank sensor reads
   >= 72.0 C, PCM-side exchange cut at 62.0 C if a PCM sensor exists; hard limits 75 C water / 65 C PCM. Verify by heating
   the sensor, not by software.
2. **Sensor commissioning.** Calibrate every `measurable: true` sensor; log 7 days at the sampling interval; check
   ranges against the simulator's training distribution (normalisation file).
3. **Estimator.** T_pcm, f_melt, stored and recoverable energy are NOT measurable (contract). Build and validate an
   estimator from tank temperatures before the policy may use them; until then feed their training means (the O3
   `pcm_dropout` sensor-fault scenario measured the cost of doing so).
4. **Shadow mode.** Run the policy in parallel with always-on operation (B0) without actuating; log both decisions.
5. **Software-in-the-loop replay of the shadow log** through the O3 simulator (`app/streamlit_app.py`, re-run tab).
6. **Hardware-in-the-loop.** Policy actuates the pump on the bench rig with the interlock active; pass = zero hard-limit
   events and delivery not worse than B0 by more than the O3 validation gap (59 kWh/yr window).

Simulation result to beat (O3, validation 2021-2023): B0 delivers more window energy than this policy; the policy's
benefit is fewer shield hours. Deploy only if the project values that trade-off.
