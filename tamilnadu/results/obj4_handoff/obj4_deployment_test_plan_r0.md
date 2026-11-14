# Objective 4 Deployment Test Plan — tamilnadu regime 0

**Policy:** dqn  
**Status:** SPECIFICATION ONLY — NOT AUTHORISED FOR FIELD DEPLOYMENT  
**Exit gate:** No embedded deployment until hardware-in-the-loop tests pass (deployment_test_plan step 5).

| Step | Test | Pass gate |
|---|---|---|
| 1 | Bench sensor calibration | every sensor within datasheet tolerance vs reference |
| 2 | Actuator dry-run | pump + valve respond to all 3 commands, measured latency < control_cycle_s |
| 3 | Shield interlock test (policy disabled) | forced over-temp triggers BYPASS and latches fault without the policy |
| 4 | Software-in-the-loop replay on hardware controller | controller reproduces simulator actions bit-for-bit on logged weather |
| 5 | Hardware-in-the-loop (HIL) with emulated tank | closed-loop stable, zero hard-limit breaches over a stress week |
| 6 | Instrumented field trial | measured delivery >= B0 baseline, zero safety violations — THIS is the first real experimental validation (Objective 4) |

## Fail-safe (hardware interlock, independent of the policy)

- IF T_water >= bypass_water_C: force BYPASS (isolate collector, pump OFF).
- IF T_pcm >= bypass_pcm_C AND T_water > T_pcm: block charge heat path.
- IF any sensor stale > stale_timeout_s: fail to BYPASS (safe default).
- IF f_melt estimator diverges from T_pcm-implied enthalpy: fall back to reactive rule B1 and raise an alarm.
- IF hard limit exceeded despite shield: latch fault, pump OFF, alarm, require manual reset.
