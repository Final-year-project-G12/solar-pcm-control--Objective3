# Objective 4 Deployment Test Plan — rajasthan regime 1

**Policy:** dqn  
**Status:** SPECIFICATION ONLY — NOT AUTHORISED FOR FIELD DEPLOYMENT  
**Exit gate:** No embedded deployment until hardware-in-the-loop tests pass (deployment_test_plan step 5).

| Step | Test | Pass gate |
|---|---|---|
| 1 | Bench sensor calibration | every sensor within datasheet tolerance vs reference |
| 2 | Actuator dry-run | pump + valve respond to all 3 commands, measured latency < control cycle |
| 3 | Shield interlock test (policy disabled) | forced over-temp triggers BYPASS and latches fault without the policy |
| 4 | Software-in-the-loop replay on hardware controller | controller reproduces simulator actions bit-for-bit on logged weather |
| 5 | Hardware-in-the-loop (HIL) with emulated tank | closed-loop stable, zero hard-limit breaches over a stress week |
| 6 | Instrumented field trial | measured delivery >= B0 baseline, zero safety violations — THIS is the first real experimental validation (Objective 4) |

## Fail-safe (hardware interlock, independent of the policy)

- IF T_water_top >= bypass_water_C: force BYPASS (isolate collector, pump OFF).
- IF T_pcm >= bypass_pcm_C AND T_water > T_pcm: block the charge heat path.
- IF GHI < pump_off_below_irradiance_Wm2: pump OFF.
- IF pump commanded ON and the flow sensor reads ~0 (dry run): stop the pump, raise a fault.
- IF any sensor stale > stale_timeout_s: fail to BYPASS (safe default), raise a fault.
- IF the f_melt estimator diverges from the T_pcm-implied enthalpy: fall back to reactive rule B1 and raise an alarm.
- IF a hard limit is exceeded despite the shield: latch fault, pump OFF, alarm, require manual reset.
