# Objective 3 — Complete implementation plan

## Climate-adaptive deep-reinforcement-learning controller for PCM solar water heating
**Project:** Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating
**Scope:** Rajasthan, Assam, Tamil Nadu and Uttarakhand, implemented separately first and then unified through one state-parameterised codebase
**Dependency:** Objective 3 consumes the frozen Objective 2 hardware design and simulator; it must not reselect the PCM or redesign the capsule system.

---

## 0. Executive decision and boundary
Objective 3 should be implemented as a **safe, hybrid-control reinforcement-learning environment**. The action is the operating decision—charge, discharge, bypass and, where hardware permits, a continuous flow command. Objective 3 must not change the PCM identity, capsule geometry, capsule count, tank volume, collector size, hard safety limits or released thermal physics. The uploaded project documents define this separation explicitly: Objective 1 selects the PCM, Objective 2 selects the physical design and operating envelope, Objective 3 selects the real-time action, and Objective 4 validates the integrated stack on hardware. [&#91;1&#93;](#ref-unknown0000), [&#91;2&#93;](#ref-unknown0000a)

The current Tamil Nadu Objective 2 documentation reports three regime-specific designs, with C25/C26 PCMs, 32–35 capsules and radial or single-layer arrangements, and states that the contract contains the required static design, action, reward and acceptance-test fields. [&#91;3&#93;](#ref-unknown0000b) The current objective summary also describes a 30-stage Objective 2 pipeline and the intended O3 boundary. [&#91;1&#93;](#ref-unknown0000)

**Important implementation status rule:** this plan treats the supplied current Objective 2 summaries as the starting specification, but it does not silently merge contradictory historical documents. The uploaded records contain older and newer tank sizes, contracts, surrogate winners and held-out-test statuses. Before O3 training, select one dated Objective 2 result registry and one contract per state; archive all superseded contracts. [&#91;3&#93;](#ref-unknown0000b), [&#91;4&#93;](#ref-unknown0000c)

---

# 1. Objective 3 research question and measurable outcomes

### Primary research question
Can a safe deep-reinforcement-learning controller improve the operation of the frozen PCM solar-water-heating design over the existing reactive bypass/charge-block shield while preserving temperature, hydraulic and energy-balance constraints under the four Indian climate locations?

### Secondary questions

1. Does a controller trained separately for Rajasthan, Assam, Tamil Nadu and Uttarakhand outperform one pooled controller with a state/regime context variable?

2. Does a recurrent policy using recent weather and thermal history outperform a memoryless policy during cloudy sequences and delayed PCM discharge?

3. Can a model-predictive or physics-informed policy reduce unnecessary safety-shield activation without increasing unmet demand or temperature violations?

4. Does the policy remain safe under unseen years, member points within the same regime, sensor noise, forecast error and live-data gaps?

### Primary success criteria
The final policy must satisfy all of the following on the untouched test set:

- zero hard safety violations, or a formally enforced safety shield that overrides every unsafe action;

- energy-balance residual within the released simulator tolerance;

- delivery reliability no worse than the Objective 2 baseline;

- lower shield-activation frequency or better discharge timing than the reactive baseline;

- no command outside the Objective 2 flow and pressure envelope;

- reproducible trajectories under fixed seeds;

- improvement reported with confidence intervals over the same weather, demand and initial-state scenarios.

Do not define success only as higher reward. A policy can obtain high reward by exploiting a simulator defect, reducing useful delivery, or approaching a safety boundary too aggressively.

---

# 2. What Objective 3 receives from Objective 2

## 2.1 Contract-loading stage
The first program must load, validate and hash `obj3_environment_contract_<state>.json`. It must refuse to train if the contract is missing, malformed, internally inconsistent or different from the simulator/configuration hash. The controller must not re-run the Objective 2 optimiser.

The contract should contain:

- state and regime identifier;

- selected PCM and complete thermophysical property record;

- capsule dimensions, number, arrangement, PCM mass and conduction distance;

- tank volume, collector area, insulation, losses and hydraulic limits;

- nominal flow and minimum/maximum permitted flow;

- water and PCM hard limits;

- guard-band thresholds;

- released simulator version and configuration hash;

- weather and demand scenario manifests;

- observation definitions and sensor/estimator flags;

- action space and action duration;

- reset scenarios;

- reward components and starting weights;

- acceptance tests and provenance hashes.

The supplied Objective 3 material already proposes this contract structure, including discrete charge/discharge/bypass actions, a continuous-flow hybrid, reset scenarios, a reward function and a pre-training acceptance checklist. [&#91;3&#93;](#ref-unknown0000b)

## 2.2 Contract consistency checks
Before any RL training, automatically test:

1. PCM ID in the contract equals the PCM ID in the Objective 2 result registry.

2. Capsule count, arrangement, diameter and flow are identical in the simulator and contract.

3. Tank/collector dimensions match the latest accepted Objective 2 run. The older handoff document contains 50 L/1.5 m² values, whereas later Tamil Nadu materials report a 100 L/2.5 m² configuration; only the latest signed contract may be used. [&#91;3&#93;](#ref-unknown0000b), [&#91;4&#93;](#ref-unknown0000c)

4. Safety guard values are below hard limits: water bypass at 72 °C and PCM charge block at 62 °C for hard limits of 75 °C and 65 °C, if these are the accepted state configuration. [&#91;4&#93;](#ref-unknown0000c), [&#91;5&#93;](#ref-unknown0000d)

5. `run_year()` and the future `step(action)` produce identical trajectories when the stepper is given the baseline fixed action.

6. The contract’s reward weights are treated as starting values, not as experimentally established optimum weights.

---

# 3. Refactor the annual simulator into a step-based environment

## 3.1 Preserve the physics
Refactor the inner loop of the existing annual simulator; do not rewrite the enthalpy, heat-transfer, collector, tank-loss or hydraulic equations. The uploaded O3 instructions correctly state that O3 needs `step(action) -> (observation, reward, done, info)` while reusing Objective 2 physics. [&#91;3&#93;](#ref-unknown0000b)

Each environment instance should maintain:

- current date, time and timestep;

- water temperatures for all tank nodes;

- PCM temperature and enthalpy for each group;

- liquid/melt fraction;

- collector and outlet temperatures;

- current flow and pressure drop;

- demand draw and unmet demand;

- cumulative delivered energy, pumping energy and safety events;

- weather forecast/observation buffer;

- previous action and switching counters.

## 3.2 Recommended observation vector
Use three observation versions so the contribution of complexity can be measured.

**V1: minimal physical state**

`[GHI, ambient temperature, top-water temperature, bottom-water temperature, PCM temperature, melt fraction, demand, mains temperature, hour, day-of-year, previous mode]`

**V2: engineered state**

Add:

- short-term irradiance history;

- 3 h and 6 h irradiance trend;

- predicted next-demand volume;

- recoverable energy above the delivery temperature;

- temperature margin to the water and PCM guard bands;

- recent shield activations;

- pressure-drop margin;

- regime ID and state ID.

**V3: recurrent/raw-history state**

Provide the last 12–24 observations to an LSTM/GRU policy rather than relying entirely on engineered summaries. This is recommended for Assam cloud persistence, Rajasthan high irradiance, Tamil Nadu monsoon timing and Uttarakhand cold-inlet conditions.

Every variable must be normalised using training-set statistics only. Do not leak future weather, future demand or test-year statistics into normalisation.

## 3.3 Action space
Implement and compare the following:

**A0 — fixed baseline:** Objective 2’s selected nominal flow and its existing shield.

**A1 — discrete control:**

`{charge, discharge, bypass}`

The environment maps each mode to a validated flow/valve command.

**A2 — hybrid control:**

`[mode, flow_rate]`, where flow is clipped to the state-specific Objective 2 envelope.

**A3 — residual control:**

The safety shield and a conservative rule-based controller produce a safe baseline action; the RL policy learns only a bounded correction such as flow adjustment or an earlier bypass request. This is the recommended first learning formulation because it reduces unsafe exploration.

The action mask must disable actions that are physically unavailable, but the safety shield must remain inside the environment and cannot be bypassed by the policy.

---

# 4. Safety architecture: shielded and constrained RL
A conventional unconstrained DQN/PPO/SAC agent should not control the raw simulator directly. Use a layered architecture:

1. **Policy proposal:** RL outputs a candidate action.

2. **Action validator:** checks flow, valve, pressure and mode constraints.

3. **Safety shield:** overrides any action predicted to cross a guard band during the next step.

4. **Physics transition:** simulator advances one step.

5. **Emergency termination:** records the event and terminates only if the protected environment still violates a hard limit.

The present shield is a reactive instantaneous rule. O3’s research contribution is to learn better timing and operation around it, not to remove it. The supplied materials explicitly describe the shield as safe but blunt and identify anticipatory control as the purpose of O3. [&#91;3&#93;](#ref-unknown0000b)

Use the following safety methods as progressively complex experiments:

- action masking;

- rule-based shield;

- one-step reachable-set check;

- model-predictive safety filter using the grey-box model;

- constrained PPO with a cost critic;

- Lagrangian SAC or constrained SAC for continuous flow;

- conservative offline pretraining from safe trajectories;

- domain-randomised safety evaluation.

The final thesis should compare at least the unshielded proposal, shielded policy and rule-based controller, but the deployable policy must remain shielded.

---

# 5. Baselines and algorithms to implement
Do not select a complex algorithm only because it sounds advanced. Implement a controlled benchmark ladder.

## 5.1 Non-learning baselines

- **B0:** fixed nominal flow with current Objective 2 shield;

- **B1:** reactive rule-based controller using irradiance, demand time, PCM melt fraction and temperature margin;

- **B2:** forecast-aware rule-based controller using a short irradiance forecast;

- **B3:** receding-horizon MPC using the frozen grey-box model;

- **B4:** random safe policy, used only as a diagnostic.

B1 must pass the O3 acceptance test before DRL training. The uploaded O3 plan correctly requires this baseline first. [&#91;1&#93;](#ref-unknown0000)

## 5.2 Discrete algorithms

- **DQN:** baseline value-learning model;

- **Double DQN + duelling network + prioritised replay:** stronger discrete benchmark;

- **Distributional Rainbow DQN:** estimates a return distribution and can represent risk;

- **Recurrent Rainbow DQN:** tests memory for cloud sequences and delayed storage response.

## 5.3 Continuous/hybrid algorithms

- **PPO:** first recommended policy-gradient method because it is stable and easy to constrain;

- **Recurrent PPO/LSTM-PPO:** recommended for partial observability;

- **SAC:** recommended for continuous flow because entropy regularisation helps exploration;

- **Constrained SAC:** use safety cost and Lagrangian multiplier;

- **TD3:** optional continuous-control comparison where action smoothness matters;

- **Hybrid action PPO/SAC:** discrete mode plus continuous flow head.

## 5.4 Physics-informed and model-based algorithms

- **MPC-guided PPO:** use MPC action as a teacher, warm start or safety fallback;

- **PETS/MBPO-style model-based RL:** learn a residual model only for uncertain disturbances, never replace the released physics silently;

- **offline conservative RL:** train from safe rule/MPC trajectories before online simulation interaction;

- **multi-objective RL:** report separate policies for delivery, energy and safety rather than hiding all goals in one arbitrary reward.

The recommended final comparison is **rule-based control vs MPC vs recurrent PPO vs constrained SAC vs MPC-guided recurrent PPO**. DQN/Rainbow should be included if the discrete action version is retained. A single final algorithm should be chosen from test results, not assumed beforehand.

Published precedents support this method family. The IEEE NSGA-II paper establishes constrained elitist multi-objective search, while the IEEE surrogate-assisted work supports comparing multiple surrogate models instead of assuming one is best. [&#91;6&#93;](#ref-deb2002), [&#91;7&#93;](#ref-bhattacharjee2016) The project’s own reference file includes DRL solar-thermal control and ANN-MPC precedents, but these should support method selection rather than substitute for testing the present environment. [&#91;8&#93;](#ref-unknown0000e)

---

# 6. Reward design
Start from the contract’s reward defaults, then conduct a declared sensitivity study. Do not tune reward weights on the final test years.

A suitable step reward is:

`r_t = w_E E_delivered,t - w_U E_unmet,t - w_P E_pump,t - w_S C_safety,t - w_B C_bypass,t - w_A C_action-change,t`

where:

- `E_delivered` rewards hot water above the required delivery temperature;

- `E_unmet` penalises undelivered demand;

- `E_pump` penalises pumping energy only if the model calculates it meaningfully;

- `C_safety` gives a very large penalty for guard-band activation and an effectively prohibitive penalty for hard-limit violation;

- `C_bypass` penalises unnecessary collector bypass;

- `C_action-change` limits chattering.

Use reward versions:

- R1: delivery and unmet demand only;

- R2: add pumping/action-switching costs;

- R3: constrained reward with separate safety cost critic;

- R4: risk-sensitive distributional reward, penalising poor lower-tail performance.

Report whether a policy’s improvement comes from more useful delivery, fewer shield events, lower switching, lower pump energy or simply changed reward scaling.

---

# 7. Climate and data design for four separate locations

## 7.1 State-specific first implementation
Create four independent contracts and four experiment folders:

- `obj3/rajasthan/`

- `obj3/assam/`

- `obj3/tamilnadu/`

- `obj3/uttarakhand/`

Use the same environment API, but state-specific configuration. The four states should not be concatenated blindly because their climate signatures, regime counts, elevation, monsoon behaviour, PCM candidates and thermal targets differ. The Objective 1 framework describes the intended contrasts: hot-dry Rajasthan, monsoon/cloud-dominated Assam, north-east-monsoon/coastal-interior Tamil Nadu, and cold/elevation-dominated Uttarakhand. [&#91;9&#93;](#ref-unknown0000f)

## 7.2 Required implementation differences
**Rajasthan**

- include high irradiance, high collector-temperature and strong diurnal-cycle scenarios;

- test overheating and unnecessary bypass as primary control problems;

- include long dry sequences and high evening demand;

- preserve the documented limitation that flow optima can be model artefacts if collector efficiency is flow-independent. [&#91;5&#93;](#ref-unknown0000d)

**Assam**

- include long cloudy and consecutive-low-clearness sequences;

- increase the importance of uncertainty, partial charging and unmet demand;

- test humidity/corrosion-related fixed hardware constraints, but do not let the controller change the PCM or enclosure;

- use recurrent policies and forecast uncertainty because cloud persistence is central to the operating problem. [&#91;2&#93;](#ref-unknown0000a), [&#91;10&#93;](#ref-unknown0000g)

**Tamil Nadu**

- model north-east monsoon timing separately from south-west-monsoon states;

- preserve the current three-regime structure only after the current contract/configuration is frozen;

- test whether selected actions differ by coastal, interior and high-relief regimes;

- use the current O2 designs as fixed hardware, not as a new optimisation problem. [&#91;11&#93;](#ref-unknown0000h)

**Uttarakhand**

- retain elevation, cold mains temperature and mountain-representative weather;

- test cold-start, incomplete charging and high latent-heat demand;

- include sensor/communication gaps and forecast uncertainty;

- do not reuse Tamil Nadu’s target, PCM or tank assumptions. The Uttarakhand pipeline specifically identifies diagonal GMM, k=5 and elevation-sensitive climate structure. [&#91;12&#93;](#ref-unknown0000i)

## 7.3 Unified versus separate policy
Train three policy forms:

1. **State-specific policy:** one policy per state and regime;

2. **State-conditioned policy:** one policy with state ID and regime features;

3. **meta-policy:** shared encoder plus state-specific policy heads.

Use the state-specific policies as the primary scientific comparison. Use the pooled policy only if it performs no worse in every state and does not conceal poor performance in Assam or Uttarakhand. A pooled model must be evaluated per state, regime, season and weather year.

---

# 8. Training, validation and testing protocol

## 8.1 Dataset construction
Generate step-level transitions from the frozen simulator. Store:

`episode_id, state, regime, year, point, season, observation, action, next_observation, reward, safety_cost, done, simulator_hash, contract_hash`.

Generate episodes from:

- fully solid PCM;

- partially charged PCM;

- fully liquid PCM;

- cold-start tank;

- representative daily demand;

- high-demand and shifted-evening demand;

- low-irradiance and cloudy sequences;

- sensor-noise and forecast-error scenarios.

The O2 reports emphasise that weather roles and held-out tests must be locked. Preserve this principle in O3: no test year may influence reward selection, hyperparameter tuning, normalisation, early stopping or algorithm choice. [&#91;5&#93;](#ref-unknown0000d)

## 8.2 Recommended split
Use a time- and location-aware split for each state:

- **training:** 2016–2020;

- **validation:** 2021–2023;

- **final test:** 2011–2015 only if the state’s held-out lock confirms it was not used for O3 tuning;

- **stress test:** 2024–2025 or newly acquired years, but only after the primary test is locked;

- **member-point test:** held-out medoid-member locations within each climate regime.

Because the supplied documents contain different claims about which years were already used for Objective 2, the exact O3 test years must be read from the final contract and registry rather than assumed. [&#91;4&#93;](#ref-unknown0000c), [&#91;5&#93;](#ref-unknown0000d)

## 8.3 Training procedure

1. Validate the contract and simulator hash.

2. Run unit tests for enthalpy, energy balance, time indexing, action bounds and shield overrides.

3. Run B0–B3 baselines.

4. Generate safe demonstrations using B1 and MPC.

5. Train a discrete agent on A1 if discrete control is selected.

6. Train PPO and SAC on A2/A3 with action shielding.

7. Add recurrence only after the memoryless baseline is stable.

8. Run 10–20 independent random seeds.

9. Select hyperparameters using validation performance and safety—not test performance.

10. Freeze the policy, normalisation statistics and reward configuration.

11. Run the locked test once.

12. Run stress tests separately and label them exploratory.

Use early stopping based on validation episodes, not training reward. Save the best checkpoint by a lexicographic rule: first zero hard violations, then delivery reliability, then useful energy, then shield/switching reduction.

## 8.4 Evaluation metrics
Report mean, standard deviation and 95% confidence intervals for:

- delivered energy;

- unmet demand;

- delivery success probability;

- solar fraction;

- water peak temperature;

- PCM peak temperature;

- guard-band activations;

- hard safety violations;

- bypass duration;

- action-switch count;

- charge/discharge timing error;

- pump energy, if valid;

- energy-balance residual;

- robustness under property/weather/demand perturbations.

Report per state, regime, season, year, initial PCM state and controller. Never report only pooled averages.

---

# 9. Live climate data and demonstration mode

## 9.1 Can live data be used before hardware is available?
**Yes, but only as a software-in-the-loop demonstration and not as evidence of field validation.** Live or near-real-time weather can drive a demonstration adapter that feeds irradiance, ambient temperature and forecast values into the frozen simulator. The simulator supplies the unmeasured tank/PCM state. This demonstrates the data pipeline and policy inference, but it does not prove that a real tank follows the simulated state.

## 9.2 Correct live-data architecture
Use three modes:

**Replay mode:** recorded historical weather; used for reproducible research results.

**Live-demo mode:** live weather API plus simulator state estimator; used for dashboards and demonstrations. Log the provider, timestamp, timezone, missing values, latency and fallback source.

**Hardware mode:** sensor readings from the actual collector/tank; used only after Objective 4 instrumentation exists.

The live adapter must include:

- timestamp and timezone conversion;

- unit conversion;

- missing-value detection;

- stale-data timeout;

- fallback to persistence or last verified forecast;

- forecast horizon and uncertainty;

- data-quality flags;

- immutable raw-data logging;

- policy action log;

- simulator-versus-live provenance label.

Do not mix live data into the training set unless it is archived, quality-controlled and assigned to a declared train/validation/test role. Do not call a live dashboard “real-time validation” before sensors measure the physical system.

## 9.3 Demonstration workflow

1. Select a state and regime contract.

2. Fetch current irradiance, ambient temperature and forecast data.

3. Map the location to the nearest supported climate regime.

4. Initialise the simulator from a declared PCM/tank state.

5. Run `step(action)` at the demonstration interval.

6. Show predicted water/PCM temperature, melt fraction, demand, action and shield status.

7. Log every input and output.

8. If data becomes stale, switch visibly to fallback mode.

9. Repeat the same demonstration using a historical replay so results remain reproducible.

---

# 10. Changes needed in Objective 1 and Objective 2 before or alongside O3
These are improvement actions, not permission for O3 to redesign the frozen hardware during training.

## Objective 1 changes

1. Freeze one definition of delivery temperature, PCM target temperature, approach temperature, charging cap and PCM safety limit. The supplied O1 and O2 documents currently use different target and delivery conventions. [&#91;5&#93;](#ref-unknown0000d), [&#91;9&#93;](#ref-unknown0000f)

2. Reconcile state-specific GMM covariance and cluster counts before training a pooled O3 policy. Do not call all four pipelines identical when their uploaded records report different choices. [&#91;10&#93;](#ref-unknown0000g), [&#91;12&#93;](#ref-unknown0000i), [&#91;13&#93;](#ref-unknown0000j)

3. Complete external climate validation where claimed; distinguish internal clustering metrics from Köppen/NBC validation. The master overview explicitly notes missing external validation in the current pipeline copy. [&#91;13&#93;](#ref-unknown0000j)

4. Replace “physics-validated ranking” with the weaker current claim where rank–physics agreement is weak. [&#91;13&#93;](#ref-unknown0000j)

5. Complete the PCM property provenance and uncertainty record. Missing properties must remain flagged rather than becoming false precision. [&#91;2&#93;](#ref-unknown0000a)

6. Keep Top-2/Top-3 candidates and their uncertainty, but pass only the selected O2 design into each O3 contract.

7. Preserve the two-tier climate signature and daily-integral provenance; three sun-event samples alone cannot represent daily storage autonomy. [&#91;9&#93;](#ref-unknown0000f)

## Objective 2 changes

1. Produce one signed contract per state and regime and remove obsolete contracts from the training path.

2. Verify that `step(action)` reproduces `run_year()` under fixed baseline control.

3. Clarify whether the reported selected design is the final current design or only the best within a binding 30% loading cap. The current documentation itself warns that the cap is binding. [&#91;1&#93;](#ref-unknown0000)

4. Correct or qualify flow optimisation while collector efficiency is flow-independent; otherwise the controller may learn a model artefact. [&#91;1&#93;](#ref-unknown0000), [&#91;5&#93;](#ref-unknown0000d)

5. Separate nominal, Monte Carlo, held-out, CFD and experimental claims. The supplied CFD results are model checks, not hardware validation. [&#91;4&#93;](#ref-unknown0000c)

6. Complete the Objective 2 state-by-state comparison before claiming one universal design method. Rajasthan reports a mature multi-stage pipeline, while Tamil Nadu’s current contract and design values differ. [&#91;5&#93;](#ref-unknown0000d), [&#91;11&#93;](#ref-unknown0000h)

7. Add uncertainty for pump efficiency, heat-transfer coefficient, manufacturing tolerance and sensor error before using O3 robustness claims; the current O3 notes identify some of these as not yet sampled. [&#91;3&#93;](#ref-unknown0000b)

---

# 11. Complete implementation workflow and deliverables

## Phase O3-0 — Freeze and audit
**Inputs:** four current Objective 2 contracts, result registries, simulator/configuration hashes.
**Outputs:** `contract_audit.json`, `state_regime_manifest.csv`, archived superseded contracts.
**Exit gate:** no contract mismatch.

## Phase O3-1 — Environment refactor
**Work:** implement stateful reset/step API, preserve annual simulator, add deterministic replay.
**Outputs:** `environment.py`, `step_trace.csv`, unit-test report.
**Exit gate:** fixed-action stepper equals annual batch within tolerance.

## Phase O3-2 — Baselines and safety
**Work:** implement B0–B3, action validator and shield.
**Outputs:** baseline metrics, shield-event report, safety-test report.
**Exit gate:** rule-based controller passes the contract acceptance checklist.

## Phase O3-3 — Dataset and demonstrations
**Work:** generate safe trajectories across four states, regimes, seasons and reset conditions.
**Outputs:** transition dataset, data dictionary, split manifest, demonstration policy.
**Exit gate:** no time, location or test-year leakage.

## Phase O3-4 — Algorithm benchmark
**Work:** train Double/Duelling/Prioritised DQN, PPO, recurrent PPO, SAC and constrained SAC; optionally MPC-guided PPO.
**Outputs:** seed-wise learning curves, validation table, safety-cost table.
**Exit gate:** at least one policy beats the baseline without safety degradation.

## Phase O3-5 — Robustness and ablation
**Work:** weather perturbation, property uncertainty, demand shifts, sensor noise, missing data, forecast error, initial PCM-state variation.
**Outputs:** robustness distributions and ablation report.
**Exit gate:** policy remains safe and its benefit is not confined to one regime.

## Phase O3-6 — Locked evaluation
**Work:** freeze the selected policy and run the untouched test set once.
**Outputs:** `obj3_final_test_results.csv`, policy hash, environment hash, reproducibility manifest.
**Exit gate:** all primary success criteria reported with uncertainty.

## Phase O3-7 — Live software demonstration
**Work:** connect archived/live weather adapter to simulator-state estimator; show action and predicted thermal state.
**Outputs:** live-demo log, data-quality report, replay-equivalent demonstration.
**Exit gate:** live mode is clearly labelled simulation-only until hardware exists.

## Phase O3-8 — Objective 4 handoff
**Work:** define sensor list, sampling interval, actuator interface, fail-safe behaviour and embedded policy export.
**Outputs:** hardware interface specification and deployment test plan.
**Exit gate:** no embedded deployment until hardware-in-the-loop tests pass.

---

# 12. Suggested final contribution statement

> Objective 3 will develop a contract-constrained, safety-shielded adaptive controller for the hardware selected by Objective 2. The study will compare rule-based control, model-predictive control, discrete deep reinforcement learning, recurrent PPO, constrained SAC and an MPC-guided policy across Rajasthan, Assam, Tamil Nadu and Uttarakhand. Training will use simulator-generated transitions and archived weather; validation will use unseen years and member points; and a locked test set will evaluate safety, delivery reliability, useful energy, shield activation and action stability. Live weather will be used first in a reproducible software-in-the-loop demonstration, not as a substitute for hardware validation. Objective 3 will not alter the PCM, capsule design, tank size, safety limits or released physics model.

---

# 13. Core algorithm references to add

1. K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” *IEEE Transactions on Evolutionary Computation*, 2002, DOI: 10.1109/4235.996017. [&#91;6&#93;](#ref-deb2002)

2. K. S. Bhattacharjee, H. K. Singh, T. Ray and J. Branke, “Multiple surrogate assisted multiobjective optimization using improved pre-selection,” *2016 IEEE Congress on Evolutionary Computation*, DOI: 10.1109/CEC.2016.7744340. [&#91;7&#93;](#ref-bhattacharjee2016)

3. N. Sampath Kumar and S. Srinivas, “Design Optimization of PMaSynRM Using Kriging Surrogate Modeling and NSGA-II Integrated with Taguchi Method,” *TENCON 2024*, DOI: 10.1109/TENCON61640.2024.10902704. [&#91;14&#93;](#ref-kumar2024)

4. E. Bradford, A. M. Schweidtmann and A. A. Lapkin, “Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm,” *Journal of Global Optimization*, 2018, DOI: 10.1007/s10898-018-0609-2. [&#91;15&#93;](#ref-bradford2018)

5. P. Gadhave, C. L. Prabhune and F. Pathan, “Selection of Phase Change Material for Domestic Water Heating Using Multi Criteria Decision Approach,” *Australian Journal of Mechanical Engineering*, 2020, DOI: 10.1080/14484846.2020.1842297. [&#91;16&#93;](#ref-gadhave2020)

6. The project’s supplied references also include direct solar-thermal DRL, ANN-MPC and AI/PCM studies that should be checked for exact applicability before being used as evidence for performance claims. [&#91;8&#93;](#ref-unknown0000e)

**Evidence limit:** This is a complete implementation plan based on the supplied Objective 1, Objective 2 and Objective 3 documents and indexed academic abstracts. It specifies the work to perform; it does not claim that the O3 environment, trained policies, live adapter or hardware tests already exist.

## References

<a id="ref-unknown0000"></a>

&#91;1&#93; “OBJECTIVES\_AND\_ALGORITHMS.” Public source link unavailable.

<a id="ref-unknown0000a"></a>

&#91;2&#93; “Objective2\_PCM\_design\_optimization\_workflow.” Public source link unavailable.

<a id="ref-unknown0000b"></a>

&#91;3&#93; “OBJECTIVE3\_INPUTS\_AND\_NEXT\_STEPS.” Public source link unavailable.

<a id="ref-unknown0000c"></a>

&#91;4&#93; “obj2\_RESULTS.” Public source link unavailable.

<a id="ref-unknown0000d"></a>

&#91;5&#93; “O2\_rajasthan\_pipeline\_report.” Public source link unavailable.

<a id="ref-deb2002"></a>

&#91;6&#93; K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” *IEEE Transactions on Evolutionary Computation*, Apr. 2002, doi: [10.1109/4235.996017](https://doi.org/10.1109/4235.996017).

<a id="ref-bhattacharjee2016"></a>

&#91;7&#93; K. S. Bhattacharjee, H. K. Singh, T. Ray, and J. Branke, “Multiple surrogate assisted multiobjective optimization using improved pre-selection,” in *2016 IEEE congress on evolutionary computation*, July 2016. doi: [10.1109/cec.2016.7744340](https://doi.org/10.1109/cec.2016.7744340).

<a id="ref-unknown0000e"></a>

&#91;8&#93; “References.” Public source link unavailable.

<a id="ref-unknown0000f"></a>

&#91;9&#93; “Objective1\_PCM\_climate\_framework\_plan\_v3.” Public source link unavailable.

<a id="ref-unknown0000g"></a>

&#91;10&#93; “ASSAM\_FULL\_PIPELINE.” Public source link unavailable.

<a id="ref-unknown0000h"></a>

&#91;11&#93; “O2\_TamilNadu\_pipeline\_report.” Public source link unavailable.

<a id="ref-unknown0000i"></a>

&#91;12&#93; “UTTARAKHAND\_FULL\_PIPELINE.” Public source link unavailable.

<a id="ref-unknown0000j"></a>

&#91;13&#93; “00\_MASTER\_OVERVIEW.” Public source link unavailable.

<a id="ref-kumar2024"></a>

&#91;14&#93; N. S. Kumar and S. Srinivas, “Design optimization of PMaSynRM using kriging surrogate modeling and NSGA-II integrated with taguchi method,” in *TENCON 2024 - 2024 IEEE region 10 conference*, Dec. 2024. doi: [10.1109/tencon61640.2024.10902704](https://doi.org/10.1109/tencon61640.2024.10902704).

<a id="ref-bradford2018"></a>

&#91;15&#93; E. Bradford, A. M. Schweidtmann, and A. A. Lapkin, “Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm,” *Journal of Global Optimization*, Feb. 2018, doi: [10.1007/s10898-018-0609-2](https://doi.org/10.1007/s10898-018-0609-2).

<a id="ref-gadhave2020"></a>

&#91;16&#93; P. Gadhave, C. L. Prabhune, and F. Pathan, “Selection of phase change material for domestic water heating using multi criteria decision approach,” *Australian Journal of Mechanical Engineering*, Nov. 2020, doi: [10.1080/14484846.2020.1842297](https://doi.org/10.1080/14484846.2020.1842297).

<!-- answerthis-citation-style: ieee -->
