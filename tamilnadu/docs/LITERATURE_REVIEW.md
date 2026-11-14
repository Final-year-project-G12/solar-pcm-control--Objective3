# Literature Review — Climate-Adaptive DRL Control of PCM Solar Water Heating

Paper-ready related-work survey for Objective 3. It groups the project
bibliography (`docs/references.bib`) into themes, summarises each cluster, and
states the **research gap** it leaves and **how this work addresses it**. Cite
keys match the `.bib` file, so a paragraph here drops straight into a manuscript
(`\cite{key}`). Numbers describing *our* results are from the as-built
evaluation (see `METHODOLOGY_AND_FINDINGS.md`); claims about prior work are kept
at the scope their titles/abstracts support, without inventing specific figures.

> **One-line positioning.** Prior AI-for-PCM work is overwhelmingly about
> *predicting* behaviour or *optimising the design*; this work instead learns a
> *safe real-time operating policy* for a frozen PCM-SWH design, and shows that
> value-based / off-policy deep RL (DQN, SAC) — not the on-policy policy-gradient
> family — is the method that learns anticipatory charge/idle/bypass control.

---

## 1. PCM for solar thermal energy storage (the application context)

Phase-change materials are now an established route to latent thermal storage
for solar systems, and recent reviews map the materials, encapsulation, and
system integration landscape [Hamza2025PCMReview; Rathore2024ThermalPCM;
Abdellatif2025PCMModeling; Martinez2025PCMIndustrial], with dedicated treatments
of PCM in solar **water heating** [Singh2025PCMSWH] and building-integrated solar
heating with heat pipes and PCM [Kou2025BIHPPCM]. The consensus is that PCM can
raise solar fraction and buffer the diurnal mismatch between collection and
demand, but that real deployments are constrained by overheating, degradation,
and the gap between idealised and in-situ performance.

**Gap.** These works characterise *materials and designs*; they do not address
how a *controller* should operate a fixed PCM store hour-by-hour to exploit the
storage while respecting temperature limits.

**This work.** We take the PCM design as frozen (the output of Objectives 1–2 —
PureTemp 58 / RT57HC in a 100 L tank, three Tamil Nadu climate regimes) and study
the *operating* problem on top of it.

## 2. Design optimisation of PCM-SWH systems (Taguchi / GRA / MCDM / ML)

A large body of work optimises the *static design* of PCM-SWH systems — collector
and nanofluid parameters via Taguchi + Grey Relational Analysis
[Chen2025TaguchiGRAPCM], ML-driven PCM-SWH sizing and dynamic simulation
[Barqawi2025MLPCMSWH], and ML-based multi-objective optimisation of PCM solar
collectors [Assareh2023MLPCMSolarCollector].

**Gap.** Design optimisation fixes *what to build*; it is a one-time, offline
decision. It says nothing about the *sequential* decision of when to charge or
bypass under a weather sequence the designer never saw.

**This work.** This is exactly the boundary we respect: our Objectives 1–2 used
funnel/consensus selection to fix the design, and Objective 3 adds the online
control layer — a complementary, not competing, contribution.

## 3. AI / ML in thermal and PCM systems — prediction vs. optimisation

Reviews of AI in thermal-energy systems and PCM storage frame the field as moving
"from prediction to optimization" [Liu2025AIPCMTES; Nems2025AIPCMReview;
Mohammed2025AIthermal]. Concretely, ML is used as a **surrogate/predictor**:
ANN models of water-PCM domestic hot-water stores [Eldokaishi2022ANNPCMSWHModel],
prediction of PCM **melting response time** in triplex-tube stores
[Yan2025MLMeltingTime], and ML performance prediction for PCM solar-chimney drying
[Rajamurugu2025PCMML]; evolutionary/neural regression also appears in PCM solar
drying [Barghi2026SolarDryingPCM].

**Gap.** These models answer "what will happen / what is optimal" but are not
closed-loop controllers. A predictor of melt time or outlet temperature still
needs a *policy* to decide the actuator setting each step, and most leave that
policy as a fixed rule or an offline optimiser.

**This work.** We use learning for **control**, not prediction: the network maps
the observed thermal/weather state directly to an operating action, trained to
maximise delivered energy while minimising guard-band (shield) activations.

## 4. Deep reinforcement learning for solar / thermal control (closest prior art)

DRL has begun to appear for solar-thermal operation — e.g. smart control of a
solar-driven power cycle with thermal storage [Emami2026DRLSolar] — and the broader
DRL-control literature shows that algorithm choice materially changes behaviour,
with systematic comparisons of DRL controllers in other domains
[Sivaraj2023DRLControl].

**Gap.** DRL for solar is sparse and concentrated on utility-scale power cycles;
we found no DRL controller for **domestic PCM solar water heating** that (i) keeps
a hard safety shield *inside* the environment so the policy can never create an
unsafe state, and (ii) compares value-based, on-policy, and off-policy families
to justify the algorithm rather than assuming one.

**This work.** We fill both. A shielded Gymnasium environment wraps the frozen
simulator; six algorithms (PPO, DQN, RecurrentPPO, QRDQN, MaskablePPO, SAC) are
compared on identical weather. The finding — *value-based DQN and off-policy SAC
escape the "always-charge" attractor via sustained exploration, while the
on-policy policy-gradient family collapses to it* — is the kind of
algorithm-justification [Sivaraj2023DRLControl] argues is necessary, grounded in
observed behaviour.

## 5. AI in solar water heating, broadly

A recent review catalogues AI across SWH research [OdoiYorke2025AISWH], against the
state-of-the-art of SWH systems themselves [AlMamun2023SWHReview].

**Gap.** The review-level view confirms AI-for-SWH is dominated by forecasting,
fault detection, and design, with real-time *supervisory control* under
explicit safety constraints under-represented.

**This work.** Our contribution sits in that under-represented cell:
safety-constrained supervisory control of a PCM-SWH store.

## 6. Forecast-aware control and irradiance forecasting

Anticipatory control depends on knowing what the sun will do; recent surveys cover
irradiance forecasting across horizons and methods [Ghodusinejad2026SolarForecast]
and multimodal time-series forecasting for renewable systems
[Mansouri2025Multimodal].

**Gap.** Forecasting is studied largely as a standalone prediction task, not as a
signal consumed by a safety-aware operating policy.

**This work.** Our **B2 forecast-aware baseline** uses a 4-hour irradiance
look-ahead to pre-emptively bypass before the PCM over-charges, and we include an
imperfect-forecast sensitivity mode — directly connecting the forecasting
literature to the control decision. (The learned DRL policy achieves comparable
anticipation *without* an explicit forecast, from the observed state alone.)

## 7. Model-based control alternatives (ANN-MPC)

The nearest control-theoretic alternative is ANN-based model-predictive control,
e.g. for a shallow solar pond with experimental validation
[Terfai2025SolarPondANNMPC].

**Gap.** ANN-MPC needs an accurate forward model and solves an optimisation each
step; its safety depends on the model and the solver. It is powerful but
model- and compute-heavy for an embedded domestic controller.

**This work.** We take the model-free DRL route with a *structural* safety shield:
safety is guaranteed by construction (the shield overrides any unsafe action
inside the environment), and inference is a single network forward pass — a
contrast worth drawing explicitly in the paper against [Terfai2025SolarPondANNMPC].

## 8. Techno-economic and environmental framing

Techno-economic and environmental assessments of solar thermal systems
[Chopra2023MonteCarloETC; Duraivel2025DSTS] motivate *why* squeezing more useful
delivery from the same hardware matters.

**Gap / this work.** These justify the objective (more delivery, fewer protective
trips, same hardware) and are the natural home for a future cost/CO₂ analysis of
the controller's shield-reduction benefit — flagged as future work.

## 9. Data and material resources

Weather and material inputs draw on public resources: solar/irradiance data for
India [ISROSolarCalculator; GlobalSolarAtlasIndia; NITIAayogEnergyDashboard],
renewable time-series tools [RenewablesNinja], and PCM vendor datasheets
[RubithermPCM; PlussPCM]. Our training/validation/test weather is historical
NASA POWER reanalysis for Tamil Nadu (year-locked split), and the live demo
streams present-day weather from the Open-Meteo API.

---

## 10. Consolidated research gap → contribution

| # | Gap in the literature | Representative refs | How Objective 3 addresses it |
|---|---|---|---|
| G1 | PCM reviews cover materials/design, not *operation* | [Hamza2025PCMReview; Singh2025PCMSWH; Rathore2024ThermalPCM] | Online operating policy on a **frozen** design |
| G2 | Design optimisation is offline & one-shot | [Chen2025TaguchiGRAPCM; Assareh2023MLPCMSolarCollector; Barqawi2025MLPCMSWH] | Sequential hour-by-hour control under unseen weather |
| G3 | ML in PCM = prediction, not closed-loop control | [Liu2025AIPCMTES; Eldokaishi2022ANNPCMSWHModel; Yan2025MLMeltingTime] | State→action **control** policy, not a surrogate predictor |
| G4 | DRL for solar is sparse, no PCM-SWH, no in-env safety, no family comparison | [Emami2026DRLSolar; Sivaraj2023DRLControl] | Shielded env + **6-algorithm** comparison; value/off-policy wins |
| G5 | Safety-constrained supervisory control under-represented in AI-SWH | [OdoiYorke2025AISWH; AlMamun2023SWHReview] | Structural safety shield the policy can never bypass |
| G6 | Forecasting studied standalone, not as a control signal | [Ghodusinejad2026SolarForecast; Mansouri2025Multimodal] | Forecast-aware B2 baseline + forecast-free anticipatory DRL |
| G7 | Control alternative (ANN-MPC) is model/compute-heavy | [Terfai2025SolarPondANNMPC] | Model-free DRL, single forward pass, safety by construction |

## 11. Suggested paper framing (abstract skeleton)

> Latent-heat storage with PCM can shift solar hot-water delivery into the demand
> window, but existing AI work optimises the *design* or *predicts* thermal
> behaviour rather than *operating* the store safely in real time
> [Liu2025AIPCMTES; Nems2025AIPCMReview]. We present a climate-adaptive deep
> reinforcement learning controller that chooses charge / idle / bypass each hour
> for a fixed PCM solar-water-heating design across three Tamil Nadu climate
> regimes, with a safety shield enforced inside the environment so the learned
> policy can never breach the temperature limits. Comparing six algorithms on
> identical historical weather, we find that value-based (DQN) and off-policy
> entropy-regularised (SAC) agents learn *anticipatory* control — matching the
> always-on baseline's delivery while cutting guard-band activations by 26–54 %
> with zero safety violations on a held-out test period — whereas on-policy
> policy-gradient methods collapse to the trivial always-charge policy. A
> software-in-the-loop demonstration streams present-day weather from a public API
> to visualise the controller's decisions.

See `METHODOLOGY_AND_FINDINGS.md` for the result tables that back the quantitative
claims above, and `ALGORITHMS.md` for the per-algorithm detail.
