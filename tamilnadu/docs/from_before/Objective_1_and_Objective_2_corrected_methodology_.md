# Objective 1 and Objective 2: corrected methodology, results, and priority audit

## Critical corrections — put these first in the project write-up

1. **Do not claim that Objective 1’s PCM ranking is validated by physics.** The current Tamil Nadu summary reports mean Spearman agreement of only +0.169 across regimes, with cluster-wise values +0.381, +0.103 and +0.024; it also states that none is statistically significant at the available candidate counts. This supports the conclusion that the rankings were *tested against* a physics model, but not that the ranking reliably predicts physical performance. State the weak/uncertain validation plainly. [&#91;1&#93;](#ref-unknown0000)

2. **Reconcile the melting-temperature and safety specifications before calling O1/O2 consistent.** The O1 overview reports a current target of 67 °C, while the earlier O1 framework derives targets from a 50 °C delivery temperature plus an approach temperature, and the O2 reports use a 45 °C delivery metric with a 65 °C maximum PCM temperature. Those are not interchangeable assumptions. Freeze one configuration and document how target melting temperature, charge feasibility, delivery setpoint, and maximum allowable PCM temperature relate; do not claim the 67 °C target is operationally validated under a 65 °C PCM ceiling. [&#91;1&#93;](#ref-unknown0000), [&#91;2&#93;](#ref-unknown0000a), [&#91;3&#93;](#ref-unknown0000b)

3. **Treat the project documents as version-conflicted, not as one already reconciled four-state pipeline.** The uploaded plans and state pipelines report different covariance structures, numbers of clusters, filters and weighting methods. For example, Tamil Nadu’s current summary reports diagonal-covariance GMM and k=3; the Assam pipeline describes full-covariance GMM; Uttarakhand reports diagonal GMM and k=5. The cross-state method cannot be presented as identical until the state-specific code/config/results are reconciled. [&#91;4&#93;](#ref-unknown0000c), [&#91;5&#93;](#ref-unknown0000d), [&#91;6&#93;](#ref-unknown0000e)

4. **Use the implemented algorithm record—not the older Objective 2 workflow as evidence of what ran.** The September workflow document is explicitly labelled an implementation plan, not evidence that each listed stage ran. The newer algorithm note describes DOE, surrogate comparison, NSGA-II, active learning and held-out testing as implemented, while the October 30 results file says the held-out run had not yet been performed. The Tamil Nadu report also says stacked models won all nine pairs, whereas the newer algorithm summary says XGBoost won 21 of 27 groups. These cannot all describe the same run/version. Do not publish “held-out validated,” the winning surrogate, or final values until one dated registry and code commit settle the conflicts. [&#91;4&#93;](#ref-unknown0000c), [&#91;7&#93;](#ref-unknown0000f), [&#91;8&#93;](#ref-unknown0000g), [&#91;9&#93;](#ref-unknown0000h)

5. **Objective 2 has not established an unconstrained global optimum.** The current algorithm note says the 30% PCM-fraction cap is binding and that a 40% test gave larger gains; it also flags the apparent low-flow optimum as an artefact because collector efficiency is flow-independent in the model. Report the selected design as “best within the declared search box and selection rule,” not “the system optimum,” and identify loading and flow-model limitations. [&#91;4&#93;](#ref-unknown0000c)

6. **Do not call the CFD check independent experimental validation.** The current summary describes the CFD as a model check, reports zero melt during the short current simulation windows, and explicitly notes geometry and buoyancy limitations. It can assess selected modelling assumptions, but it does not experimentally validate the annual tank-level performance. [&#91;9&#93;](#ref-unknown0000h)

7. **Do not state that every requested correction has been made in code.** The supplied materials are plans, status reports and editable reports, not the runnable source repository, configuration files, data registry, or scripts. This note corrects the *written synthesis* and identifies the exact conflicts; it cannot safely change or rerun the underlying implementation. The final verification of code/results remains incomplete until the canonical repository and versioned outputs are checked.

## 1. What is correct in Objective 1
The project’s basic separation is sound: O1 constructs climate regimes, screens and ranks PCMs; O2 optimises the physical capsule/tank design for the inherited PCM shortlist; later control work must not silently reselect the material or redesign the hardware. The earlier O1 plan also correctly distinguishes sun-event features from daily integrals and identifies that three daily observations alone cannot support daily-energy or true diurnal-range indices. [&#91;2&#93;](#ref-unknown0000a), [&#91;4&#93;](#ref-unknown0000c)

The master overview describes the two-tier climate signature and GMM clustering. [&#91;1&#93;](#ref-unknown0000) It reports feasibility filtering, four-method MCDM and grey-box physics validation as later stages. [&#91;1&#93;](#ref-unknown0000) The Tamil Nadu full-pipeline stage map also lists GMM clustering, feasibility filtering and MCDM ranking. [&#91;10&#93;](#ref-unknown0000i) The master overview also reports repairs to accumulated-variable handling, a 1000× flow-rate error, GMM covariance choice, tank losses, and execution-order/path issues. Those are meaningful corrections, but the paper should cite the corrected run/version and not repeat values from superseded runs. [&#91;1&#93;](#ref-unknown0000) The Tamil Nadu pipeline maps its MCDM methods to named methodological references; check this mapping against the citations used in the final paper. [&#91;10&#93;](#ref-unknown0000i)

GMM is a reasonable model to test when climate signatures overlap and probabilistic assignment is useful; weather-regime studies describe probabilistic mixtures as one option for representing regime uncertainty. [&#91;11&#93;](#ref-baldo2022) Comparative work also finds that the selected clustering method can change the recovered regimes. [&#91;12&#93;](#ref-babanov2023) BIC selects a mixture’s density approximation, which is not always the same thing as a substantively meaningful number of climate regimes. [&#91;13&#93;](#ref-baudry2010) Keep the current k-selection and stability diagnostics, report the competing criteria and membership uncertainty, and add external validation before making strong climate-zone claims.

The four-method O1 stack—TOPSIS, GRA, PROMETHEE II and VIKOR—with Borda aggregation and Kendall’s W is a reasonable robustness design if each method is applied to the same eligible alternatives and a correctly defined decision matrix. [&#91;4&#93;](#ref-unknown0000c) The direct PCM domestic-water-heating literature also uses multi-criteria ranking methods and mixed criteria weights. [&#91;14&#93;](#ref-gadhave2020) Method agreement is *internal robustness*, not independent proof that the recommended PCM performs best; the project’s own weak physical correlations make that distinction essential. [&#91;1&#93;](#ref-unknown0000)

### O1 corrections to apply in the master thesis/report

- Replace “physics-validated ranking” with: **“The MCDM ranking was compared with the project’s lumped-enthalpy physics model; agreement was weak and non-significant in the reported run, so predictive validity was not established.”** [&#91;1&#93;](#ref-unknown0000)

- Describe the input selection precisely as **population-weighted point selection/coverage** if the implementation uses population to select or summarise points. Do not imply that the GMM likelihood itself was population-weighted unless the code actually applies weights; the project notes explicitly caution against double-weighting. [&#91;2&#93;](#ref-unknown0000a)

- State the actual per-state cluster count and covariance structure from the matching run. The current master summary reports diagonal GMM and k=3; the Assam and Uttarakhand files report different covariance/k choices, so do not generalise one state’s setting to all states. [&#91;1&#93;](#ref-unknown0000), [&#91;5&#93;](#ref-unknown0000d), [&#91;6&#93;](#ref-unknown0000e)

- Name the implemented PCM-weight method precisely. The Tamil Nadu master describes entropy-based weights with a capped supercooling weight; some state documents describe an entropy/AHP blend, while the older framework calls for guide-elicited AHP pairwise judgements. Unless the actual pairwise matrix and consistency ratio exist for the reported run, write **“entropy weights with the stated prior/cap”** rather than “AHP-elicited weights.” The supplied state audit itself flags that the current weights are priors, not a completed AHP elicitation. [&#91;1&#93;](#ref-unknown0000), [&#91;6&#93;](#ref-unknown0000e), [&#91;15&#93;](#ref-unknown0000j)

- State the actual target, share assumption, PCM safety limit and water-delivery metric together in one versioned table. Remove older, inconsistent targets from templates and handoff documents. [&#91;1&#93;](#ref-unknown0000), [&#91;2&#93;](#ref-unknown0000a), [&#91;3&#93;](#ref-unknown0000b)

- Keep external-validation claims bounded. The Tamil Nadu overview says Köppen–Geiger validation is not wired in; the Rajasthan pipeline reports an external comparison but says NBC/ECBC validation remains unwired. Report precisely what was computed for each state. [&#91;1&#93;](#ref-unknown0000), [&#91;15&#93;](#ref-unknown0000j)

## 2. What is correct in Objective 2
The latest algorithm summary describes a simulator as evaluator, screening and space-filling DOE, a surrogate bake-off, simulator-confirmed active-learning samples, constrained NSGA-II search, coverage checks, and a predeclared selection funnel. [&#91;4&#93;](#ref-unknown0000c) The project reports paired comparisons against a water-only twin at matching flow, weather and demand, which is the right comparison structure for attributing gains to PCM. [&#91;8&#93;](#ref-unknown0000g)

The algorithm choices have credible published precedents. The original NSGA-II paper establishes non-dominated sorting, elitist parent–offspring selection and constrained multi-objective optimisation. [&#91;16&#93;](#ref-deb2002) An IEEE conference study combines Taguchi sampling, Kriging surrogates and NSGA-II in an engineering design problem, supporting the general method combination—not the specific correctness of this project’s parameter bounds. [&#91;17&#93;](#ref-kumar2024) A *Journal of Global Optimization* paper presents Gaussian-process, spectral-sampling and NSGA-II methods for expensive multi-objective black-box problems (DOI prefix 10.1007). [&#91;18&#93;](#ref-bradford2018) These are appropriate methods references to add beside the actual algorithms used.

### O2 corrections to apply in the master thesis/report

- Say **“surrogate-assisted, constrained multi-objective search”** only for the run whose code and results demonstrate it. The Objective 2 workflow is labelled an implementation plan, and the Objective 2 results file marks earlier results as superseded. [&#91;7&#93;](#ref-unknown0000f), [&#91;9&#93;](#ref-unknown0000h)

- Report the surrogate winner and the held-out status only from the same frozen run. The attached TN report and the newer summary disagree on which model won; the October results file and later algorithm note disagree on whether the one-shot held-out test was run. Until that is reconciled, label the affected results **“status/version not verified from supplied records.”** [&#91;4&#93;](#ref-unknown0000c), [&#91;8&#93;](#ref-unknown0000g), [&#91;9&#93;](#ref-unknown0000h)

- Report Pareto fronts and the selected compromise separately. NSGA-II returns a set of trade-off solutions; it does not itself justify one unique “best” design. The tie-break/funnel must be frozen before seeing the results and described explicitly. [&#91;4&#93;](#ref-unknown0000c), [&#91;7&#93;](#ref-unknown0000f)

- Use “selected design within the 2–30% search bound,” not “global optimum,” unless the upper-bound investigation is completed. Retain the flow-rate caveat until the collector model makes efficiency flow-dependent or an independent hydraulic/thermal model validates the flow effect. [&#91;4&#93;](#ref-unknown0000c)

- Distinguish simulator verification, CFD cross-checking, and experimental validation. The attached materials report energy-balance/tests and CFD checks, but no field measurements; model-to-model agreement is not measured validation. [&#91;3&#93;](#ref-unknown0000b), [&#91;9&#93;](#ref-unknown0000h)

- Update the Objective 3 handoff: one supplied handoff document still lists a 50 L tank and 1.5 m² collector, while later O2/Tamil Nadu materials describe 100 L and 2.5 m². Do not use the older contract/configuration as the current O2 design. [&#91;4&#93;](#ref-unknown0000c), [&#91;9&#93;](#ref-unknown0000h), [&#91;19&#93;](#ref-unknown0000k)

## 3. Recommended algorithm-reference additions
Use the following sources as methodological precedents and cite them next to the methods actually used:

- **NSGA-II (foundational, IEEE):** “A fast and elitist multiobjective genetic algorithm: NSGA-II,” *IEEE Transactions on Evolutionary Computation*, DOI 10.1109/4235.996017. Cite for non-dominated sorting, elitism, diversity preservation and constrained MOO. [&#91;16&#93;](#ref-deb2002)

- **Taguchi + Kriging + NSGA-II (IEEE):** “Design Optimization of PMaSynRM Using Kriging Surrogate Modeling and NSGA-II Integrated with Taguchi Method,” *TENCON 2024*, DOI 10.1109/TENCON61640.2024.10902704. Cite as a precedent for the combination, not as evidence that the project’s DOE is sufficient by itself. [&#91;17&#93;](#ref-kumar2024)

- **Surrogate-assisted MOO (IEEE):** “Multiple surrogate assisted multiobjective optimization using improved pre-selection,” *2016 IEEE Congress on Evolutionary Computation*, DOI 10.1109/CEC.2016.7744340. Cite for multi-surrogate model management and simulator-efficient search. [&#91;20&#93;](#ref-bhattacharjee2016)

- **Gaussian-process multi-objective optimisation (Springer journal; DOI prefix 10.1007):** “Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm,” *Journal of Global Optimization*, DOI 10.1007/s10898-018-0609-2. Cite for GP/surrogate methods in expensive black-box MOO; do not describe it as the exact expected-improvement implementation unless your code matches. [&#91;18&#93;](#ref-bradford2018)

- **PCM in domestic water heating (application-specific):** “Selection of Phase Change Material for Domestic Water Heating Using Multi Criteria Decision Approach.” This is directly aligned to O1’s PCM-selection context and complements generic algorithm citations. [&#91;14&#93;](#ref-gadhave2020)

The attached `references.txt` already contains useful PCM, solar-water-heating, Taguchi/GRA, AI/PCM and Monte Carlo sources. Add the above entries to the project’s canonical bibliography only after checking for duplicates and confirming complete bibliographic metadata. [&#91;21&#93;](#ref-unknown0000l)

## 4. Corrected status statement ready to use

> **Objective 1.** The master overview reports GMM climate grouping, four-method MCDM ranking and a separate lumped-enthalpy physics comparison. [&#91;1&#93;](#ref-unknown0000) It records specific data-processing and orchestration fixes. [&#91;1&#93;](#ref-unknown0000) Its physics comparison reports weak, non-significant agreement. [&#91;1&#93;](#ref-unknown0000) The state records do not describe one common GMM configuration: the current master summary reports diagonal covariance and k=3, Assam reports full covariance, and Uttarakhand reports diagonal covariance and k=5. [&#91;1&#93;](#ref-unknown0000), [&#91;5&#93;](#ref-unknown0000d), [&#91;6&#93;](#ref-unknown0000e)
>
> **Objective 2.** The latest algorithm record describes DOE, surrogate comparison, NSGA-II, active learning and simulator re-evaluation. [&#91;4&#93;](#ref-unknown0000c) The Tamil Nadu report says the stacked surrogate won all nine pairs, while the later algorithm summary says XGBoost won 21 of 27 groups. [&#91;4&#93;](#ref-unknown0000c), [&#91;8&#93;](#ref-unknown0000g) The October 30 results file says the held-out test had not been run; the later algorithm record describes a five-year held-out test and reports its gates. [&#91;4&#93;](#ref-unknown0000c), [&#91;9&#93;](#ref-unknown0000h) Reconcile these versions before selecting a final result table. The current algorithm summary reports a 30% PCM-fraction cap and a flow-model caveat, so describe the design as selected within tested bounds, not as a proven global optimum. [&#91;4&#93;](#ref-unknown0000c)

## 5. Work still required before “all corrected and done” is accurate

1. Select one authoritative repository commit and result registry for each state/run; rebuild a single current O1/O2 status table from those records.

2. Freeze one coherent delivery-temperature / `Tm_target` / charging-cap / PCM-safety configuration and regenerate affected O1/O2 outputs if the change alters eligibility or simulation.

3. Reconcile the TN surrogate-winner and held-out-status contradictions; mark all superseded tables and handoff contracts prominently.

4. Verify state-specific GMM covariance/k, MCDM weighting/filters, and external validation against the actual scripts and outputs; do not treat the older framework plan as implementation evidence.

5. Extend the O2 search bound or justify the 30% administrative cap, correct/validate the flow-dependent collector model, and keep selected designs distinct from a proven global optimum.

6. Add the IEEE/Springer references to the final paper’s bibliography and cite them where each corresponding method is described.

**Evidence limit:** Literature citations above were assessed from indexed abstracts, and project-status statements from the supplied reports/notes. The runnable source code and canonical output registry were not supplied, so no code edits, test reruns, new performance calculations, or final run-level reconciliation are claimed here. Several state-specific pipeline documents contain their own version histories; the limitations above are therefore part of the correction, not evidence that an issue is absent from the underlying repository.

## References

<a id="ref-unknown0000"></a>

&#91;1&#93; “00\_MASTER\_OVERVIEW.” Public source link unavailable.

<a id="ref-unknown0000a"></a>

&#91;2&#93; “Objective1\_PCM\_climate\_framework\_plan\_v3.” Public source link unavailable.

<a id="ref-unknown0000b"></a>

&#91;3&#93; “O2\_rajasthan\_pipeline\_report.” Public source link unavailable.

<a id="ref-unknown0000c"></a>

&#91;4&#93; “OBJECTIVES\_AND\_ALGORITHMS.” Public source link unavailable.

<a id="ref-unknown0000d"></a>

&#91;5&#93; “ASSAM\_FULL\_PIPELINE.” Public source link unavailable.

<a id="ref-unknown0000e"></a>

&#91;6&#93; “UTTARAKHAND\_FULL\_PIPELINE.” Public source link unavailable.

<a id="ref-unknown0000f"></a>

&#91;7&#93; “Objective2\_PCM\_design\_optimization\_workflow.” Public source link unavailable.

<a id="ref-unknown0000g"></a>

&#91;8&#93; “O2\_TamilNadu\_pipeline\_report.” Public source link unavailable.

<a id="ref-unknown0000h"></a>

&#91;9&#93; “obj2\_RESULTS.” Public source link unavailable.

<a id="ref-unknown0000i"></a>

&#91;10&#93; “TAMILNADU\_FULL\_PIPELINE.” Public source link unavailable.

<a id="ref-baldo2022"></a>

&#91;11&#93; A. Baldo and R. Locatelli, “A probabilistic view on modelling weather regimes,” *International Journal of Climatology*, Nov. 2022, doi: [10.1002/joc.7942](https://doi.org/10.1002/joc.7942).

<a id="ref-babanov2023"></a>

&#91;12&#93; B. A. Babanov, В. А. Семенов, and И. И. Мохов, “Comparison of cluster analysis methods for identifying weather regimes in the euro-atlantic region for winter and summer seasons,” *Izvestiya Atmospheric and Oceanic Physics*, Dec. 2023, doi: [10.1134/s0001433823060026](https://doi.org/10.1134/s0001433823060026).

<a id="ref-baudry2010"></a>

&#91;13&#93; J.-P. Baudry, A. E. Raftery, G. Celeux, K. Lo, and R. Gottardo, “Combining mixture components for clustering,” *Journal of computational and graphical statistics : a joint publication of American Statistical Association, Institute of Mathematical Statistics, Interface Foundation of North America*, June 2010, doi: [10.1198/jcgs.2010.08111](https://doi.org/10.1198/jcgs.2010.08111).

<a id="ref-gadhave2020"></a>

&#91;14&#93; P. Gadhave, C. L. Prabhune, and F. Pathan, “Selection of phase change material for domestic water heating using multi criteria decision approach,” *Australian Journal of Mechanical Engineering*, Nov. 2020, doi: [10.1080/14484846.2020.1842297](https://doi.org/10.1080/14484846.2020.1842297).

<a id="ref-unknown0000j"></a>

&#91;15&#93; “RAJASTHAN\_FULL\_PIPELINE.” Public source link unavailable.

<a id="ref-deb2002"></a>

&#91;16&#93; K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” *IEEE Transactions on Evolutionary Computation*, Apr. 2002, doi: [10.1109/4235.996017](https://doi.org/10.1109/4235.996017).

<a id="ref-kumar2024"></a>

&#91;17&#93; N. S. Kumar and S. Srinivas, “Design optimization of PMaSynRM using kriging surrogate modeling and NSGA-II integrated with taguchi method,” in *TENCON 2024 - 2024 IEEE region 10 conference*, Dec. 2024. doi: [10.1109/tencon61640.2024.10902704](https://doi.org/10.1109/tencon61640.2024.10902704).

<a id="ref-bradford2018"></a>

&#91;18&#93; E. Bradford, A. M. Schweidtmann, and A. A. Lapkin, “Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm,” *Journal of Global Optimization*, Feb. 2018, doi: [10.1007/s10898-018-0609-2](https://doi.org/10.1007/s10898-018-0609-2).

<a id="ref-unknown0000k"></a>

&#91;19&#93; “OBJECTIVE3\_INPUTS\_AND\_NEXT\_STEPS.” Public source link unavailable.

<a id="ref-bhattacharjee2016"></a>

&#91;20&#93; K. S. Bhattacharjee, H. K. Singh, T. Ray, and J. Branke, “Multiple surrogate assisted multiobjective optimization using improved pre-selection,” in *2016 IEEE congress on evolutionary computation*, July 2016. doi: [10.1109/cec.2016.7744340](https://doi.org/10.1109/cec.2016.7744340).

<a id="ref-unknown0000l"></a>

&#91;21&#93; “References.” Public source link unavailable.

<!-- answerthis-citation-style: ieee -->
