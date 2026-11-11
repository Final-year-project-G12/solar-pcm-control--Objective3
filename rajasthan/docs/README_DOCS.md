# Documentation — Objective 3 (Rajasthan)

| file | what |
|---|---|
| `OBJECTIVE_3_README.md` | overview, regimes, phases, status, directory map |
| `METHODOLOGY_AND_FINDINGS.md` | the question, method, what is measured and what is not, threats to validity |
| `ALGORITHMS.md` | every baseline and agent, config, reference numbers |
| `HOW_TO_RUN.md` | commands, `--n-jobs` / `--n-envs`, measured speed-ups, compute estimates |
| `CORRECTIONS.md` | every deviation from Tamil Nadu / the plan, with reasons; open items |
| `objective3_plans/` | the plan (FINAL.docx) |

## Status as of 2026-11-11
* Objective 1 / 2 (Rajasthan): complete; O3 reads O2 through the v3 contract
  (`OBJECTIVE2/.../results/obj3_environment_contract_rajasthan_v3.json`, built by the new
  `src/handoff/build_obj3_contract_rajasthan_v3.py`; no O2 value changed).
* Objective 3: implemented (phases 0–8), 117 tests. Run: baselines 2016–2023 (all regimes), B1 tuning, headroom gate G2
  (**failed: no on/off policy beats B0**), PPO 3 seeds × 300k (regime 1, collapses to B0), robustness, fresh 2006–2010 test
  weather, Streamlit app. **Not run:** the locked test (lock unspent), plan-Q2 weight grid, other algorithms / regimes at full budget.

## The O1 / O2 / O3 boundary
O1 picks the PCM shortlist; O2 picks the released design and owns the simulator, shield and limits; O3 only chooses, hour by
hour, whether the collector loop circulates. O3 never edits O2.
