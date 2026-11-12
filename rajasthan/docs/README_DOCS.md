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
* Objective 3: implemented (phases 0–9, plan §19 Generation-2), tests passing. Run: baselines (all regimes), B1 tuning, headroom
  gate G2 (failed on delivery for rules), PPO 3 seeds (collapses to B0), Gen-2 DQN study (regime 1: a policy at 98.2 % of B0 delivery
  with 57 % fewer shield hours; regimes 0/2 not), deployment metrics, fresh 2006–2010 test weather, Streamlit app. **Locked test spent** on the selected DQN (98.0 % of B0, shield -61 %). Ablation (3 seeds): forecast essential; training randomisation not shown to matter. **Not complete:** SAC weight sweep / regimes 0 and 2 for SAC (running), multi-seed ablations.

## The O1 / O2 / O3 boundary
O1 picks the PCM shortlist; O2 picks the released design and owns the simulator, shield and limits; O3 only chooses, hour by
hour, whether the collector loop circulates. O3 never edits O2.
