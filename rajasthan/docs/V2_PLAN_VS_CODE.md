# Plan v2 versus the Rajasthan code - mismatches found (2026-11-13)

Source: `.claude/Objective_3_Implementation_Plan_FINAL_v2.docx`. That document is the **Tamil Nadu as-built record** with
the new section 12.2; the Rajasthan tree differs from it in the places below. "Fixed" = code or docs changed in this pass;
"Flagged" = a difference that is real but left alone, with the reason.

| # | Plan v2 says | Rajasthan code / files | Status |
|---|---|---|---|
| 1 | Section 12.2: `forecast_mode` perfect/smart/none, B2 on the same forecast, `--forecast-mode`, five tests | Not implemented before this pass | **Fixed** - implemented (see CORRECTIONS.md) |
| 2 | Weather loader must carry `GHI_clearsky_Wm2` (12.2.2) | Column was in the CSVs but dropped by `weather_roles._read` | **Fixed** |
| 3 | Live mode has no clear-sky column; use an analytic envelope (12.2.2) | Live adapter fell back to plain persistence | **Fixed** (`smart_persistence_forecast`) |
| 4 | Delivery rule: annual >= 95 % of B0 only (Tamil Nadu rule) | Rajasthan also gated on window delivery | **Fixed** - see CORRECTIONS.md |
| 5 | Section 4.3: V2 = 15 values | Rajasthan V2 = 13 (contract fields), V3 = 17 | Flagged - same name, different vector; renaming would invalidate saved models |
| 6 | Section 7: shield weight 10 | `W_SHIELD_BY_REGIME = {0: 3.9, 1: 1.7, 2: 1.7}` | Flagged - re-derived for Rajasthan, documented in `pcm_env.py` |
| 7 | Section 3: 100 L tank, 2.5 m2, Tamil Nadu PCMs | 50 L, 1.5 m2, savE OM55 / PureTemp 60 / PureTemp 58 | Flagged - Tamil Nadu document; the Rajasthan values are in the contract |
| 8 | Section 8.2: locked test 2011-2015; train includes 2024-2025 | Locked test 2006-2010; 2011-2015 and 2024-2025 are refused (`used_o2`) | Flagged - stricter in Rajasthan, deliberately |
| 9 | Section 6.2: six algorithms trained | PPO, DQN, SAC trained; QR-DQN, RecurrentPPO, MaskablePPO only smoke-tested | Flagged - scope decided: DQN and SAC only |
| 10 | Section 10: Generation-2 "designed; to be re-run" | Generation-2 was run for Rajasthan (DQN all regimes, SAC regime 1 sweep) | Flagged - the Rajasthan results are in CORRECTIONS.md |
| 11 | Section 11: `make_live_dashboard.py`, `streamlit_portal.py` | `app/streamlit_app.py` is now the Tamil-Nadu-style portal (2026-11-15; shared UI in `app/portal_ui.py`) | Fixed - same design, own adapter |
| 12 | Section 11: hand-off exports an 11-signal sensor list | `phase8_o4_handoff.py` exports 13 (V2) or 17 (V3, via `--run`) signals | Flagged - Rajasthan is the newer version |
| 13 | Section 8.4: checkpoint-resume exists for DQN and SAC | Rajasthan trainers had none | **Fixed** - added to `o3/agents/gen2.py` |
| 14 | Section 16 cites `docs/LITERATURE_REVIEW.md`, `docs/references.bib` | Neither exists in `rajasthan/docs/` | Flagged - belongs to the paper, not the controller code |
| 15 | `requirements.txt` lists sb3-contrib | Not installed in the venv; 3 smoke tests failed | **Fixed** - installed 2.6.0 (matches SB3 2.6.0) |
