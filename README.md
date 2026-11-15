# Solar PCM Control — Objective 3

Objective 3 develops a climate-adaptive controller for solar water heating
with phase-change-material (PCM) thermal storage. The controller operates the
fixed hardware and physics selected in Objective 2; it does not change the
PCM, tank, collector, capsule geometry, safety limits, or simulator.

The project contains three state-specific implementations:

| Folder | State | Selected agent | Documentation |
|---|---|---|---|
| [`rajasthan/`](rajasthan/) | Rajasthan | DQN (locked test, regime 1) and SAC (validation, regimes 0–2) | [`docs/OBJECTIVE_3_README.md`](rajasthan/docs/OBJECTIVE_3_README.md) |
| [`tamilnadu/`](tamilnadu/) | Tamil Nadu | SAC, trained with a realistic smart-persistence forecast | [`docs/OBJECTIVE_3_README.md`](tamilnadu/docs/OBJECTIVE_3_README.md) |
| [`uttarakhand/`](uttarakhand/) | Uttarakhand | PPO (recommended over DQN and SAC) | [`docs/FINDINGS.md`](uttarakhand/docs/FINDINGS.md) |

Headline results (each against an always-charge baseline, B0, with zero hard-limit violations): Tamil Nadu SAC keeps 97–99% of B0 delivery with 33–54% fewer safety-shield activations; Rajasthan's locked-test DQN keeps 98% with 61% fewer shield hours; Uttarakhand's PPO removes about 99% of shield time but gives up 8–11% of delivery. The states use different shield metrics and pass rules, so verdicts are not directly comparable. Forecasts are smart persistence on the clear-sky index rather than a perfect look-ahead.

Both implementations provide:

- a Gymnasium-compatible environment around the frozen Objective 2 simulator;
- a safety shield that remains active regardless of the learned policy;
- rule-based baselines for comparison;
- PPO, DQN, recurrent, distributional, masked, and SAC training options;
- robustness evaluation, locked test evaluation, plots, and a software-in-the-loop demo.

## Combined dashboard (all states)

Live app: [https://solar-pcm-control--objective3-ohrrdnt9lzp2fgngxmmjdb.streamlit.app/](https://solar-pcm-control--objective3-ohrrdnt9lzp2fgngxmmjdb.streamlit.app/).

One Streamlit app at the repo root shows every state: `streamlit run streamlit_app.py` (needs `pip install -r requirements.txt`).
Rajasthan and Uttarakhand run their own unmodified apps (`hub/state_runner.py` keeps their two `o3` packages apart); Tamil Nadu shows its
saved result dashboards. Deployment to Streamlit Community Cloud, the CI/CD workflow and the known limits are in
[`deploy/README.md`](deploy/README.md).

## Getting started

Choose a state directory and create or activate a Python 3.10+ environment.
Then install that state's dependencies:

```powershell
cd rajasthan
python -m pip install -r requirements.txt
```

Tamil Nadu and Uttarakhand have the same project structure; use each state's documentation and dependency configuration when running it.

Run the Rajasthan verification and test suite with:

```powershell
cd rajasthan
python scripts/phase0_audit.py
python -m pytest -q
```

For complete phase descriptions, commands, data splits, safety details, and
troubleshooting, see the state-specific documentation linked above. Run
commands from the selected state directory. Generated models, plots, and
evaluation outputs are written under that state's `results/` directory.

## Scope and safety

This repository contains simulation and software-in-the-loop work only. It is
not a hardware-control system and its results are not field validation.
Objective 4 covers hardware integration and experimental validation.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).

## Project links

| Part | Repository |
|---|---|
| Objective 1: PCM selection | [https://github.com/Final-year-project-G12/PCM-Selection-ML-model](https://github.com/Final-year-project-G12/PCM-Selection-ML-model) |
| Objective 2: design optimisation | [https://github.com/Final-year-project-G12/Objective---2](https://github.com/Final-year-project-G12/Objective---2) |
| Objective 3: DRL controller | [https://github.com/Final-year-project-G12/solar-pcm-control--Objective3](https://github.com/Final-year-project-G12/solar-pcm-control--Objective3) |
| Objective 4: software-in-the-loop | [https://github.com/Final-year-project-G12/Objective--4_software-in-loop](https://github.com/Final-year-project-G12/Objective--4_software-in-loop) |
| Documentation portal | [https://github.com/Final-year-project-G12/Documentation-Portal](https://github.com/Final-year-project-G12/Documentation-Portal) · [live site](https://Final-year-project-G12.github.io/Documentation-Portal) |
| Objective 3 live dashboard | [Streamlit app](https://solar-pcm-control--objective3-ohrrdnt9lzp2fgngxmmjdb.streamlit.app/) |
