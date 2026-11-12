# Solar PCM Control — Objective 3

Objective 3 develops a climate-adaptive controller for solar water heating
with phase-change-material (PCM) thermal storage. The controller operates the
fixed hardware and physics selected in Objective 2; it does not change the
PCM, tank, collector, capsule geometry, safety limits, or simulator.

The project currently contains two state-specific implementations:

| Folder | State | Documentation |
|---|---|---|
| [`rajasthan/`](rajasthan/) | Rajasthan | [`docs/OBJECTIVE_3_README.md`](rajasthan/docs/OBJECTIVE_3_README.md) |
| [`tamilnadu/`](tamilnadu/) | Tamil Nadu | [`docs/OBJECTIVE_3_README.md`](tamilnadu/docs/OBJECTIVE_3_README.md) |

Both implementations provide:

- a Gymnasium-compatible environment around the frozen Objective 2 simulator;
- a safety shield that remains active regardless of the learned policy;
- rule-based baselines for comparison;
- PPO, DQN, recurrent, distributional, masked, and SAC training options;
- robustness evaluation, locked test evaluation, plots, and a software-in-the-loop demo.

## Getting started

Choose a state directory and create or activate a Python 3.10+ environment.
Then install that state's dependencies:

```powershell
cd rajasthan
python -m pip install -r requirements.txt
```

Tamil Nadu has the same project structure; use its documentation and available
dependency configuration when running that implementation.

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
