# Deploying the combined dashboard (Streamlit Community Cloud, free)

The combined app is `streamlit_app.py` at the repo root. Every state page runs that state's own portal (the Tamil Nadu design: a Live Controller tab
and a Deployment & ML Models tab): `rajasthan/app/streamlit_app.py`, `uttarakhand/app/streamlit_app.py` and `tamilnadu/scripts/streamlit_portal.py`.
The Tamil Nadu page also keeps its saved result dashboards in an expander.

## What is in the repo for this
| Path | Purpose |
|---|---|
| `streamlit_app.py`, `hub/` | the multipage app and `hub/state_runner.py`, which runs one state's app at a time with a clean module table |
| `requirements.txt` | single requirements file (CPU-only torch via the PyTorch CPU index) |
| `.streamlit/config.toml` | headless, no file watcher, no telemetry |
| `deploy/vendor/objective2-tamilnadu/` | the part of the Objective-2 Tamil Nadu tree that Tamil Nadu's portal reads (`src/`, `config.py`, `configs/`, Objective-1 CSVs without the 150 MB raw-weather and suntime files, validation-and-training weather 2016-2023). About 21 MB. `hub/state_runner.py` points Tamil Nadu at it with `O2_ROOT` |
| `deploy/vendor/objective2-rajasthan/` | the part of the Objective-2 Rajasthan tree that Rajasthan's O3 reads (`src/`, `config.py`, `configs/`, contract and selection files, train+validation weather 2016-2023, O1 shortlist). About 20 MB. Rajasthan's `config_O2.py` normally looks for this tree OUTSIDE the repo; `hub/state_runner.py` points it here with `O2_RAJASTHAN_ROOT` |
| `tests/test_hub_app.py` | smoke test: every page loads, each state runs on its own `o3`, vendored files present |
| `.github/workflows/deploy-dashboard.yml` | test on every push / PR; on a passing push to `main`, copy that commit to the `deploy` branch |

Test years (2006-2010) and the O2 "used" years are not in the vendored weather, and the apps refuse them anyway.

## One-time setup on Streamlit Community Cloud
1. Push this repo to GitHub (the workflow needs it). Let the workflow run once so the `deploy` branch exists.
2. share.streamlit.io -> New app -> pick this repository, **branch `deploy`**, main file path `streamlit_app.py`.
3. Advanced settings -> Python version **3.12**.
4. Optional: in GitHub -> Settings -> Variables, add `STREAMLIT_APP_URL` (the app's URL); the workflow then pings it after a deploy
   so a sleeping app wakes up.

After that every passing push to `main` redeploys the app; a failing test never reaches it.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
pytest tests/test_hub_app.py
```

## Known limits (read before relying on it)
* **Not verified on Streamlit Cloud itself.** Tested locally on Windows / Python 3.10 (4 tests pass). The workflow's Linux / Python 3.12 run is the first check on the target platform.
* **Memory and CPU.** The free tier is about 1 GB RAM and one shared core. torch + numba + the simulator are heavy; the first view of a
  state compiles numba code, and switching state re-imports that state's packages (about 30 s the first time for Uttarakhand locally).
  If the app is killed for memory, run it on a small VM or Hugging Face Spaces instead; nothing else changes.
* **One state runs at a time per server process.** `hub/state_runner.py` serialises state pages with a lock because both states
  define a package named `o3`. Two visitors on different states queue behind each other.
* **Repo size.** The git pack is about 515 MB (many tracked training checkpoints under `uttarakhand/results/`). Streamlit Cloud clones the
  whole branch. If the clone is too slow or refused, drop old checkpoints from the `deploy` branch.
* **Live mode** calls Open-Meteo from the server; it works from a cloud host but needs outbound internet.
* **Saved Uttarakhand PPO models** were written by Python 3.12; loading them on another Python prints a harmless "Could not deserialize
  lr_schedule" warning (inference is unaffected).
