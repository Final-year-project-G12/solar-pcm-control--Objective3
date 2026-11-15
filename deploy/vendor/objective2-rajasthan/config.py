"""
Objective 2 — shared paths and configuration (Rajasthan).

Mirrors objective2-tamilnadu/config.py exactly — only the state name and
the Objective 1 source-folder name differ. Assumes this folder
(objective2-rajasthan/) sits as a SIBLING to the Rajasthan Objective 1
pipeline folder:

    project-root/
      era5-rajasthan/          <- Objective 1 pipeline (READ-ONLY from here on)
      objective2-rajasthan/    <- this project (everything Obj2 writes lives here)

Objective 1's outputs have already been frozen and copied under
data/objective1/, data/weather/, data/demand/ (see data/README.md), so the
OBJ1_* paths below are only needed if you ever re-run the build scripts
against a fresh Objective 1 run. If your Objective 1 folder has a different
name, change OBJ1_ROOT below — nothing else needs to change.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# ── Objective 1 pipeline (READ-ONLY — Objective 2 must never write here) ──
# NOT a sibling of this folder — Objective 1 actually lives under
# PCM-Selection-ML-model/, two levels up from OBJECTIVE2/Objective---2/.
# Fixed 2026-09-13 (same fix applied to objective2-tamilnadu/config.py):
# this used to point at a nonexistent PROJECT_ROOT/era5-rajasthan sibling,
# which only went unnoticed because data/objective1/ was already frozen
# from an earlier successful run.
OBJ1_ROOT = PROJECT_ROOT.parent.parent / "PCM-Selection-ML-model" / "era5-rajasthan"
OBJ1_DATA_DIR = OBJ1_ROOT / "data"
OBJ1_PROCESSED_DIR = OBJ1_DATA_DIR / "processed"
OBJ1_PREPROCESSED_DIR = OBJ1_DATA_DIR / "preprocessed"
OBJ1_RAW_POWER_DIR = OBJ1_DATA_DIR / "raw" / "nasapower"

# ── Objective 2's own tree ─────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
OBJ1_FROZEN_DIR = DATA_DIR / "objective1"            # D2.1: frozen copies land here
OBJ1_FROZEN_WEATHER_DIR = OBJ1_FROZEN_DIR / "raw_weather"   # medoid points' hourly cache
WEATHER_DIR = DATA_DIR / "weather"                   # per-regime frozen weather profiles
DEMAND_DIR = DATA_DIR / "demand"                     # canonical draw profile
PCM_DIR = DATA_DIR / "pcm"
PROCESSED_DIR = DATA_DIR / "processed"               # Obj2's own DOE/surrogate/optimizer output

CONFIGS_DIR = BASE_DIR / "configs"                    # *_shared.yaml + states/rajasthan.yaml
RESULTS_DIR = BASE_DIR / "results"

MANIFEST_FILE = OBJ1_FROZEN_DIR / "manifest.json"


def ensure_dirs():
    for d in (DATA_DIR, OBJ1_FROZEN_DIR, OBJ1_FROZEN_WEATHER_DIR, WEATHER_DIR,
              DEMAND_DIR, PCM_DIR, PROCESSED_DIR, CONFIGS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
