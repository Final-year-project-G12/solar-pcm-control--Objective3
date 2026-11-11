"""
config_O2.py — paths for the Objective-3 Rajasthan controller.

This file only says WHERE things are; it never holds a physical number. Every Rajasthan
number (PCM, capsule count, tank, collector, limits, weather, demand) comes from the
Objective-2 Rajasthan tree through the v3 contract and the O2 config files.

Layout
    <repo>/solar-pcm-control--Objective3/rajasthan/        <- O3_ROOT (this folder)
    <repo>/OBJECTIVE2/Objective---2/objective2-rajasthan/  <- O2_ROOT (frozen simulator, never edited by O3)

Override the O2 location with the environment variable O2_RAJASTHAN_ROOT.

Why this file is called config_O2 and not config: the O2 tree has its own `config.py`
(imported by its `src` package as `config`). Using another name here avoids a clash when
both roots are on sys.path.
"""

import os
import sys
from pathlib import Path

STATE = "rajasthan"

O3_ROOT = Path(__file__).resolve().parent
_DEFAULT_O2 = O3_ROOT.parents[1] / "OBJECTIVE2" / "Objective---2" / "objective2-rajasthan"
O2_ROOT = Path(os.environ.get("O2_RAJASTHAN_ROOT", _DEFAULT_O2)).resolve()

CONTRACTS_DIR = O3_ROOT / "contracts"
# O3_RESULTS_DIR redirects every script's output (smoke tests / dry runs must not touch results/)
RESULTS_DIR = Path(os.environ.get("O3_RESULTS_DIR", O3_ROOT / "results"))
DOCS_DIR = O3_ROOT / "docs"
CONTRACT_FILE = CONTRACTS_DIR / "obj3_environment_contract_rajasthan_v3.json"

# O2 files the contract points at (read-only for O3)
O2_RESULTS_DIR = O2_ROOT / "results"
O2_CONTRACT_V3 = O2_RESULTS_DIR / "obj3_environment_contract_rajasthan_v3.json"
O2_WEATHER_DIR = O2_ROOT / "data" / "weather" / "point_years" / "rajasthan"


def setup_paths() -> None:
    """Put the O2 root and the O3 root at the front of sys.path (idempotent).

    O2 root first: its `src` package does `from config import ...`, so its config.py must win.
    """
    for p in (str(O3_ROOT), str(O2_ROOT)):          # insert O3 first, then O2 -> O2 ends up at index 0
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if not O2_ROOT.exists():
        raise FileNotFoundError(f"O2 Rajasthan tree not found at {O2_ROOT}; set O2_RAJASTHAN_ROOT")
