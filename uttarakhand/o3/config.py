"""o3/config.py -- the O3 config file, its hash, and the per-regime normalisation files."""

from __future__ import annotations

import json
from functools import lru_cache

import yaml

from o3 import o2_bridge as ob

CONFIG_PATH = ob.UK_ROOT / "configs" / "o3_config.yaml"
RESULTS = ob.UK_ROOT / "results"


@lru_cache(maxsize=1)
def load() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash() -> str:
    return ob.sha256_file(CONFIG_PATH)


def normalisation_path(regime: int):
    return RESULTS / f"normalisation_r{regime}.json"


def load_normalisation(regime: int) -> dict:
    p = normalisation_path(regime)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/m3_baselines.py first")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)
