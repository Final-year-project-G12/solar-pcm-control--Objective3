"""
o3/roles_o3.py
==============
Objective 3's own weather-year roles and its one-shot test lock (plan section 15.8, Uttarakhand version).

    2016-2020  train      (O2 'explore')  policy learning, normalisation statistics
    2021-2023  validate   (O2 'select')   reward weights, B1 thresholds, checkpoint choice
    2024-2025  o2_used    O2's held-out test, run once (lock heldout_c75ac0e6fb78): NEVER loadable here
    2006-2010  test       O3-only test at the four O2 medoids, behind results/o3_test.lock

Train/validate weather is O2's own accepted point-year file (o2_frozen/data/weather/point_years).
Test weather is downloaded by O3 (scripts/m0_fetch_test_weather.py), built with O2's acceptance rules
(scripts/m0_build_test_inventory.py) into data/weather_test_2006_2010/, and can only be read after
acquire_o3_test_lock() in the same process. This is a guard rail against accidents, not security
(as O2 says of its own lock); deleting the lock must be logged in docs/PROGRESS.md.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from o3 import o2_bridge as ob
from src.io_utils import drop_leap_day   # noqa: E402  (O2 path set up by o2_bridge)

MEDOIDS = {0: "UKP_0014", 1: "UKP_0029", 2: "UKP_0025", 3: "UKP_0041"}
YEAR_ROLES = {**{y: "train" for y in range(2016, 2021)},
              **{y: "validate" for y in range(2021, 2024)},
              **{y: "o2_used" for y in (2024, 2025)},
              **{y: "test" for y in range(2006, 2011)}}
PURPOSES = ("train", "validate", "test")

O2_WEATHER_DIR = ob.O2_ROOT / "data" / "weather" / "point_years" / "uttarakhand"
TEST_WEATHER_DIR = ob.UK_ROOT / "data" / "weather_test_2006_2010"
TEST_INVENTORY = TEST_WEATHER_DIR / "weather_inventory_test.csv"
TEST_LOCK_PATH = ob.UK_ROOT / "results" / "o3_test.lock"


class O3LockError(PermissionError):
    pass


_lock_held = False


def role_of(year: int) -> str:
    try:
        return YEAR_ROLES[int(year)]
    except KeyError:
        raise ValueError(f"year {year} has no O3 role") from None


def years_for(purpose: str) -> list[int]:
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    return sorted(y for y, r in YEAR_ROLES.items() if r == purpose)


def accepted_test_years(regime: int) -> list[int]:
    """Test years whose point-year file passed O2's acceptance rules (empty until M0 is done)."""
    if not TEST_INVENTORY.exists():
        return []
    inv = pd.read_csv(TEST_INVENTORY)
    ok = inv[(inv.cluster_id == regime) & (inv.accepted)]
    return sorted(int(y) for y in ok.year)


def acquire_o3_test_lock(run_id: str, fingerprint: str) -> Path:
    """Exclusive-create results/o3_test.lock (same pattern as O2's roles.acquire_test_lock)."""
    global _lock_held
    TEST_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(TEST_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise O3LockError(f"O3 test already run (lock {TEST_LOCK_PATH}): {TEST_LOCK_PATH.read_text()[:300]}") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({"run_id": run_id, "fingerprint": fingerprint, "utc": datetime.now(timezone.utc).isoformat()}, fh)
    _lock_held = True
    return TEST_LOCK_PATH


def attach_existing_lock(expected_fingerprint: str) -> dict:
    """POST-LOCK, EXPLORATORY ONLY (plan 8.3 step 8). Allow test-year reads in this process only if the one-shot test
    has ALREADY been run (results/o3_test.lock exists) with exactly the given policy fingerprint. Never creates a lock,
    so it cannot be used to run the test for the first time or to evaluate a different (re-trained) policy set.
    Every output produced this way must be labelled exploratory."""
    global _lock_held
    if not TEST_LOCK_PATH.exists():
        raise O3LockError("no O3 test lock: run scripts/m6_locked_test.py first; attach_existing_lock never creates one")
    rec = json.loads(TEST_LOCK_PATH.read_text())
    if rec.get("fingerprint") != expected_fingerprint:
        raise O3LockError(f"lock fingerprint {rec.get('fingerprint')} != frozen policies {expected_fingerprint}")
    _lock_held = True
    return rec


def load_weather(regime: int, year: int, purpose: str) -> pd.DataFrame:
    """Hourly weather (8760 h, 29 Feb removed) for the regime's medoid, refusing any role mismatch."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    role = role_of(year)
    if role == "o2_used":
        raise PermissionError(f"{year} is O2's used held-out year; O3 may not load it")
    if role != purpose:
        raise PermissionError(f"year {year} is a '{role}' year and may not be used for '{purpose}'")
    pid = MEDOIDS[regime]
    if role == "test":
        if not _lock_held:
            raise O3LockError("O3 test years can only be loaded after acquire_o3_test_lock() in the same process")
        path = TEST_WEATHER_DIR / f"cluster{regime}_{pid}_{year}.csv"
    else:
        path = O2_WEATHER_DIR / f"cluster{regime}_{pid}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (point-year rejected or not built)")
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    return drop_leap_day(df.sort_values("timestamp_utc").reset_index(drop=True))
