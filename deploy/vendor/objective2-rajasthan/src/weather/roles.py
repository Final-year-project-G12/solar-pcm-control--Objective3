"""
src/weather/roles.py
=======================
O2 G5 / plan WP4 + decision D5 (2026-10-26) — enforce, in code, what each
weather year may be used for:

    2016-2020  "explore"  design exploration (DOE, surrogate training, optimisation)
    2021-2023  "select"   choosing the winner among explored designs
    2011-2015  "test"     held-out test, run exactly ONCE (v3, cap 30 %; downloaded 2026-10-27 after the v1 test)
    2024-2025  "used_test_v1"  the v1 (cap 20 %) held-out test years: already used, labelled so, accepted for NO purpose

`load_point_year(..., purpose=...)` refuses a year whose role differs from the stated
purpose. The test years (2011-2015) additionally require `acquire_test_lock()`, which atomically
creates results/held_out_test.lock and refuses if it already exists -- so the
held-out evaluation cannot be repeated or tuned against by accident. This is a guard
rail inside the codebase, not a security boundary: the cached CSVs are ordinary files
and anyone can open them directly. Deleting the lock to re-run is a deliberate,
visible act that must be logged in docs/PROGRESS.md.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.io_utils import BASE_DIR, drop_leap_day

WEATHER_POINT_YEARS_DIR = BASE_DIR / "data" / "weather/point_years"
INVENTORY_PATH = WEATHER_POINT_YEARS_DIR / "weather_inventory.csv"
TEST_LOCK_PATH = BASE_DIR / "results" / "held_out_test.lock"

YEAR_ROLES = {**{y: "test" for y in range(2011, 2016)},
              **{y: "explore" for y in range(2016, 2021)},
              **{y: "select" for y in range(2021, 2024)},
              **{y: "used_test_v1" for y in (2024, 2025)}}
PURPOSES = ("explore", "select", "test")


class HeldOutLockError(PermissionError):
    pass


_lock_held_by_this_process = False


def role_of(year: int) -> str:
    try:
        return YEAR_ROLES[int(year)]
    except KeyError:
        raise ValueError(f"year {year} is outside the 2011-2025 record") from None


def years_for(purpose: str):
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    return [y for y, r in YEAR_ROLES.items() if r == purpose]


def acquire_test_lock(run_id: str, selection_fingerprint: str) -> Path:
    """Create the held-out lock (atomic, exclusive). Call once, immediately before the
    single held-out evaluation; `selection_fingerprint` ties it to the frozen selection."""
    global _lock_held_by_this_process
    TEST_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(TEST_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise HeldOutLockError(
            f"held-out test ({min(years_for('test'))}-{max(years_for('test'))}) already run "
            f"(lock {TEST_LOCK_PATH}); it may be run exactly once. "
            f"Contents: {TEST_LOCK_PATH.read_text()[:300]}") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({"run_id": run_id, "selection_fingerprint": selection_fingerprint,
                   "utc": datetime.now(timezone.utc).isoformat()}, fh)
    _lock_held_by_this_process = True
    return TEST_LOCK_PATH


def point_year_path(state: str, cluster_id: int, point_id: str, year: int) -> Path:
    return WEATHER_POINT_YEARS_DIR / state / f"cluster{cluster_id}_{point_id}_{year}.csv"


def load_point_year(state: str, cluster_id: int, point_id: str, year: int, purpose: str) -> pd.DataFrame:
    """Hourly weather for one accepted point-year, 29 Feb removed (8760 h)."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    role = role_of(year)
    if role != purpose:
        raise PermissionError(f"year {year} is a '{role}' year and may not be used for '{purpose}'")
    if role == "test" and not _lock_held_by_this_process:
        raise HeldOutLockError("test years can only be loaded after acquire_test_lock() in the same process")
    path = point_year_path(state, cluster_id, point_id, year)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- point-year rejected or not built (see weather_inventory.csv)")
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    return drop_leap_day(df.sort_values("timestamp_utc").reset_index(drop=True))
