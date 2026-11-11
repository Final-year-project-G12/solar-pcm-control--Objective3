"""
o3/utils/weather_roles.py — year roles and the O3 test lock (Rajasthan).

O2's own role map (src/weather/roles.py) is NOT used for loading, because O2's held-out lock
already exists and O2's loader would (correctly) refuse the test years. O3 needs its own map and
its own lock. Same pattern as O2: exclusive-create lock file, guard rail not security.

    train      2016-2020   policy learning, normalisation, reward-weight and B1 tuning ARE NOT done here
    validate   2021-2023   reward weights, B1 thresholds, checkpoint choice, early stopping
    test       2006-2010   FRESH O3-only years (plan 8.2, Q1), downloaded by scripts/phase0_fetch_test_weather.py into
                           data/weather_test_2006_2010/ and accepted by the O2 inventory rules (15 of 15 point-years).
                           O3 evaluates ONCE behind results/o3_test.lock (phase6).
    used_o2    2011-2015   O2's held-out test years (O2 ran its one-shot test on them) and
               2024-2025   O2's v1 test years. Never usable by O3.

load_weather(regime_id, year, purpose) refuses a year whose role differs from `purpose`.
Test-year loading additionally needs the lock to be held by this process
(acquire_test_lock) or by a parent process that passes its run_id to a worker
(assume_test_lock) — that is how phase6 runs seeds/regimes in parallel.
"""

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config_O2

TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020]
VAL_YEARS = [2021, 2022, 2023]
TEST_YEARS = [2006, 2007, 2008, 2009, 2010]
USED_O2_YEARS = [2011, 2012, 2013, 2014, 2015, 2024, 2025]

YEAR_ROLES = {**{y: "train" for y in TRAIN_YEARS},
              **{y: "validate" for y in VAL_YEARS},
              **{y: "test" for y in TEST_YEARS},
              **{y: "used_o2" for y in USED_O2_YEARS}}
PURPOSES = ("train", "validate", "test")

MEDOID_POINTS = {0: "RJP_0132", 1: "RJP_0192", 2: "RJP_0083"}   # contract weather_sequences.medoid_points


def default_lock_path() -> Path:
    return Path(os.environ.get("O3_TEST_LOCK", config_O2.RESULTS_DIR / "o3_test.lock"))


class TestLockError(PermissionError):
    __test__ = False          # not a pytest class


_lock_run_id: str | None = None          # set when this process holds / has been told about the lock


def role_of(year: int) -> str:
    try:
        return YEAR_ROLES[int(year)]
    except KeyError:
        raise ValueError(f"year {year} is outside the 2006-2025 record") from None


def years_for(purpose: str) -> list[int]:
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    return [y for y, r in YEAR_ROLES.items() if r == purpose]


def acquire_test_lock(run_id: str, fingerprint: str, lock_path: Path | None = None) -> Path:
    """Create the O3 test lock (atomic, exclusive). Call once, immediately before the single
    test evaluation. `fingerprint` = hash of policy + normalisation + reward config (+ git commit)."""
    global _lock_run_id
    lock_path = Path(lock_path) if lock_path else default_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise TestLockError(
            f"O3 test evaluation already run (lock {lock_path}); it may be run exactly once. "
            f"Contents: {lock_path.read_text()[:300]}") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({"run_id": run_id, "fingerprint": fingerprint,
                   "utc": datetime.now(timezone.utc).isoformat()}, fh)
    _lock_run_id = run_id
    return lock_path


def assume_test_lock(run_id: str, lock_path: Path | None = None) -> None:
    """In a worker process: declare that the parent holds the lock. Verified against the lock file."""
    global _lock_run_id
    lock_path = Path(lock_path) if lock_path else default_lock_path()
    if not lock_path.exists():
        raise TestLockError("no O3 test lock on disk; the parent must acquire it first")
    held = json.loads(lock_path.read_text()).get("run_id")
    if held != run_id:
        raise TestLockError(f"lock run_id {held!r} does not match {run_id!r}")
    _lock_run_id = run_id


def release_for_tests() -> None:
    """Forget the in-process lock flag (unit tests only; does not touch the file)."""
    global _lock_run_id
    _lock_run_id = None


def test_lock_held() -> bool:
    return _lock_run_id is not None


def weather_path(regime_id: int, year: int) -> Path:
    pid = MEDOID_POINTS[int(regime_id)]
    if int(year) in TEST_YEARS:
        return config_O2.O3_ROOT / "data" / "weather_test_2006_2010" / f"cluster{int(regime_id)}_{pid}_{int(year)}.csv"
    return config_O2.O2_WEATHER_DIR / f"cluster{int(regime_id)}_{pid}_{int(year)}.csv"


@lru_cache(maxsize=64)
def _read(regime_id: int, year: int) -> dict:
    from src.io_utils import drop_leap_day

    p = weather_path(regime_id, year)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - point-year rejected or not built (see O2 weather_inventory.csv)")
    df = pd.read_csv(p, parse_dates=["timestamp_utc"])
    df = drop_leap_day(df.sort_values("timestamp_utc").reset_index(drop=True))
    return {
        "GHI_Wm2": df["GHI_Wm2"].to_numpy(dtype=float),
        "T_amb_C": df["T_amb_C"].to_numpy(dtype=float),
        "local_hour": df["local_hour"].to_numpy(dtype=np.int64),
    }


def load_weather(regime_id: int, year: int, purpose: str | None = None) -> dict:
    """Hourly arrays {GHI_Wm2, T_amb_C, local_hour} for one regime-year (29 Feb removed, 8760 h).

    purpose=None infers the purpose from the year's role (so a validation year is simply loadable
    as validation) but the guards still apply: 'used_o2' years are always refused and test years
    need the lock.
    """
    role = role_of(year)
    if role == "used_o2":
        raise PermissionError(f"year {year} was used by O2's tests (2011-2015 held-out, 2024-2025 v1); O3 may not use it")
    if purpose is not None and purpose != role:
        raise PermissionError(f"year {year} is a '{role}' year and may not be used for '{purpose}'")
    if role == "test" and not test_lock_held():
        raise TestLockError("test years (2006-2010) can only be loaded after acquire_test_lock() "
                            "(or assume_test_lock() in a worker)")
    d = _read(int(regime_id), int(year))
    return {k: v.copy() for k, v in d.items()}
