"""
src/weather/inventory.py
===========================
O2 G5 / plan WP4 (2026-10-26) — build the per point-year weather inventory
and the cleaned hourly files from Objective 1's raw NASA POWER cache.

Representative points (decision D5): the CURRENT Objective 1 medoid of each cluster of the
K = 4 clustering (PCM-Selection-ML-model commit 6be0dcd, 2026-10-03) -- the "medoid point" printed by
O1's 09_recommendation_cards.py (closest member to the cluster's mean lat/lon), each with membership
probability 1.0 in cluster_assignments_uttarakhand.csv. Read-only access to the raw JSON.

Acceptance rule (decision D6): a point-year is accepted if >= 98 % of its hours are valid
(GHI and T2M present, not the -999 fill) AND no gap exceeds 3 consecutive hours; gaps of up
to 3 h are filled by linear interpolation and flagged in the hourly file (`filled`).
Rejected point-years stay in the inventory with their reason; nothing is silently dropped.

Hourly resolution is verified per file: the timestamp keys must form an unbroken hourly
UTC sequence Jan 1 00:00 .. Dec 31 23:00.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OBJ1_RAW_POWER_DIR
from src.weather.roles import INVENTORY_PATH, WEATHER_POINT_YEARS_DIR, point_year_path, role_of

RAW_DIR = OBJ1_RAW_POWER_DIR
CURRENT_MEDOIDS = {0: "UKP_0014", 1: "UKP_0029", 2: "UKP_0025", 3: "UKP_0041"}   # O1 K=4 medoids, verified against cluster_assignments
REGIMES = tuple(sorted(CURRENT_MEDOIDS))
YEARS = range(2016, 2026)
MIN_VALID_FRACTION = 0.98
MAX_FILLABLE_GAP_H = 3
RENAME = {"ALLSKY_SFC_SW_DWN": "GHI_Wm2", "CLRSKY_SFC_SW_DWN": "GHI_clearsky_Wm2", "T2M": "T_amb_C",
          "RH2M": "RH_pct", "WS10M": "wind_ms"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gap_runs(invalid: np.ndarray):
    """Lengths of consecutive True runs."""
    runs, n = [], 0
    for v in invalid:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def build_point_year(cluster_id: int, point_id: str, year: int, raw_dir: Path = RAW_DIR, write: bool = True):
    """Returns (inventory_row dict, hourly DataFrame or None)."""
    path = raw_dir / f"power_{point_id}_{year}.json"
    row = {"state": "uttarakhand", "cluster_id": cluster_id, "point_id": point_id, "year": year,
           "role": role_of(year), "raw_file": str(path.name)}
    if not path.exists():
        return {**row, "accepted": False, "reject_reason": "raw file missing"}, None
    row["raw_sha256"] = _sha256(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    params = data["properties"]["parameter"]
    row["time_standard"] = data.get("header", {}).get("time_standard", "")
    keys = list(params["ALLSKY_SFC_SW_DWN"].keys())
    stamps = pd.to_datetime(keys, format="%Y%m%d%H", utc=True)
    expected = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz="UTC")
    row["n_hours_expected"] = len(expected)
    row["n_keys_found"] = len(stamps)
    row["hourly_sequence_ok"] = bool(len(stamps) == len(expected) and (stamps == expected).all())
    df = pd.DataFrame({RENAME[k]: pd.Series(list(v.values()), index=pd.to_datetime(list(v.keys()), format="%Y%m%d%H", utc=True))
                       for k, v in params.items() if k in RENAME}).reindex(expected)
    df = df.replace(-999.0, np.nan)
    invalid = df[["GHI_Wm2", "T_amb_C"]].isna().any(axis=1).to_numpy()
    runs = _gap_runs(invalid)
    row["n_invalid_hours"] = int(invalid.sum())
    row["pct_valid"] = 100.0 * (1.0 - invalid.mean())
    row["n_gaps"] = len(runs)
    row["max_gap_h"] = max(runs) if runs else 0
    row["n_other_vars_missing_hours"] = int(df[["RH_pct", "wind_ms", "GHI_clearsky_Wm2"]].isna().any(axis=1).sum())
    reasons = []
    if not row["hourly_sequence_ok"]:
        reasons.append("hourly sequence broken")
    if row["pct_valid"] < 100 * MIN_VALID_FRACTION:
        reasons.append(f"valid {row['pct_valid']:.2f}% < {100 * MIN_VALID_FRACTION:.0f}%")
    if row["max_gap_h"] > MAX_FILLABLE_GAP_H:
        reasons.append(f"gap {row['max_gap_h']} h > {MAX_FILLABLE_GAP_H} h")
    row["accepted"] = not reasons
    row["reject_reason"] = "; ".join(reasons)
    if reasons:
        return row, None

    filled = invalid.copy()
    df = df.interpolate(method="linear", limit=MAX_FILLABLE_GAP_H, limit_direction="both")
    row["n_filled_hours"] = int(filled.sum())
    row["n_negative_ghi"] = int((df["GHI_Wm2"] < 0).sum())
    df["GHI_Wm2"] = df["GHI_Wm2"].clip(lower=0.0)
    row["n_ghi_above_clearsky_1p1"] = int((df["GHI_Wm2"] > 1.1 * df["GHI_clearsky_Wm2"] + 5.0).sum())
    ist = df.index.tz_convert("Asia/Kolkata")
    out = df.reset_index().rename(columns={"index": "timestamp_utc"})
    out["WS_ms"] = out["wind_ms"]
    out["local_hour"] = ist.hour
    out["local_date"] = ist.date.astype(str)
    out["point_id"], out["cluster_id"], out["year"] = point_id, cluster_id, year
    out["filled"] = filled
    cols = ["timestamp_utc", "GHI_Wm2", "T_amb_C", "RH_pct", "WS_ms", "local_hour", "local_date", "point_id",
            "cluster_id", "year", "wind_ms", "GHI_clearsky_Wm2", "filled"]
    out = out[cols]
    leap_free = out[~out["local_date"].str.endswith("-02-29")]
    row["GHI_annual_kWh_m2"] = float(leap_free["GHI_Wm2"].sum() / 1000.0)
    row["T_amb_mean_C"] = float(leap_free["T_amb_C"].mean())
    row["n_hours_after_leap_drop"] = len(leap_free)
    if write:
        p = point_year_path("uttarakhand", cluster_id, point_id, year)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    return row, out


def build_inventory(write: bool = True, years=None) -> pd.DataFrame:
    """`years` limits the (re)build to those years; the other rows of the existing inventory are kept as they are
    (and their hourly CSVs are not touched), so adding years never rewrites files that running jobs read."""
    rows = []
    for cid, pid in CURRENT_MEDOIDS.items():
        for y in (YEARS if years is None else years):
            r, _ = build_point_year(cid, pid, y, raw_dir=RAW_DIR, write=write)
            rows.append(r)
    inv = pd.DataFrame(rows)
    if years is not None and INVENTORY_PATH.exists():
        old = pd.read_csv(INVENTORY_PATH)
        inv = pd.concat([old[~old["year"].isin(list(years))], inv], ignore_index=True)
        inv = inv.sort_values(["cluster_id", "year"]).reset_index(drop=True)
    if write:
        WEATHER_POINT_YEARS_DIR.mkdir(parents=True, exist_ok=True)
        inv.to_csv(INVENTORY_PATH, index=False)
    return inv
