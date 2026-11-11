"""
Phase O3-0 (M0) — fresh O3-only test weather, 2006-2010 at the three Rajasthan medoid points.

Plan §8.2 / Q1 (decided by the student 2026-11-09): every year 2011-2025 is already used by O2, so the O3 test set is
2006-2010 at RJP_0132, RJP_0192, RJP_0083 — a temporal hold-out, same three sites. Same source and request as O2's
g19_fetch_test2_weather.py: NASA POWER hourly point API, community RE, ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, T2M, RH2M,
WS10M, UTC. The student approved the download in chat on 2026-11-11.

This script only DOWNLOADS and runs the O2 weather-inventory ACCEPTANCE RULES (>= 98 % valid hours, no gap > 3 h, unbroken
hourly UTC sequence; gaps <= 3 h interpolated and flagged). It never prints or plots weather values: output is file names,
counts and pass/fail reasons only (plan 15.8). Rules are the same as O2's src/weather/inventory.py (re-stated here because
O2's loader only knows 2011-2025 and must not be edited).

Writes (inside this repo):
    data/weather_test_2006_2010/raw/power_<point>_<year>.json
    data/weather_test_2006_2010/cluster<k>_<point>_<year>.csv        (accepted point-years only; same columns as O2's)
    data/weather_test_2006_2010/inventory.csv                         (all 15 rows, accepted flag, reason, SHA-256 of the raw file)

Usage:   python scripts/phase0_fetch_test_weather.py [--no-download]     (--no-download: re-run the quality rules only)
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args  # noqa: E402

OUT = REPO_ROOT / "data" / "weather_test_2006_2010"
RAW = OUT / "raw"
POINTS = {0: ("RJP_0132", 24.375, 74.125), 1: ("RJP_0192", 26.875, 73.375), 2: ("RJP_0083", 26.625, 75.875)}
YEARS = range(2006, 2011)
BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M"
RENAME = {"ALLSKY_SFC_SW_DWN": "GHI_Wm2", "CLRSKY_SFC_SW_DWN": "GHI_clearsky_Wm2", "T2M": "T_amb_C", "RH2M": "RH_pct", "WS10M": "wind_ms"}
MIN_VALID_FRACTION, MAX_FILLABLE_GAP_H = 0.98, 3


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, (pid, lat, lon) in POINTS.items():
        for y in YEARS:
            f = RAW / f"power_{pid}_{y}.json"
            if f.exists() and f.stat().st_size > 10_000:
                print("skip", f.name)
                continue
            q = urllib.parse.urlencode({"start": f"{y}0101", "end": f"{y}1231", "latitude": lat, "longitude": lon,
                                        "community": "RE", "parameters": PARAMS, "format": "JSON", "time-standard": "UTC"})
            for attempt in range(1, 4):
                try:
                    with urllib.request.urlopen(f"{BASE}?{q}", timeout=120) as r:
                        d = json.loads(r.read())
                    if not d.get("properties", {}).get("parameter"):
                        raise ValueError("empty parameter block")
                    f.write_text(json.dumps(d), encoding="utf-8")
                    print("ok", f.name, round(f.stat().st_size / 1e3), "KB", flush=True)
                    break
                except Exception as e:      # noqa: BLE001
                    print("fail", attempt, f.name, repr(e)[:160], flush=True)
                    time.sleep(20)
            else:
                sys.exit(f"could not download {f.name}")
            time.sleep(1.0)


def gap_runs(invalid):
    runs, n = [], 0
    for v in invalid:
        if v:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return runs


def build_point_year(cid: int, pid: str, year: int) -> dict:
    """O2 inventory rule, no value inspection beyond validity counts."""
    path = RAW / f"power_{pid}_{year}.json"
    row = {"cluster_id": cid, "point_id": pid, "year": year, "raw_file": path.name}
    if not path.exists():
        return {**row, "accepted": False, "reject_reason": "raw file missing"}
    row["raw_sha256"] = sha256(path)
    params = json.loads(path.read_text(encoding="utf-8"))["properties"]["parameter"]
    stamps = pd.to_datetime(list(params["ALLSKY_SFC_SW_DWN"].keys()), format="%Y%m%d%H", utc=True)
    expected = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz="UTC")
    row["hourly_sequence_ok"] = bool(len(stamps) == len(expected) and (stamps == expected).all())
    df = pd.DataFrame({RENAME[k]: pd.Series(list(v.values()), index=pd.to_datetime(list(v.keys()), format="%Y%m%d%H", utc=True))
                       for k, v in params.items() if k in RENAME}).reindex(expected).replace(-999.0, np.nan)
    invalid = df[["GHI_Wm2", "T_amb_C"]].isna().any(axis=1).to_numpy()
    runs = gap_runs(invalid)
    row.update(n_invalid_hours=int(invalid.sum()), pct_valid=100.0 * (1.0 - invalid.mean()), n_gaps=len(runs),
               max_gap_h=max(runs) if runs else 0)
    why = []
    if not row["hourly_sequence_ok"]:
        why.append("hourly sequence broken")
    if row["pct_valid"] < 100 * MIN_VALID_FRACTION:
        why.append(f"valid {row['pct_valid']:.2f}% < {100 * MIN_VALID_FRACTION:.0f}%")
    if row["max_gap_h"] > MAX_FILLABLE_GAP_H:
        why.append(f"gap {row['max_gap_h']} h > {MAX_FILLABLE_GAP_H} h")
    row["accepted"], row["reject_reason"] = not why, "; ".join(why)
    if why:
        return row
    filled = invalid.copy()
    df = df.interpolate(method="linear", limit=MAX_FILLABLE_GAP_H, limit_direction="both")
    df["GHI_Wm2"] = df["GHI_Wm2"].clip(lower=0.0)
    ist = df.index.tz_convert("Asia/Kolkata")
    out = df.reset_index().rename(columns={"index": "timestamp_utc"})
    out["WS_ms"], out["local_hour"], out["local_date"] = out["wind_ms"], ist.hour, ist.date.astype(str)
    out["point_id"], out["cluster_id"], out["year"], out["filled"] = pid, cid, year, filled
    out = out[["timestamp_utc", "GHI_Wm2", "T_amb_C", "RH_pct", "WS_ms", "local_hour", "local_date", "point_id", "cluster_id",
               "year", "wind_ms", "GHI_clearsky_Wm2", "filled"]]
    out.to_csv(OUT / f"cluster{cid}_{pid}_{year}.csv", index=False)
    row["n_filled_hours"] = int(filled.sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    add_parallel_args(ap)      # accepted for CLI uniformity; 15 polite sequential requests, nothing to parallelise
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        download()
    rows = [build_point_year(cid, pid, y) for cid, (pid, _, _) in POINTS.items() for y in YEARS]
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    for r in rows:
        print(f"{r['point_id']} {r['year']}: {'ACCEPTED' if r['accepted'] else 'REJECTED ' + r['reject_reason']}")
    n_ok = int(inv["accepted"].sum())
    print(f"\n{n_ok} of {len(rows)} point-years accepted -> {OUT / 'inventory.csv'}")
    if n_ok < len(rows):
        print("Rejected years shrink the test set; report that in the paper (plan risk 2).")


if __name__ == "__main__":
    main()
