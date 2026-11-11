"""
scripts/m0_fetch_test_weather.py -- milestone M0: download the O3-only test years 2006-2010.

Every year 2016-2025 is already used by O2 in Uttarakhand (explore / select / held-out test run once), so the O3
test set is 2006-2010 at the same four O2 medoids (plan Q1, Uttarakhand version; also what O2's
06_OBJECTIVE3_HANDOFF.md says Uttarakhand would need). Same source and request as O2's original download and as
O2's g19 script: NASA POWER hourly point API, community RE, parameters ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M,
time-standard UTC. Coordinates: the medoid's lat/lon in O1's climate_signature_uttarakhand.csv.

Writes data/weather_test_2006_2010/raw/power_<point>_<year>.json (20 files). ONLY downloads: never prints, plots
or summarises a value (the years are fixed before anything is inspected).
Usage: python scripts/m0_fetch_test_weather.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "weather_test_2006_2010" / "raw"
SIG = ROOT / "o2_frozen" / "data" / "objective1" / "climate_signature_uttarakhand.csv"
MEDOIDS = ["UKP_0014", "UKP_0029", "UKP_0025", "UKP_0041"]
YEARS = range(2006, 2011)
BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIG).set_index("point_id")
    s = requests.Session()
    for pid in MEDOIDS:
        lat, lon = float(sig.loc[pid, "lat"]), float(sig.loc[pid, "lon"])
        for y in YEARS:
            f = OUT / f"power_{pid}_{y}.json"
            if f.exists() and f.stat().st_size > 10_000:
                print("skip", f.name)
                continue
            q = {"start": f"{y}0101", "end": f"{y}1231", "latitude": lat, "longitude": lon, "community": "RE",
                 "parameters": PARAMS, "format": "JSON", "time-standard": "UTC"}
            for attempt in range(1, 4):
                try:
                    r = s.get(BASE, params=q, timeout=120)
                    r.raise_for_status()
                    d = r.json()
                    if not d.get("properties", {}).get("parameter"):
                        raise ValueError("empty parameter block")
                    f.write_text(json.dumps(d), encoding="utf-8")
                    print("ok", f.name, round(f.stat().st_size / 1e3), "KB", flush=True)
                    break
                except Exception as e:                                  # noqa: BLE001
                    print("fail", attempt, f.name, repr(e)[:200], flush=True)
                    time.sleep(20)
            else:
                sys.exit(f"could not download {f.name}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
