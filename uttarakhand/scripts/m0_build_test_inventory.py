"""
scripts/m0_build_test_inventory.py -- milestone M0: accept or reject each 2006-2010 point-year with O2's rules.

Reuses O2's own src/weather/inventory.build_point_year (acceptance rule D6: >= 98 % valid hours, no gap > 3 h,
gaps <= 3 h linearly filled and flagged, unbroken hourly UTC sequence, local time Asia/Kolkata). O2's function calls
roles.role_of(year), which only knows 2016-2025, so it is called with role_of temporarily answering 'o3_test' --
no O2 file is changed. Output goes to data/weather_test_2006_2010/ (O3's folder), not O2's.

The inventory records only the acceptance fields and SHA-256 of each raw file; the annual GHI / mean temperature
columns that O2's inventory carries are DROPPED here, so building the test set reveals no climate value.
Usage: python scripts/m0_build_test_inventory.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import o2_bridge as ob     # noqa: E402,F401  (O2 path)
from o3 import roles_o3            # noqa: E402
import src.weather.inventory as inv   # noqa: E402

RAW = roles_o3.TEST_WEATHER_DIR / "raw"
VALUE_COLS = ["GHI_annual_kWh_m2", "T_amb_mean_C", "n_negative_ghi", "n_ghi_above_clearsky_1p1"]


def main():
    rows = []
    orig = inv.role_of
    inv.role_of = lambda y: "o3_test"
    try:
        for cid, pid in roles_o3.MEDOIDS.items():
            for y in roles_o3.years_for("test"):
                row, df = inv.build_point_year(cid, pid, y, raw_dir=RAW, write=False)
                if df is not None:
                    p = roles_o3.TEST_WEATHER_DIR / f"cluster{cid}_{pid}_{y}.csv"
                    df.to_csv(p, index=False)
                    row["csv_sha256"] = ob.sha256_file(p)
                row["state"] = "uttarakhand"
                rows.append({k: v for k, v in row.items() if k not in VALUE_COLS})
                print(pid, y, "accepted" if row["accepted"] else f"REJECTED: {row['reject_reason']}", flush=True)
    finally:
        inv.role_of = orig
    out = pd.DataFrame(rows)
    out.to_csv(roles_o3.TEST_INVENTORY, index=False)
    print(f"{int(out.accepted.sum())} / {len(out)} point-years accepted -> {roles_o3.TEST_INVENTORY}")


if __name__ == "__main__":
    main()
