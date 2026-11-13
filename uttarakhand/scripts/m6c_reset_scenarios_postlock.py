"""
scripts/m6c_reset_scenarios_postlock.py -- POST-LOCK, EXPLORATORY (plan 8.3 step 8). Run only after M6.

Plan sections 1 and 8.4 ask for results per reset scenario. The one-shot M6 run used the fully_solid start only. This
script attaches read-only to the existing lock (frozen-policy fingerprint must match; never creates a lock) and runs the
SAME frozen controllers on the SAME test years for the two other contract reset scenarios (partially_charged,
fully_liquid), deterministic demand, each paired with its water-only twin. Labelled EXPLORATORY; M6's numbers are not
changed or replaced.
Writes results/m6c_reset_scenarios_EXPLORATORY.csv
Usage: python scripts/m6c_reset_scenarios_postlock.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3 import o2_bridge as ob                     # noqa: E402
from o3 import roles_o3                            # noqa: E402
from o3.controllers import B0AlwaysOn              # noqa: E402
from o3.evaluate import evaluate                   # noqa: E402

SCENARIOS = ("partially_charged", "fully_liquid")


def main():
    cfg = o3c.load()
    ob.verify_contract()
    fz = {k: frozen.load_frozen(k) for k in cfg["regimes"]["pcm"]}
    fp = ob.hashlib.sha256(json.dumps({str(k): v["fingerprint"] for k, v in fz.items()}, sort_keys=True).encode()).hexdigest()
    lock = roles_o3.attach_existing_lock(fp)
    print("attached to", lock["run_id"], "(EXPLORATORY)", flush=True)
    out = []
    for k in cfg["regimes"]["pcm"] + cfg["regimes"]["water_only"]:
        yrs = roles_o3.accepted_test_years(k)
        for c in [B0AlwaysOn(), frozen.b1(k)] + (frozen.ppo_policies(k) if k in fz else []):
            out.append(evaluate(c, k, yrs, "test", SCENARIOS))
        print(f"r{k} done", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["label"] = "EXPLORATORY post-lock supplement (policies, config, years unchanged)"
    df["lock_run_id"], df["o3_config_sha256"] = lock["run_id"], o3c.config_hash()
    df.to_csv(o3c.RESULTS / "m6c_reset_scenarios_EXPLORATORY.csv", index=False)
    df["group"] = df.controller.str.replace(r"_seed\d+$", "", regex=True)
    cols = ["E_del_window_kWh", "g_window_kWh", "overheat_h", "hours_over_limit", "bypass_switches"]
    print(df.groupby(["regime", "scenario", "group"])[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
