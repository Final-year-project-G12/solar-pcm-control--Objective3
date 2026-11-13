"""
scripts/m6_locked_test.py -- milestone M6: the ONE evaluation on the O3-only test years 2006-2010.

Refuses unless (a) the contract and O2 snapshot check pass, (b) every PCM regime has a frozen policy whose hashes
verify, (c) the test inventory exists and is accepted. Then acquires results/o3_test.lock (exclusive create; fails if
it exists), fingerprint = hash of every regime's frozen fingerprint, and evaluates once:
  regimes 0-2: B0, B1, every frozen PPO seed;  regime 3: B0, B1 (water-only regime, no PPO)
on all accepted test years, fully_solid, deterministic demand, each paired with its water-only twin.
Anything run on these years afterwards must be labelled exploratory (plan 8.3 step 8).
Writes results/m6_test_results.csv and results/m6_test_summary.json
Usage: python scripts/m6_locked_test.py --confirm-run-once
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3 import o2_bridge as ob                     # noqa: E402
from o3 import roles_o3                            # noqa: E402
from o3.controllers import B0AlwaysOn              # noqa: E402
from o3.evaluate import evaluate                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run-once", action="store_true", required=True)
    ap.parse_args()
    cfg = o3c.load()
    ob.verify_contract()
    fz = {k: frozen.load_frozen(k) for k in cfg["regimes"]["pcm"]}
    years = {k: roles_o3.accepted_test_years(k) for k in cfg["regimes"]["pcm"] + cfg["regimes"]["water_only"]}
    if not all(years.values()):
        sys.exit(f"test inventory incomplete: {years}")
    fingerprint = ob.hashlib.sha256(json.dumps({str(k): v["fingerprint"] for k, v in fz.items()},
                                               sort_keys=True).encode()).hexdigest()
    run_id = "o3test_" + uuid.uuid4().hex[:12]
    roles_o3.acquire_o3_test_lock(run_id, fingerprint)
    print("lock acquired", run_id, fingerprint, flush=True)

    out = []
    for k, yrs in years.items():
        ctrls = [B0AlwaysOn(), frozen.b1(k)] + (frozen.ppo_policies(k) if k in fz else [])
        for c in ctrls:
            out.append(evaluate(c, k, yrs, "test"))
            print(f"r{k} {c.name} done", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["run_id"], df["fingerprint"], df["o3_config_sha256"] = run_id, fingerprint, o3c.config_hash()
    df.to_csv(o3c.RESULTS / "m6_test_results.csv", index=False)
    cols = ["E_del_window_kWh", "g_window_kWh", "g_annual_kWh", "overheat_h", "shield_water_h", "shield_pcm_h",
            "guard_hours", "hours_over_limit", "t_water_max_C", "t_pcm_max_C", "bypass_switches", "residual_pct"]
    df["group"] = df.controller.str.replace(r"_seed\d+$", "", regex=True)
    summ = df.groupby(["regime", "group"])[cols].agg(["mean", "std"]).round(3)
    (o3c.RESULTS / "m6_test_summary.json").write_text(json.dumps({
        "run_id": run_id, "fingerprint": fingerprint, "years": years, "o3_config_sha256": o3c.config_hash(),
        "summary": {f"r{k}|{g}": {f"{c}_{s}": v for (c, s), v in row.items()}
                    for (k, g), row in summ.iterrows()}}, indent=1, default=float))
    print(df.groupby(["regime", "group"])[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
