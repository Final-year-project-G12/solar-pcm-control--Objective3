"""
scripts/m13b_postlock_test.py -- POST-LOCK, EXPLORATORY. Run only after M6 and the m13 study.

The O3 one-shot test (results/o3_test.lock, run o3test_e6ea0abf636a) is spent; its numbers in results/m6_test_results.csv
are the official ones and are not replaced. This script attaches read-only to that lock (after checking the frozen
policies' fingerprint matches it) and scores the policies recommended by the m13 study -- PPO with the
smart-persistence forecast, V2 observation, trained with domain randomisation, best validation checkpoint per seed --
on the same test years, reset scenario and demand as M6, so they can be read next to M6's B0 / B1 / frozen PPO.
Because the choice of agent was made after the test years had been seen once, every number here is EXPLORATORY.
Writes results/m13b_postlock_test_EXPLORATORY.csv (per year) and results/m13b_postlock_test_summary_EXPLORATORY.csv
Usage: python scripts/m13b_postlock_test.py
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
from o3.controllers import SB3Policy               # noqa: E402
from o3.evaluate import evaluate                   # noqa: E402

KEYS = ["E_del_window_kWh", "E_del_annual_kWh", "g_window_kWh", "overheat_h", "hours_over_limit", "guard_hours",
        "bypass_switches", "t_water_max_C", "t_pcm_max_C"]
STUDY_TAG = {0: "ppo_w2_3_w4_0.3_V2_fc_smart_DR", 1: "ppo_w2_0_w4_0.3_V2_fc_smart_DR", 2: "ppo_w2_0_w4_0.3_V2_fc_smart_DR"}


def main():
    cfg = o3c.load()
    ob.verify_contract()
    fz = {k: frozen.load_frozen(k) for k in cfg["regimes"]["pcm"]}
    fp = ob.hashlib.sha256(json.dumps({str(k): v["fingerprint"] for k, v in fz.items()}, sort_keys=True).encode()).hexdigest()
    lock = roles_o3.attach_existing_lock(fp)
    print("attached to", lock["run_id"], "(EXPLORATORY post-lock scoring)", flush=True)
    m6 = pd.read_csv(o3c.RESULTS / "m6_test_results.csv")
    rows = []
    for k in cfg["regimes"]["pcm"]:
        yrs = roles_o3.accepted_test_years(k)
        runs = sorted((o3c.RESULTS / "ext" / "fcalgo" / f"r{k}" / STUDY_TAG[k]).glob("seed*/best_model.zip"))
        if len(runs) != 3:
            raise SystemExit(f"regime {k}: expected 3 seeds, found {len(runs)}")
        for p in runs:
            pol = SB3Policy(p, "ppo", name="PPO_smart", env_kwargs={"obs_version": "V2", "forecast_mode": "smart"})
            df = evaluate(pol, k, yrs, "test")
            df = df.assign(controller="PPO_smart", regime=k, seed=int(p.parent.name[4:]))
            rows.append(df)
            print(f"r{k} {p.parent.name}: E_win {df.E_del_window_kWh.mean():.1f} overheat {df.overheat_h.mean():.1f} "
                  f"over-limit {df.hours_over_limit.sum():.1f}", flush=True)
    new = pd.concat(rows, ignore_index=True)
    new.assign(o3_config_sha256=o3c.config_hash(), label="EXPLORATORY post-lock; official test = m6_test_results.csv") \
       .to_csv(o3c.RESULTS / "m13b_postlock_test_EXPLORATORY.csv", index=False)

    ref = m6[m6.regime.isin(cfg["regimes"]["pcm"])].copy()
    is_ppo = ref["controller"].str.startswith("PPO_seed")                 # M6 keeps the PPO seed in the name
    ref.loc[is_ppo, "seed"] = ref.loc[is_ppo, "controller"].str[len("PPO_seed"):].astype(int)
    ref.loc[is_ppo, "controller"] = "PPO_frozen"
    both = pd.concat([ref.assign(source="M6 locked test"), new.assign(source="m13b EXPLORATORY")], ignore_index=True)
    per_seed = both.groupby(["regime", "controller", "source", "seed"])[KEYS].mean().reset_index()
    s = per_seed.groupby(["regime", "controller", "source"]).agg(
        **{f"{x}_mean": (x, "mean") for x in KEYS}, E_del_window_kWh_sd=("E_del_window_kWh", "std"),
        hours_over_limit_max=("hours_over_limit", "max"), n_seeds=("seed", "nunique")).reset_index()
    b0 = s[s.controller == "B0"].set_index("regime")
    s["delivery_pct_of_B0"] = 100 * s.E_del_window_kWh_mean / s.regime.map(b0.E_del_window_kWh_mean)
    s["overheat_cut_pct_vs_B0"] = 100 * (1 - s.overheat_h_mean / s.regime.map(b0.overheat_h_mean))
    s.assign(o3_config_sha256=o3c.config_hash(), label="EXPLORATORY post-lock") \
     .to_csv(o3c.RESULTS / "m13b_postlock_test_summary_EXPLORATORY.csv", index=False)
    print(s[["regime", "controller", "source", "E_del_window_kWh_mean", "E_del_window_kWh_sd", "delivery_pct_of_B0",
             "g_window_kWh_mean", "overheat_h_mean", "overheat_cut_pct_vs_B0", "hours_over_limit_max",
             "bypass_switches_mean", "n_seeds"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
