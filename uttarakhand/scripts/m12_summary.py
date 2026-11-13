"""
scripts/m12_summary.py -- collect the post-lock extension study (validation 2021-2023 only) with the M4 / baseline
reference points into results/m12_extensions.csv (one row per run) and results/m12_extensions_summary.csv
(mean / SD over seeds). Every row's score is the run's best checkpoint on the validation years, chosen by the same
lexicographic rule as M4. DIAGNOSTIC rows (w3 below the dominance rule) are flagged and never selectable.
Usage: python scripts/m12_summary.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402

KEYS = ["E_del_window_kWh", "E_del_annual_kWh", "overheat_h", "hours_over_limit", "bypass_hours", "bypass_switches"]


def main():
    R = o3c.RESULTS
    rows = []
    for p in sorted((R / "ext").glob("*/r*/*/seed*/best.json")):
        b = json.loads(p.read_text())
        rows.append({"study": b["study"], "regime": b["regime"], "algo": b["algo"].upper(), "w2": b["w2"], "w4": b["w4"],
                     "w3": b["weights"]["w3"], "obs": b["obs_version"], "domain_randomize": b["domain_randomize"],
                     "diagnostic": b.get("diagnostic_w3_override") is not None, "seed": b["seed"],
                     "best_steps": b["best"]["steps"], **{k: b["best"][k] for k in KEYS}})
    for p in sorted((R / "ppo").glob("r*/*/seed*/best.json")):            # M4 runs (frozen study), for reference
        b = json.loads(p.read_text())
        rows.append({"study": "m4", "regime": b["regime"], "algo": "PPO", "w2": b["w2"], "w4": b["w4"],
                     "w3": b["weights"]["w3"], "obs": "V1", "domain_randomize": False, "diagnostic": False,
                     "seed": b["seed"], "best_steps": b["best"]["steps"], **{k: b["best"][k] for k in KEYS}})
    for k in range(4):                                                    # baselines on the same years
        m3 = pd.read_csv(R / f"m3_baselines_r{k}.csv")
        m3 = m3[(m3.purpose == "validate") & m3.controller.isin(["B0", "B1"])]
        for c, g in m3.groupby("controller"):
            rows.append({"study": "baseline", "regime": k, "algo": c, "diagnostic": False, "seed": 0,
                         **{x: g[x].mean() for x in KEYS}})
        b2 = R / f"m9_b2_summary_r{k}.csv"
        if b2.exists():
            g = pd.read_csv(b2)
            g = g[g.controller == "B2_perfect"]
            rows.append({"study": "baseline", "regime": k, "algo": "B2", "diagnostic": False, "seed": 0,
                         **{x: g[x].mean() for x in KEYS}})
    d = pd.DataFrame(rows).assign(o3_config_sha256=o3c.config_hash(), label="validation 2021-2023; extensions post-lock")
    d.to_csv(R / "m12_extensions.csv", index=False)
    grp = ["study", "regime", "algo", "w2", "w4", "w3", "obs", "domain_randomize", "diagnostic"]
    s = d.fillna({"w2": -1, "w4": -1, "w3": -1, "obs": "-", "domain_randomize": False}).groupby(grp)[KEYS].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index().assign(n_seeds=d.fillna({"w2": -1, "w4": -1, "w3": -1, "obs": "-", "domain_randomize": False})
                               .groupby(grp).size().to_numpy(), o3_config_sha256=o3c.config_hash())
    s.to_csv(R / "m12_extensions_summary.csv", index=False)
    print(s[["study", "regime", "algo", "w2", "w3", "obs", "domain_randomize", "diagnostic", "E_del_window_kWh_mean",
             "E_del_window_kWh_std", "overheat_h_mean", "bypass_switches_mean", "n_seeds"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
