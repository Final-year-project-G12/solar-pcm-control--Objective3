"""
scripts/m6b_postlock_checks.py -- POST-LOCK, EXPLORATORY (plan 8.3 step 8). Run only after M6.

Attaches (read-only) to the existing results/o3_test.lock, after recomputing the frozen-policy fingerprint and checking
it equals the one in the lock, then:
  (a) M7 exit gate: re-runs every M6 controller on every test year through the DEMO code path (o3/live.py FrameEnv +
      simulate) and checks the window delivery, shield hours, hard-limit hours and switches equal results/m6_test_results.csv
      exactly. This proves the demo reproduces the locked numbers; it does not change them.
  (b) O2-definition night serve metrics (tau_serve, serve_frac; o3/night.py, checked against O2 in tests/test_night.py)
      on the same years / controllers / water-only twin. M6 reported only the in-run tau variant; these columns are a
      supplement computed after the lock, with policies, config and years unchanged, and labelled EXPLORATORY.
Writes results/m6b_demo_reproduces_m6.csv, results/m6b_night_metrics_EXPLORATORY.csv, results/m6b_summary.json
Usage: python scripts/m6b_postlock_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3 import live                                # noqa: E402
from o3 import o2_bridge as ob                     # noqa: E402
from o3 import roles_o3                            # noqa: E402
from o3.controllers import B0AlwaysOn              # noqa: E402
from o3.evaluate import episode_metrics, evaluate  # noqa: E402

KEYS = ["E_del_window_kWh", "E_del_annual_kWh", "overheat_h", "hours_over_limit", "bypass_switches"]


def main():
    cfg = o3c.load()
    ob.verify_contract()
    fz = {k: frozen.load_frozen(k) for k in cfg["regimes"]["pcm"]}
    fp = ob.hashlib.sha256(json.dumps({str(k): v["fingerprint"] for k, v in fz.items()}, sort_keys=True).encode()).hexdigest()
    lock = roles_o3.attach_existing_lock(fp)
    print("attached to", lock["run_id"], "(EXPLORATORY post-lock checks)", flush=True)
    m6 = pd.read_csv(o3c.RESULTS / "m6_test_results.csv")
    regimes = cfg["regimes"]["pcm"] + cfg["regimes"]["water_only"]

    rep, night = [], []
    for k in regimes:
        yrs = roles_o3.accepted_test_years(k)
        ctrls = [B0AlwaysOn(), frozen.b1(k)] + (frozen.ppo_policies(k) if k in fz else [])
        for y in yrs:
            frame = roles_o3.load_weather(k, y, "test")
            runs = live.simulate(k, frame, ctrls)
            for c in ctrls:
                m = episode_metrics(runs[c.name])
                ref = m6[(m6.regime == k) & (m6.year == y) & (m6.controller == c.name)].iloc[0]
                rep.append({"regime": k, "year": y, "controller": c.name,
                            **{f"{x}_demo": m[x] for x in KEYS}, **{f"{x}_m6": ref[x] for x in KEYS},
                            "identical": bool(all(np.isclose(m[x], ref[x], rtol=0, atol=1e-9) for x in KEYS))})
        for c in ctrls:
            night.append(evaluate(c, k, yrs, "test", night=True))
        print(f"r{k} done", flush=True)

    rep = pd.DataFrame(rep)
    rep["o3_config_sha256"] = o3c.config_hash()
    rep.to_csv(o3c.RESULTS / "m6b_demo_reproduces_m6.csv", index=False)
    nt = pd.concat(night, ignore_index=True)
    nt["label"] = "EXPLORATORY post-lock supplement (policies, config, years unchanged)"
    nt["lock_run_id"], nt["o3_config_sha256"] = lock["run_id"], o3c.config_hash()
    nt.to_csv(o3c.RESULTS / "m6b_night_metrics_EXPLORATORY.csv", index=False)
    nt["group"] = nt.controller.str.replace(r"_seed\d+$", "", regex=True)
    cols = ["tau_serve_o2_mean_h", "twin_tau_serve_o2_mean_h", "gain_tau_serve_o2_h", "serve_frac_o2_mean"]
    tbl = nt.groupby(["regime", "group"])[cols].mean()
    summ = {"lock_run_id": lock["run_id"], "fingerprint": fp, "label": "EXPLORATORY",
            "o3_config_sha256": o3c.config_hash(),
            "demo_reproduces_m6_all": bool(rep.identical.all()), "n_rows_compared": len(rep),
            "same_E_del_window_as_m6": bool(np.allclose(nt.E_del_window_kWh.to_numpy(),
                                                         m6.set_index(["regime", "controller", "year"]).loc[
                                                             list(zip(nt.regime, nt.controller, nt.year)),
                                                             "E_del_window_kWh"].to_numpy(), rtol=0, atol=1e-9)),
            "night": {f"r{k}|{g}": {c: float(v) for c, v in row.items()} for (k, g), row in tbl.iterrows()}}
    (o3c.RESULTS / "m6b_summary.json").write_text(json.dumps(summ, indent=1))
    print("demo reproduces M6:", summ["demo_reproduces_m6_all"], f"({len(rep)} rows);",
          "night run E_del equals M6:", summ["same_E_del_window_as_m6"])
    print(tbl.round(3).to_string())


if __name__ == "__main__":
    main()
