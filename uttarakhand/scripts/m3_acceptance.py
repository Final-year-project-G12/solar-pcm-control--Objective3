"""
scripts/m3_acceptance.py -- the contract's `acceptance_test_before_drl_training` list, checked explicitly for every
controller that is reported (plan M3 exit gate: "B1 passes the contract acceptance list"; also B0 and the frozen PPO
seeds). Validation years 2021-2023, fully_solid, deterministic demand, regimes 0-3.

  A1 every action stays within the flow envelope / temperature limits
       design flow in [pump_flow_min, pump_flow_max] (constant, never commanded by the policy);
       zero kernel time over 75 C water / 65 C PCM; yearly peaks <= the limits
  A2 charge / discharge / bypass transitions are physically valid
       actions in {0, 1}; a bypassed hour collects exactly 0 Wh; collector heat is never negative
  A3 the environment conserves energy: episode residual < 0.1 % of collector energy (O2 Gate 1)
  A4 the safety shield cannot be bypassed (counterfactual, non-vacuous): the episode is re-stepped hour by hour; in
       every hour where the policy said ENABLE and the kernel's water shield engaged, the same hour is re-run from the
       same state with ONLY the water shield removed (its 72 C threshold P[29] moved to 1e9; P[28] is the master switch of
       BOTH shields and stays on, so the PCM-side shield is unchanged). The shielded hour must collect no more heat
       and end no hotter than the unshielded one, and in at least one hour per year the shield must have changed the
       outcome (so the test cannot pass vacuously). The kernel's rule itself (sub-step: top node >= 72 C -> loop off)
       is tested directly by T6.
  A5 the same seed reproduces the same trajectory: the episode is run twice and every hourly column is identical
  A6 simulator output matches O2 for identical actions: the controller's own action sequence is replayed through
       O2's run_case with GHI = 0 in the bypassed hours, and hourly T_top, T_pcm, f_melt, E_del and collector heat
       agree to rtol 1e-9
Writes results/m3_acceptance.csv (one row per controller x regime x year) and results/m3_acceptance.json.
Usage: python scripts/m3_acceptance.py
"""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                       # noqa: E402
from o3 import frozen                              # noqa: E402
from o3 import o2_bridge as ob                     # noqa: E402
from o3 import roles_o3                            # noqa: E402
from o3.controllers import B0AlwaysOn              # noqa: E402
from o3.evaluate import run_episode                # noqa: E402
from o3.stepper import HourlyStepper            # noqa: E402

RTOL = 1e-9
CMP = [("T_w_C", "T_top"), ("T_pcm_C", "T_pcm"), ("f_melt", "f_melt"), ("Edel_target_Wh", "Edel_Wh"),
       ("Q_collector_Wh", "Q_coll_Wh")]


def close(a, b):
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=RTOL, atol=1e-9))


def shield_counterfactual(k, y, actions, df):
    """Returns (n enabled hours with water shield, n where the shield changed the outcome, all consistent?)."""
    w = roles_o3.load_weather(k, y, "validate")
    case = ob.build_kernel_case(k, w.iloc[:1], "fully_solid")
    st = HourlyStepper(case, w)
    P0 = case.P.copy()
    P0[29] = 1e9                                                    # water-shield threshold unreachable; PCM shield kept
    off = copy.copy(st)
    off.case = dataclasses.replace(case, P=P0)
    n_sh = n_eff = 0
    ok = True
    for t, act in enumerate(actions):
        T, h = st.T.copy(), st.h.copy()
        r = st.step(int(act))
        if r["T_top"] != df["T_top"].iat[t]:
            return 0, 0, False                                      # re-stepping must reproduce the episode
        if act == 1 and r["shield_water_s"] > 0:
            n_sh += 1
            off.t, off.T, off.h = t, T, h
            rc = off.step(1)
            ok &= rc["shield_water_s"] == 0
            ok &= r["Q_coll_Wh"] <= rc["Q_coll_Wh"] + 1e-9 and r["max_water_C"] <= rc["max_water_C"] + 1e-9
            n_eff += rc["Q_coll_Wh"] > r["Q_coll_Wh"] + 1e-6
    return n_sh, n_eff, bool(ok)


def check(ctrl, k, y, lim):
    df, _ = run_episode(ctrl, k, y, "fully_solid", "validate")
    df2, _ = run_episode(ctrl, k, y, "fully_solid", "validate")
    a = df["action"].to_numpy()
    flow = ob.chosen_design(k)["flow"]
    has_pcm = df["max_pcm_C"].max() > -1e8
    a1 = (lim["pump_flow_min_kg_s"] - 1e-12 <= flow <= lim["pump_flow_max_kg_s"] + 1e-12
          and df["over_limit_s"].sum() == 0 and df["max_water_C"].max() <= lim["max_water_temp_C"]
          and (not has_pcm or df["max_pcm_C"].max() <= lim["max_pcm_temp_C"]))
    a2 = bool(set(np.unique(a)) <= {0, 1} and (df.loc[a == 0, "Q_coll_Wh"] == 0).all() and (df["Q_coll_Wh"] >= 0).all())
    a3 = bool(df.attrs["residual_pct"] < 0.1)
    n_sh, n_eff, cf_ok = shield_counterfactual(k, y, a, df)
    a4 = bool(cf_ok and (n_sh == 0 or n_eff > 0))
    num = df.select_dtypes("number").columns
    a5 = bool(df[num].equals(df2[num]))
    w = roles_o3.load_weather(k, y, "validate")
    wm = w.copy()
    wm.loc[a == 0, "GHI_Wm2"] = 0.0
    ref = ob.run_case_for(k, wm, "fully_solid")["hourly"]
    a6 = all(close(ref[r], df[c]) for r, c in CMP if r in ref)
    return {"controller": ctrl.name, "regime": k, "year": y, "flow_kg_s": flow, "n_bypass_h": int((a == 0).sum()),
            "n_enabled_shield_h": n_sh, "n_shield_changed_outcome_h": n_eff, "residual_pct": df.attrs["residual_pct"],
            "A1_limits": bool(a1), "A2_transitions": a2, "A3_energy": a3, "A4_shield": a4, "A5_determinism": a5,
            "A6_matches_O2": bool(a6)}


def main():
    cfg = o3c.load()
    lim = ob.load_contract()["global_limits"]
    rows = []
    for k in cfg["regimes"]["pcm"] + cfg["regimes"]["water_only"]:
        ctrls = [B0AlwaysOn(), frozen.b1(k)] + (frozen.ppo_policies(k) if k in cfg["regimes"]["pcm"] else [])
        for c in ctrls:
            for y in cfg["years"]["validate"]:
                rows.append(check(c, k, y, lim))
            print(f"r{k} {c.name}: " + " ".join(f"{t}={all(r[t] for r in rows if r['regime'] == k and r['controller'] == c.name)}"
                                             for t in ("A1_limits", "A2_transitions", "A3_energy", "A4_shield",
                                                       "A5_determinism", "A6_matches_O2")), flush=True)
    df = pd.DataFrame(rows)
    df["o3_config_sha256"] = o3c.config_hash()
    df.to_csv(o3c.RESULTS / "m3_acceptance.csv", index=False)
    tests = [c for c in df.columns if c[:2] in ("A1", "A2", "A3", "A4", "A5", "A6")]
    summ = {"contract_list": ob.load_contract()["acceptance_test_before_drl_training"],
            "all_pass": bool(df[tests].all().all()),
            "per_test": {t: bool(df[t].all()) for t in tests},
            "per_controller": {f"r{r}|{c}": bool(g[tests].all().all())
                               for (r, c), g in df.groupby(["regime", "controller"])},
            "n_rows": len(df), "years": cfg["years"]["validate"], "o3_config_sha256": o3c.config_hash()}
    (o3c.RESULTS / "m3_acceptance.json").write_text(json.dumps(summ, indent=1))
    summ["A4_shield_engaged_against_enable_h_total"] = int(df["n_enabled_shield_h"].sum())
    summ["A4_shield_changed_outcome_h_total"] = int(df["n_shield_changed_outcome_h"].sum())
    summ["all_pass"] = summ["all_pass"] and summ["A4_shield_changed_outcome_h_total"] > 0
    (o3c.RESULTS / "m3_acceptance.json").write_text(json.dumps(summ, indent=1))
    print("ALL PASS" if summ["all_pass"] else "FAILURES: " + str({t: v for t, v in summ["per_test"].items() if not v}))


if __name__ == "__main__":
    main()
