"""
scripts/m4_select.py -- milestone M4 exit: choose the reward-weight set per regime and freeze the policy.

For every (regime, w2, w4) of the Q2 grid: the best checkpoint of each seed (already chosen on the validation years
by the lexicographic rule in m4_train_ppo.py), then the mean and spread over seeds. The weight set is chosen by the
same lexicographic rule applied to the SEED MEANS (so a lucky seed cannot carry a weight set). DIAGNOSTIC runs
(w3 override) are reported but never eligible. Every weight set tried is written out (Q2: report all).

Freezing: results/frozen/r{k}/ gets the chosen set's seed models, the normalisation file, and frozen.json with the
SHA-256 of each policy, the normalisation file, the O3 config and the git commit; m6 refuses to run without it.
Usage: python scripts/m4_select.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o3 import config as o3c                    # noqa: E402
from o3 import o2_bridge as ob                  # noqa: E402
from o3.evaluate import lexicographic_best       # noqa: E402

KEYS = ["E_del_window_kWh", "g_window_kWh", "E_del_annual_kWh", "overheat_h", "shield_water_h", "shield_pcm_h",
        "guard_hours", "hours_over_limit", "bypass_hours", "bypass_switches", "t_pcm_max_C", "t_water_max_C", "reward"]


def git_commit() -> dict:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ob.UK_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "."], cwd=ob.UK_ROOT, text=True).strip())
        return {"commit": h, "uncommitted_changes": dirty}
    except Exception as e:                                   # noqa: BLE001
        return {"commit": None, "error": repr(e)}


def main():
    cfg = o3c.load()
    tie = cfg["selection"]["delivery_tie_kWh_per_yr"]
    rows = []
    for bj in sorted((o3c.RESULTS / "ppo").glob("r*/*/seed*/best.json")):
        b = json.loads(bj.read_text())
        rows.append({"regime": b["regime"], "set": bj.parent.parent.name, "w2": b["w2"], "w4": b["w4"],
                     "w3": b["weights"]["w3"], "diagnostic": b.get("diagnostic_w3_override") is not None,
                     "seed": b["seed"], "ckpt": b["best"]["ckpt"], "steps": b["best"]["steps"],
                     "model": str(bj.parent / "best_model.zip"), **{k: b["best"][k] for k in KEYS}})
    runs = pd.DataFrame(rows)
    runs.to_csv(o3c.RESULTS / "m4_runs.csv", index=False)
    agg = runs.groupby(["regime", "set", "w2", "w4", "w3", "diagnostic"])[KEYS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = runs.groupby(["regime", "set"]).size().reindex(pd.MultiIndex.from_frame(agg[["regime", "set"]])).to_numpy()
    agg.to_csv(o3c.RESULTS / "m4_weight_sets.csv", index=False)

    frozen_all = {}
    for k, g in agg[~agg.diagnostic].groupby("regime"):
        cands = [{**r, "E_del_window_kWh": r["E_del_window_kWh_mean"], "overheat_h": r["overheat_h_mean"],
                  "hours_over_limit": r["hours_over_limit_mean"], "bypass_switches": r["bypass_switches_mean"]}
                 for r in g.to_dict("records")]
        best = lexicographic_best(cands, tie)
        seeds = runs[(runs.regime == k) & (runs.set == best["set"])]
        primary = lexicographic_best(seeds.to_dict("records"), tie)
        out = o3c.RESULTS / "frozen" / f"r{k}"
        out.mkdir(parents=True, exist_ok=True)
        pols = {}
        for r in seeds.to_dict("records"):
            dst = out / f"policy_seed{r['seed']}.zip"
            shutil.copy(r["model"], dst)
            pols[f"seed{r['seed']}"] = {"file": dst.name, "sha256": ob.sha256_file(dst), "checkpoint": r["ckpt"]}
        shutil.copy(o3c.normalisation_path(k), out / "normalisation.json")
        frozen = {"regime": int(k), "weight_set": best["set"], "w2": best["w2"], "w4": best["w4"], "w3": best["w3"],
                  "validation_seed_means": {kk: best[f"{kk}_mean"] for kk in KEYS},
                  "policies": pols, "primary_seed": int(primary["seed"]),
                  "normalisation_sha256": ob.sha256_file(out / "normalisation.json"),
                  "o3_config_sha256": o3c.config_hash(), "contract_sha256": ob.sha256_file(ob.CONTRACT_PATH),
                  "git": git_commit(), "rule": f"plan 8.3 lexicographic on seed means, tie {tie} kWh/yr"}
        frozen["fingerprint"] = ob.hashlib.sha256(json.dumps(
            [p["sha256"] for p in pols.values()] + [frozen["normalisation_sha256"], frozen["o3_config_sha256"],
                                                   frozen["contract_sha256"], str(frozen["git"].get("commit"))]).encode()).hexdigest()
        (out / "frozen.json").write_text(json.dumps(frozen, indent=1, default=float))
        frozen_all[int(k)] = frozen
        print(f"r{k}: chosen {best['set']}  E_win {best['E_del_window_kWh']:.1f}  overheat {best['overheat_h']:.0f} h  "
              f"switches {best['bypass_switches']:.0f}  primary seed {frozen['primary_seed']}")
    cols = ["regime", "set", "diagnostic", "E_del_window_kWh_mean", "E_del_window_kWh_std", "overheat_h_mean",
            "overheat_h_std", "guard_hours_mean", "bypass_switches_mean", "hours_over_limit_mean"]
    print(agg[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
