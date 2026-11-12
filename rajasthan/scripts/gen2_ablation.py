"""
gen2_ablation.py — what the Generation-2 ingredients are worth (plan 19.1 items 2 and 5; fills the gap CORRECTIONS.md recorded).

Three DQN runs, same regime / seed / shield weight / 300k steps, differing in ONE thing from the selected configuration:
    full         V3 (forecast) + training-time domain randomisation         [the study's run]
    no_forecast  V2 observation (no look-ahead)  + domain randomisation
    no_dr        V3 observation + NO domain randomisation
Train the two missing variants first (about 10 min each):
    python scripts/train_one.py --algo dqn --regime 1 --seed 42 --shield-weight 1.7 --obs-version V2
    python scripts/train_one.py --algo dqn --regime 1 --seed 42 --shield-weight 1.7 --obs-version V3 --no-domain-randomize
Then:  python scripts/gen2_ablation.py --regime 1 --n-jobs 6
Scores each on the validation years (nominal weather) and on the 13-scenario robustness suite (phase 5), so plot 16 can show
whether training-time randomisation tightens the robustness band. ONE seed per variant: a pointer, not a significance test.
Outputs: results/gen2/gen2_ablation_dqn_r<regime>.csv (per variant) and results/gen2/gen2_ablation_robustness_dqn_r<regime>.csv.
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()
from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.agents.gen2 import evaluate_run, evaluate_baseline_as, gen2_dir, default_tag, run_model_path  # noqa: E402
from o3.evaluation.robustness import run_robustness_suite  # noqa: E402
from o3.environment.pcm_env import default_w_shield  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 123])
    ap.add_argument("--algo", default="dqn")
    add_parallel_args(ap, envs=False)
    a = ap.parse_args()
    n_jobs, _ = resolve(a)
    w, g = default_w_shield(a.regime), gen2_dir()
    spec_of = {"full": ("V3", True), "no_forecast": ("V2", True), "no_dr": ("V3", False)}
    variants = {}                                         # (variant, seed) -> tag, only runs that exist
    for k, (ov, dr) in spec_of.items():
        for sd in a.seeds:
            t = default_tag(a.algo, a.regime, sd, w, ov, 60, dr)
            if (g / t / "run_meta.json").exists():
                variants[(k, sd)] = t
            else:
                print(f"[skip] {k} seed {sd}: not trained ({t}); see the header of this script")
    if not variants:
        sys.exit(1)
    rows = run_parallel(evaluate_run, [dict(run_dir=str(g / t), year=y) for t in variants.values() for y in VAL_YEARS], n_jobs=n_jobs)
    base = run_parallel(evaluate_baseline_as, [dict(baseline="b0", regime_id=a.regime, year=y) for y in VAL_YEARS], n_jobs=n_jobs)
    b0 = float(np.mean([b["total_delivered_kWh"] for b in base]))
    b0w = float(np.mean([b["E_del45_window_kWh"] for b in base]))
    b0s = float(np.mean([b["shield_water_h"] + b["shield_pcm_h"] for b in base]))
    out = []
    for (k, sd), t in variants.items():
        rs = [r for r in rows if r["tag"] == t]
        e, win = float(np.mean([r["total_delivered_kWh"] for r in rs])), float(np.mean([r["E_del45_window_kWh"] for r in rs]))
        sh = float(np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in rs]))
        out.append({"variant": k, "seed": sd, "tag": t, "E45_kWh": e, "pct_of_B0_delivery": 100 * e / b0, "pct_of_B0_window": 100 * win / b0w,
                    "shield_h": sh, "shield_cut_vs_B0_pct": 100 * (1 - sh / b0s), "violations": int(sum(r["n_safety_violations"] for r in rs)),
                    "switches": float(np.mean([r["action_switches"] for r in rs]))})
    p = g / f"gen2_ablation_{a.algo}_r{a.regime}.csv"
    with open(p, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out[0]))
        w_.writeheader()
        w_.writerows(out)
    print(f"{'variant':12s} {'seed':>5s} {'E45':>7s} {'%B0':>6s} {'win%B0':>7s} {'shield h':>8s} {'cut%':>6s} {'switch':>7s} {'viol':>5s}")
    for o in out:
        print(f"{o['variant']:12s} {o['seed']:5d} {o['E45_kWh']:7.1f} {o['pct_of_B0_delivery']:6.1f} {o['pct_of_B0_window']:7.1f} {o['shield_h']:8.0f} "
              f"{o['shield_cut_vs_B0_pct']:6.1f} {o['switches']:7.0f} {o['violations']:5d}")
    rob = []                                              # robustness for the first seed only (plot 16)
    sd0 = next(sd for (_, sd) in variants)
    for (k, sd), t in variants.items():
        if sd != sd0:
            continue
        rd = g / t
        m = json.loads((rd / "run_meta.json").read_text())
        spec = ("model", m["algo"], str(run_model_path(rd)), dict(obs_version=m["obs_version"], control_step_min=60))
        rob += run_robustness_suite(policy_name=k, regime_id=a.regime, years=VAL_YEARS[:2], policy_spec=spec, n_jobs=n_jobs, verbose=False)
    p2 = g / f"gen2_ablation_robustness_{a.algo}_r{a.regime}.csv"
    with open(p2, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rob[0]))
        w_.writeheader()
        w_.writerows(rob)
    print(f"-> {p}\n-> {p2}  (robustness: seed {sd0} only)")


if __name__ == "__main__":
    main()
