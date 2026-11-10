"""
Phase O3-4 re-evaluation — recompute the comparison table from SAVED models.

Use this after fixing the eval harness (shield/safety info keys) to refresh
results/phase4_comparison_r{regime}.csv WITHOUT retraining. It loads each
algorithm's best_model checkpoint and re-runs it on the validation years,
then re-runs the B0/B1/B2 baselines.

Run from the objective3 root:
    python scripts/phase4_reeval.py --regime 0
"""

import argparse
import pathlib
import sys
import csv

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.environment.gym_wrapper import VAL_YEARS
from scripts.phase4_train import (
    collect_baselines, run_rl_eval, run_recurrent_eval,
    run_maskable_eval, run_sac_eval, print_comparison, RESULTS_DIR,
)

# algo key -> (results subdir, loader)
ALGO_DIRS = {
    "ppo":          "ppo",
    "dqn":          "dqn",
    "recurrent":    "recurrent_ppo",
    "qrdqn":        "qrdqn",
    "maskable":     "maskable_ppo",
    "sac":          "sac",
}


def _load(algo: str, regime_id: int):
    base = RESULTS_DIR / ALGO_DIRS[algo] / f"r{regime_id}"
    best = base / "best_model" / "best_model"
    final = base / f"{ALGO_DIRS[algo]}_final"
    path = best if best.with_suffix(".zip").exists() else final
    if not path.with_suffix(".zip").exists():
        return None
    if algo == "recurrent":
        from sb3_contrib import RecurrentPPO; return RecurrentPPO.load(str(path))
    if algo == "qrdqn":
        from sb3_contrib import QRDQN; return QRDQN.load(str(path))
    if algo == "maskable":
        from sb3_contrib import MaskablePPO; return MaskablePPO.load(str(path))
    if algo == "ppo":
        from stable_baselines3 import PPO; return PPO.load(str(path))
    if algo == "dqn":
        from stable_baselines3 import DQN; return DQN.load(str(path))
    if algo == "sac":
        from stable_baselines3 import SAC; return SAC.load(str(path))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--algo", type=str, nargs="+",
                    default=["ppo", "dqn", "recurrent", "qrdqn", "maskable"],
                    choices=list(ALGO_DIRS) + ["all"])
    args = ap.parse_args()
    algos = list(ALGO_DIRS) if "all" in args.algo else args.algo

    print("=" * 70)
    print(f"Phase O3-4 RE-EVAL (corrected harness) — regime {args.regime}")
    print("=" * 70)

    rows = []
    print("\n--- Baselines (B0, B1, B2) ---")
    rows += collect_baselines(args.regime, VAL_YEARS)

    for algo in algos:
        m = _load(algo, args.regime)
        if m is None:
            print(f"  [skip] no saved model for {algo}")
            continue
        name = {"ppo":"PPO","dqn":"DQN","recurrent":"RecurrentPPO",
                "qrdqn":"QRDQN","maskable":"MaskablePPO","sac":"SAC"}[algo]
        print(f"\n--- {name} (re-eval) ---")
        if algo == "recurrent":
            rows += run_recurrent_eval(m, name, args.regime, VAL_YEARS)
        elif algo == "maskable":
            rows += run_maskable_eval(m, name, args.regime, VAL_YEARS)
        elif algo == "sac":
            rows += run_sac_eval(m, name, args.regime, VAL_YEARS)
        else:
            rows += run_rl_eval(
                lambda obs: int(m.predict(obs.astype("float32"),
                                          deterministic=True)[0]),
                name, args.regime, VAL_YEARS)

    print_comparison(rows)
    out = RESULTS_DIR / f"phase4_comparison_r{args.regime}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n[INFO] Corrected results -> {out}")


if __name__ == "__main__":
    main()
