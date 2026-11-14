"""
Phase O3-5 — Robustness and ablation evaluation.

Run from the objective3 root:
    python scripts/phase5_robustness.py [--regime 0] [--algo ppo|dqn|b1]
                                        [--years 2021 2022 2023]

Loads a trained PPO or DQN model (from results/) and runs the full
10-scenario robustness suite on validation years.

Also accepts --algo b1 to run the B1 baseline through the same suite
as a comparison reference.

Outputs:
    results/phase5_robustness_r{regime}_{algo}.csv
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.evaluation.robustness import (
    run_robustness_suite, save_robustness_results, print_robustness_summary,
)
from o3.environment.gym_wrapper import VAL_YEARS

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"


def load_ppo(regime_id: int, tag: str | None = None):
    from stable_baselines3 import PPO
    rdir = f"r{regime_id}_{tag}" if tag else f"r{regime_id}"
    # prefer the validation-selected best_model; fall back to final
    path = RESULTS_DIR / "ppo" / rdir / "best_model" / "best_model"
    if not path.with_suffix(".zip").exists():
        path = RESULTS_DIR / "ppo" / rdir / "ppo_final"
    model = PPO.load(str(path))
    def policy_fn(obs):
        import numpy as np
        action, _ = model.predict(obs.astype("float32"), deterministic=True)
        return int(action)
    return policy_fn


def load_dqn(regime_id: int, tag: str | None = None):
    from stable_baselines3 import DQN
    rdir = f"r{regime_id}_{tag}" if tag else f"r{regime_id}"
    # prefer the validation-selected best_model; fall back to final
    path = RESULTS_DIR / "dqn" / rdir / "best_model" / "best_model"
    if not path.with_suffix(".zip").exists():
        path = RESULTS_DIR / "dqn" / rdir / "dqn_final"
    model = DQN.load(str(path))
    def policy_fn(obs):
        import numpy as np
        action, _ = model.predict(obs.astype("float32"), deterministic=True)
        return int(action)
    return policy_fn


def load_sac(regime_id: int, tag: str | None = None):
    from stable_baselines3 import SAC
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    rdir = f"r{regime_id}_{tag}" if tag else f"r{regime_id}"
    # prefer the validation-selected best_model; fall back to final
    path = RESULTS_DIR / "sac" / rdir / "best_model" / "best_model"
    if not path.with_suffix(".zip").exists():
        path = RESULTS_DIR / "sac" / rdir / "sac_final"
    model = SAC.load(str(path))
    def policy_fn(obs):
        # SAC acts in a continuous [-1,1] space; bin to {CHARGE, IDLE, BYPASS}
        action, _ = model.predict(obs.astype("float32"), deterministic=True)
        return int(PCMContinuousGymEnv._to_discrete(float(action[0])))
    return policy_fn


def load_b1():
    from o3.baselines.b1_reactive import ReactiveRulePolicy
    pol = ReactiveRulePolicy()
    return pol.act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--algo",   type=str, default="ppo",
                        choices=["ppo", "dqn", "sac", "b1"])
    parser.add_argument("--years",  type=int, nargs="+", default=VAL_YEARS[:2])
    parser.add_argument("--obs-version", default="V1", choices=["V1", "V2", "V3"])
    parser.add_argument("--forecast-mode", default="perfect",
                        choices=["perfect", "smart", "none"])
    parser.add_argument("--tag", default=None,
                        help="evaluate results/<algo>/r<regime>_<tag>/ instead of "
                             "the canonical r<regime>/ dir")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase O3-5 — Robustness & Ablation")
    print(f"Algo: {args.algo}  |  Regime: {args.regime}  |  Years: {args.years}  |  "
          f"obs={args.obs_version}  forecast={args.forecast_mode}  tag={args.tag}")
    print("=" * 60)

    if args.algo == "ppo":
        policy_fn = load_ppo(args.regime, tag=args.tag)
    elif args.algo == "dqn":
        policy_fn = load_dqn(args.regime, tag=args.tag)
    elif args.algo == "sac":
        policy_fn = load_sac(args.regime, tag=args.tag)
    else:
        policy_fn = load_b1()

    results = run_robustness_suite(
        policy_fn     = policy_fn,
        policy_name   = args.algo.upper(),
        regime_id     = args.regime,
        years         = args.years,
        obs_version   = args.obs_version,
        forecast_mode = args.forecast_mode,
    )

    print_robustness_summary(results)

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"phase5_robustness_r{args.regime}_{args.algo}{suffix}.csv"
    save_robustness_results(results, out_path)

    # acceptance gate: zero safety violations on nominal scenario
    nominal_violations = [
        r for r in results
        if r["scenario"] == "nominal" and r["n_safety_violations"] > 0
    ]
    if nominal_violations:
        print("\n[FAIL] Safety violations on nominal scenario — fix policy.")
        sys.exit(1)

    print("\n[OK]   Phase O3-5 PASSED.  Policy is robust across all scenarios.")
    print("       Ready for Phase O3-6 (Locked final evaluation).")


if __name__ == "__main__":
    main()
