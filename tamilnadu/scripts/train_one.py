"""
Phase O3-10 — universal single-run trainer for the research-extension studies:
    (a) Pareto sweep over the shield-penalty weight (reward-weight x seed)
    (b) sub-hourly control-step ablation
    (c) regime 1/2 coverage
    (d) forecast observation (V2) + training-time domain randomisation

One process per run, torch pinned to few threads so several can run truly in
parallel (12 logical CPUs here) without oversubscription.

Usage:
    python scripts/train_one.py --algo sac --regime 0 --steps 300000 --seed 42 \
        --shield-weight 10 --obs-version V2 --control-step-min 60 \
        --domain-randomize --tag pareto_w10_s42

Writes:
    results/<algo>/r<regime>_<tag>/              (checkpoints, best_model)
    results/phase10_<tag>.csv                    (validation-year eval rows)
"""

import argparse
import pathlib
import sys
import time

import torch
torch.set_num_threads(2)

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.environment.gym_wrapper import VAL_YEARS

RESULTS_DIR = REPO_ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["sac", "dqn"])
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shield-weight", type=float, default=10.0)
    ap.add_argument("--obs-version", default="V2", choices=["V1", "V2", "V3"])
    ap.add_argument("--forecast-mode", default="perfect",
                    choices=["perfect", "smart", "none"])
    ap.add_argument("--control-step-min", type=int, default=60)
    ap.add_argument("--domain-randomize", action="store_true")
    ap.add_argument("--tag", required=True,
                    help="run tag; output dir = results/<algo>/r<regime>_<tag>")
    args = ap.parse_args()

    reward_weights = {"shield": args.shield_weight}
    out_dir = RESULTS_DIR / args.algo / f"r{args.regime}_{args.tag}"

    print(f"[train_one] {args.algo} r{args.regime} tag={args.tag} "
          f"steps={args.steps} seed={args.seed} shield_w={args.shield_weight} "
          f"obs={args.obs_version} forecast={args.forecast_mode} "
          f"ctrl={args.control_step_min}min "
          f"domain_rand={args.domain_randomize}")

    t0 = time.time()
    common = dict(
        regime_id=args.regime, total_timesteps=args.steps, seed=args.seed,
        out_dir=out_dir, reward_weights=reward_weights,
        obs_version=args.obs_version, forecast_mode=args.forecast_mode,
        control_step_min=args.control_step_min,
        domain_randomize=args.domain_randomize,
    )
    if args.algo == "sac":
        from o3.agents.sac_trainer import train_sac, evaluate_policy
        model = train_sac(**common)
        rows = evaluate_policy(model, args.regime, VAL_YEARS,
                               reward_weights=reward_weights,
                               obs_version=args.obs_version,
                               forecast_mode=args.forecast_mode,
                               control_step_min=args.control_step_min)
    else:
        from o3.agents.dqn_trainer import train_dqn, evaluate_policy
        model = train_dqn(**common)
        rows = evaluate_policy(model, args.regime, VAL_YEARS,
                               reward_weights=reward_weights,
                               obs_version=args.obs_version,
                               forecast_mode=args.forecast_mode,
                               control_step_min=args.control_step_min)

    for r in rows:
        r.update({"seed": args.seed, "shield_weight": args.shield_weight,
                  "obs_version": args.obs_version,
                  "forecast_mode": args.forecast_mode,
                  "control_step_min": args.control_step_min,
                  "domain_randomize": args.domain_randomize, "tag": args.tag,
                  "regime": args.regime, "algo": args.algo})

    import csv
    csv_path = RESULTS_DIR / f"phase10_{args.tag}_r{args.regime}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    el = (time.time() - t0) / 60
    mean_del = sum(r["total_delivered_kWh"] for r in rows) / len(rows)
    mean_sh = sum(r["total_shield_water"] + r["total_shield_pcm"] for r in rows) / len(rows)
    mean_rw = sum(r["total_reward"] for r in rows) / len(rows)
    print(f"[train_one] DONE {args.tag} in {el:.1f} min | "
          f"mean del={mean_del:.1f} kWh shields={mean_sh:.0f} reward={mean_rw:.0f} "
          f"-> {csv_path}")


if __name__ == "__main__":
    main()
