"""
train_one.py — the universal single-run CLI for the Generation-2 environment (plan section 19.2).

    python scripts/train_one.py --algo dqn --regime 1 --steps 300000 --seed 42 --shield-weight 1.7 \
           --obs-version V3 --control-step-min 60 --domain-randomize --tag my_run

Writes results/gen2/<tag>/ (final_model.zip, best_model/, eval_logs/, run_meta.json) and prints the validation-year table.
Defaults: V3 observation (13 contract fields + 3 h GHI look-ahead + next-hour demand), hourly control, domain
randomisation ON in training (never in validation), the regime's default shield weight.
One run = one process; --n-jobs / --n-envs are accepted for uniformity and unused (use scripts/gen2_study.py for many runs).
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args  # noqa: E402
from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.agents.gen2 import train_run, evaluate_run, gen2_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["dqn", "sac", "qrdqn", "ppo", "maskable", "recurrent"])
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shield-weight", type=float, default=None, help="default: the regime's value in pcm_env.W_SHIELD_BY_REGIME")
    ap.add_argument("--obs-version", default="V3", choices=["V1", "V2", "V3"])
    ap.add_argument("--control-step-min", type=int, default=60, choices=[60, 30, 20, 15, 10, 5])
    ap.add_argument("--domain-randomize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results-dir", default=None)
    add_parallel_args(ap)
    a = ap.parse_args()
    meta = train_run(a.algo, a.regime, a.steps, a.seed, a.shield_weight, a.obs_version, a.control_step_min,
                     a.domain_randomize, a.tag, a.results_dir)
    run_dir = gen2_dir(a.results_dir) / meta["tag"]
    for y in VAL_YEARS:
        r = evaluate_run(str(run_dir), y)
        print(f"  {meta['tag']} y={y}  E45={r['total_delivered_kWh']:.1f} kWh  window={r['E_del45_window_kWh']:.1f}  "
              f"shield_h={r['shield_water_h'] + r['shield_pcm_h']:.0f}  safety={r['n_safety_violations']}")


if __name__ == "__main__":
    main()
