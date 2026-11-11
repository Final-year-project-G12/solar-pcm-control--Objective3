"""
Phase O3-7 — Software-in-the-loop demo (Rajasthan).

Runs the policy against live, replayed or stub weather and prints a real-time action/state table.

    *** Software-in-the-loop simulation. Not field validation. ***
    All thermal state (T_top, T_pcm, f_melt) comes from the frozen O2 simulator, not from physical sensors.
    This cannot substitute for hardware-in-the-loop validation (Objective 4).

Run from the rajasthan O3 root:
    # Stub mode (no network, deterministic)
    python scripts/phase7_demo.py --mode stub --algo b1 --steps 48
    # Replay mode (recorded Rajasthan weather, validation year; test years are refused without the O3 lock)
    python scripts/phase7_demo.py --mode replay --year 2021 --algo ppo --regime 1 --steps 168
    # Live mode (internet; Open-Meteo at the chosen medoid: 0 Udaipur, 1 Nagaur, 2 Jaipur)
    python scripts/phase7_demo.py --mode live --algo ppo --regime 1 --steps 24 --interval 3600

--n-jobs / --n-envs are accepted for uniformity; the demo is one sequential simulation and uses neither.
Never mix live data into training. The demo never touches the O3 test years before the lock.
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.demo.live_adapter import LiveDemoRunner, DemoMode, BANNER, MEDOIDS  # noqa: E402
from o3.utils.cli import add_parallel_args  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="stub", choices=["stub", "replay", "live"])
    parser.add_argument("--algo", type=str, default="b1",
                        choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac", "b0", "b1", "b2"])
    parser.add_argument("--regime", type=int, default=1, choices=[0, 1, 2], help="O2 medoid: 0 Udaipur, 1 Nagaur, 2 Jaipur")
    parser.add_argument("--year", type=int, default=2021, help="replay year (train/validate roles only)")
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--interval", type=float, default=0.0, help="seconds between steps (live demo: 3600)")
    parser.add_argument("--model", type=str, default=None, help="explicit model .zip (default: results/<algo>/r<regime>)")
    add_parallel_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print(f"*** {BANNER} ***")
    print(f"Phase O3-7 — Demo | mode={args.mode} | algo={args.algo} | regime {args.regime} ({MEDOIDS[args.regime]['place']})")
    print("=" * 70)
    runner = LiveDemoRunner(algo=args.algo, regime_id=args.regime, mode=DemoMode(args.mode), replay_year=args.year,
                            model_path=args.model)
    frames = runner.run(n_steps=args.steps, interval_s=args.interval, verbose=True)
    delivered = sum(f["E_del45_Wh"] for f in frames) / 1000.0
    shield = sum(1 for f in frames if f["shield_override"])
    unsafe = sum(f["safety_violation"] for f in frames)
    print(f"\n[SUMMARY] {len(frames)} h | delivered to 45 C: {delivered:.2f} kWh | shield hours: {shield} | safety violations: {unsafe}")
    print(f"*** {BANNER} ***")
    print("[OK]   Phase O3-7 demo complete.")


if __name__ == "__main__":
    main()
