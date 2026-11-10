"""
Phase O3-7 — Software-in-the-loop live demo.

Runs the trained policy against live or replayed weather and prints
a real-time action/state table.

⚠️  This is a SIMULATION-ONLY demonstration.
    All thermal state (T_water, T_pcm, f_melt) comes from the frozen
    simulator, not from physical sensors. This cannot substitute for
    hardware-in-the-loop validation (Objective 4).

Run from the objective3 root:
    # Stub mode (no network, deterministic)
    python scripts/phase7_demo.py --mode stub --algo recurrent --steps 48

    # Replay mode (historical Tamil Nadu weather)
    python scripts/phase7_demo.py --mode replay --year 2021 --algo recurrent

    # Live mode (requires internet — fetches Open-Meteo for Madurai, TN)
    python scripts/phase7_demo.py --mode live --algo recurrent --steps 24
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.demo.live_adapter import LiveDemoRunner, DemoMode


def get_replay_csv(year: int, regime_id: int) -> pathlib.Path:
    """Find the O2 hourly weather CSV for replay."""
    base = (_O2_ROOT / "data" / "weather" /
            f"weather_regime_tamilnadu_cluster{regime_id}_hourly_{year}.csv")
    return base if base.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   type=str,  default="stub",
                        choices=["stub", "replay", "live"])
    parser.add_argument("--algo",   type=str,  default="recurrent",
                        choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"])
    parser.add_argument("--regime", type=int,  default=0)
    parser.add_argument("--year",   type=int,  default=2021,
                        help="Year for replay mode (must be in val set 2021-2023)")
    parser.add_argument("--steps",  type=int,  default=72,
                        help="Number of hourly steps to simulate (default 72 = 3 days)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Seconds between steps for real-time display (0=fast)")
    args = parser.parse_args()

    if args.mode == "replay" and args.year in {2011,2012,2013,2014,2015}:
        print("[ERROR] Test years (2011-2015) cannot be used for demo.")
        sys.exit(1)

    replay_csv = get_replay_csv(args.year, args.regime) \
        if args.mode == "replay" else None
    if args.mode == "replay" and replay_csv is None:
        print(f"[WARN] Replay CSV not found for year {args.year} "
              f"regime {args.regime}; falling back to stub.")
        args.mode = "stub"

    print("=" * 70)
    print("Phase O3-7 — Software-in-the-loop Demo")
    print(f"Mode: {args.mode.upper()}  |  Algo: {args.algo}  |  "
          f"Regime: {args.regime}  |  Steps: {args.steps}")
    if args.mode == "live":
        print("⚠  SIMULATION ONLY — thermal state from frozen simulator, NOT sensors")
    print("=" * 70)
    print()

    runner = LiveDemoRunner(
        algo       = args.algo,
        regime_id  = args.regime,
        mode       = DemoMode(args.mode),
        replay_csv = replay_csv,
        log_dir    = REPO_ROOT / "results" / "demo_logs",
    )

    frames = runner.run(
        n_steps    = args.steps,
        interval_s = args.interval,
        verbose    = True,
    )

    # summary
    total_del   = sum(f["E_delivered_Wh"]   for f in frames) / 1000.0
    total_unmet = sum(f["E_unmet_Wh"]       for f in frames) / 1000.0
    n_shield    = sum(1 for f in frames if f["shield_override"])
    n_safety    = sum(1 for f in frames if f["safety_violation"])
    action_counts = {}
    for f in frames:
        action_counts[f["action"]] = action_counts.get(f["action"], 0) + 1

    print(f"\n--- {args.steps}-step demo summary ---")
    print(f"  Delivered:   {total_del:.2f} kWh")
    print(f"  Unmet:       {total_unmet:.2f} kWh")
    print(f"  Shield fires: {n_shield}")
    print(f"  Safety viol:  {n_safety}  {'⚠' if n_safety else '✓'}")
    print(f"  Actions:      {action_counts}")
    if args.mode != "stub":
        print(f"\n  Mode = {args.mode.upper()} — this is a simulation, "
              "NOT hardware validation.")

    print("\n[OK]   Phase O3-7 demo complete.")


if __name__ == "__main__":
    main()
