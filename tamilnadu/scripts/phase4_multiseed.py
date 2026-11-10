"""
Phase O3-4 — Multi-seed training for statistical comparison.

Trains each algorithm with N seeds, computes mean ± std across seeds,
and produces a publishable comparison table with confidence intervals.

Run from the objective3 root:
    python scripts/phase4_multiseed.py [--regime 0] [--steps 300000] [--seeds 42 0 123]

This is the scientifically rigorous version for the thesis.
Single-seed results (from phase4_train.py) are for development only.
"""

import argparse
import pathlib
import sys
import csv
import json
import numpy as np

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.environment.gym_wrapper   import VAL_YEARS
from o3.environment.pcm_env       import PCMEnv
from o3.baselines.b0_fixed         import FixedChargePolicy, run_episode as run_b0
from o3.baselines.b1_reactive      import ReactiveRulePolicy, run_episode as run_b1
from o3.baselines.b2_forecast      import ForecastAwarePolicy, run_episode as run_b2
from o3.baselines.b4_random        import RandomSafePolicy, run_episode as run_b4

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALGO_MAP = {
    "ppo":      ("PPO",          "o3.agents.ppo_trainer",          "train_ppo"),
    "dqn":      ("DQN",          "o3.agents.dqn_trainer",          "train_dqn"),
    "recurrent":("RecurrentPPO", "o3.agents.recurrent_ppo_trainer","train_recurrent_ppo"),
    "qrdqn":    ("QRDQN",        "o3.agents.qrdqn_trainer",        "train_qrdqn"),
    "maskable": ("MaskablePPO",  "o3.agents.maskable_ppo_trainer", "train_maskable_ppo"),
    "sac":      ("SAC",          "o3.agents.sac_trainer",          "train_sac"),
}


# ---------------------------------------------------------------------------
# Deterministic baselines (run once — no seed variation)
# ---------------------------------------------------------------------------

def collect_baselines(regime_id: int, years: list[int]) -> list[dict]:
    env = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
    rows = []
    for name, pol, runner in [
        ("B0_fixed",    FixedChargePolicy(),   run_b0),
        ("B1_reactive", ReactiveRulePolicy(),  run_b1),
        ("B2_forecast", ForecastAwarePolicy(), run_b2),
        ("B4_random",   RandomSafePolicy(),    run_b4),
    ]:
        for year in years:
            r = runner(env, year=year)
            rows.append({**r, "seed": "N/A", "algo_key": name})
    return rows


# ---------------------------------------------------------------------------
# RL eval helpers
# ---------------------------------------------------------------------------

def _eval_standard(model, regime_id, years):
    env = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
    rows = []
    for year in years:
        obs, _ = env.reset(year=year)
        done = False
        total_reward = total_del = total_unmet = 0.0
        sw = sp = sv = steps = 0
        while not done:
            action, _ = model.predict(obs.astype("float32"), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            sw += int(info.get("n_shield_water", 0))
            sp += int(info.get("n_shield_pcm", 0))
            sv += int(info.get("n_safety_violations", 0))
            steps += 1
        rows.append({
            "regime_id": regime_id, "year": year, "n_steps": steps,
            "total_reward": total_reward,
            "total_delivered_kWh": total_del / 1000.0,
            "total_unmet_kWh": total_unmet / 1000.0,
            "total_shield_water": sw, "total_shield_pcm": sp,
            "n_safety_violations": sv,
        })
    return rows


def _eval_recurrent(model, regime_id, years):
    from o3.environment.gym_wrapper import PCMGymEnv
    rows = []
    for year in years:
        env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
        obs, _ = env.reset(); lstm_state = None; ep_start = True
        done = False
        total_reward = total_del = total_unmet = 0.0
        sw = sp = sv = steps = 0
        while not done:
            action, lstm_state = model.predict(obs, state=lstm_state,
                                               episode_start=ep_start,
                                               deterministic=True)
            ep_start = False
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            sw += int(info.get("n_shield_water", 0))
            sp += int(info.get("n_shield_pcm", 0))
            sv += int(info.get("n_safety_violations", 0))
            steps += 1
        rows.append({
            "regime_id": regime_id, "year": year, "n_steps": steps,
            "total_reward": total_reward,
            "total_delivered_kWh": total_del / 1000.0,
            "total_unmet_kWh": total_unmet / 1000.0,
            "total_shield_water": sw, "total_shield_pcm": sp,
            "n_safety_violations": sv,
        })
    return rows


def _eval_maskable(model, regime_id, years):
    from o3.environment.gym_wrapper import PCMGymEnv
    rows = []
    for year in years:
        env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
        obs, _ = env.reset()
        done = False
        total_reward = total_del = total_unmet = 0.0
        sw = sp = sv = steps = 0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            sw += int(info.get("n_shield_water", 0))
            sp += int(info.get("n_shield_pcm", 0))
            sv += int(info.get("n_safety_violations", 0))
            steps += 1
        rows.append({
            "regime_id": regime_id, "year": year, "n_steps": steps,
            "total_reward": total_reward,
            "total_delivered_kWh": total_del / 1000.0,
            "total_unmet_kWh": total_unmet / 1000.0,
            "total_shield_water": sw, "total_shield_pcm": sp,
            "n_safety_violations": sv,
        })
    return rows


def _eval_sac(model, regime_id, years):
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    rows = []
    for year in years:
        env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
        obs, _ = env.reset()
        done = False
        total_reward = total_del = total_unmet = 0.0
        sw = sp = sv = steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            sw += int(info.get("n_shield_water", 0))
            sp += int(info.get("n_shield_pcm", 0))
            sv += int(info.get("n_safety_violations", 0))
            steps += 1
        rows.append({
            "regime_id": regime_id, "year": year, "n_steps": steps,
            "total_reward": total_reward,
            "total_delivered_kWh": total_del / 1000.0,
            "total_unmet_kWh": total_unmet / 1000.0,
            "total_shield_water": sw, "total_shield_pcm": sp,
            "n_safety_violations": sv,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def compute_stats(rows_per_seed: list[list[dict]]) -> dict:
    """Given N seed runs, compute mean ± std of key metrics."""
    metrics = ["total_delivered_kWh", "total_unmet_kWh",
               "total_shield_water", "total_shield_pcm", "n_safety_violations",
               "total_reward"]
    # average over years within each seed first
    seed_means = []
    for seed_rows in rows_per_seed:
        m = {k: np.mean([r[k] for r in seed_rows]) for k in metrics}
        seed_means.append(m)
    result = {}
    for k in metrics:
        vals = [m[k] for m in seed_means]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"]  = float(np.std(vals))
        result[f"{k}_min"]  = float(np.min(vals))
        result[f"{k}_max"]  = float(np.max(vals))
    return result


def print_summary_table(stats_by_algo: dict) -> None:
    print("\n" + "=" * 100)
    print(f"{'Algorithm':20s}  {'Del kWh (mean±std)':22s}  "
          f"{'Unmet kWh (mean±std)':22s}  {'Safety viol':11s}  {'Reward mean':12s}")
    print("-" * 100)
    for algo, s in stats_by_algo.items():
        print(
            f"{algo:20s}  "
            f"{s['total_delivered_kWh_mean']:7.1f} ± {s['total_delivered_kWh_std']:5.1f}  "
            f"  {s['total_unmet_kWh_mean']:7.1f} ± {s['total_unmet_kWh_std']:5.1f}  "
            f"  {s['n_safety_violations_mean']:>6.1f} ± {s['n_safety_violations_std']:.1f}  "
            f"  {s['total_reward_mean']:10.0f}"
        )
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--steps",  type=int, default=300_000)
    parser.add_argument("--seeds",  type=int, nargs="+", default=[42, 0, 123])
    parser.add_argument("--algo",   type=str, nargs="+",
                        default=list(ALGO_MAP.keys()),
                        choices=list(ALGO_MAP.keys()) + ["all"])
    args = parser.parse_args()

    algos = list(ALGO_MAP.keys()) if "all" in args.algo else args.algo

    print("=" * 70)
    print("Phase O3-4 Multi-Seed — Algorithm Comparison with Statistics")
    print(f"Regime: {args.regime}  Steps: {args.steps:,}  "
          f"Seeds: {args.seeds}  Algos: {algos}")
    print("=" * 70)

    all_raw: list[dict] = []
    stats_by_algo: dict = {}

    # --- deterministic baselines (single run) ---
    print("\n[Baselines]")
    base_rows = collect_baselines(args.regime, VAL_YEARS)
    for name in ["B0_fixed", "B1_reactive", "B2_forecast", "B4_random"]:
        rows = [r for r in base_rows if r["policy"].startswith(name.split("_")[0])]
        stats_by_algo[name] = compute_stats([rows])  # single "seed"
    all_raw += base_rows

    # --- RL algorithms over multiple seeds ---
    for algo_key in algos:
        display_name, module_path, fn_name = ALGO_MAP[algo_key]
        import importlib
        module    = importlib.import_module(module_path)
        train_fn  = getattr(module, fn_name)

        seed_results: list[list[dict]] = []
        for seed in args.seeds:
            print(f"\n[{display_name}] seed={seed}")
            out_dir = RESULTS_DIR / algo_key / f"r{args.regime}_seed{seed}"
            model   = train_fn(args.regime, args.steps, seed=seed, out_dir=out_dir)

            if algo_key == "recurrent":
                rows = _eval_recurrent(model, args.regime, VAL_YEARS)
            elif algo_key == "maskable":
                rows = _eval_maskable(model, args.regime, VAL_YEARS)
            elif algo_key == "sac":
                rows = _eval_sac(model, args.regime, VAL_YEARS)
            else:
                rows = _eval_standard(model, args.regime, VAL_YEARS)

            for r in rows:
                r["policy"]   = display_name
                r["algo_key"] = algo_key
                r["seed"]     = seed
            all_raw += rows
            seed_results.append(rows)

        stats_by_algo[display_name] = compute_stats(seed_results)

    # --- summary ---
    print_summary_table(stats_by_algo)

    # save raw per-seed per-year results
    raw_path = RESULTS_DIR / f"phase4_multiseed_raw_r{args.regime}.csv"
    if all_raw:
        keys = list(dict.fromkeys(k for r in all_raw for k in r.keys()))
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_raw)
        print(f"\n[INFO] Raw results -> {raw_path}")

    # save aggregated stats (for thesis table)
    stats_path = RESULTS_DIR / f"phase4_stats_r{args.regime}.json"
    with open(stats_path, "w") as f:
        json.dump(stats_by_algo, f, indent=2)
    print(f"[INFO] Stats table -> {stats_path}")

    # check safety
    violated = {k for k, s in stats_by_algo.items()
                if k not in ("B4_random",) and s["n_safety_violations_mean"] > 0}
    if violated:
        print(f"\n[WARN] Safety violations in: {violated}")
    else:
        print("\n[OK]   All RL policies safe on validation set across all seeds.")


if __name__ == "__main__":
    main()
