"""
Phase O3-4 — Full DRL algorithm benchmark.

Trains 5 algorithms and compares them vs B0/B1 baselines on val years.

Algorithms:
  ppo          — PPO (memoryless, MLP policy)
  dqn          — DQN (double DQN, value-based)
  recurrent    — RecurrentPPO (LSTM, handles temporal patterns)
  qrdqn        — QRDQN (distributional, risk-aware)
  maskable     — MaskablePPO (safety-constrained action masking)

Run from the objective3 root:
    python scripts/phase4_train.py [--regime 0] [--steps 200000] [--seed 42]
    python scripts/phase4_train.py --algo ppo recurrent maskable
    python scripts/phase4_train.py --algo all

Outputs:
    results/{algo}/r{regime}/  — model checkpoints
    results/phase4_comparison_r{regime}.csv
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
from o3.environment.pcm_env     import PCMEnv
from o3.baselines.b0_fixed       import FixedChargePolicy
from o3.baselines.b1_reactive    import ReactiveRulePolicy
from o3.baselines.b2_forecast    import ForecastAwarePolicy
from o3.baselines.b0_fixed       import run_episode as run_b0
from o3.baselines.b1_reactive    import run_episode as run_b1
from o3.baselines.b2_forecast    import run_episode as run_b2

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_ALGOS = ["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def collect_baselines(regime_id: int, years: list[int]) -> list[dict]:
    env  = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
    rows = []
    for year in years:
        for name, pol, runner in [
            ("B0_fixed",    FixedChargePolicy(),   run_b0),
            ("B1_reactive", ReactiveRulePolicy(),  run_b1),
            ("B2_forecast", ForecastAwarePolicy(), run_b2),
        ]:
            r = runner(env, year=year)
            rows.append(r)
            print(f"  {name:15s} y={year} "
                  f"del={r['total_delivered_kWh']:.1f} kWh  "
                  f"safety={r['n_safety_violations']}")
    return rows


# ---------------------------------------------------------------------------
# Generic RL evaluator (for PPO, DQN, QRDQN — no special state needed)
# ---------------------------------------------------------------------------

def run_rl_eval(policy_fn, policy_name: str, regime_id: int,
                years: list[int]) -> list[dict]:
    from o3.environment.pcm_env import PCMEnv
    env = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
    rows = []
    for year in years:
        obs, _ = env.reset(year=year)
        done = False
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            shield_w     += int(info.get("n_shield_water", 0))
            shield_p     += int(info.get("n_shield_pcm", 0))
            safety_v     += int(info.get("n_safety_violations", 0))
            steps        += 1
        rows.append({
            "policy":               policy_name,
            "regime_id":             regime_id,
            "year":                  year,
            "n_steps":               steps,
            "total_reward":          total_reward,
            "total_delivered_kWh":   total_del / 1000.0,
            "total_unmet_kWh":       total_unmet / 1000.0,
            "total_shield_water":    shield_w,
            "total_shield_pcm":      shield_p,
            "n_safety_violations":   safety_v,
        })
        print(f"  {policy_name:15s} y={year} "
              f"del={rows[-1]['total_delivered_kWh']:.1f} kWh  "
              f"safety={safety_v}")
    return rows


# ---------------------------------------------------------------------------
# RecurrentPPO evaluator (needs lstm_state)
# ---------------------------------------------------------------------------

def run_recurrent_eval(model, policy_name: str, regime_id: int,
                       years: list[int]) -> list[dict]:
    from o3.environment.gym_wrapper import PCMGymEnv
    rows = []
    for year in years:
        env  = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
        obs, _   = env.reset()
        lstm_state = None
        ep_start   = True
        done = False
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0
        while not done:
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=ep_start, deterministic=True
            )
            ep_start = False
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            shield_w     += int(info.get("n_shield_water", 0))
            shield_p     += int(info.get("n_shield_pcm", 0))
            safety_v     += int(info.get("n_safety_violations", 0))
            steps        += 1
        rows.append({
            "policy":               policy_name,
            "regime_id":             regime_id,
            "year":                  year,
            "n_steps":               steps,
            "total_reward":          total_reward,
            "total_delivered_kWh":   total_del / 1000.0,
            "total_unmet_kWh":       total_unmet / 1000.0,
            "total_shield_water":    shield_w,
            "total_shield_pcm":      shield_p,
            "n_safety_violations":   safety_v,
        })
        print(f"  {policy_name:15s} y={year} "
              f"del={rows[-1]['total_delivered_kWh']:.1f} kWh  "
              f"safety={safety_v}")
    return rows


# ---------------------------------------------------------------------------
# MaskablePPO evaluator (needs action_masks)
# ---------------------------------------------------------------------------

def run_maskable_eval(model, policy_name: str, regime_id: int,
                      years: list[int]) -> list[dict]:
    from o3.environment.gym_wrapper import PCMGymEnv
    rows = []
    for year in years:
        env  = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
        obs, _ = env.reset()
        done = False
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0
        while not done:
            masks  = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            shield_w     += int(info.get("n_shield_water", 0))
            shield_p     += int(info.get("n_shield_pcm", 0))
            safety_v     += int(info.get("n_safety_violations", 0))
            steps        += 1
        rows.append({
            "policy":               policy_name,
            "regime_id":             regime_id,
            "year":                  year,
            "n_steps":               steps,
            "total_reward":          total_reward,
            "total_delivered_kWh":   total_del / 1000.0,
            "total_unmet_kWh":       total_unmet / 1000.0,
            "total_shield_water":    shield_w,
            "total_shield_pcm":      shield_p,
            "n_safety_violations":   safety_v,
        })
        print(f"  {policy_name:15s} y={year} "
              f"del={rows[-1]['total_delivered_kWh']:.1f} kWh  "
              f"safety={safety_v}")
    return rows


# ---------------------------------------------------------------------------
# SAC evaluator (continuous-action wrapper; model emits a Box action)
# ---------------------------------------------------------------------------

def run_sac_eval(model, policy_name: str, regime_id: int,
                 years: list[int]) -> list[dict]:
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    rows = []
    for year in years:
        env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=regime_id,
                                  fixed_year=year)
        obs, _ = env.reset()
        done = False
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            shield_w     += int(info.get("n_shield_water", 0))
            shield_p     += int(info.get("n_shield_pcm", 0))
            safety_v     += int(info.get("n_safety_violations", 0))
            steps        += 1
        rows.append({
            "policy":               policy_name,
            "regime_id":             regime_id,
            "year":                  year,
            "n_steps":               steps,
            "total_reward":          total_reward,
            "total_delivered_kWh":   total_del / 1000.0,
            "total_unmet_kWh":       total_unmet / 1000.0,
            "total_shield_water":    shield_w,
            "total_shield_pcm":      shield_p,
            "n_safety_violations":   safety_v,
        })
        print(f"  {policy_name:15s} y={year} "
              f"del={rows[-1]['total_delivered_kWh']:.1f} kWh  "
              f"safety={safety_v}")
    return rows


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(rows: list[dict]) -> None:
    print("\n" + "=" * 108)
    print(f"{'Policy':20s} {'Year':5s} {'Regime':6s} {'Del kWh':8s} "
          f"{'Unmet kWh':10s} {'Shield_w':9s} {'Shield_p':9s} {'Safety':7s} {'Reward':10s}")
    print("-" * 108)
    for r in rows:
        print(
            f"{r['policy']:20s} {r['year']:5d} {r['regime_id']:6d} "
            f"{r['total_delivered_kWh']:8.1f} {r['total_unmet_kWh']:10.1f} "
            f"{r['total_shield_water']:9d} {r['total_shield_pcm']:9d} "
            f"{r['n_safety_violations']:7d} {r['total_reward']:10.0f}"
        )
    print("=" * 108)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--steps",  type=int, default=200_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--algo",   type=str, nargs="+",
                        default=["ppo", "recurrent", "qrdqn", "maskable"],
                        choices=ALL_ALGOS + ["all"],
                        help="Algorithms to train (space-separated, or 'all')")
    args = parser.parse_args()

    algos = ALL_ALGOS if "all" in args.algo else args.algo

    print("=" * 70)
    print("Phase O3-4 — Complex DRL Algorithm Benchmark")
    print(f"Regime: {args.regime}  |  Steps: {args.steps:,}  |  "
          f"Seed: {args.seed}  |  Algos: {algos}")
    print("=" * 70)

    all_results: list[dict] = []

    # baselines
    print("\n--- Baselines (B0, B1) ---")
    all_results += collect_baselines(args.regime, VAL_YEARS)

    # --- PPO ---
    if "ppo" in algos:
        from o3.agents.ppo_trainer import train_ppo
        print("\n--- PPO ---")
        m = train_ppo(args.regime, args.steps, seed=args.seed,
                      out_dir=RESULTS_DIR / "ppo" / f"r{args.regime}")
        all_results += run_rl_eval(
            lambda obs: int(m.predict(obs.astype("float32"), deterministic=True)[0]),
            "PPO", args.regime, VAL_YEARS,
        )

    # --- DQN ---
    if "dqn" in algos:
        from o3.agents.dqn_trainer import train_dqn
        print("\n--- DQN ---")
        m = train_dqn(args.regime, args.steps, seed=args.seed,
                      out_dir=RESULTS_DIR / "dqn" / f"r{args.regime}")
        all_results += run_rl_eval(
            lambda obs: int(m.predict(obs.astype("float32"), deterministic=True)[0]),
            "DQN", args.regime, VAL_YEARS,
        )

    # --- RecurrentPPO ---
    if "recurrent" in algos:
        from o3.agents.recurrent_ppo_trainer import train_recurrent_ppo
        print("\n--- RecurrentPPO (LSTM) ---")
        m = train_recurrent_ppo(args.regime, args.steps, seed=args.seed,
                                out_dir=RESULTS_DIR / "recurrent_ppo" / f"r{args.regime}")
        all_results += run_recurrent_eval(m, "RecurrentPPO", args.regime, VAL_YEARS)

    # --- QRDQN ---
    if "qrdqn" in algos:
        from o3.agents.qrdqn_trainer import train_qrdqn
        print("\n--- QRDQN (Distributional) ---")
        m = train_qrdqn(args.regime, args.steps, seed=args.seed,
                        out_dir=RESULTS_DIR / "qrdqn" / f"r{args.regime}")
        all_results += run_rl_eval(
            lambda obs: int(m.predict(obs.astype("float32"), deterministic=True)[0]),
            "QRDQN", args.regime, VAL_YEARS,
        )

    # --- MaskablePPO ---
    if "maskable" in algos:
        from o3.agents.maskable_ppo_trainer import train_maskable_ppo
        print("\n--- MaskablePPO (Action Masking) ---")
        m = train_maskable_ppo(args.regime, args.steps, seed=args.seed,
                               out_dir=RESULTS_DIR / "maskable_ppo" / f"r{args.regime}")
        all_results += run_maskable_eval(m, "MaskablePPO", args.regime, VAL_YEARS)

    # --- SAC ---
    if "sac" in algos:
        from o3.agents.sac_trainer import train_sac
        print("\n--- SAC (Soft Actor-Critic, continuous) ---")
        m = train_sac(args.regime, args.steps, seed=args.seed,
                      out_dir=RESULTS_DIR / "sac" / f"r{args.regime}")
        all_results += run_sac_eval(m, "SAC", args.regime, VAL_YEARS)

    # --- summary ---
    print_comparison(all_results)

    out_path = RESULTS_DIR / f"phase4_comparison_r{args.regime}.csv"
    # unify keys
    all_keys = list(dict.fromkeys(k for r in all_results for k in r.keys()))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\n[INFO] Results saved -> {out_path}")

    # acceptance gate
    rl_names  = set(algos)
    rl_viols  = [r for r in all_results
                 if r["policy"] in rl_names and r["n_safety_violations"] > 0]
    if rl_viols:
        print("\n[WARN] Safety violations found in RL policies on val set:")
        for r in rl_viols:
            print(f"  {r['policy']} y={r['year']}: {r['n_safety_violations']}")
    else:
        print("\n[OK]   Phase O3-4 PASSED.  Ready for Phase O3-5 (Robustness).")


if __name__ == "__main__":
    main()
