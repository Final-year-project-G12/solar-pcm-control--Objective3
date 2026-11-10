"""
Phase O3-6 — Locked final evaluation on held-out test set (2011-2015).

⚠️  RUN THIS ONLY ONCE — after policy selection is frozen.
⚠️  Test years 2011-2015 must NOT have influenced training, reward
    shaping, normalisation, early stopping, or algorithm selection.

Usage:
    python scripts/phase6_locked_eval.py --algo recurrent [--regime 0 1 2]

Outputs:
    results/phase6_final_test_r{regime}_{algo}.csv  — per-year metrics
    results/phase6_summary.json                      — aggregated publishable table
    results/phase6_policy_hash.txt                   — model SHA for reproducibility
"""

import argparse
import pathlib
import sys
import csv
import json
import hashlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).parents[1]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from o3.environment.pcm_env     import PCMEnv
from o3.environment.gym_wrapper import PCMGymEnv
from o3.baselines.b0_fixed       import FixedChargePolicy, run_episode as run_b0
from o3.baselines.b1_reactive    import ReactiveRulePolicy, run_episode as run_b1

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"
RESULTS_DIR   = REPO_ROOT / "results"
TEST_YEARS    = [2011, 2012, 2013, 2014, 2015]
PCM_INITS     = ["solid", "liquid", "partial"]


def sha256_of_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# algo key -> results subdirectory (recurrent/maskable use the *_ppo dir names)
_ALGO_DIR = {
    "ppo": "ppo", "dqn": "dqn", "recurrent": "recurrent_ppo",
    "qrdqn": "qrdqn", "maskable": "maskable_ppo", "sac": "sac",
}


def load_model(algo: str, regime_id: int):
    """Load the best trained model for a given algorithm and regime."""
    subdir = _ALGO_DIR.get(algo, algo)
    base = RESULTS_DIR / subdir / f"r{regime_id}"
    best = base / "best_model" / "best_model"
    final = base / f"{subdir}_final"
    # prefer best_model checkpoint, fall back to final
    path = best if best.with_suffix(".zip").exists() else final
    if not path.with_suffix(".zip").exists():
        # also check seed-specific dirs
        seed_dirs = sorted(base.parent.glob(f"r{regime_id}_seed*"))
        for sd in seed_dirs:
            p = sd / "best_model" / "best_model"
            if p.with_suffix(".zip").exists():
                path = p
                break
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"No trained model found for {algo} regime {regime_id}.\n"
            f"Run phase4_train.py first."
        )
    return path


def eval_episode_standard(model, env_pcm: PCMEnv, year: int,
                           pcm_init: str = "solid") -> dict:
    obs, _ = env_pcm.reset(year=year, initial_pcm_state=pcm_init)
    done = False
    total_reward = total_del = total_unmet = 0.0
    sw = sp = sv = steps = 0
    while not done:
        action, _ = model.predict(obs.astype("float32"), deterministic=True)
        obs, reward, terminated, truncated, info = env_pcm.step(int(action))
        done = terminated or truncated
        total_reward += reward
        total_del    += info.get("E_delivered_Wh", 0.0)
        total_unmet  += info.get("E_unmet_Wh",    0.0)
        sw += int(info.get("n_shield_water", 0))
        sp += int(info.get("n_shield_pcm", 0))
        sv += int(info.get("n_safety_violations", 0))
        steps += 1
    return {
        "n_steps": steps, "total_reward": total_reward,
        "total_delivered_kWh": total_del / 1000.0,
        "total_unmet_kWh": total_unmet / 1000.0,
        "total_shield_water": sw, "total_shield_pcm": sp,
        "n_safety_violations": sv,
    }


def eval_episode_recurrent(model, regime_id: int, year: int,
                            pcm_init: str = "solid") -> dict:
    env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
    obs, _ = env.inner.reset(year=year, initial_pcm_state=pcm_init)
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
    return {
        "n_steps": steps, "total_reward": total_reward,
        "total_delivered_kWh": total_del / 1000.0,
        "total_unmet_kWh": total_unmet / 1000.0,
        "total_shield_water": sw, "total_shield_pcm": sp,
        "n_safety_violations": sv,
    }


def eval_episode_sac(model, regime_id: int, year: int,
                     pcm_init: str = "solid") -> dict:
    from o3.environment.gym_wrapper import PCMContinuousGymEnv
    env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year)
    env.inner.reset(year=year, initial_pcm_state=pcm_init)
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
    return {
        "n_steps": steps, "total_reward": total_reward,
        "total_delivered_kWh": total_del / 1000.0,
        "total_unmet_kWh": total_unmet / 1000.0,
        "total_shield_water": sw, "total_shield_pcm": sp,
        "n_safety_violations": sv,
    }


def eval_episode_maskable(model, env_gym: PCMGymEnv, year: int,
                           pcm_init: str = "solid") -> dict:
    env_gym.inner.reset(year=year, initial_pcm_state=pcm_init)
    obs, _ = env_gym.reset()
    done = False
    total_reward = total_del = total_unmet = 0.0
    sw = sp = sv = steps = 0
    while not done:
        masks = env_gym.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, info = env_gym.step(action)
        done = terminated or truncated
        total_reward += reward
        total_del    += info.get("E_delivered_Wh", 0.0)
        total_unmet  += info.get("E_unmet_Wh",    0.0)
        sw += int(info.get("n_shield_water", 0))
        sp += int(info.get("n_shield_pcm", 0))
        sv += int(info.get("n_safety_violations", 0))
        steps += 1
    return {
        "n_steps": steps, "total_reward": total_reward,
        "total_delivered_kWh": total_del / 1000.0,
        "total_unmet_kWh": total_unmet / 1000.0,
        "total_shield_water": sw, "total_shield_pcm": sp,
        "n_safety_violations": sv,
    }


def run_locked_eval(algo: str, regime_id: int) -> list[dict]:
    model_path = load_model(algo, regime_id)
    model_hash = sha256_of_file(model_path.with_suffix(".zip"))
    print(f"  Model: {model_path}  SHA: {model_hash[:12]}...")

    # load model
    if algo == "recurrent":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(model_path))
    elif algo == "qrdqn":
        from sb3_contrib import QRDQN
        model = QRDQN.load(str(model_path))
    elif algo == "maskable":
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(model_path))
    elif algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))
    elif algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(str(model_path))
    elif algo == "sac":
        from stable_baselines3 import SAC
        model = SAC.load(str(model_path))
    else:
        raise ValueError(f"Unknown algo: {algo}")

    env_pcm = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
    env_gym = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=TEST_YEARS[0])

    rows = []
    for year in TEST_YEARS:
        for pcm_init in PCM_INITS:
            if algo == "recurrent":
                ep = eval_episode_recurrent(model, regime_id, year, pcm_init)
            elif algo == "maskable":
                ep = eval_episode_maskable(model, env_gym, year, pcm_init)
            elif algo == "sac":
                ep = eval_episode_sac(model, regime_id, year, pcm_init)
            else:
                ep = eval_episode_standard(model, env_pcm, year, pcm_init)

            row = {
                "policy": algo, "regime_id": regime_id,
                "year": year, "pcm_init": pcm_init,
                "model_sha": model_hash[:12],
                **ep,
            }
            rows.append(row)
            print(f"  {algo:12s} r{regime_id} y={year} init={pcm_init:7s} "
                  f"del={ep['total_delivered_kWh']:.1f} kWh  "
                  f"safety={ep['n_safety_violations']}")
    return rows


def run_baselines_locked(regime_id: int) -> list[dict]:
    env = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
    rows = []
    for year in TEST_YEARS:
        for pcm_init in PCM_INITS:
            for name, pol, runner in [
                ("B0_fixed",    FixedChargePolicy(),  run_b0),
                ("B1_reactive", ReactiveRulePolicy(), run_b1),
            ]:
                r = runner(env, year=year, initial_pcm_state=pcm_init)
                rows.append(r)
    return rows


def print_primary_success_criteria(rows: list[dict], algo: str) -> bool:
    """Check the O3 primary success criteria (from implementation plan §1)."""
    print(f"\n--- Primary success criteria: {algo} ---")
    passed = True

    # 1. Zero hard safety violations
    total_sv = sum(r["n_safety_violations"] for r in rows)
    ok = total_sv == 0
    print(f"  {'✓' if ok else '✗'} Zero hard safety violations: {total_sv}")
    passed = passed and ok

    # 2. Delivery reliability ≥ B0
    algo_del = np.mean([r["total_delivered_kWh"] for r in rows
                        if r["policy"] == algo])
    b0_del   = np.mean([r["total_delivered_kWh"] for r in rows
                        if r["policy"] == "B0_fixed_charge"])
    ok = algo_del >= b0_del * 0.95
    print(f"  {'✓' if ok else '✗'} Delivery ≥ 95% of B0: "
          f"{algo_del:.1f} vs {b0_del:.1f} kWh")
    passed = passed and ok

    # 3. Shield events ≤ B0
    algo_sh = np.mean([r["total_shield_water"] + r["total_shield_pcm"]
                       for r in rows if r["policy"] == algo])
    b0_sh   = np.mean([r["total_shield_water"] + r["total_shield_pcm"]
                       for r in rows if r["policy"] == "B0_fixed_charge"])
    ok = algo_sh <= b0_sh * 1.05
    print(f"  {'✓' if ok else '✗'} Shield events ≤ B0+5%: "
          f"{algo_sh:.0f} vs {b0_sh:.0f}")

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",   type=str, default="recurrent",
                        choices=["ppo", "dqn", "recurrent", "qrdqn", "maskable", "sac"])
    parser.add_argument("--regime", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    print("=" * 70)
    print("Phase O3-6 — LOCKED FINAL EVALUATION  (test years 2011-2015)")
    print(f"Algorithm: {args.algo}  |  Regimes: {args.regime}")
    print("WARNING: results from this run should NOT be used for further tuning.")
    print("=" * 70)

    all_rows: list[dict] = []

    for regime_id in args.regime:
        print(f"\n=== Regime {regime_id} ===")
        all_rows += run_baselines_locked(regime_id)
        all_rows += run_locked_eval(args.algo, regime_id)

    # save CSV
    out_csv = RESULTS_DIR / f"phase6_final_test_{args.algo}.csv"
    if all_rows:
        keys = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[INFO] Results -> {out_csv}")

    # primary success criteria
    print_primary_success_criteria(all_rows, args.algo)

    # aggregate summary for JSON
    summary = {}
    for policy in set(r["policy"] for r in all_rows):
        p_rows = [r for r in all_rows if r["policy"] == policy]
        summary[policy] = {
            "delivered_kWh_mean": float(np.mean([r["total_delivered_kWh"] for r in p_rows])),
            "delivered_kWh_std":  float(np.std( [r["total_delivered_kWh"] for r in p_rows])),
            "unmet_kWh_mean":     float(np.mean([r["total_unmet_kWh"]     for r in p_rows])),
            "shield_mean":        float(np.mean([r["total_shield_water"] + r["total_shield_pcm"]
                                                  for r in p_rows])),
            "safety_violations":  int(sum(r["n_safety_violations"] for r in p_rows)),
        }
    summary_path = RESULTS_DIR / f"phase6_summary_{args.algo}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary JSON -> {summary_path}")
    print("\n[OK]   Phase O3-6 complete.")


if __name__ == "__main__":
    main()
