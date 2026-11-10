"""
Phase O3-4 — SAC trainer using Stable-Baselines3.

Soft Actor-Critic is the implementation plan's recommended algorithm for
continuous flow control (plan §5.3). The PCM control task is natively
discrete {CHARGE, IDLE, BYPASS}, so SAC trains on the continuous-action
view (`PCMContinuousGymEnv`), which exposes a 1-D action in [-1, 1] and
bins it back to the three modes. The in-env safety shield is unchanged.

SAC brings entropy-regularised exploration and an off-policy replay
buffer — a different learning regime from PPO (on-policy) and DQN
(value-only), so it is a meaningful comparison point for the thesis.
"""

import pathlib
import time
import csv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from o3.environment.gym_wrapper import (
    PCMContinuousGymEnv, TRAIN_YEARS, VAL_YEARS,
)

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def train_sac(
    regime_id: int     = 0,
    total_timesteps: int = 200_000,
    seed:    int       = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None  = None,
) -> SAC:
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "sac" / f"r{regime_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(PCMContinuousGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        years=TRAIN_YEARS, reward_weights=reward_weights,
    ))
    eval_env = Monitor(PCMContinuousGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        fixed_year=VAL_YEARS[0], reward_weights=reward_weights,
    ))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(out_dir / "best_model"),
        log_path             = str(out_dir / "eval_logs"),
        eval_freq            = 8760 * 2,
        n_eval_episodes      = len(VAL_YEARS),
        deterministic        = True,
        verbose              = 1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate       = 3e-4,
        buffer_size         = 100_000,
        learning_starts     = 8760,
        batch_size          = 256,
        tau                 = 0.005,
        gamma               = 0.99,
        train_freq          = 1,
        gradient_steps      = 1,
        ent_coef            = "auto",
        verbose             = 1,
        seed                = seed,
        device              = "cpu",
    )

    print(f"\n[SAC] Training — regime {regime_id}, "
          f"{total_timesteps:,} steps, seed {seed}")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                progress_bar=False)
    print(f"[SAC] Done in {(time.time()-t0)/60:.1f} min")

    model.save(str(out_dir / "sac_final"))
    return model


def evaluate_policy(model: SAC, regime_id: int, years: list[int],
                    reward_weights: dict | None = None) -> list[dict]:
    results = []
    for year in years:
        env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=regime_id,
                                  fixed_year=year, reward_weights=reward_weights)
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
            total_unmet  += info.get("E_unmet_Wh", 0.0)
            shield_w     += int(info.get("n_shield_water", 0))
            shield_p     += int(info.get("n_shield_pcm", 0))
            safety_v     += int(info.get("n_safety_violations", 0))
            steps        += 1
        results.append({
            "policy":              "SAC",
            "regime_id":            regime_id,
            "year":                 year,
            "n_steps":              steps,
            "total_reward":         total_reward,
            "total_delivered_kWh":  total_del / 1000.0,
            "total_unmet_kWh":      total_unmet / 1000.0,
            "total_shield_water":   shield_w,
            "total_shield_pcm":     shield_p,
            "n_safety_violations":  safety_v,
        })
    return results


def save_results(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Results saved -> {path}")
