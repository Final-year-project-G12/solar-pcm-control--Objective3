"""
Phase O3-4 — DQN trainer using Stable-Baselines3.

Double DQN with the default MlpPolicy for the discrete 3-action space.
Serves as the value-learning baseline beside PPO.
"""

import pathlib
import time
import csv

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS, VAL_YEARS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def train_dqn(
    regime_id: int     = 0,
    total_timesteps: int = 200_000,
    seed:    int       = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None  = None,
) -> DQN:
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "dqn" / f"r{regime_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(PCMGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        years=TRAIN_YEARS, reward_weights=reward_weights,
    ))
    eval_env = Monitor(PCMGymEnv(
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

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate       = 1e-4,
        buffer_size         = 50_000,
        learning_starts     = 8760,
        batch_size          = 256,
        tau                 = 1.0,
        gamma               = 0.99,
        train_freq          = 4,
        gradient_steps      = 1,
        target_update_interval = 1000,
        exploration_fraction   = 0.3,
        exploration_final_eps  = 0.05,
        verbose             = 1,
        seed                = seed,
        device              = "cpu",
    )

    print(f"\n[DQN] Training — regime {regime_id}, "
          f"{total_timesteps:,} steps, seed {seed}")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                progress_bar=False)
    elapsed = time.time() - t0
    print(f"[DQN] Training done in {elapsed/60:.1f} min")

    model.save(str(out_dir / "dqn_final"))
    return model


def evaluate_policy(model: DQN, regime_id: int, years: list[int],
                    reward_weights: dict | None = None) -> list[dict]:
    results = []
    for year in years:
        env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id,
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
            "policy":              "DQN",
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
