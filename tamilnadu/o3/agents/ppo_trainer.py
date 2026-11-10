"""
Phase O3-4 — PPO trainer using Stable-Baselines3.

Trains a Proximal Policy Optimisation agent on the PCM environment
across all training years (2016-2020), evaluates on val years,
and saves checkpoints + a metrics CSV.

Usage:
    from o3.agents.ppo_trainer import train_ppo, evaluate_policy
"""

import pathlib
import time
import numpy as np
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS, VAL_YEARS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def make_env(regime_id: int, years: list[int], fixed_year: int | None = None,
             reward_weights: dict | None = None) -> Monitor:
    env = PCMGymEnv(
        contract_path  = CONTRACT_PATH,
        regime_id      = regime_id,
        years          = years,
        fixed_year     = fixed_year,
        reward_weights = reward_weights,
    )
    return Monitor(env)


def train_ppo(
    regime_id: int   = 0,
    total_timesteps: int = 200_000,
    n_steps:  int    = 8760,       # one full year per rollout
    seed:     int    = 42,
    out_dir:  pathlib.Path | None = None,
    reward_weights: dict | None   = None,
) -> PPO:
    """Train a PPO agent and return it.

    Parameters
    ----------
    regime_id       : Tamil Nadu climate regime (0, 1 or 2)
    total_timesteps : training budget in env steps
    n_steps         : rollout buffer size (one year = 8760 h)
    seed            : random seed for reproducibility
    out_dir         : directory for checkpoints and results
    reward_weights  : override default reward shaping weights
    """
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "ppo" / f"r{regime_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(regime_id, TRAIN_YEARS, reward_weights=reward_weights)
    eval_env  = make_env(regime_id, VAL_YEARS, fixed_year=VAL_YEARS[0],
                         reward_weights=reward_weights)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(out_dir / "best_model"),
        log_path             = str(out_dir / "eval_logs"),
        eval_freq            = n_steps * 2,
        n_eval_episodes      = len(VAL_YEARS),
        deterministic        = True,
        verbose              = 1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps          = n_steps,
        batch_size       = 256,
        n_epochs         = 10,
        learning_rate    = 3e-4,
        gamma            = 0.99,
        gae_lambda       = 0.95,
        clip_range       = 0.2,
        ent_coef         = 0.01,
        verbose          = 1,
        seed             = seed,
        device           = "cpu",
    )

    print(f"\n[PPO] Training — regime {regime_id}, "
          f"{total_timesteps:,} steps, seed {seed}")
    t0 = time.time()
    model.learn(
        total_timesteps  = total_timesteps,
        callback         = eval_cb,
        progress_bar     = False,
    )
    elapsed = time.time() - t0
    print(f"[PPO] Training done in {elapsed/60:.1f} min")

    model.save(str(out_dir / "ppo_final"))
    print(f"[PPO] Model saved -> {out_dir / 'ppo_final'}")
    return model


def evaluate_policy(
    model: PPO,
    regime_id: int,
    years: list[int],
    n_episodes_per_year: int = 1,
    reward_weights: dict | None = None,
) -> list[dict]:
    """Run the trained policy and collect metrics per episode."""
    results = []
    for year in years:
        for ep in range(n_episodes_per_year):
            env = PCMGymEnv(
                CONTRACT_PATH, regime_id=regime_id,
                fixed_year=year, reward_weights=reward_weights,
            )
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            total_del = 0.0
            total_unmet = 0.0
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
                "policy":           "PPO",
                "regime_id":         regime_id,
                "year":              year,
                "episode":           ep,
                "n_steps":           steps,
                "total_reward":      total_reward,
                "total_delivered_kWh": total_del / 1000.0,
                "total_unmet_kWh":   total_unmet / 1000.0,
                "total_shield_water": shield_w,
                "total_shield_pcm":   shield_p,
                "n_safety_violations": safety_v,
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
