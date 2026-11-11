"""
Phase O3-4 — PPO trainer using Stable-Baselines3.

Trains a Proximal Policy Optimisation agent on the PCM environment across the training years
(2016-2020), evaluates on the validation years (2021-2023) with EvalCallback, and saves the best
checkpoint (by validation reward) plus the final model.

Parallel: experience comes from `n_envs` worker processes (SubprocVecEnv). `n_steps` is the TOTAL
number of samples per update (default 8760, one simulated year, as in the TN reference); each
environment contributes n_steps / n_envs of them. See o3/agents/common.py.

Usage:
    from o3.agents.ppo_trainer import train_ppo, evaluate_policy
"""

import csv
import pathlib
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from o3.agents import common as C
from o3.evaluation.rollout import evaluate_policy_object, KIND_STANDARD

CONTRACT_PATH = C.CONTRACT_PATH


def make_env(regime_id: int, years: list[int], fixed_year: int | None = None,
             reward_weights: dict | None = None):
    return C.make_env(regime_id, years, fixed_year, False, reward_weights)


def train_ppo(
    regime_id: int = 0,
    total_timesteps: int = 200_000,
    n_steps: int = 8760,            # TOTAL samples per update (one simulated year)
    seed: int = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None = None,
    n_envs: int = 1,
    vec: str = "auto",
    eval_every_episodes: float = 2.0,
    verbose: int = 1,
) -> PPO:
    """Train a PPO agent and return it.

    regime_id       : Rajasthan climate regime (0, 1 or 2)
    total_timesteps : training budget in env steps (summed over all envs)
    n_steps         : samples per policy update, summed over envs
    n_envs          : worker processes collecting experience (1 = in-process)
    """
    out_dir = pathlib.Path(out_dir) if out_dir else C.default_out_dir("ppo", regime_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.pin_main_threads(1)

    train_env = C.make_train_vec(regime_id, n_envs, vec, reward_weights)
    eval_env = C.make_eval_vec(regime_id, reward_weights)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out_dir / "best_model"), log_path=str(out_dir / "eval_logs"),
        eval_freq=C.eval_freq_calls(eval_every_episodes, n_envs) if total_timesteps >= 8760 * eval_every_episodes else max(1, total_timesteps // max(1, n_envs)),
        n_eval_episodes=3, deterministic=True, verbose=verbose,
    )
    model = PPO(
        "MlpPolicy", train_env, n_steps=C.rollout_steps_per_env(n_steps, n_envs), batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=verbose, seed=seed, device="cpu",
    )
    print(f"\n[PPO] Training — regime {regime_id}, {total_timesteps:,} steps, seed {seed}, n_envs {n_envs}")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    finally:
        train_env.close()
        eval_env.close()
    print(f"[PPO] Training done in {(time.time() - t0) / 60:.1f} min")
    model.save(str(out_dir / "ppo_final"))
    print(f"[PPO] Model saved -> {out_dir / 'ppo_final'}")
    return model


def evaluate_policy(model: PPO, regime_id: int, years: list[int], n_episodes_per_year: int = 1,
                    reward_weights: dict | None = None) -> list[dict]:
    """Run the trained policy and collect metrics per episode."""
    rows = []
    for ep in range(n_episodes_per_year):
        for r in evaluate_policy_object(model, KIND_STANDARD, regime_id, years, "PPO", reward_weights=reward_weights):
            r["episode"] = ep
            rows.append(r)
    return rows


def save_results(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in results for k in r)))
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Results saved -> {path}")
