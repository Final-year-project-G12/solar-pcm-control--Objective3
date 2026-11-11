"""
Phase O3-4 — Quantile Regression DQN (QRDQN) trainer.

QRDQN (Dabney et al. 2017) learns the full *distribution* of returns, not just the expectation: each
action's value is N quantiles of its return distribution. Why here: solar irradiance is stochastic —
the same action has very different returns on a clear and a cloudy day — and QRDQN captures that and
can be used risk-sensitively. Available via sb3-contrib. Off-policy: ONE environment (see common.py).
Used in place of the plan's Rainbow DQN, which SB3 does not provide (plan 5, deferred item).
"""

import pathlib
import time

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback

from o3.agents import common as C
from o3.agents.ppo_trainer import save_results  # noqa: F401
from o3.evaluation.rollout import evaluate_policy_object, KIND_STANDARD

CONTRACT_PATH = C.CONTRACT_PATH


def train_qrdqn(
    regime_id: int = 0,
    total_timesteps: int = 200_000,
    n_quantiles: int = 50,
    seed: int = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None = None,
    n_envs: int = 1,
    eval_every_episodes: float = 2.0,
    verbose: int = 1,
) -> QRDQN:
    """Train a QRDQN agent. n_quantiles: 50 = standard, 200 = richer."""
    out_dir = pathlib.Path(out_dir) if out_dir else C.default_out_dir("qrdqn", regime_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.pin_main_threads(1)
    C.note_offpolicy(n_envs, "QRDQN")

    train_env = C.make_train_vec(regime_id, 1, "dummy", reward_weights)
    eval_env = C.make_eval_vec(regime_id, reward_weights)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out_dir / "best_model"), log_path=str(out_dir / "eval_logs"),
        eval_freq=C.eval_freq_calls(eval_every_episodes, 1) if total_timesteps >= 8760 * eval_every_episodes else total_timesteps,
        n_eval_episodes=3, deterministic=True, verbose=verbose,
    )
    model = QRDQN(
        "MlpPolicy", train_env, learning_rate=1e-4, buffer_size=50_000, learning_starts=min(8760, max(100, total_timesteps // 4)),
        batch_size=256, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000,
        exploration_fraction=0.3, exploration_final_eps=0.05, policy_kwargs=dict(n_quantiles=n_quantiles),
        verbose=verbose, seed=seed, device="cpu",
    )
    print(f"\n[QRDQN] Training — regime {regime_id}, {total_timesteps:,} steps, {n_quantiles} quantiles, seed {seed}")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    finally:
        train_env.close()
        eval_env.close()
    print(f"[QRDQN] Done in {(time.time() - t0) / 60:.1f} min")
    model.save(str(out_dir / "qrdqn_final"))
    return model


def evaluate_policy(model: QRDQN, regime_id: int, years: list[int], reward_weights: dict | None = None) -> list[dict]:
    return evaluate_policy_object(model, KIND_STANDARD, regime_id, years, "QRDQN", reward_weights=reward_weights)
