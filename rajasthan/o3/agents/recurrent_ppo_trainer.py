"""
Phase O3-4 — Recurrent PPO (LSTM) trainer.

Uses sb3-contrib's RecurrentPPO (LSTM policy). The agent keeps a hidden state across timesteps so it
can learn temporal patterns that a memoryless policy cannot: delayed PCM discharge after a charge,
multi-day cloudy sequences, day-ahead demand anticipation (the contract observation has no forecast).

Parallel: SubprocVecEnv with n_envs workers (RecurrentPPO supports vec envs; each env has its own
LSTM state). n_steps is the total samples per update, split over the envs.
"""

import pathlib
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback

from o3.agents import common as C
from o3.agents.ppo_trainer import save_results  # noqa: F401
from o3.evaluation.rollout import evaluate_policy_object, KIND_RECURRENT

CONTRACT_PATH = C.CONTRACT_PATH


def train_recurrent_ppo(
    regime_id: int = 0,
    total_timesteps: int = 200_000,
    n_steps: int = 8760,
    lstm_hidden: int = 64,
    seed: int = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None = None,
    n_envs: int = 1,
    vec: str = "auto",
    eval_every_episodes: float = 2.0,
    verbose: int = 1,
) -> RecurrentPPO:
    """Train a Recurrent PPO (LSTM) agent. lstm_hidden: 64 = fast, 128 = stronger."""
    out_dir = pathlib.Path(out_dir) if out_dir else C.default_out_dir("recurrent_ppo", regime_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.pin_main_threads(1)

    train_env = C.make_train_vec(regime_id, n_envs, vec, reward_weights)
    eval_env = C.make_eval_vec(regime_id, reward_weights)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out_dir / "best_model"), log_path=str(out_dir / "eval_logs"),
        eval_freq=C.eval_freq_calls(eval_every_episodes, n_envs) if total_timesteps >= 8760 * eval_every_episodes else max(1, total_timesteps // max(1, n_envs)),
        n_eval_episodes=3, deterministic=True, verbose=verbose,
    )
    policy_kwargs = dict(lstm_hidden_size=lstm_hidden, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True)
    model = RecurrentPPO(
        "MlpLstmPolicy", train_env, n_steps=C.rollout_steps_per_env(n_steps, n_envs), batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device="cpu",
    )
    print(f"\n[RecurrentPPO] Training — regime {regime_id}, {total_timesteps:,} steps, LSTM hidden={lstm_hidden}, "
          f"seed {seed}, n_envs {n_envs}")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    finally:
        train_env.close()
        eval_env.close()
    print(f"[RecurrentPPO] Done in {(time.time() - t0) / 60:.1f} min")
    model.save(str(out_dir / "recurrent_ppo_final"))
    return model


def evaluate_policy(model: RecurrentPPO, regime_id: int, years: list[int], reward_weights: dict | None = None) -> list[dict]:
    return evaluate_policy_object(model, KIND_RECURRENT, regime_id, years, "RecurrentPPO", reward_weights=reward_weights)
