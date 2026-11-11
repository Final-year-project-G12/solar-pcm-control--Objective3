"""
Phase O3-4 — SAC trainer using Stable-Baselines3.

Soft Actor-Critic is the implementation plan's recommended continuous-control algorithm (plan 5.3 of
the original plan; plan v-final lists it as deferred because it needs continuous actions). The PCM
control task is natively discrete {CHARGE, IDLE, BYPASS}, so SAC trains on the continuous-action view
(`PCMContinuousGymEnv`): a 1-D action in [-1, 1] binned back to the three modes. The in-kernel safety
shield is unchanged. Off-policy (replay buffer): ONE environment (DummyVecEnv, see common.py).
"""

import pathlib
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from o3.agents import common as C
from o3.agents.ppo_trainer import save_results  # noqa: F401
from o3.evaluation.rollout import evaluate_policy_object, KIND_SAC

CONTRACT_PATH = C.CONTRACT_PATH


def train_sac(
    regime_id: int = 0,
    total_timesteps: int = 200_000,
    seed: int = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None = None,
    n_envs: int = 1,
    eval_every_episodes: float = 2.0,
    verbose: int = 1,
) -> SAC:
    out_dir = pathlib.Path(out_dir) if out_dir else C.default_out_dir("sac", regime_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.pin_main_threads(1)
    C.note_offpolicy(n_envs, "SAC")

    train_env = C.make_train_vec(regime_id, 1, "dummy", reward_weights, continuous=True)
    eval_env = C.make_eval_vec(regime_id, reward_weights, continuous=True)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out_dir / "best_model"), log_path=str(out_dir / "eval_logs"),
        eval_freq=C.eval_freq_calls(eval_every_episodes, 1) if total_timesteps >= 8760 * eval_every_episodes else total_timesteps,
        n_eval_episodes=3, deterministic=True, verbose=verbose,
    )
    model = SAC(
        "MlpPolicy", train_env, learning_rate=3e-4, buffer_size=100_000,
        learning_starts=min(8760, max(100, total_timesteps // 4)), batch_size=256, tau=0.005, gamma=0.99,
        train_freq=1, gradient_steps=1, ent_coef="auto", verbose=verbose, seed=seed, device="cpu",
    )
    print(f"\n[SAC] Training — regime {regime_id}, {total_timesteps:,} steps, seed {seed}")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    finally:
        train_env.close()
        eval_env.close()
    print(f"[SAC] Done in {(time.time() - t0) / 60:.1f} min")
    model.save(str(out_dir / "sac_final"))
    return model


def evaluate_policy(model: SAC, regime_id: int, years: list[int], reward_weights: dict | None = None) -> list[dict]:
    return evaluate_policy_object(model, KIND_SAC, regime_id, years, "SAC", reward_weights=reward_weights)
