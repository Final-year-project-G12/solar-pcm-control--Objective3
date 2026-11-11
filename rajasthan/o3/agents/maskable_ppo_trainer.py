"""
Phase O3-4 — MaskablePPO trainer (safety-aware action masking).

MaskablePPO (sb3-contrib) extends PPO to accept an action mask at each timestep. Masked actions get
-inf in the policy logits before the softmax, so the agent can NEVER select them.

Why masking here:
  - Charging with the top node >= 68 C is useless (the shield cuts the loop at 72 C anyway) and makes
    the reward signal noisy; charging with no sun does nothing.
  - Masking these structurally speeds up learning and gives a policy that respects them by construction.

The mask is computed by PCMGymEnv.action_masks() (gym_wrapper.py) from physical thresholds decoded
from the observation (68 C top node, GHI at the collector cutoff).

Parallel: SubprocVecEnv with n_envs workers; masks are collected through env_method("action_masks").
"""

import pathlib
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from o3.agents import common as C
from o3.agents.ppo_trainer import save_results  # noqa: F401  (same CSV helper)
from o3.evaluation.rollout import evaluate_policy_object, KIND_MASKABLE

CONTRACT_PATH = C.CONTRACT_PATH


def train_maskable_ppo(
    regime_id: int = 0,
    total_timesteps: int = 200_000,
    n_steps: int = 8760,
    seed: int = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None = None,
    n_envs: int = 1,
    vec: str = "auto",
    eval_every_episodes: float = 2.0,
    verbose: int = 1,
) -> MaskablePPO:
    """Train a MaskablePPO agent with action masking."""
    out_dir = pathlib.Path(out_dir) if out_dir else C.default_out_dir("maskable_ppo", regime_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.pin_main_threads(1)

    train_env = C.make_train_vec(regime_id, n_envs, vec, reward_weights)
    eval_env = C.make_eval_vec(regime_id, reward_weights)
    eval_cb = MaskableEvalCallback(
        eval_env, best_model_save_path=str(out_dir / "best_model"), log_path=str(out_dir / "eval_logs"),
        eval_freq=C.eval_freq_calls(eval_every_episodes, n_envs) if total_timesteps >= 8760 * eval_every_episodes else max(1, total_timesteps // max(1, n_envs)),
        n_eval_episodes=3, deterministic=True, verbose=verbose,
    )
    model = MaskablePPO(
        "MlpPolicy", train_env, n_steps=C.rollout_steps_per_env(n_steps, n_envs), batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005,      # lower entropy: masking already forces exploration
        verbose=verbose, seed=seed, device="cpu",
    )
    print(f"\n[MaskablePPO] Training — regime {regime_id}, {total_timesteps:,} steps, seed {seed}, n_envs {n_envs}")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    finally:
        train_env.close()
        eval_env.close()
    print(f"[MaskablePPO] Done in {(time.time() - t0) / 60:.1f} min")
    model.save(str(out_dir / "maskable_ppo_final"))
    return model


def evaluate_policy(model: MaskablePPO, regime_id: int, years: list[int], reward_weights: dict | None = None) -> list[dict]:
    return evaluate_policy_object(model, KIND_MASKABLE, regime_id, years, "MaskablePPO", reward_weights=reward_weights)
