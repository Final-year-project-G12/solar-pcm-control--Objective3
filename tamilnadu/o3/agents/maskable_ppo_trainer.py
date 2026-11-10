"""
Phase O3-4 — MaskablePPO trainer (safety-aware action masking).

MaskablePPO (sb3-contrib) extends PPO to accept an action mask at each
timestep. Masked actions receive −∞ in the policy logits before softmax,
so the agent can NEVER select them.

Why action masking for PCM solar control:
  - Charging when T_water > 68°C is both thermodynamically useless
    (heat flows from hot water to collector, not in) and forces the
    safety shield to fire immediately (wasted step, noisy reward signal).
  - Charging at night / zero-GHI is physically meaningless.
  - By masking these before the policy logits, the agent explores only
    physically valid actions. This speeds up learning and produces a
    policy that respects constraints structurally rather than by learning
    to avoid a large safety penalty.

The mask is computed by PCMGymEnv.action_masks() (in gym_wrapper.py).
"""

import pathlib
import time
import csv
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS, VAL_YEARS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def train_maskable_ppo(
    regime_id: int   = 0,
    total_timesteps: int = 200_000,
    n_steps:  int    = 8760,
    seed:     int    = 42,
    out_dir:  pathlib.Path | None = None,
    reward_weights: dict | None   = None,
) -> MaskablePPO:
    """Train a MaskablePPO agent with action masking."""
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "maskable_ppo" / f"r{regime_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(PCMGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        years=TRAIN_YEARS, reward_weights=reward_weights,
    ))
    eval_env = Monitor(PCMGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        fixed_year=VAL_YEARS[0], reward_weights=reward_weights,
    ))

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path = str(out_dir / "best_model"),
        log_path             = str(out_dir / "eval_logs"),
        eval_freq            = n_steps * 2,
        n_eval_episodes      = len(VAL_YEARS),
        deterministic        = True,
        verbose              = 1,
    )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        n_steps          = n_steps,
        batch_size        = 256,
        n_epochs          = 10,
        learning_rate     = 3e-4,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.2,
        ent_coef          = 0.005,    # lower entropy: masking already forces exploration
        verbose           = 1,
        seed              = seed,
        device            = "cpu",
    )

    print(f"\n[MaskablePPO] Training — regime {regime_id}, "
          f"{total_timesteps:,} steps, seed {seed}")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                progress_bar=False)
    print(f"[MaskablePPO] Done in {(time.time()-t0)/60:.1f} min")

    model.save(str(out_dir / "maskable_ppo_final"))
    return model


def evaluate_policy(model: MaskablePPO, regime_id: int,
                    years: list[int], reward_weights: dict | None = None) -> list[dict]:
    results = []
    for year in years:
        env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id,
                        fixed_year=year, reward_weights=reward_weights)
        obs, _ = env.reset()
        done   = False
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0
        masked_count = 0

        while not done:
            masks  = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            masked_count += int(not masks[int(action)])  # should be 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_del    += info.get("E_delivered_Wh", 0.0)
            total_unmet  += info.get("E_unmet_Wh",    0.0)
            shield_w     += int(info.get("n_shield_water", 0))
            shield_p     += int(info.get("n_shield_pcm", 0))
            safety_v     += int(info.get("n_safety_violations", 0))
            steps        += 1

        results.append({
            "policy":               "MaskablePPO",
            "regime_id":             regime_id,
            "year":                  year,
            "n_steps":               steps,
            "total_reward":          total_reward,
            "total_delivered_kWh":   total_del / 1000.0,
            "total_unmet_kWh":       total_unmet / 1000.0,
            "total_shield_water":    shield_w,
            "total_shield_pcm":      shield_p,
            "n_safety_violations":   safety_v,
            "masked_violations":     masked_count,  # should always be 0
        })
    return results
