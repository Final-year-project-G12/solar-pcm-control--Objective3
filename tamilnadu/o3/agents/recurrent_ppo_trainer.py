"""
Phase O3-4 — Recurrent PPO (LSTM) trainer.

Uses sb3-contrib's RecurrentPPO which wraps PPO with an LSTM policy
(LstmPolicy). The agent maintains a hidden state across timesteps,
allowing it to learn temporal patterns like:
  - monsoon onset / cloud sequences (Tamil Nadu NE monsoon)
  - delayed PCM discharge after charge
  - day-ahead demand anticipation

Scientific justification: a memoryless policy can only react to the
current observation. The LSTM variant can smooth over noisy GHI
measurements and anticipate energy needs from accumulated context.
"""

import pathlib
import time
import csv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS, VAL_YEARS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def train_recurrent_ppo(
    regime_id: int   = 0,
    total_timesteps: int = 200_000,
    n_steps:  int    = 8760,
    lstm_hidden: int = 64,
    seed:     int    = 42,
    out_dir:  pathlib.Path | None = None,
    reward_weights: dict | None   = None,
) -> RecurrentPPO:
    """Train a Recurrent PPO (LSTM) agent.

    Parameters
    ----------
    lstm_hidden : size of the LSTM hidden state (64 = fast, 128 = stronger)
    """
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "recurrent_ppo" / f"r{regime_id}"
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
        eval_freq            = n_steps * 2,
        n_eval_episodes      = len(VAL_YEARS),
        deterministic        = True,
        verbose              = 1,
    )

    policy_kwargs = dict(
        lstm_hidden_size   = lstm_hidden,
        n_lstm_layers      = 1,
        shared_lstm        = False,       # separate actor / critic LSTMs
        enable_critic_lstm = True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        n_steps          = n_steps,
        batch_size        = 256,
        n_epochs          = 10,
        learning_rate     = 3e-4,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.2,
        ent_coef          = 0.01,
        policy_kwargs     = policy_kwargs,
        verbose           = 1,
        seed              = seed,
        device            = "cpu",
    )

    print(f"\n[RecurrentPPO] Training — regime {regime_id}, "
          f"{total_timesteps:,} steps, LSTM hidden={lstm_hidden}, seed {seed}")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                progress_bar=False)
    print(f"[RecurrentPPO] Done in {(time.time()-t0)/60:.1f} min")

    model.save(str(out_dir / "recurrent_ppo_final"))
    return model


def evaluate_policy(model: RecurrentPPO, regime_id: int,
                    years: list[int], reward_weights: dict | None = None) -> list[dict]:
    results = []
    for year in years:
        env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id,
                        fixed_year=year, reward_weights=reward_weights)
        obs, _    = env.reset()
        lstm_state = None
        done  = False
        episode_start = True
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0

        while not done:
            action, lstm_state = model.predict(
                obs, state=lstm_state,
                episode_start=episode_start, deterministic=True,
            )
            episode_start = False
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
            "policy":               "RecurrentPPO",
            "regime_id":             regime_id,
            "year":                  year,
            "n_steps":               steps,
            "total_reward":          total_reward,
            "total_delivered_kWh":   total_del / 1000.0,
            "total_unmet_kWh":       total_unmet / 1000.0,
            "total_shield_water":    shield_w,
            "total_shield_pcm":      shield_p,
            "n_safety_violations":   safety_v,
        })
    return results
