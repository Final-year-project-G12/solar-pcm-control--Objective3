"""
Phase O3-4 — Quantile Regression DQN (QRDQN) trainer.

QRDQN (Dabney et al. 2017) learns the full *distribution* of returns,
not just the expectation. Each action's value is represented as N quantiles
of its return distribution.

Why QRDQN for PCM solar control:
  - Solar irradiance is stochastic: same action can have very different returns
    on a clear vs cloudy day.
  - QRDQN naturally captures this uncertainty and can be used risk-sensitively:
    optimise the lower tail (CVaR) to minimise worst-case hot water shortfall.
  - A distributional agent also tends to be more stable under reward noise from
    the shield's override penalties.

QRDQN is available via sb3-contrib.
"""

import pathlib
import time
import csv

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS, VAL_YEARS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def train_qrdqn(
    regime_id: int     = 0,
    total_timesteps: int = 200_000,
    n_quantiles: int   = 50,
    seed:    int       = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None  = None,
) -> QRDQN:
    """Train a QRDQN agent.

    Parameters
    ----------
    n_quantiles : number of quantiles to fit (50 = standard, 200 = richer)
    """
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "qrdqn" / f"r{regime_id}"
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

    model = QRDQN(
        "MlpPolicy",
        train_env,
        learning_rate          = 1e-4,
        buffer_size            = 50_000,
        learning_starts        = 8760,
        batch_size             = 256,
        tau                    = 1.0,
        gamma                  = 0.99,
        train_freq             = 4,
        gradient_steps         = 1,
        target_update_interval = 1000,
        exploration_fraction   = 0.3,
        exploration_final_eps  = 0.05,
        policy_kwargs          = dict(n_quantiles=n_quantiles),
        verbose                = 1,
        seed                   = seed,
        device                 = "cpu",
    )

    print(f"\n[QRDQN] Training — regime {regime_id}, "
          f"{total_timesteps:,} steps, {n_quantiles} quantiles, seed {seed}")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                progress_bar=False)
    print(f"[QRDQN] Done in {(time.time()-t0)/60:.1f} min")

    model.save(str(out_dir / "qrdqn_final"))
    return model


def evaluate_policy(model: QRDQN, regime_id: int,
                    years: list[int], reward_weights: dict | None = None,
                    risk_level: float = 0.5) -> list[dict]:
    """Evaluate QRDQN policy.

    risk_level : quantile index used for action selection
        0.5  = expectation (neutral)
        0.1  = pessimistic / risk-averse (lower tail)
        0.9  = optimistic (upper tail)
    """
    results = []
    for year in years:
        env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year,
                        reward_weights=reward_weights)
        obs, _ = env.reset()
        done   = False
        total_reward = total_del = total_unmet = 0.0
        shield_w = shield_p = safety_v = steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
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
            "policy":               f"QRDQN_q{risk_level}",
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
