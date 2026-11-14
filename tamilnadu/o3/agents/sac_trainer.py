"""
Phase O3-4 — SAC trainer using Stable-Baselines3.

Soft Actor-Critic is the implementation plan's recommended algorithm for
continuous flow control (plan §5.3). The PCM control task is natively
discrete {CHARGE, IDLE, BYPASS}, so SAC trains on the continuous-action
view (`PCMContinuousGymEnv`), which exposes a 1-D action in [-1, 1] and
bins it back to the three modes. The in-env safety shield is unchanged.

SAC brings entropy-regularised exploration and an off-policy replay
buffer — a different learning regime from PPO (on-policy) and DQN
(value-only), so it is a meaningful comparison point for the thesis.
"""

import pathlib
import time
import csv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from o3.environment.gym_wrapper import (
    PCMContinuousGymEnv, TRAIN_YEARS, VAL_YEARS,
)

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


def train_sac(
    regime_id: int     = 0,
    total_timesteps: int = 200_000,
    seed:    int       = 42,
    out_dir: pathlib.Path | None = None,
    reward_weights: dict | None  = None,
    obs_version: str = "V1",
    forecast_mode: str = "perfect",
    control_step_min: int = 60,
    domain_randomize: bool = False,
) -> SAC:
    if out_dir is None:
        out_dir = pathlib.Path(__file__).resolve().parents[2] / \
            "results" / "sac" / f"r{regime_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(PCMContinuousGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        years=TRAIN_YEARS, reward_weights=reward_weights,
        obs_version=obs_version, forecast_mode=forecast_mode,
        control_step_min=control_step_min,
        domain_randomize=domain_randomize,
    ))
    # evaluation stays nominal (no domain randomisation) even when training
    # is randomised, so the eval reward is comparable across runs. The
    # forecast_mode is NOT reset to "perfect" here — a realistic forecast
    # should degrade eval/deployment too, not just training.
    eval_env = Monitor(PCMContinuousGymEnv(
        CONTRACT_PATH, regime_id=regime_id,
        fixed_year=VAL_YEARS[0], reward_weights=reward_weights,
        obs_version=obs_version, forecast_mode=forecast_mode,
        control_step_min=control_step_min,
        domain_randomize=False,
    ))

    ticks_per_hour = 60 // control_step_min   # scale hour-denominated settings
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(out_dir / "best_model"),
        log_path             = str(out_dir / "eval_logs"),
        eval_freq            = 8760 * 2 * ticks_per_hour,
        n_eval_episodes      = len(VAL_YEARS),
        deterministic        = True,
        verbose              = 1,
    )

    # --- resume support ---------------------------------------------------
    # Long (300k-step, ~5h) SAC runs are vulnerable to the training session
    # being interrupted. A CheckpointCallback periodically saves the model
    # AND its replay buffer (true latest state, not EvalCallback's
    # best-so-far); on relaunch with the same out_dir, train_sac picks up
    # the newest checkpoint and trains only the remaining steps instead of
    # restarting from zero. Safe to ignore when nothing has been saved yet.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_freq = max(2000, (8760 * ticks_per_hour) // 2)  # ~twice per eval cycle
    ckpt_cb = CheckpointCallback(
        save_freq        = ckpt_freq,
        save_path        = str(ckpt_dir),
        name_prefix      = "latest",
        save_replay_buffer = True,
        verbose           = 0,
    )

    existing = sorted(
        ckpt_dir.glob("latest_*_steps.zip"),
        key=lambda p: int(p.stem.split("_")[-2]),
    )
    resumed_steps = 0
    if existing:
        latest_zip = existing[-1]
        resumed_steps = int(latest_zip.stem.split("_")[-2])
        buf_path = ckpt_dir / f"{latest_zip.stem}_replay_buffer.pkl"
        print(f"\n[SAC] Resuming from checkpoint: {latest_zip.name} "
              f"({resumed_steps:,} / {total_timesteps:,} steps already done)")
        model = SAC.load(str(latest_zip), env=train_env, device="cpu")
        if buf_path.exists():
            model.load_replay_buffer(str(buf_path))
            print(f"[SAC] Replay buffer restored ({buf_path.name})")
        else:
            print("[SAC] No saved replay buffer — resuming with an empty one "
                  "(step count/model weights are still correct).")
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate       = 3e-4,
            buffer_size         = 100_000,
            learning_starts     = 8760 * ticks_per_hour,
            batch_size          = 256,
            tau                 = 0.005,
            gamma               = 0.99,
            train_freq          = 4,   # match DQN's update cadence — SAC's
                                        # 347k-param actor+2-critics network
                                        # made train_freq=1 ~6x slower per env
                                        # step than DQN with no measurable
                                        # benefit at this problem's scale
            gradient_steps      = 1,
            ent_coef            = "auto",
            verbose             = 1,
            seed                = seed,
            device              = "cpu",
        )

    remaining = total_timesteps - resumed_steps
    if remaining <= 0:
        print(f"\n[SAC] Already at/past target ({resumed_steps:,} >= "
              f"{total_timesteps:,}) — skipping training, saving final.")
    else:
        print(f"\n[SAC] Training — regime {regime_id}, "
              f"{remaining:,} steps remaining of {total_timesteps:,} total, "
              f"seed {seed}")
        t0 = time.time()
        model.learn(total_timesteps=remaining, callback=[eval_cb, ckpt_cb],
                    reset_num_timesteps=(resumed_steps == 0),
                    progress_bar=False)
        print(f"[SAC] Done in {(time.time()-t0)/60:.1f} min "
              f"(this launch; {resumed_steps:,} steps were already banked "
              f"from earlier launches)")

    model.save(str(out_dir / "sac_final"))
    return model


def evaluate_policy(model: SAC, regime_id: int, years: list[int],
                    reward_weights: dict | None = None,
                    obs_version: str = "V1", forecast_mode: str = "perfect",
                    control_step_min: int = 60,
                    ) -> list[dict]:
    results = []
    for year in years:
        env = PCMContinuousGymEnv(CONTRACT_PATH, regime_id=regime_id,
                                  fixed_year=year, reward_weights=reward_weights,
                                  obs_version=obs_version, forecast_mode=forecast_mode,
                                  control_step_min=control_step_min)
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
            "policy":              "SAC",
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
