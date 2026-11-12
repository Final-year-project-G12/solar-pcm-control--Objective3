"""
o3/agents/gen2.py — one training run on the upgraded ("Generation-2", plan section 19) environment, and its scoring.

A run is fully described by (algo, regime, steps, seed, shield weight, obs_version, control_step_min, domain_randomize).
It is written to results/gen2/<tag>/ together with run_meta.json, so that any later script can rebuild the SAME
environment configuration to score the model (a V3 / 15-minute model cannot be scored in a V2 / hourly environment).

  train_run(...)        train and save; training env uses domain randomisation if asked, validation env never does
  evaluate_run(...)     score a saved run on one (year, reset scenario) — nominal weather, the run's own obs/cadence
Both are module-level and picklable: one process per run / per episode through o3.utils.parallel.run_parallel.
Hyper-parameters are the same as the per-algorithm trainers (TN values, unswept).
"""

import json
import pathlib
import time

import config_O2
from o3.agents import common as C
from o3.environment.pcm_env import PCMEnv, default_w_shield
from o3.evaluation.rollout import ModelPolicy, ALGO_KIND, ALGO_LABEL, load_model, run_policy_episode

ONPOLICY = {"ppo", "maskable", "recurrent"}


def gen2_dir(results_dir=None) -> pathlib.Path:
    return (pathlib.Path(results_dir) if results_dir else config_O2.RESULTS_DIR) / "gen2"


def default_tag(algo, regime_id, seed, shield_weight, obs_version, control_step_min, domain_randomize) -> str:
    return (f"{algo}_r{regime_id}_w{shield_weight:g}_s{seed}_{obs_version}_{control_step_min}min"
            f"{'_dr' if domain_randomize else ''}")


def _model(algo: str, env, seed: int, n_steps_total: int, learning_starts: int):
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50_000, learning_starts=learning_starts, batch_size=256,
                   tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000,
                   exploration_fraction=0.3, exploration_final_eps=0.05, verbose=0, seed=seed, device="cpu")
    if algo == "qrdqn":
        from sb3_contrib import QRDQN
        return QRDQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50_000, learning_starts=learning_starts, batch_size=256,
                     tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000,
                     exploration_fraction=0.3, exploration_final_eps=0.05, policy_kwargs=dict(n_quantiles=50),
                     verbose=0, seed=seed, device="cpu")
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000, learning_starts=learning_starts, batch_size=256,
                   tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef="auto", verbose=0, seed=seed, device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO("MlpPolicy", env, n_steps=n_steps_total, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                   gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, verbose=0, seed=seed, device="cpu")
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, n_steps=n_steps_total, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                           gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, verbose=0, seed=seed, device="cpu")
    if algo == "recurrent":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO("MlpLstmPolicy", env, n_steps=n_steps_total, batch_size=256, n_epochs=10, learning_rate=3e-4,
                            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                            policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True),
                            verbose=0, seed=seed, device="cpu")
    raise ValueError(algo)


def train_run(algo: str, regime_id: int = 1, steps: int = 300_000, seed: int = 42, shield_weight: float | None = None,
              obs_version: str = "V3", control_step_min: int = 60, domain_randomize: bool = True, tag: str | None = None,
              results_dir: str | None = None, eval_every_steps: int = 35_040) -> dict:
    """Train one run and save it under results/gen2/<tag>/. Returns the run_meta dict."""
    from stable_baselines3.common.callbacks import EvalCallback
    w = default_w_shield(regime_id) if shield_weight is None else float(shield_weight)
    tag = tag or default_tag(algo, regime_id, seed, w, obs_version, control_step_min, domain_randomize)
    out = gen2_dir(results_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    C.pin_main_threads(1)
    rw = {"shield": w}
    cont = algo == "sac"
    tph = 60 // control_step_min
    train_env = C.make_train_vec(regime_id, 1, "dummy", rw, continuous=cont, obs_version=obs_version,
                                 control_step_min=control_step_min, domain_randomize=domain_randomize)
    eval_env = C.make_eval_vec(regime_id, rw, continuous=cont, obs_version=obs_version, control_step_min=control_step_min)
    if algo == "maskable":
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as CB
    else:
        CB = EvalCallback
    cb = CB(eval_env, best_model_save_path=str(out / "best_model"), log_path=str(out / "eval_logs"),
            eval_freq=min(eval_every_steps * tph, max(1, steps)), n_eval_episodes=3, deterministic=True, verbose=0)
    model = _model(algo, train_env, seed, 8760 * tph, min(8760 * tph, max(100, steps // 4)))
    t0 = time.time()
    try:
        model.learn(total_timesteps=steps, callback=cb, progress_bar=False)
    finally:
        train_env.close()
        eval_env.close()
    model.save(str(out / "final_model"))
    meta = {"tag": tag, "algo": algo, "regime_id": regime_id, "steps": steps, "seed": seed, "shield_weight": w,
            "obs_version": obs_version, "control_step_min": control_step_min, "domain_randomize": domain_randomize,
            "train_minutes": (time.time() - t0) / 60.0, "generation": 2}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[gen2] {tag}: trained in {meta['train_minutes']:.1f} min", flush=True)
    return meta


def run_model_path(run_dir: pathlib.Path) -> pathlib.Path:
    best = run_dir / "best_model" / "best_model.zip"
    return best if best.exists() else run_dir / "final_model.zip"


def evaluate_run(run_dir: str, year: int, initial_pcm_state: str = "solid", lock_run_id: str | None = None,
                 lock_path: str | None = None, checkpoint: str = "best") -> dict:
    """Score a saved Gen-2 run on one year, in ITS OWN observation layout and control cadence, nominal weather."""
    if lock_run_id is not None:
        from o3.utils import weather_roles
        weather_roles.assume_test_lock(lock_run_id, lock_path)
    import torch
    torch.set_num_threads(1)
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    path = run_model_path(run_dir) if checkpoint == "best" else run_dir / "final_model.zip"
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=meta["regime_id"], year=year, obs_version=meta["obs_version"],
                 control_step_min=meta["control_step_min"], reward_weights={"shield": meta["shield_weight"]})
    pol = ModelPolicy(load_model(meta["algo"], path), ALGO_KIND[meta["algo"]])
    row = run_policy_episode(env, pol, year, initial_pcm_state, ALGO_LABEL[meta["algo"]])
    row.update({k: meta[k] for k in ("tag", "algo", "seed", "shield_weight", "obs_version", "control_step_min", "domain_randomize")})
    return row


def evaluate_baseline_as(baseline: str, regime_id: int, year: int, control_step_min: int = 60, shield_weight: float | None = None,
                         initial_pcm_state: str = "solid", water_only: bool = False) -> dict:
    """B0 / B1 / twin at a given control cadence (so a 15-minute agent is compared with a 15-minute baseline)."""
    from o3.evaluation.rollout import make_baseline
    rw = None if shield_weight is None else {"shield": shield_weight}
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=regime_id, year=year, control_step_min=control_step_min,
                 reward_weights=rw, water_only=water_only)
    row = run_policy_episode(env, make_baseline(baseline), year, initial_pcm_state,
                             "TWIN_water_only" if water_only else None)
    row["control_step_min"] = control_step_min
    return row
