"""
o3/evaluation/rollout.py — one function that scores a policy on one regime-year.

The Tamil Nadu code repeated the same 25-line evaluation loop in every trainer and phase script
(and later had to fix the same wrong info keys in 14 places, TN docs/CORRECTIONS.md C-5). Here there
is ONE loop; trainers, phase scripts and the robustness suite call it.

Every row has the TN column names (policy, regime_id, year, n_steps, total_reward,
total_delivered_kWh, total_unmet_kWh, total_shield_water, total_shield_pcm, n_safety_violations)
plus the O2 metrics the plan asks for (plan 1, 8.4): window delivered energy and its gain versus the
water-only twin, shield hours, hours over limit, peak temperatures, switches, energy residual.

`evaluate_model()` and `evaluate_baseline()` are module-level job functions, so they can be farmed
out one-process-per-job by o3.utils.parallel.run_parallel; each builds its own environment, which
makes the result independent of the number of workers.
"""

import pathlib
import numpy as np

import config_O2
from o3.environment.pcm_env import PCMEnv, CHARGE, IDLE, BYPASS
from o3.environment.gym_wrapper import action_mask_from_obs, PCMContinuousGymEnv

CONTRACT_PATH = config_O2.CONTRACT_FILE

# model kinds -> how to ask the model for an action
KIND_STANDARD = "standard"      # PPO, DQN, QRDQN
KIND_RECURRENT = "recurrent"    # RecurrentPPO (carries lstm_state)
KIND_MASKABLE = "maskable"      # MaskablePPO (needs action_masks)
KIND_SAC = "sac"                # SAC (continuous action binned to modes)

ALGO_KIND = {"ppo": KIND_STANDARD, "dqn": KIND_STANDARD, "qrdqn": KIND_STANDARD,
             "recurrent": KIND_RECURRENT, "maskable": KIND_MASKABLE, "sac": KIND_SAC}
ALGO_DIR = {"ppo": "ppo", "dqn": "dqn", "recurrent": "recurrent_ppo", "qrdqn": "qrdqn",
            "maskable": "maskable_ppo", "sac": "sac"}
ALGO_LABEL = {"ppo": "PPO", "dqn": "DQN", "recurrent": "RecurrentPPO", "qrdqn": "QRDQN",
              "maskable": "MaskablePPO", "sac": "SAC"}


class ModelPolicy:
    """Adapter: SB3 model -> act(obs) / reset(), whatever the algorithm needs."""

    def __init__(self, model, kind: str):
        self.model, self.kind = model, kind
        self.reset()

    def reset(self) -> None:
        self._state = None
        self._start = True

    def act(self, obs, info=None) -> int:
        o = np.asarray(obs, dtype=np.float32)
        if self.kind == KIND_STANDARD:
            return int(self.model.predict(o, deterministic=True)[0])
        if self.kind == KIND_RECURRENT:
            a, self._state = self.model.predict(o, state=self._state, episode_start=np.array([self._start]), deterministic=True)
            self._start = False
            return int(a)
        if self.kind == KIND_MASKABLE:
            return int(self.model.predict(o, action_masks=action_mask_from_obs(o), deterministic=True)[0])
        if self.kind == KIND_SAC:
            return PCMContinuousGymEnv._to_discrete(self.model.predict(o, deterministic=True)[0])
        raise ValueError(self.kind)


def run_policy_episode(env: PCMEnv, policy, year: int, initial_pcm_state: str = "solid",
                       policy_name: str | None = None, seed: int | None = None) -> dict:
    """Run one full episode of `policy` (object with act(obs, info) and reset()) and return a result row."""
    obs, info = env.reset(year=year, initial_pcm_state=initial_pcm_state, seed=seed)
    policy.reset()
    if hasattr(policy, "attach_env"):
        policy.attach_env(env)
    total_reward = 0.0
    done = False
    while not done:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    s = env.episode_summary()
    st = env._state
    return {
        "policy": policy_name or getattr(policy, "name", type(policy).__name__),
        "regime_id": env.regime_id, "year": year, "pcm_init": initial_pcm_state,
        "n_steps": s["n_steps"], "total_reward": total_reward,
        "total_delivered_kWh": s["E_del45_annual_kWh"], "total_unmet_kWh": s["E_unmet_kWh"],
        "total_shield_water": st["n_shield_water"], "total_shield_pcm": st["n_shield_pcm"],
        "n_safety_violations": st["n_safety_violations"], "action_switches": st["action_switches"],
        "final_T_w_C": s["final_T_w_C"], "final_T_pcm_C": s["final_T_pcm_C"], "final_f_melt": s["final_f_melt"],
        "E_del45_window_kWh": s["E_del45_window_kWh"], "serve_frac_45_window_insitu": s["serve_frac_45_window_insitu"],
        "shield_water_h": s["shield_water_h"], "shield_pcm_h": s["shield_pcm_h"],
        "hours_over_limit": s["hours_over_limit"], "t_top_max_C": s["t_top_max_C"], "t_pcm_max_C": s["t_pcm_max_C"],
        "bypass_switches": s["bypass_switches"], "residual_pct_of_collector": s["residual_pct_of_collector"],
        "config_sha": getattr(env, "config_sha", "")[:16],
    }


# ---------------------------------------------------------------------------------------------
# Job functions (module level -> picklable -> usable with o3.utils.parallel.run_parallel)
# ---------------------------------------------------------------------------------------------

def make_baseline(name: str, **kw):
    """Policy object for a baseline name: b0 | b1 | b2 | b4."""
    if name == "b0":
        from o3.baselines.b0_fixed import FixedChargePolicy
        return FixedChargePolicy()
    if name == "b1":
        from o3.baselines.b1_reactive import ReactiveRulePolicy
        return ReactiveRulePolicy(**kw)
    if name == "b2":
        from o3.baselines.b2_forecast import ForecastAwarePolicy
        return ForecastAwarePolicy(**kw)
    if name == "b4":
        from o3.baselines.b4_random import RandomSafePolicy
        return RandomSafePolicy(**kw)
    raise ValueError(name)


def evaluate_baseline(baseline: str, regime_id: int, year: int, initial_pcm_state: str = "solid",
                      contract_path=CONTRACT_PATH, reward_weights: dict | None = None,
                      policy_kwargs: dict | None = None, seed: int = 0) -> dict:
    env = PCMEnv(contract_path, regime_id=regime_id, year=year, reward_weights=reward_weights)
    return run_policy_episode(env, make_baseline(baseline, **(policy_kwargs or {})), year, initial_pcm_state, seed=seed)


def evaluate_twin(regime_id: int, year: int, initial_pcm_state: str = "solid", contract_path=CONTRACT_PATH,
                  reward_weights: dict | None = None) -> dict:
    """Water-only twin: same tank, flow, shield, weather and demand, zero capsules, loop always enabled."""
    from o3.baselines.b0_fixed import FixedChargePolicy
    env = PCMEnv(contract_path, regime_id=regime_id, year=year, reward_weights=reward_weights, water_only=True)
    row = run_policy_episode(env, FixedChargePolicy(), year, initial_pcm_state, policy_name="TWIN_water_only")
    return row


def load_model(algo: str, path: str | pathlib.Path):
    """Load a saved SB3 / sb3-contrib model by algorithm key (ppo, dqn, recurrent, qrdqn, maskable, sac)."""
    if algo == "ppo":
        from stable_baselines3 import PPO as M
    elif algo == "dqn":
        from stable_baselines3 import DQN as M
    elif algo == "sac":
        from stable_baselines3 import SAC as M
    elif algo == "recurrent":
        from sb3_contrib import RecurrentPPO as M
    elif algo == "qrdqn":
        from sb3_contrib import QRDQN as M
    elif algo == "maskable":
        from sb3_contrib import MaskablePPO as M
    else:
        raise ValueError(algo)
    return M.load(str(path), device="cpu")


def evaluate_model(algo: str, model_path: str, regime_id: int, year: int, initial_pcm_state: str = "solid",
                   contract_path=CONTRACT_PATH, reward_weights: dict | None = None,
                   policy_name: str | None = None, lock_run_id: str | None = None, lock_path: str | None = None) -> dict:
    """Load a saved model and score it on one regime-year (one process per call when run in parallel)."""
    if lock_run_id is not None:                       # test years: the parent holds the O3 lock
        from o3.utils import weather_roles
        weather_roles.assume_test_lock(lock_run_id, lock_path)
    import torch
    torch.set_num_threads(1)
    model = load_model(algo, model_path)
    env = PCMEnv(contract_path, regime_id=regime_id, year=year, reward_weights=reward_weights)
    pol = ModelPolicy(model, ALGO_KIND[algo])
    return run_policy_episode(env, pol, year, initial_pcm_state, policy_name or ALGO_LABEL[algo])


def evaluate_policy_object(model, kind: str, regime_id: int, years: list[int], policy_name: str,
                           initial_pcm_state: str = "solid", reward_weights: dict | None = None,
                           contract_path=CONTRACT_PATH) -> list[dict]:
    """In-process scoring of an in-memory model (used right after training and by the trainers)."""
    env = PCMEnv(contract_path, regime_id=regime_id, year=years[0], reward_weights=reward_weights)
    pol = ModelPolicy(model, kind)
    return [run_policy_episode(env, pol, y, initial_pcm_state, policy_name) for y in years]


def add_twin_gain(rows: list[dict], twin_rows: list[dict]) -> list[dict]:
    """Attach gain_window_kWh = E_del45_window(policy) - E_del45_window(water-only twin), matched on (regime, year, pcm_init)."""
    key = {(t["regime_id"], t["year"], t["pcm_init"]): t for t in twin_rows}
    for r in rows:
        t = key.get((r["regime_id"], r["year"], r["pcm_init"]))
        if t is not None:
            r["twin_E_del45_window_kWh"] = t["E_del45_window_kWh"]
            r["gain_window_kWh"] = r["E_del45_window_kWh"] - t["E_del45_window_kWh"]
            r["gain_annual_kWh"] = r["total_delivered_kWh"] - t["total_delivered_kWh"]
    return rows
