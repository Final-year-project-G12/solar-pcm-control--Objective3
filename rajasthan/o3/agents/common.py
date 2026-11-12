"""
o3/agents/common.py — shared plumbing for the six trainers (environment factories, vec-env choice).

Parallel training (Step 3):
  * ON-POLICY agents (PPO, MaskablePPO, RecurrentPPO) collect experience from `n_envs` environments
    in separate processes (SubprocVecEnv, 'spawn' start method — the only one Windows has).
    `rollout_steps_per_env(total, n_envs)` keeps the number of samples per policy update the same as
    the Tamil Nadu reference (8760 = one simulated year): with n_envs = 1, 2, 3, 4, 5, 6, 8, 10, 12, 15
    the per-update batch is exactly 8760.
  * OFF-POLICY agents (DQN, QRDQN, SAC) use a DummyVecEnv with ONE env: their cost is the replay-buffer
    gradient steps in the main process, and extra environments would change the updates-per-sample
    ratio of the algorithm. `n_envs` is accepted for a uniform CLI and ignored (a note is printed).
  * Results are reproducible for a fixed (seed, n_envs). SubprocVecEnv and DummyVecEnv with the same
    n_envs and seed give the same trajectories (checked in tests/test_parallel.py); changing n_envs
    changes the experience stream (as in any vectorised RL) and is therefore recorded in every result.
"""

import functools
import math
import pathlib

import config_O2
from o3.environment.gym_wrapper import PCMGymEnv, PCMContinuousGymEnv
from o3.utils.weather_roles import TRAIN_YEARS, VAL_YEARS

CONTRACT_PATH = config_O2.CONTRACT_FILE
RESULTS_DIR = config_O2.RESULTS_DIR


def default_out_dir(algo_dir: str, regime_id: int) -> pathlib.Path:
    return RESULTS_DIR / algo_dir / f"r{regime_id}"


def make_env(regime_id: int, years: list[int], fixed_year: int | None = None, cycle_years: bool = False,
             reward_weights: dict | None = None, continuous: bool = False, contract_path=CONTRACT_PATH,
             obs_version: str = "V2", demand_mode: str = "deterministic", control_step_min: int = 60,
             domain_randomize: bool = False):
    """A Monitor-wrapped PCMGymEnv. Module-level so it can be pickled into worker processes."""
    from stable_baselines3.common.monitor import Monitor
    cls = PCMContinuousGymEnv if continuous else PCMGymEnv
    return Monitor(cls(contract_path, regime_id=regime_id, years=years, fixed_year=fixed_year,
                       cycle_years=cycle_years, reward_weights=reward_weights, obs_version=obs_version,
                       demand_mode=demand_mode, control_step_min=control_step_min, domain_randomize=domain_randomize))


def make_train_vec(regime_id: int, n_envs: int = 1, vec: str = "auto", reward_weights: dict | None = None,
                   continuous: bool = False, years: list[int] | None = None, demand_mode: str = "deterministic",
                   obs_version: str = "V2", control_step_min: int = 60, domain_randomize: bool = False):
    """Training VecEnv. vec: 'auto' (Subproc if n_envs > 1) | 'subproc' | 'dummy'."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    fn = functools.partial(make_env, regime_id, list(years or TRAIN_YEARS), None, False, reward_weights, continuous,
                           CONTRACT_PATH, obs_version, demand_mode, control_step_min, domain_randomize)
    use_subproc = vec == "subproc" or (vec == "auto" and n_envs > 1)
    if use_subproc:
        return SubprocVecEnv([fn] * n_envs, start_method="spawn")
    return DummyVecEnv([fn] * n_envs)


def make_eval_vec(regime_id: int, reward_weights: dict | None = None, continuous: bool = False,
                  obs_version: str = "V2", control_step_min: int = 60):
    """Validation VecEnv: cycles through the validation years 2021-2023 (one env, deterministic order).
    Domain randomisation is always OFF here so validation reward stays comparable across runs (plan 19.1)."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    fn = functools.partial(make_env, regime_id, list(VAL_YEARS), None, True, reward_weights, continuous,
                           CONTRACT_PATH, obs_version, "deterministic", control_step_min, False)
    return DummyVecEnv([fn])


def rollout_steps_per_env(total_per_update: int, n_envs: int) -> int:
    """n_steps per environment so that n_steps * n_envs ~= total_per_update (exact when n_envs | 8760)."""
    return max(1, int(math.ceil(total_per_update / max(1, n_envs))))


def pin_main_threads(n: int = 1) -> None:
    import torch
    torch.set_num_threads(n)


def eval_freq_calls(episodes_between_evals: float, n_envs: int, steps_per_episode: int = 8760) -> int:
    """EvalCallback counts calls (vec steps): convert 'every k episodes of experience' to calls."""
    return max(1, int(episodes_between_evals * steps_per_episode / max(1, n_envs)))


def note_offpolicy(n_envs: int, algo: str) -> None:
    if n_envs and n_envs > 1:
        print(f"[{algo}] off-policy agent: using 1 environment (n_envs={n_envs} ignored; see o3/agents/common.py)")
