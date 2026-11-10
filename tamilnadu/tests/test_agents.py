"""
Phase O3-4 tests — smoke tests for all DRL agents.

Tests verify (without full training):
  - gymnasium wrapper is SB3-compatible
  - PPO, DQN, RecurrentPPO, QRDQN, MaskablePPO each run 500 steps
  - action space and observation space are correct
  - action masking works correctly
  - trained policy produces valid actions
"""

import pathlib
import numpy as np
import pytest

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"

SMOKE_STEPS = 500


@pytest.fixture(scope="module")
def gym_env():
    from o3.environment.gym_wrapper import PCMGymEnv
    return PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016)


# ------------------------------------------------------------------
# Gymnasium wrapper
# ------------------------------------------------------------------

def test_gym_env_spaces(gym_env):
    from gymnasium import spaces
    assert isinstance(gym_env.action_space,      spaces.Discrete)
    assert isinstance(gym_env.observation_space, spaces.Box)
    assert gym_env.action_space.n == 3
    assert gym_env.observation_space.shape == (11,)


def test_gymnasium_env_checker(gym_env):
    from stable_baselines3.common.env_checker import check_env
    check_env(gym_env, warn=True)


def test_gym_reset_obs_in_space(gym_env):
    obs, _ = gym_env.reset(seed=0)
    assert gym_env.observation_space.contains(obs)


def test_gym_step_obs_in_space(gym_env):
    obs, _ = gym_env.reset(seed=0)
    obs2, reward, terminated, truncated, info = gym_env.step(0)
    assert gym_env.observation_space.contains(obs2)
    assert isinstance(reward, float)


# ------------------------------------------------------------------
# Action masking
# ------------------------------------------------------------------

def test_action_masks_shape(gym_env):
    gym_env.reset(seed=0)
    masks = gym_env.action_masks()
    assert masks.shape == (3,)
    assert masks.dtype == bool


def test_action_masks_idle_always_allowed(gym_env):
    """IDLE and BYPASS must always be unmasked."""
    gym_env.reset(seed=0)
    for _ in range(24):   # run a day, check each step
        masks = gym_env.action_masks()
        assert masks[1], "IDLE must never be masked"
        assert masks[2], "BYPASS must never be masked"
        gym_env.step(1)


# ------------------------------------------------------------------
# PPO smoke
# ------------------------------------------------------------------

def test_ppo_smoke(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv
    env   = Monitor(PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016))
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64,
                seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action) <= 2


def test_ppo_save_load(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv
    env   = Monitor(PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016))
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64,
                seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    model.save(str(tmp_path / "ppo_test"))
    loaded = PPO.load(str(tmp_path / "ppo_test"))
    obs, _ = env.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    assert 0 <= int(action) <= 2


# ------------------------------------------------------------------
# DQN smoke
# ------------------------------------------------------------------

def test_dqn_smoke(tmp_path):
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv
    env   = Monitor(PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016))
    model = DQN("MlpPolicy", env, learning_starts=100, buffer_size=1000,
                seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action) <= 2


# ------------------------------------------------------------------
# RecurrentPPO smoke
# ------------------------------------------------------------------

def test_recurrent_ppo_smoke(tmp_path):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv
    env   = Monitor(PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016))
    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        n_steps=256, batch_size=64,
        policy_kwargs=dict(lstm_hidden_size=32, n_lstm_layers=1,
                           shared_lstm=False, enable_critic_lstm=True),
        seed=0, device="cpu", verbose=0,
    )
    model.learn(SMOKE_STEPS)
    obs, _     = env.reset()
    lstm_state = None
    action, lstm_state = model.predict(obs, state=lstm_state,
                                       episode_start=True, deterministic=True)
    assert 0 <= int(action) <= 2
    assert lstm_state is not None, "LSTM state should be non-None after first step"


# ------------------------------------------------------------------
# QRDQN smoke
# ------------------------------------------------------------------

def test_qrdqn_smoke(tmp_path):
    from sb3_contrib import QRDQN
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv
    env   = Monitor(PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016))
    model = QRDQN(
        "MlpPolicy", env,
        learning_starts=100, buffer_size=1000,
        policy_kwargs=dict(n_quantiles=10),
        seed=0, device="cpu", verbose=0,
    )
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action) <= 2


# ------------------------------------------------------------------
# MaskablePPO smoke
# ------------------------------------------------------------------

def test_maskable_ppo_smoke(tmp_path):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv
    env   = Monitor(PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016))
    model = MaskablePPO(
        "MlpPolicy", env,
        n_steps=256, batch_size=64,
        seed=0, device="cpu", verbose=0,
    )
    model.learn(SMOKE_STEPS)
    # predict with masking
    inner_env = PCMGymEnv(CONTRACT_PATH, regime_id=0, fixed_year=2016)
    obs, _    = inner_env.reset()
    masks     = inner_env.action_masks()
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert 0 <= int(action) <= 2
    # action must respect the mask
    assert masks[int(action)], \
        f"MaskablePPO selected a masked action {action}! Masks: {masks}"
