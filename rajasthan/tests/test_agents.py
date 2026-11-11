"""
Phase O3-4 tests — smoke tests for all DRL agents.

Tests verify (without full training):
  - the gymnasium wrapper is SB3-compatible and has the right spaces (13 contract fields, 3 actions)
  - PPO, DQN, RecurrentPPO, QRDQN, MaskablePPO and SAC each run a few hundred steps
  - action masking works
  - a trained policy produces valid actions and can be scored with the shared rollout code
"""

import numpy as np
import pytest

import config_O2

CONTRACT_PATH = config_O2.CONTRACT_FILE
REGIME = 1
SMOKE_STEPS = 500


@pytest.fixture(scope="module")
def gym_env():
    from o3.environment.gym_wrapper import PCMGymEnv
    return PCMGymEnv(CONTRACT_PATH, regime_id=REGIME, fixed_year=2016)


def _mon(continuous=False):
    from stable_baselines3.common.monitor import Monitor
    from o3.environment.gym_wrapper import PCMGymEnv, PCMContinuousGymEnv
    return Monitor((PCMContinuousGymEnv if continuous else PCMGymEnv)(CONTRACT_PATH, regime_id=REGIME, fixed_year=2016))


# ------------------------------------------------------------------ gymnasium wrapper

def test_gym_env_spaces(gym_env):
    from gymnasium import spaces
    assert isinstance(gym_env.action_space, spaces.Discrete) and gym_env.action_space.n == 3
    assert isinstance(gym_env.observation_space, spaces.Box) and gym_env.observation_space.shape == (13,)


def test_gymnasium_env_checker(gym_env):
    from stable_baselines3.common.env_checker import check_env
    check_env(gym_env, warn=True)


def test_gym_reset_obs_in_space(gym_env):
    obs, _ = gym_env.reset(seed=0)
    assert gym_env.observation_space.contains(obs)


def test_gym_step_obs_in_space(gym_env):
    gym_env.reset(seed=0)
    obs2, reward, terminated, truncated, info = gym_env.step(0)
    assert gym_env.observation_space.contains(obs2) and isinstance(reward, float)


def test_training_env_samples_years_and_reset_scenarios():
    from o3.environment.gym_wrapper import PCMGymEnv
    env = PCMGymEnv(CONTRACT_PATH, regime_id=REGIME)
    seen_years, seen_inits = set(), set()
    for s in range(12):
        _, info = env.reset(seed=s)
        seen_years.add(info["year"]); seen_inits.add(info["initial_pcm_state"])
    assert seen_years <= {2016, 2017, 2018, 2019, 2020} and len(seen_years) > 1
    assert seen_inits <= {"solid", "partial", "liquid"} and len(seen_inits) > 1


def test_eval_env_cycles_validation_years():
    from o3.environment.gym_wrapper import PCMGymEnv
    from o3.utils.weather_roles import VAL_YEARS
    env = PCMGymEnv(CONTRACT_PATH, regime_id=REGIME, years=VAL_YEARS, cycle_years=True)
    got = [env.reset(seed=0)[1]["year"]] + [env.reset()[1]["year"] for _ in range(3)]
    assert got == [2021, 2022, 2023, 2021]


# ------------------------------------------------------------------ action masking

def test_action_masks_shape(gym_env):
    gym_env.reset(seed=0)
    masks = gym_env.action_masks()
    assert masks.shape == (3,) and masks.dtype == bool


def test_action_masks_idle_always_allowed(gym_env):
    gym_env.reset(seed=0)
    for _ in range(24):
        masks = gym_env.action_masks()
        assert masks[1] and masks[2]
        gym_env.step(1)


# ------------------------------------------------------------------ smoke training

def test_ppo_smoke(tmp_path):
    from stable_baselines3 import PPO
    env = _mon()
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    assert 0 <= int(model.predict(obs, deterministic=True)[0]) <= 2


def test_ppo_save_load(tmp_path):
    from stable_baselines3 import PPO
    env = _mon()
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    model.save(str(tmp_path / "ppo_test"))
    loaded = PPO.load(str(tmp_path / "ppo_test"))
    obs, _ = env.reset()
    assert 0 <= int(loaded.predict(obs, deterministic=True)[0]) <= 2


def test_dqn_smoke():
    from stable_baselines3 import DQN
    env = _mon()
    model = DQN("MlpPolicy", env, learning_starts=100, buffer_size=1000, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    assert 0 <= int(model.predict(obs, deterministic=True)[0]) <= 2


def test_recurrent_ppo_smoke():
    from sb3_contrib import RecurrentPPO
    env = _mon()
    model = RecurrentPPO("MlpLstmPolicy", env, n_steps=256, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    action, state = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
    assert 0 <= int(action) <= 2 and state is not None


def test_qrdqn_smoke():
    from sb3_contrib import QRDQN
    env = _mon()
    model = QRDQN("MlpPolicy", env, learning_starts=100, buffer_size=1000, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    assert 0 <= int(model.predict(obs, deterministic=True)[0]) <= 2


def test_maskable_ppo_smoke():
    from sb3_contrib import MaskablePPO
    from o3.environment.gym_wrapper import action_mask_from_obs
    env = _mon()
    model = MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    a = int(model.predict(obs, action_masks=action_mask_from_obs(obs), deterministic=True)[0])
    assert 0 <= a <= 2


def test_sac_smoke():
    from stable_baselines3 import SAC
    env = _mon(continuous=True)
    model = SAC("MlpPolicy", env, learning_starts=100, buffer_size=1000, seed=0, device="cpu", verbose=0)
    model.learn(SMOKE_STEPS)
    obs, _ = env.reset()
    assert model.predict(obs, deterministic=True)[0].shape == (1,)


def test_trained_policy_can_be_scored_with_shared_rollout(tmp_path):
    from stable_baselines3 import PPO
    from o3.evaluation.rollout import evaluate_policy_object, KIND_STANDARD
    model = PPO("MlpPolicy", _mon(), n_steps=256, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(300)
    rows = evaluate_policy_object(model, KIND_STANDARD, REGIME, [2021], "PPO")
    assert rows[0]["n_steps"] == 8760 and rows[0]["n_safety_violations"] == 0
