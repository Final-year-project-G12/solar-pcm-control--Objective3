"""
Phase O3-4 — Gymnasium wrapper for PCMEnv.

Adds the gymnasium.Env interface (action_space, observation_space) so Stable-Baselines3 (PPO, DQN,
SAC) and sb3-contrib agents (RecurrentPPO, QRDQN, MaskablePPO) can train directly.

Training episodes randomise year from [train_years] and the PCM reset scenario from the three
contract scenarios (solid / partial / liquid; plan 8.1) on each reset. Validation / test
environments fix the year (or cycle through a fixed list) and use the 'solid' scenario.

Action masking (for MaskablePPO), physical thresholds decoded from the observation:
  CHARGE (0) masked when: top-node water >= 68 C (near the 72 C shield)  or  GHI <= the collector cutoff
  IDLE   (1) always valid
  BYPASS (2) always valid
(The TN reference compared a normalised value against 1.2, which is 78 C, although its own
docstring says 68 C. Rajasthan implements the documented 68 C — see docs/CORRECTIONS.md.)
"""

import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config_O2
from o3.environment.pcm_env import PCMEnv, N_ACTIONS, CHARGE, IDLE, BYPASS
from o3.environment import obs as _obs
from o3.utils.weather_roles import TRAIN_YEARS, VAL_YEARS, TEST_YEARS   # re-exported (TN API)

CONTRACT_PATH = config_O2.CONTRACT_FILE

OBS_DIM = _obs.OBS_DIMS[_obs.DEFAULT_VERSION]
OBS_LOW = -5.0
OBS_HIGH = 5.0

PCM_INITS = ("solid", "partial", "liquid")     # contract reset_scenarios (fully_solid, partially_charged, fully_liquid)

MASK_T_WATER_C = 68.0
MASK_GHI_WM2 = 10.0


def action_mask_from_obs(obs: np.ndarray) -> np.ndarray:
    d = _obs.decode(obs)
    charge_ok = not (d["T_water_C"] >= MASK_T_WATER_C or d["GHI_Wm2"] <= MASK_GHI_WM2)
    return np.array([charge_ok, True, True], dtype=bool)


class PCMGymEnv(gym.Env):
    """Single-regime PCM SWH environment compatible with Stable-Baselines3 and sb3-contrib.

    Parameters
    ----------
    contract_path : path-like
    regime_id     : 0, 1 or 2
    years         : years to sample from on reset (None -> TRAIN_YEARS)
    fixed_year    : if set, every episode uses this year (overrides years)
    cycle_years   : if True, `years` are visited in order instead of at random (used for evaluation
                    on all validation years inside EvalCallback)
    pcm_inits     : reset scenarios to sample from (default: all three when training, 'solid' when
                    fixed_year / cycle_years is used)
    reward_weights: dict of w_energy, w_unmet, w_safety, w_switching, w_shield
    obs_version   : "V2" (default) or "V1"
    demand_mode   : "deterministic" | "stochastic"
    """

    metadata = {"render_modes": []}

    def __init__(self, contract_path=CONTRACT_PATH, regime_id: int = 0, years: list[int] | None = None,
                 fixed_year: int | None = None, reward_weights: dict | None = None, cycle_years: bool = False,
                 pcm_inits: tuple | list | None = None, obs_version: str = _obs.DEFAULT_VERSION,
                 demand_mode: str = "deterministic"):
        super().__init__()
        self._years = list(years or TRAIN_YEARS)
        self._fixed_year = fixed_year
        self._cycle = cycle_years
        self._cycle_i = 0
        eval_like = fixed_year is not None or cycle_years
        self._pcm_inits = tuple(pcm_inits) if pcm_inits else (("solid",) if eval_like else PCM_INITS)
        self._rng = np.random.default_rng()
        self._obs_dim = _obs.OBS_DIMS[obs_version]
        self._last_obs = np.zeros(self._obs_dim, dtype=np.float32)

        init_year = fixed_year if fixed_year else self._years[0]
        self._env = PCMEnv(contract_path=contract_path, regime_id=regime_id, year=init_year,
                           obs_version=obs_version, reward_weights=reward_weights, demand_mode=demand_mode)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, shape=(self._obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._cycle_i = 0
        if self._fixed_year is not None:
            year = self._fixed_year
        elif self._cycle:
            year = self._years[self._cycle_i % len(self._years)]
            self._cycle_i += 1
        else:
            year = int(self._rng.choice(self._years))
        pcm_init = str(self._rng.choice(self._pcm_inits)) if len(self._pcm_inits) > 1 else self._pcm_inits[0]
        ep_seed = int(self._rng.integers(0, 2 ** 31 - 1))
        obs, info = self._env.reset(year=year, initial_pcm_state=pcm_init, seed=ep_seed)
        obs = np.clip(obs, OBS_LOW, OBS_HIGH).astype(np.float32)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        obs = np.clip(obs, OBS_LOW, OBS_HIGH).astype(np.float32)
        self._last_obs = obs
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Boolean mask, True = allowed (MaskablePPO). Conservative: tighter than the real shield."""
        return action_mask_from_obs(self._last_obs)

    @property
    def inner(self) -> PCMEnv:
        return self._env


class PCMContinuousGymEnv(PCMGymEnv):
    """Continuous-action view of PCMEnv for off-policy continuous RL (SAC/TD3).

    The underlying control is discrete {CHARGE, IDLE, BYPASS}. SAC requires a Box action space, so a
    1-D action in [-1, 1] is binned:  a < -1/3 -> CHARGE;  |a| <= 1/3 -> IDLE;  a > 1/3 -> BYPASS.
    The in-kernel safety shield is unchanged and still authoritative.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    @staticmethod
    def _to_discrete(action) -> int:
        a = float(np.asarray(action).reshape(-1)[0])
        if a < -1.0 / 3.0:
            return CHARGE
        if a > 1.0 / 3.0:
            return BYPASS
        return IDLE

    def step(self, action):
        return super().step(self._to_discrete(action))
