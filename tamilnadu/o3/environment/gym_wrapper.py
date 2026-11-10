"""
Phase O3-4 — Gymnasium wrapper for PCMEnv.

Adds gymnasium.Env interface (action_space, observation_space) so
Stable-Baselines3 (PPO, DQN, SAC) and sb3-contrib agents
(RecurrentPPO, QRDQN, MaskablePPO) can train directly.

Training episodes randomise year from [train_years] on each reset so the
agent sees diverse weather. Validation/test environments fix the year.

Action masking (for MaskablePPO):
  CHARGE (0) masked when: T_water > 68°C  or  GHI < 10 normalised
  IDLE   (1) always valid
  BYPASS (2) always valid
"""

import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from o3.environment.pcm_env import PCMEnv, N_ACTIONS, CHARGE, IDLE, BYPASS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"

# canonical train / val / test splits (year-locked)
TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020]
VAL_YEARS   = [2021, 2022, 2023]
TEST_YEARS  = [2011, 2012, 2013, 2014, 2015]   # never touch until final eval

OBS_DIM  = 11
OBS_LOW  = -5.0
OBS_HIGH =  5.0

# normalised thresholds used for action masking (obs[0]=GHI, obs[2]=T_water)
_GHI_MASK_THRESH      = 0.0   # obs[0] ≤ 0 -> no sun -> mask CHARGE
_TWATER_MASK_THRESH   = 1.2   # obs[2] ≥ 1.2 -> near shield -> mask CHARGE


class PCMGymEnv(gym.Env):
    """Single-regime PCM SWH environment compatible with Stable-Baselines3
    and sb3-contrib (RecurrentPPO, QRDQN, MaskablePPO).

    Parameters
    ----------
    contract_path : path-like
    regime_id     : 0, 1 or 2
    years         : list of years to sample from on reset (None -> TRAIN_YEARS)
    fixed_year    : if set, every episode uses this year (overrides years)
    reward_weights: dict of w_energy, w_unmet, w_safety, w_switching
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        contract_path=CONTRACT_PATH,
        regime_id: int = 0,
        years: list[int] | None = None,
        fixed_year: int | None = None,
        reward_weights: dict | None = None,
    ):
        super().__init__()
        self._years      = years or TRAIN_YEARS
        self._fixed_year = fixed_year
        self._rng        = np.random.default_rng()
        self._last_obs   = np.zeros(OBS_DIM, dtype=np.float32)

        init_year = fixed_year if fixed_year else self._years[0]
        self._env = PCMEnv(
            contract_path  = contract_path,
            regime_id      = regime_id,
            year           = init_year,
            reward_weights = reward_weights,
        )

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low   = OBS_LOW,
            high  = OBS_HIGH,
            shape = (OBS_DIM,),
            dtype = np.float32,
        )

    # ------------------------------------------------------------------
    # Core gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        year = self._fixed_year if self._fixed_year is not None \
               else int(self._rng.choice(self._years))

        obs, info = self._env.reset(year=year)
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
    # Action masking (required by MaskablePPO from sb3-contrib)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return boolean mask: True = action is allowed.

        Conservative pre-shield mask — tighter than the actual safety shield
        so the agent learns to avoid near-limit states proactively.
        """
        obs  = self._last_obs
        ghi  = obs[0]    # normalised GHI
        tw   = obs[2]    # normalised T_water

        charge_ok = not (tw >= _TWATER_MASK_THRESH or ghi <= _GHI_MASK_THRESH)
        return np.array([charge_ok, True, True], dtype=bool)

    # ------------------------------------------------------------------

    @property
    def inner(self) -> PCMEnv:
        return self._env


class PCMContinuousGymEnv(PCMGymEnv):
    """Continuous-action view of PCMEnv for off-policy continuous RL (SAC/TD3).

    The underlying control is discrete {CHARGE, IDLE, BYPASS}. SAC requires a
    Box action space, so we expose a 1-D action in [-1, 1] and bin it:

        a < -1/3            -> CHARGE (0)
        -1/3 ≤ a ≤ +1/3     -> IDLE   (1)
        a > +1/3            -> BYPASS (2)

    This is the standard continuous-to-discrete mapping used when comparing a
    continuous-control algorithm against discrete ones on the same task. The
    in-env safety shield is unchanged and still authoritative.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

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
