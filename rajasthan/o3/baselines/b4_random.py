"""
B4 — Random safe policy.

Selects uniformly from physically valid actions at each step. Used as a diagnostic: any trained
policy must beat random. Also the 'random search' half of the plan's headroom gate G2.
"""

import numpy as np
from o3.environment.pcm_env import CHARGE, IDLE, BYPASS, N_ACTIONS
from o3.environment.gym_wrapper import action_mask_from_obs


class RandomSafePolicy:
    """Uniform random over non-masked actions (CHARGE skipped at night / near the shield)."""

    name = "B4_random_safe"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        mask = action_mask_from_obs(obs)
        allowed = np.flatnonzero(mask)
        return int(self._rng.choice(allowed))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)      # same seed -> same trajectory (T5)


def run_episode(env, policy=None, year: int = 2016, initial_pcm_state: str = "solid") -> dict:
    from o3.evaluation.rollout import run_policy_episode
    if policy is None:
        policy = RandomSafePolicy()
    return run_policy_episode(env, policy, year, initial_pcm_state)
