"""
B4 — Random safe policy.

Selects uniformly from physically valid actions at each step.
Used as a diagnostic: any trained policy must beat random.
"""

import numpy as np
from o3.environment.pcm_env import CHARGE, IDLE, BYPASS, N_ACTIONS


class RandomSafePolicy:
    """Uniform random over non-masked actions.

    If action masking is provided (via obs context), CHARGE is skipped
    at night / near-overtemp. Otherwise, fully uniform over all 3 actions.
    """

    name = "B4_random_safe"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        ghi = obs[0]    # normalised GHI
        tw  = obs[2]    # normalised T_water
        # mask CHARGE when no sun or near shield (same as gym_wrapper mask)
        if tw >= 1.2 or ghi <= 0.0:
            return int(self._rng.choice([IDLE, BYPASS]))
        return int(self._rng.integers(0, N_ACTIONS))

    def reset(self) -> None:
        pass


def run_episode(env, policy=None, year: int = 2016,
                initial_pcm_state: str = "solid") -> dict:
    if policy is None:
        policy = RandomSafePolicy()
    policy.reset()
    obs, _ = env.reset(year=year, initial_pcm_state=initial_pcm_state)
    done = False
    total_reward = total_del = total_unmet = 0.0
    shield_w = shield_p = safety_v = steps = 0
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_del    += info.get("E_delivered_Wh", 0.0)
        total_unmet  += info.get("E_unmet_Wh",    0.0)
        shield_w     += int(info.get("n_shield_water", 0))
        shield_p     += int(info.get("n_shield_pcm", 0))
        safety_v     += int(info.get("n_safety_violations", 0))
        steps        += 1
    return {
        "policy":               policy.name,
        "regime_id":             env.regime_id,
        "year":                  year,
        "n_steps":               steps,
        "total_reward":          total_reward,
        "total_delivered_kWh":   total_del / 1000.0,
        "total_unmet_kWh":       total_unmet / 1000.0,
        "total_shield_water":    shield_w,
        "total_shield_pcm":      shield_p,
        "n_safety_violations":   safety_v,
    }
