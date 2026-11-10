"""
Phase O3-2 Baseline B0 — fixed nominal flow with the existing O2 safety shield.

This is the exact O2 operating point: pump always circulating at the
selected flow rate, the rule-based shield (bypass_water_C=72, bypass_pcm_C=62)
overrides when necessary.  It is the minimum acceptable reference — any RL
policy must beat this to claim improvement.

Usage:
    policy = FixedChargePolicy()
    action = policy.act(obs, info)   # always returns CHARGE (0)
"""

from o3.environment.pcm_env import CHARGE


class FixedChargePolicy:
    """B0: always request CHARGE (circulate pump at nominal flow).

    The safety shield inside PCMEnv will still fire when needed.
    This reproduces the O2 baseline exactly.
    """

    name = "B0_fixed_charge"

    def act(self, obs, info: dict | None = None) -> int:
        return CHARGE

    def reset(self) -> None:
        pass


def run_episode(env, policy=None, year: int = 2016,
                initial_pcm_state: str = "solid") -> dict:
    """Run one full episode with the given policy (defaults to B0).

    Returns a summary dict with totals for delivered energy, unmet demand,
    shield activations and safety violations.
    """
    if policy is None:
        policy = FixedChargePolicy()

    obs, info = env.reset(year=year, initial_pcm_state=initial_pcm_state)
    policy.reset()

    total_delivered_Wh   = 0.0
    total_unmet_Wh       = 0.0
    total_shield_water   = 0
    total_shield_pcm     = 0
    total_safety_viol    = 0
    total_reward         = 0.0
    n_steps              = 0

    done = False
    while not done:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_delivered_Wh += info["E_delivered_Wh"]
        total_unmet_Wh     += info["E_unmet_Wh"]
        total_shield_water += info["n_shield_water"]
        total_shield_pcm   += info["n_shield_pcm"]
        total_safety_viol  += 0   # tracked in env._state
        total_reward       += reward
        n_steps            += 1

    s = env._state
    return {
        "policy":               policy.name,
        "year":                 year,
        "regime_id":            env.regime_id,
        "n_steps":              n_steps,
        "total_delivered_kWh":  total_delivered_Wh / 1000.0,
        "total_unmet_kWh":      total_unmet_Wh / 1000.0,
        "total_shield_water":   s["n_shield_water"],
        "total_shield_pcm":     s["n_shield_pcm"],
        "n_safety_violations":  s["n_safety_violations"],
        "action_switches":      s["action_switches"],
        "total_reward":         total_reward,
        "final_T_w_C":          s["T_w"],
        "final_T_pcm_C":        s["T_pcm"],
        "final_f_melt":         s["f_melt"],
    }
