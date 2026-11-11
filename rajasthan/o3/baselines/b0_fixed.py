"""
Phase O3-2 Baseline B0 — loop always enabled, with the existing O2 safety shield.

This is the exact O2 operating point (the released passive PCM design): the collector loop
circulates whenever there is sun, and the shield (top node 72 C, PCM 62 C) overrides inside the
kernel when necessary. It is the minimum acceptable reference — any RL policy must beat it to claim
improvement. test_T1 proves that B0 stepped for a year equals one O2 run_year_multinode call.

Usage:
    policy = FixedChargePolicy()
    action = policy.act(obs, info)   # always returns CHARGE (0)
"""

from o3.environment.pcm_env import CHARGE


class FixedChargePolicy:
    """B0: always request CHARGE (collector loop enabled at the design flow)."""

    name = "B0_fixed_charge"

    def act(self, obs, info: dict | None = None) -> int:
        return CHARGE

    def reset(self) -> None:
        pass


def run_episode(env, policy=None, year: int = 2016, initial_pcm_state: str = "solid") -> dict:
    """Run one full episode with the given policy (defaults to B0) and return the result row
    (delivered energy, unmet demand, shield and safety counts, O2 window metrics)."""
    from o3.evaluation.rollout import run_policy_episode
    if policy is None:
        policy = FixedChargePolicy()
    return run_policy_episode(env, policy, year, initial_pcm_state)
