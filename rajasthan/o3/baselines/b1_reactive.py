"""
Phase O3-2 Baseline B1 — reactive rule-based controller.

Uses the current state (decoded from the observation) to decide the action. Rules are ordered by
priority; the first that matches fires.

Rules (highest to lowest priority):
  R1. top node >= bypass_guard (70 C)                    -> BYPASS  (pre-shield overheat protection)
  R2. PCM >= pcm_guard (60 C) and strong sun             -> BYPASS  (avoid overcharging)
  R3. evening window (17-21 h) and f_melt > 0.15         -> IDLE    (let the PCM discharge)
  R4. GHI > solar_thresh (200 W/m2) and f_melt < 0.9     -> CHARGE
  R5. f_melt < low_melt (0.05) and some sun              -> CHARGE
  R6. default                                            -> IDLE

Thresholds are the TN reference values (2 C below the shield on both sides). Plan 15.6 says B1's
thresholds are tuned on the VALIDATION years only by grid search: `tune_b1()` (phase2 --tune) does
that; the defaults here are the untuned starting point. Rajasthan's shield guard values are the same
as TN's (72 / 62 C) because they come from the shared O2 config.

This policy must pass the O3 acceptance checklist before DRL training begins.
"""

import numpy as np
from o3.environment.pcm_env import CHARGE, IDLE, BYPASS
from o3.environment.obs import decode


class ReactiveRulePolicy:
    """B1: reactive rule-based controller."""

    name = "B1_reactive_rule"

    def __init__(
        self,
        bypass_water_guard_C: float = 70.0,
        bypass_pcm_guard_C: float = 60.0,
        evening_start_h: int = 17,
        evening_end_h: int = 21,
        f_melt_discharge_min: float = 0.15,
        f_melt_charge_max: float = 0.90,
        f_melt_low: float = 0.05,
        ghi_charge_thresh_W: float = 200.0,
        ghi_strong_W: float = 500.0,
    ):
        self.bypass_water_guard_C = bypass_water_guard_C
        self.bypass_pcm_guard_C = bypass_pcm_guard_C
        self.evening_start_h = evening_start_h
        self.evening_end_h = evening_end_h
        self.f_melt_discharge_min = f_melt_discharge_min
        self.f_melt_charge_max = f_melt_charge_max
        self.f_melt_low = f_melt_low
        self.ghi_charge_thresh_W = ghi_charge_thresh_W
        self.ghi_strong_W = ghi_strong_W

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        d = decode(obs)
        ghi_Wm2, T_w_C, T_pcm_C, f_melt, hour_of_day = d["GHI_Wm2"], d["T_water_C"], d["T_pcm_C"], d["f_melt"], d["hour"]

        if T_w_C >= self.bypass_water_guard_C:
            return BYPASS
        if T_pcm_C >= self.bypass_pcm_guard_C and ghi_Wm2 >= self.ghi_strong_W:
            return BYPASS
        in_evening = self.evening_start_h <= hour_of_day < self.evening_end_h
        if in_evening and f_melt >= self.f_melt_discharge_min:
            return IDLE
        if ghi_Wm2 >= self.ghi_charge_thresh_W and f_melt < self.f_melt_charge_max:
            return CHARGE
        if f_melt < self.f_melt_low and ghi_Wm2 > 50.0:
            return CHARGE
        return IDLE

    def reset(self) -> None:
        pass


def run_episode(env, policy=None, year: int = 2016, initial_pcm_state: str = "solid") -> dict:
    from o3.baselines.b0_fixed import run_episode as _run
    if policy is None:
        policy = ReactiveRulePolicy()
    return _run(env, policy=policy, year=year, initial_pcm_state=initial_pcm_state)


# ---------------------------------------------------------------------------------------------
# B1 grid tuning on VALIDATION years only (plan 15.6). Job function: one grid point x one year.
# ---------------------------------------------------------------------------------------------

B1_GRID = {
    "bypass_water_guard_C": [68.0, 70.0],
    "bypass_pcm_guard_C": [58.0, 60.0],
    "ghi_charge_thresh_W": [100.0, 200.0, 300.0],
    "f_melt_charge_max": [0.8, 0.9],
}


def grid_points() -> list[dict]:
    import itertools
    keys = list(B1_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*B1_GRID.values())]
