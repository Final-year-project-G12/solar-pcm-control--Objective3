"""
Phase O3-2 Baseline B1 — reactive rule-based controller.

Uses current state (from observation + info) to decide the action.
Rules are ordered by priority; the first that matches fires.

Rules (highest to lowest priority):
  R1. If T_water >= bypass_guard (70°C) -> BYPASS  (pre-shield overheat protection)
  R2. If T_pcm >= pcm_guard (60°C) and solar is strong -> BYPASS (avoid overcharging)
  R3. If evening demand window (17–21h) and f_melt > 0.15 -> IDLE (let PCM discharge)
  R4. If GHI > solar_thresh (200 W/m2) and f_melt < 0.9 -> CHARGE
  R5. If f_melt < low_melt (0.05) -> CHARGE (always try to charge when nearly empty)
  R6. Default -> IDLE

This policy must pass the O3 acceptance checklist before DRL training begins.
"""

import numpy as np
from o3.environment.pcm_env import CHARGE, IDLE, BYPASS


class ReactiveRulePolicy:
    """B1: reactive rule-based controller.

    Parameters mirror the threshold values in the O2 contract's safety shield
    (bypass_water_C=72, bypass_pcm_C=62) with 2°C pre-emptive guard bands.
    """

    name = "B1_reactive_rule"

    def __init__(
        self,
        bypass_water_guard_C: float  = 70.0,  # 2°C below shield's 72°C
        bypass_pcm_guard_C: float    = 60.0,  # 2°C below shield's 62°C
        evening_start_h: int         = 17,
        evening_end_h: int           = 21,
        f_melt_discharge_min: float  = 0.15,
        f_melt_charge_max: float     = 0.90,
        f_melt_low: float            = 0.05,
        ghi_charge_thresh_W: float   = 200.0,
        ghi_strong_W: float          = 500.0,
    ):
        self.bypass_water_guard_C   = bypass_water_guard_C
        self.bypass_pcm_guard_C     = bypass_pcm_guard_C
        self.evening_start_h        = evening_start_h
        self.evening_end_h          = evening_end_h
        self.f_melt_discharge_min   = f_melt_discharge_min
        self.f_melt_charge_max      = f_melt_charge_max
        self.f_melt_low             = f_melt_low
        self.ghi_charge_thresh_W    = ghi_charge_thresh_W
        self.ghi_strong_W           = ghi_strong_W

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        """Decide action from the V1 observation vector.

        V1 layout (see pcm_env.py):
            0  GHI / 1000
            1  (T_amb - 15) / 25
            2  (T_w - 30) / 40
            3  (T_pcm - 30) / 40
            4  f_melt
            5  demand_L / 20
            6  (T_mains - 20) / 20
            7  hour / 24
            8  day / 365
            9  prev_action / 2
            10 margin / 10
        """
        ghi_Wm2  = float(obs[0]) * 1000.0
        T_w_C    = float(obs[2]) * 40.0 + 30.0
        T_pcm_C  = float(obs[3]) * 40.0 + 30.0
        f_melt   = float(obs[4])
        hour_of_day = int(round(float(obs[7]) * 24)) % 24

        # R1: imminent water overheat — bypass now, before the shield
        if T_w_C >= self.bypass_water_guard_C:
            return BYPASS

        # R2: PCM near ceiling and strong solar — bypass to avoid overcharge
        if T_pcm_C >= self.bypass_pcm_guard_C and ghi_Wm2 >= self.ghi_strong_W:
            return BYPASS

        # R3: evening demand window and PCM has stored energy — let it discharge
        in_evening = self.evening_start_h <= hour_of_day < self.evening_end_h
        if in_evening and f_melt >= self.f_melt_discharge_min:
            return IDLE

        # R4: solar available and PCM not fully charged — charge
        if ghi_Wm2 >= self.ghi_charge_thresh_W and f_melt < self.f_melt_charge_max:
            return CHARGE

        # R5: very low melt fraction — always try to charge when sun exists
        if f_melt < self.f_melt_low and ghi_Wm2 > 50.0:
            return CHARGE

        # R6: default
        return IDLE

    def reset(self) -> None:
        pass


# --- convenience wrapper reusing the b0 run_episode helper ---------------

def run_episode(env, policy=None, year: int = 2016,
                initial_pcm_state: str = "solid") -> dict:
    from o3.baselines.b0_fixed import run_episode as _run
    if policy is None:
        policy = ReactiveRulePolicy()
    return _run(env, policy=policy, year=year, initial_pcm_state=initial_pcm_state)
