"""
Phase O3-2 Baseline B2 — forecast-aware rule-based controller.

B1 is purely reactive (it sees only the current hour). B2 adds a short irradiance forecast: it looks
ahead `horizon_h` hours and makes *anticipatory* decisions the reactive controller cannot:

  - pre-emptive BYPASS when strong sun is coming and the PCM is already near full (avoids charging
    into the shield later in the day);
  - aggressive CHARGE when this is the last useful sun before a cloudy / night stretch and the PCM
    is not yet charged;
  - hold (IDLE) through the evening demand window so stored heat is discharged when it is needed.

The forecast is read from the environment's own weather series (`attach_env` -> env.forecast_ghi).
This is a *perfect-forecast* baseline by default; pass `forecast_noise_frac` > 0 for an imperfect one.
Without an attached env, B2 falls back to a daylight-curve proxy (sun rises ~6 h, peaks ~13 h, sets ~18 h).
Note the plan lists B2/B3 as deferred (plan 5); B2 is kept because TN built it and it is cheap.
B2 uses information the real controller would not have unless a forecast feed exists (O4 question).
"""

import numpy as np
from o3.environment.pcm_env import CHARGE, IDLE, BYPASS
from o3.environment.obs import decode


class ForecastAwarePolicy:
    """B2: forecast-aware rule-based controller."""

    name = "B2_forecast_rule"

    def __init__(
        self,
        horizon_h: int = 4,
        bypass_water_guard_C: float = 70.0,
        bypass_pcm_guard_C: float = 60.0,
        evening_start_h: int = 17,
        evening_end_h: int = 21,
        f_melt_discharge_min: float = 0.15,
        f_melt_nearly_full: float = 0.85,
        f_melt_charge_max: float = 0.90,
        ghi_charge_thresh_W: float = 200.0,
        ghi_strong_W: float = 500.0,
        forecast_noise_frac: float = 0.0,
        seed: int = 0,
    ):
        self.horizon_h = horizon_h
        self.bypass_water_guard_C = bypass_water_guard_C
        self.bypass_pcm_guard_C = bypass_pcm_guard_C
        self.evening_start_h = evening_start_h
        self.evening_end_h = evening_end_h
        self.f_melt_discharge_min = f_melt_discharge_min
        self.f_melt_nearly_full = f_melt_nearly_full
        self.f_melt_charge_max = f_melt_charge_max
        self.ghi_charge_thresh_W = ghi_charge_thresh_W
        self.ghi_strong_W = ghi_strong_W
        self.forecast_noise_frac = forecast_noise_frac
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._env = None

    def attach_env(self, env) -> None:
        """Give B2 read access to the weather series for the forecast."""
        self._env = env

    def _forecast_ghi(self, hour_of_day: int) -> np.ndarray:
        """Return the next `horizon_h` hours of GHI (W/m2)."""
        env = self._env
        if env is not None and getattr(env, "_state", None) is not None:
            ghi = np.array([env.forecast_ghi(k) for k in range(1, self.horizon_h + 1)], dtype=float)
            if self.forecast_noise_frac > 0:
                ghi = ghi * (1.0 + self._rng.normal(0.0, self.forecast_noise_frac, size=ghi.shape))
                ghi = np.clip(ghi, 0.0, None)
            return ghi
        out = []
        for k in range(1, self.horizon_h + 1):
            h = (hour_of_day + k) % 24
            out.append(max(0.0, 900.0 * np.sin((h - 6) / 12 * np.pi)) if 6 <= h <= 18 else 0.0)
        return np.array(out, dtype=float)

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        d = decode(obs)
        ghi_now, T_w_C, T_pcm_C, f_melt, hod = d["GHI_Wm2"], d["T_water_C"], d["T_pcm_C"], d["f_melt"], d["hour"]

        fc = self._forecast_ghi(hod)
        strong_coming = bool(np.any(fc >= self.ghi_strong_W))
        sun_coming = bool(np.any(fc >= self.ghi_charge_thresh_W))

        if T_w_C >= self.bypass_water_guard_C:                                   # R1
            return BYPASS
        if T_pcm_C >= self.bypass_pcm_guard_C and ghi_now >= self.ghi_strong_W:  # R2
            return BYPASS
        if f_melt >= self.f_melt_nearly_full and strong_coming:                  # R3 (forecast)
            return BYPASS
        if self.evening_start_h <= hod < self.evening_end_h and f_melt >= self.f_melt_discharge_min:   # R4
            return IDLE
        if ghi_now >= self.ghi_charge_thresh_W and not sun_coming and f_melt < self.f_melt_charge_max:  # R5 (forecast)
            return CHARGE
        if ghi_now >= self.ghi_charge_thresh_W and f_melt < self.f_melt_charge_max:                    # R6
            return CHARGE
        return IDLE                                                                                    # R7

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)


def run_episode(env, policy=None, year: int = 2016, initial_pcm_state: str = "solid") -> dict:
    from o3.baselines.b0_fixed import run_episode as _run
    if policy is None:
        policy = ForecastAwarePolicy()
    policy.attach_env(env)
    return _run(env, policy=policy, year=year, initial_pcm_state=initial_pcm_state)
