"""
Phase O3-2 Baseline B2 — forecast-aware rule-based controller.

B1 is purely reactive (it sees only the current hour). B2 adds a short
irradiance forecast: it looks ahead `horizon_h` hours and makes
*anticipatory* decisions the reactive controller cannot:

  - pre-emptive BYPASS when strong sun is coming and the PCM is already
    near full (avoids charging into the shield later in the day);
  - aggressive CHARGE when this is the last useful sun before a cloudy /
    night stretch and the PCM is not yet charged;
  - hold (IDLE) through the evening demand window so stored heat is
    discharged when it is actually needed.

The forecast is read from the environment's own weather series
(`attach_env`). This is a *perfect-forecast* baseline by default; pass
`forecast_noise_frac` > 0 to study an imperfect forecast. If no env is
attached, B2 falls back to a daylight-curve proxy (sun rises ~6h, peaks
~13h, sets ~18h) so it still runs standalone.

B2 must respect the same safety priorities as B1 — the in-env shield is
the hard guarantee; these rules only improve *timing* around it.
"""

import numpy as np
from o3.environment.pcm_env import CHARGE, IDLE, BYPASS


class ForecastAwarePolicy:
    """B2: forecast-aware rule-based controller."""

    name = "B2_forecast_rule"

    def __init__(
        self,
        horizon_h: int              = 4,      # look-ahead window
        bypass_water_guard_C: float = 70.0,
        bypass_pcm_guard_C: float   = 60.0,
        evening_start_h: int        = 17,
        evening_end_h: int          = 21,
        f_melt_discharge_min: float = 0.15,
        f_melt_nearly_full: float   = 0.85,
        f_melt_charge_max: float    = 0.90,
        ghi_charge_thresh_W: float  = 200.0,
        ghi_strong_W: float         = 500.0,
        forecast_noise_frac: float  = 0.0,    # 0 = perfect forecast
        seed: int                   = 0,
    ):
        self.horizon_h             = horizon_h
        self.bypass_water_guard_C  = bypass_water_guard_C
        self.bypass_pcm_guard_C    = bypass_pcm_guard_C
        self.evening_start_h       = evening_start_h
        self.evening_end_h         = evening_end_h
        self.f_melt_discharge_min  = f_melt_discharge_min
        self.f_melt_nearly_full    = f_melt_nearly_full
        self.f_melt_charge_max     = f_melt_charge_max
        self.ghi_charge_thresh_W   = ghi_charge_thresh_W
        self.ghi_strong_W          = ghi_strong_W
        self.forecast_noise_frac   = forecast_noise_frac
        self._rng                  = np.random.default_rng(seed)
        self._env                  = None

    def attach_env(self, env) -> None:
        """Give B2 read access to the weather series for the forecast."""
        self._env = env

    # ------------------------------------------------------------------

    def _forecast_ghi(self, hour_of_day: int) -> np.ndarray:
        """Return the next `horizon_h` hours of GHI (W/m2)."""
        env = self._env
        if env is not None and getattr(env, "_state", None) is not None:
            i = env._state["hour_idx"]
            w = env._weather
            ghis = []
            for k in range(1, self.horizon_h + 1):
                j = i + k
                ghis.append(float(w.iloc[j]["GHI_Wm2"]) if j < len(w) else 0.0)
            ghi = np.array(ghis, dtype=float)
            if self.forecast_noise_frac > 0:
                ghi = ghi * (1.0 + self._rng.normal(
                    0.0, self.forecast_noise_frac, size=ghi.shape))
                ghi = np.clip(ghi, 0.0, None)
            return ghi
        # fallback: smooth daylight curve as a forecast proxy
        out = []
        for k in range(1, self.horizon_h + 1):
            h = (hour_of_day + k) % 24
            out.append(max(0.0, 900.0 * np.sin((h - 6) / 12 * np.pi))
                       if 6 <= h <= 18 else 0.0)
        return np.array(out, dtype=float)

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        ghi_now = float(obs[0]) * 1000.0
        T_w_C   = float(obs[2]) * 40.0 + 30.0
        T_pcm_C = float(obs[3]) * 40.0 + 30.0
        f_melt  = float(obs[4])
        hod     = int(round(float(obs[7]) * 24)) % 24

        fc        = self._forecast_ghi(hod)
        strong_coming = bool(np.any(fc >= self.ghi_strong_W))
        sun_coming    = bool(np.any(fc >= self.ghi_charge_thresh_W))

        # R1: imminent water overheat — bypass now (same hard priority as B1)
        if T_w_C >= self.bypass_water_guard_C:
            return BYPASS

        # R2: PCM near ceiling with strong sun now — bypass to avoid overcharge
        if T_pcm_C >= self.bypass_pcm_guard_C and ghi_now >= self.ghi_strong_W:
            return BYPASS

        # R3 (forecast): PCM already nearly full AND more strong sun is coming —
        #     pre-emptively bypass so we don't spend the afternoon in the shield
        if f_melt >= self.f_melt_nearly_full and strong_coming:
            return BYPASS

        # R4: evening demand window with stored energy — hold to discharge
        if self.evening_start_h <= hod < self.evening_end_h \
                and f_melt >= self.f_melt_discharge_min:
            return IDLE

        # R5 (forecast): sun now but none coming soon and PCM not full —
        #     this is the last useful charge window; take it
        if ghi_now >= self.ghi_charge_thresh_W and not sun_coming \
                and f_melt < self.f_melt_charge_max:
            return CHARGE

        # R6: normal charge when sun is available and PCM not full
        if ghi_now >= self.ghi_charge_thresh_W and f_melt < self.f_melt_charge_max:
            return CHARGE

        # R7: default
        return IDLE

    def reset(self) -> None:
        pass


def run_episode(env, policy=None, year: int = 2016,
                initial_pcm_state: str = "solid") -> dict:
    from o3.baselines.b0_fixed import run_episode as _run
    if policy is None:
        policy = ForecastAwarePolicy()
    policy.attach_env(env)
    return _run(env, policy=policy, year=year, initial_pcm_state=initial_pcm_state)
