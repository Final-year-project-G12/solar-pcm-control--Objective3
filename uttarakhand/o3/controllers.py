"""
o3/controllers.py
=================
Controllers compared in Objective 3 (plan section 5 / 15.6). Each maps the RAW observation (and, for the
learned policy, the normalised one) to an action: 0 = bypass, 1 = enable.

  B0  always enable              = O2's passive PCM design as released, with its in-kernel shield
  B1  reactive rule              bypass when the top node is at/above theta_top, or the PCM is nearly all
                                 liquid (f_melt >= theta_f) and the top node is at/above theta_hot; thresholds
                                 tuned on the validation years only (scripts/m3_baselines.py)
  B2  forecast-aware rule        B1 plus a look-ahead: pre-emptive bypass when the PCM is nearly full, the tank is hot
                                 and strong sun is forecast in the next `horizon_h` hours (plan 11.1 / Tamil Nadu
                                 section 16.2). Forecast = the episode's own future GHI (perfect) or with multiplicative
                                 noise `noise_frac`; thresholds tuned on the validation years only (scripts/m9_b2_forecast.py)
  B4  random                     diagnostic floor
  PPO frozen SB3 policy          deterministic actions
The water-only twin is B0 on the same tank with zero capsules (o3/evaluate.py).
"""

from __future__ import annotations

import numpy as np

from o3.env import OBS_FIELDS

I = {k: i for i, k in enumerate(OBS_FIELDS)}


class B0AlwaysOn:
    name = "B0"
    needs_norm = False

    def act(self, raw, norm=None):
        return 1


class B1Reactive:
    name = "B1"
    needs_norm = False

    def __init__(self, theta_top: float = 72.0, theta_f: float = 1.01, theta_hot: float = 99.0):
        self.theta_top, self.theta_f, self.theta_hot = float(theta_top), float(theta_f), float(theta_hot)

    @property
    def params(self):
        return {"theta_top": self.theta_top, "theta_f": self.theta_f, "theta_hot": self.theta_hot}

    def act(self, raw, norm=None):
        top, f = raw[I["T_water_top_C"]], raw[I["f_melt"]]
        if top >= self.theta_top or (f >= self.theta_f and top >= self.theta_hot):
            return 0
        return 1


class B4Random:
    name = "B4"
    needs_norm = False

    def __init__(self, seed: int = 0, p_enable: float = 0.5):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.p = p_enable

    def act(self, raw, norm=None):
        return int(self.rng.random() < self.p)


class PPOPolicy:
    name = "PPO"
    needs_norm = True

    def __init__(self, model_path, name: str = "PPO"):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path), device="cpu")
        self.name = name
        self.path = model_path

    def act(self, raw, norm=None):
        a, _ = self.model.predict(norm, deterministic=True)
        return int(a)


class B2Forecast:
    """Forecast-aware rule. Sees the next `horizon_h` hours of GHI (perfect forecast, or perfect x (1 + noise))."""
    name = "B2"
    needs_norm = False

    def __init__(self, theta_top: float = 72.0, theta_f: float = 0.99, theta_hot: float = 60.0,
                 ghi_ahead_Wm2: float = 400.0, horizon_h: int = 4, noise_frac: float = 0.0, seed: int = 0):
        self.theta_top, self.theta_f, self.theta_hot = float(theta_top), float(theta_f), float(theta_hot)
        self.ghi_ahead, self.horizon, self.noise = float(ghi_ahead_Wm2), int(horizon_h), float(noise_frac)
        self.seed = seed
        self._env = None

    @property
    def params(self):
        return {"theta_top": self.theta_top, "theta_f": self.theta_f, "theta_hot": self.theta_hot,
                "ghi_ahead_Wm2": self.ghi_ahead, "horizon_h": self.horizon, "noise_frac": self.noise}

    def bind(self, env):
        self._env = env
        ghi = env.stepper.ghi
        if self.noise > 0:
            rng = np.random.default_rng(self.seed)
            ghi = np.maximum(ghi * (1.0 + rng.normal(0.0, self.noise, len(ghi))), 0.0)
        self._fc = ghi

    def act(self, raw, norm=None):
        top, f = raw[I["T_water_top_C"]], raw[I["f_melt"]]
        if top >= self.theta_top:
            return 0
        t = self._env.stepper.t
        ahead = self._fc[t + 1: t + 1 + self.horizon]
        if f >= self.theta_f and top >= self.theta_hot and len(ahead) and ahead.mean() >= self.ghi_ahead:
            return 0                                   # store nearly full, hot, strong sun coming: bypass now
        return 1


class SB3Policy:
    """Any Stable-Baselines3 policy (PPO / DQN) from the post-lock extension study, with the environment settings it was
    trained on (env_kwargs, e.g. obs_version="V2"); evaluation never uses domain randomisation."""
    needs_norm = True

    def __init__(self, model_path, algo: str = "ppo", name: str | None = None, env_kwargs: dict | None = None):
        import stable_baselines3 as sb3
        self.model = {"ppo": sb3.PPO, "dqn": sb3.DQN}[algo].load(str(model_path), device="cpu")
        self.name = name or algo.upper()
        self.path = model_path
        self.env_kwargs = {k: v for k, v in (env_kwargs or {}).items() if k != "domain_randomize"}

    def act(self, raw, norm=None):
        a, _ = self.model.predict(norm, deterministic=True)
        return int(a)
