"""
o3/controllers.py
=================
Controllers compared in Objective 3 (plan section 5 / 15.6). Each maps the RAW observation (and, for the
learned policy, the normalised one) to an action: 0 = bypass, 1 = enable.

  B0  always enable              = O2's passive PCM design as released, with its in-kernel shield
  B1  reactive rule              bypass when the top node is at/above theta_top, or the PCM is nearly all
                                 liquid (f_melt >= theta_f) and the top node is at/above theta_hot; thresholds
                                 tuned on the validation years only (scripts/m3_baselines.py)
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

    def act(self, raw, norm=None):
        a, _ = self.model.predict(norm, deterministic=True)
        return int(a)
