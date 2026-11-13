"""
scripts/m12_train_ext.py -- POST-LOCK research extensions (Tamil Nadu plan sections 16.3 / 19.1), Uttarakhand.
One training run: algorithm (PPO or DQN), regime, reward weights (w2, w4; w3 by the dominance rule, unchanged),
seed, observation version (V1 / V2 forecast) and training-time domain randomisation. Same budget, gamma, network,
validation scoring (every 87,600 steps on 2021-2023, fully_solid, nominal) and lexicographic checkpoint rule as M4.
Writes ONLY to results/ext/<study>/r{k}/<tag>/seed{s}/ so it can never enter the frozen M4 selection; no test year
is loaded. Hyper-parameters: SB3 defaults except gamma (config) and, for DQN, none changed.
Usage: python scripts/m12_train_ext.py --study dqn --algo dqn --regime 1 --w2 0 --w4 0.3 --seed 0
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stable_baselines3 as sb3                                             # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback                 # noqa: E402
from stable_baselines3.common.monitor import Monitor                        # noqa: E402

from o3 import config as o3c                                                # noqa: E402
from o3.controllers import SB3Policy                                        # noqa: E402
from o3.env import reward_weights                                           # noqa: E402
from o3.env_ext import UKPCMEnvExt                                          # noqa: E402
from o3.evaluate import evaluate, lexicographic_best, summarise              # noqa: E402

ALGOS = {"ppo": sb3.PPO, "dqn": sb3.DQN}


def run_dir(study, regime, algo, w2, w4, obs, dr, seed, w3=None) -> Path:
    tag = f"{algo}_w2_{w2:g}_w4_{w4:g}_{obs}" + ("_DR" if dr else "") + ("" if w3 is None else f"_w3_{w3:g}_DIAGNOSTIC")
    return o3c.RESULTS / "ext" / study / f"r{regime}" / tag / f"seed{seed}"


class Val(BaseCallback):
    def __init__(self, out, a, weights, env_kwargs, every, years):
        super().__init__()
        self.out, self.a, self.weights, self.ekw, self.every, self.years = out, a, weights, env_kwargs, every, years
        self.rows, self.t0 = [], time.time()

    def score(self):
        p = self.out / f"ckpt_{self.num_timesteps}.zip"
        self.model.save(p)
        pol = SB3Policy(p, self.a.algo, env_kwargs=self.ekw)
        s = summarise(evaluate(pol, self.a.regime, self.years, "validate", weights=self.weights))
        s.update({"steps": self.num_timesteps, "ckpt": p.name, "wall_s": time.time() - self.t0})
        self.rows.append(s)
        pd.DataFrame(self.rows).assign(o3_config_sha256=o3c.config_hash()).to_csv(self.out / "val_scores.csv", index=False)
        print(f"[{self.num_timesteps:>7}] val E_win {s['E_del_window_kWh']:.1f} overheat {s['overheat_h']:.0f} "
              f"bypass {s['bypass_hours']:.0f} sw {s['bypass_switches']:.0f}  ({s['wall_s']:.0f}s)", flush=True)

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self.score()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--algo", choices=list(ALGOS), required=True)
    ap.add_argument("--regime", type=int, required=True)
    ap.add_argument("--w2", type=float, required=True)
    ap.add_argument("--w4", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--obs-version", default="V1", choices=["V1", "V2"])
    ap.add_argument("--domain-randomize", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--w3", type=float, default=None,
                    help="DIAGNOSTIC: override the dominance-rule w3 (Q2 forbids tuning it; never eligible for selection)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = o3c.load()
    pc = cfg["ppo"]
    out = run_dir(a.study, a.regime, a.algo, a.w2, a.w4, a.obs_version, a.domain_randomize, a.seed, a.w3)
    out.mkdir(parents=True, exist_ok=True)
    weights = reward_weights(a.regime, a.w2, a.w4)
    if a.w3 is not None:
        weights["w3"] = a.w3
    ekw = {"obs_version": a.obs_version}
    env = Monitor(UKPCMEnvExt(regime=a.regime, purpose="train", weights=weights, domain_randomize=a.domain_randomize,
                              **ekw))
    kw = dict(gamma=pc["gamma"], policy_kwargs=dict(pc["policy_kwargs"]), seed=a.seed, device="cpu", verbose=0)
    model = ALGOS[a.algo]("MlpPolicy", env, **kw)
    cb = Val(out, a, weights, ekw, pc["eval_every_steps"], cfg["years"]["validate"])
    model.learn(total_timesteps=a.steps or pc["total_timesteps"], callback=cb)
    if not cb.rows or cb.rows[-1]["steps"] != model.num_timesteps:
        cb.score()
    best = lexicographic_best(cb.rows, cfg["selection"]["delivery_tie_kWh_per_yr"])
    shutil.copy(out / best["ckpt"], out / "best_model.zip")
    (out / "best.json").write_text(json.dumps({
        "study": a.study, "algo": a.algo, "regime": a.regime, "w2": a.w2, "w4": a.w4, "seed": a.seed,
        "obs_version": a.obs_version, "domain_randomize": a.domain_randomize, "weights": weights, "best": best,
        "diagnostic_w3_override": a.w3,
        "total_timesteps": model.num_timesteps, "label": "POST-LOCK EXTENSION (validation only)",
        "o3_config_sha256": o3c.config_hash()}, indent=1, default=float))
    print("best", best["ckpt"], {k: round(best[k], 2) for k in ("E_del_window_kWh", "overheat_h", "bypass_switches")})


if __name__ == "__main__":
    main()
