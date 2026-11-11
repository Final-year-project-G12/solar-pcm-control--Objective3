"""
scripts/m4_train_ppo.py -- milestone M4: train one PPO run (regime, reward-weight set, seed).

Stable-Baselines3 PPO, MlpPolicy [64, 64], SB3 defaults except gamma (configs/o3_config.yaml). Episodes: one
simulated TRAIN year (2016-2020), random year and reset scenario, deterministic demand. Every `eval_every_steps`
the deterministic policy is scored on the VALIDATION years (2021-2023, fully_solid) and saved; the run's checkpoint
is chosen by the plan 8.3 lexicographic rule over those scores. No test year is ever loaded here.

Writes results/ppo/r{k}/w2_{w2}_w4_{w4}/seed{s}/ : ckpt_<steps>.zip, val_scores.csv, best.json, best_model.zip
Usage: python scripts/m4_train_ppo.py --regime 1 --w2 0.3 --w4 0.3 --seed 0 [--steps N]
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
from stable_baselines3 import PPO                                           # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback                 # noqa: E402
from stable_baselines3.common.monitor import Monitor                        # noqa: E402

from o3 import config as o3c                                                # noqa: E402
from o3.controllers import PPOPolicy                                        # noqa: E402
from o3.env import UKPCMEnv, reward_weights                                 # noqa: E402
from o3.evaluate import evaluate, lexicographic_best, summarise              # noqa: E402


def run_dir(regime, w2, w4, seed, w3=None) -> Path:
    tag = f"w2_{w2:g}_w4_{w4:g}" + ("" if w3 is None else f"_w3_{w3:g}_DIAGNOSTIC")
    return o3c.RESULTS / "ppo" / f"r{regime}" / tag / f"seed{seed}"


class ValCallback(BaseCallback):
    def __init__(self, out: Path, regime: int, weights: dict, every: int, years):
        super().__init__()
        self.out, self.regime, self.weights, self.every, self.years = out, regime, weights, every, years
        self.rows = []
        self.t0 = time.time()

    def _score(self):
        p = self.out / f"ckpt_{self.num_timesteps}.zip"
        self.model.save(p)
        pol = PPOPolicy(p)
        s = summarise(evaluate(pol, self.regime, self.years, "validate", weights=self.weights))
        s.update({"steps": self.num_timesteps, "ckpt": p.name, "wall_s": time.time() - self.t0})
        self.rows.append(s)
        pd.DataFrame(self.rows).to_csv(self.out / "val_scores.csv", index=False)
        print(f"[{self.num_timesteps:>7}] val E_win {s['E_del_window_kWh']:.1f} overheat {s['overheat_h']:.0f} "
              f"guard {s['guard_hours']:.0f} bypass {s['bypass_hours']:.0f} sw {s['bypass_switches']:.0f} "
              f"R {s['reward']:.0f}  ({s['wall_s']:.0f}s)", flush=True)

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self._score()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, required=True)
    ap.add_argument("--w2", type=float, required=True)
    ap.add_argument("--w4", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--w3", type=float, default=None,
                    help="DIAGNOSTIC ONLY: override the dominance-rule w3 (never eligible for selection)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = o3c.load()
    pc = cfg["ppo"]
    steps = a.steps or pc["total_timesteps"]
    out = run_dir(a.regime, a.w2, a.w4, a.seed, a.w3)
    out.mkdir(parents=True, exist_ok=True)
    weights = reward_weights(a.regime, a.w2, a.w4)
    if a.w3 is not None:
        weights["w3"] = a.w3
    env = Monitor(UKPCMEnv(regime=a.regime, purpose="train", weights=weights))
    model = PPO("MlpPolicy", env, gamma=pc["gamma"], policy_kwargs=dict(pc["policy_kwargs"]), seed=a.seed,
                device="cpu", verbose=0)
    cb = ValCallback(out, a.regime, weights, pc["eval_every_steps"], cfg["years"]["validate"])
    model.learn(total_timesteps=steps, callback=cb)
    if not cb.rows or cb.rows[-1]["steps"] != model.num_timesteps:
        cb._score()
    tie = cfg["selection"]["delivery_tie_kWh_per_yr"]
    best = lexicographic_best(cb.rows, tie)
    shutil.copy(out / best["ckpt"], out / "best_model.zip")
    (out / "best.json").write_text(json.dumps({
        "regime": a.regime, "w2": a.w2, "w4": a.w4, "seed": a.seed, "weights": weights, "best": best,
        "diagnostic_w3_override": a.w3,
        "total_timesteps": model.num_timesteps, "o3_config_sha256": o3c.config_hash(),
        "normalisation_sha256": o3c.ob.sha256_file(o3c.normalisation_path(a.regime))}, indent=1, default=float))
    print("best", best["ckpt"], {k: round(best[k], 2) for k in ("E_del_window_kWh", "overheat_h", "bypass_switches")})


if __name__ == "__main__":
    main()
