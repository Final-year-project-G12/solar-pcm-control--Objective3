"""
scripts/m13_forecast_algo_study.py -- POST-LOCK: which DRL algorithm suits Uttarakhand once the controller only has
a forecast a real installation could make?

Grid: algorithm {PPO, DQN, SAC} x forecast mode {perfect, smart persistence, none} x seed {0, 1, 2}, one regime,
the frozen M4 reward weights of that regime (w2, w4; w3 by the dominance rule), V2 observation, training-time domain
randomisation. Each run is scripts/m12_train_ext.py (same budget, validation scoring and checkpoint rule as M4),
written under results/ext/fcalgo/r{k}/. Validation years only; no test year is loaded.
Usage: python scripts/m13_forecast_algo_study.py --regime 1 --jobs 18
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_W = {0: (3.0, 0.3), 1: (0.0, 0.3), 2: (0.0, 0.3)}       # M4-selected (w2, w4) per regime
ALGOS, MODES, SEEDS = ("sac", "dqn", "ppo"), ("perfect", "smart", "none"), (0, 1, 2)


def one(regime, algo, mode, seed):
    w2, w4 = FROZEN_W[regime]
    log = ROOT / "results" / "logs" / f"m13_r{regime}_{algo}_{mode}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "m12_train_ext.py"), "--study", "fcalgo", "--algo", algo,
           "--regime", str(regime), "--w2", str(w2), "--w4", str(w4), "--seed", str(seed),
           "--obs-version", "V2", "--forecast-mode", mode, "--domain-randomize"]
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"done r{regime} {algo} {mode} s{seed} rc={rc}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=18)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    a = ap.parse_args()
    jobs = [(a.regime, al, m, s) for al in a.algos for m in a.modes for s in SEEDS]   # SAC first (slowest)
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda j: one(*j), jobs))
    bad = [j for j, rc in zip(jobs, rcs) if rc]
    print("failed:", bad if bad else "none")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
