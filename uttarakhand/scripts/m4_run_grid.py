"""
scripts/m4_run_grid.py -- run the Q2 reward-weight grid (w2 x w4 x seeds x regimes) as parallel processes.

Grid from configs/o3_config.yaml (reward.sweep, ppo.seeds, regimes.pcm); plus, with --diagnostic, one w3 = 0 set
(w2 = w4 = 0) on the primary regime, labelled DIAGNOSTIC and never eligible for selection.
Usage: python scripts/m4_run_grid.py [--jobs 3] [--diagnostic] [--regimes 0 1 2]
"""
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from o3 import config as o3c   # noqa: E402


def run(args):
    r, w2, w4, s, w3 = args
    tag = f"r{r}_w2_{w2:g}_w4_{w4:g}_s{s}" + ("" if w3 is None else f"_w3_{w3:g}")
    log = ROOT / "results" / "logs" / f"m4_{tag}.log"
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "m4_train_ppo.py"), "--regime", str(r), "--w2", str(w2),
           "--w4", str(w4), "--seed", str(s)] + ([] if w3 is None else ["--w3", str(w3)])
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return tag, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--regimes", type=int, nargs="+", default=None)
    a = ap.parse_args()
    cfg = o3c.load()
    regs = a.regimes or cfg["regimes"]["pcm"]
    jobs = [(r, w2, w4, s, None) for r, w2, w4, s in itertools.product(
        regs, cfg["reward"]["sweep"]["w2"], cfg["reward"]["sweep"]["w4"], cfg["ppo"]["seeds"])]
    if a.diagnostic:
        jobs += [(cfg["regimes"]["primary"], 0.0, 0.0, s, 0.0) for s in cfg["ppo"]["seeds"]]
    print(len(jobs), "runs", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for tag, rc in ex.map(run, jobs):
            print(("ok  " if rc == 0 else f"FAIL {rc} ") + tag, flush=True)


if __name__ == "__main__":
    main()
