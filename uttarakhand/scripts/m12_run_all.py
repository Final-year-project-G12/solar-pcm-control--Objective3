"""
scripts/m12_run_all.py -- the post-lock extension study (21 runs), 2 in parallel (RAM limit, see HOW_TO_RUN).
  dqn     DQN, regimes 0-2 x seeds 0-2, each at its regime's FROZEN PPO weight set (like-for-like with PPO)
  pareto  PPO, regime 1, w4 = 0.3, w2 in {1, 10, 30} x seeds 0-2 (w2 = 0 and 3 already exist from M4, same budget)
  v2dr    PPO, regime 1, frozen weights, V2 forecast observation + training-time domain randomisation, seeds 0-2
  w3diag  PPO, regime 1, frozen w2 / w4, w3 in {0, 3, 10} (below the dominance value) x seeds 0-2: DIAGNOSTIC
          delivery-vs-shield frontier (the w2 sweep is flat because the dominance-rule w3 already removes shield time)
Usage: python scripts/m12_run_all.py [--jobs 2] [--only dqn pareto v2dr]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from o3 import config as o3c   # noqa: E402


def jobs(only):
    js = []
    seeds = o3c.load()["ppo"]["seeds"]
    fz = {k: json.loads((o3c.RESULTS / "frozen" / f"r{k}" / "frozen.json").read_text()) for k in (0, 1, 2)}
    if "dqn" in only:
        js += [("dqn", "dqn", k, fz[k]["w2"], fz[k]["w4"], s, []) for k in (0, 1, 2) for s in seeds]
    if "pareto" in only:
        js += [("pareto", "ppo", 1, w2, 0.3, s, []) for w2 in (1.0, 10.0, 30.0) for s in seeds]
    if "v2dr" in only:
        js += [("v2dr", "ppo", 1, fz[1]["w2"], fz[1]["w4"], s, ["--obs-version", "V2", "--domain-randomize"]) for s in seeds]
    if "w3diag" in only:
        js += [("w3diag", "ppo", 1, fz[1]["w2"], fz[1]["w4"], s, ["--w3", str(w3)]) for w3 in (0.0, 3.0, 10.0) for s in seeds]
    return js


def run(j):
    study, algo, k, w2, w4, s, extra = j
    tag = f"{study}_r{k}_{algo}_w2_{w2:g}_w4_{w4:g}_s{s}" + ("_" + "_".join(x.strip("-") for x in extra) if extra else "")
    with open(ROOT / "results" / "logs" / f"m12_{tag}.log", "w") as fh:
        rc = subprocess.call([sys.executable, "-u", str(ROOT / "scripts" / "m12_train_ext.py"), "--study", study,
                              "--algo", algo, "--regime", str(k), "--w2", str(w2), "--w4", str(w4), "--seed", str(s)]
                             + extra, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return tag, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="+", default=["dqn", "pareto", "v2dr"])
    a = ap.parse_args()
    js = jobs(a.only)
    print(len(js), "runs", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for tag, rc in ex.map(run, js):
            print(("ok  " if rc == 0 else f"FAIL {rc} ") + tag, flush=True)


if __name__ == "__main__":
    main()
