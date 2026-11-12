"""
gen2_study.py — the Generation-2 training study (plan section 19.3), Rajasthan.

  A  Pareto sweep      : --algo at shield weights {0, 0.5, 1, 1.5, 2} x the regime default, seeds 42 / 0 / 123   (15 runs)
                         -> delivery-vs-shield frontier and seed spread
  B  Sub-hourly ablation: the same configuration at control_step_min = 15 (vs the 60-minute run already in A)   (1 run)
  C  Regime coverage   : the selected weight and seed on the other two regimes                                   (2 runs)

All runs: V3 observation (forecast fields), domain randomisation in TRAINING only, 300k steps. TN's weights {0,5,10,15,20}
are NOT reused: they are multiples of TN's default 10; here the same multiples of Rajasthan's per-regime default are used
(docs/CORRECTIONS.md D4).

    python scripts/gen2_study.py --algo dqn --study A B C --regime 1 --n-jobs 6
    python scripts/gen2_study.py --algo sac --study A --seeds 42 0 123 --multipliers 1 --n-jobs 3      # SAC is ~20x slower
    python scripts/gen2_study.py --algo dqn --summarise-only

One process per run (--n-jobs), resumable: a run whose run_meta.json exists is not retrained. After training every run is
scored on the validation years in its own observation layout / cadence, together with B0, B1 and the water-only twin at
the same cadence. Selection (plan 18.1): zero hard violations -> delivery >= 95 % of B0 -> maximum shield reduction.
Outputs: results/gen2/gen2_results_<algo>.csv (per run-year), gen2_summary_<algo>.csv (per run), gen2_selected_<algo>.json.
--n-envs is unused (off-policy agents; one environment per run).
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args, resolve  # noqa: E402
from o3.utils.parallel import run_parallel  # noqa: E402
from o3.utils.weather_roles import VAL_YEARS  # noqa: E402
from o3.utils.models import sha256_of_file  # noqa: E402
from o3.environment.pcm_env import default_w_shield  # noqa: E402
from o3.agents.gen2 import train_run, evaluate_run, evaluate_baseline_as, gen2_dir, default_tag, run_model_path  # noqa: E402

MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEEDS = [42, 0, 123]
MIN_DELIVERY_FRAC = 0.95


def plan_runs(algo, studies, regime, seeds, mults, steps) -> list[dict]:
    w0 = default_w_shield(regime)
    runs = []
    if "A" in studies:
        runs += [dict(algo=algo, regime_id=regime, steps=steps, seed=s, shield_weight=round(m * w0, 4), study="A") for m in mults for s in seeds]
    if "B" in studies:
        runs.append(dict(algo=algo, regime_id=regime, steps=steps, seed=seeds[0], shield_weight=w0, control_step_min=15, study="B"))
        runs.append(dict(algo=algo, regime_id=regime, steps=steps, seed=seeds[0], shield_weight=w0, study="B-ref"))
    if "C" in studies:
        runs += [dict(algo=algo, regime_id=r, steps=steps, seed=seeds[0], shield_weight=default_w_shield(r), study="C")
                 for r in (0, 1, 2) if r != regime]
    seen, out = set(), []
    for r in runs:
        r.setdefault("control_step_min", 60)
        r["tag"] = default_tag(r["algo"], r["regime_id"], r["seed"], r["shield_weight"], "V3", r["control_step_min"], True)
        if r["tag"] not in seen:
            seen.add(r["tag"])
            out.append(r)
    return out


def _train(study, **kw):
    return train_run(obs_version="V3", domain_randomize=True, **kw)


def summarise(algo: str, res_dir, n_jobs: int) -> None:
    g = gen2_dir(res_dir)
    def _in_study(m):          # the main study is V3 + training-time randomisation; ablation runs (gen2_ablation.py) are scored apart
        return m["algo"] == algo and m.get("obs_version") == "V3" and m.get("domain_randomize", True)
    run_dirs = sorted(d for d in g.iterdir() if (d / "run_meta.json").exists() and _in_study(json.loads((d / "run_meta.json").read_text())))
    if not run_dirs:
        print("no finished runs")
        return
    metas = [json.loads((d / "run_meta.json").read_text()) for d in run_dirs]
    rows = run_parallel(evaluate_run, [dict(run_dir=str(d), year=y) for d in run_dirs for y in VAL_YEARS], n_jobs=n_jobs)
    combos = sorted({(m["regime_id"], m["control_step_min"]) for m in metas})
    bjobs = [dict(baseline=b, regime_id=r, year=y, control_step_min=c, water_only=(b == "twin")) for r, c in combos for y in VAL_YEARS
             for b in ("twin", "b0", "b1")]
    for j in bjobs:
        if j["baseline"] == "twin":
            j["baseline"] = "b0"
    base = run_parallel(evaluate_baseline_as, bjobs, n_jobs=n_jobs)
    ref = {}
    for r, c in combos:
        sel = [b for b in base if b["regime_id"] == r and b["control_step_min"] == c]
        mean = lambda pol, k: float(np.mean([b[k] for b in sel if b["policy"].startswith(pol)]))      # noqa: E731
        ref[(r, c)] = {p: {"E45": mean(p, "total_delivered_kWh"), "win": mean(p, "E_del45_window_kWh"),
                           "shield_h": mean(p, "shield_water_h") + mean(p, "shield_pcm_h"),
                           "n_shield": mean(p, "total_shield_water") + mean(p, "total_shield_pcm")} for p in ("TWIN", "B0", "B1")}
    summ = []
    for d, m in zip(run_dirs, metas):
        rs = [r for r in rows if r["tag"] == m["tag"]]
        b = ref[(m["regime_id"], m["control_step_min"])]
        e45, win = float(np.mean([r["total_delivered_kWh"] for r in rs])), float(np.mean([r["E_del45_window_kWh"] for r in rs]))
        sh = float(np.mean([r["shield_water_h"] + r["shield_pcm_h"] for r in rs]))
        summ.append({**{k: m[k] for k in ("tag", "algo", "regime_id", "seed", "shield_weight", "control_step_min", "train_minutes")},
                     "E45_kWh": e45, "window_kWh": win, "gain_window_vs_twin_kWh": win - b["TWIN"]["win"],
                     "pct_of_B0_delivery": 100 * e45 / b["B0"]["E45"], "pct_of_B0_window": 100 * win / b["B0"]["win"],
                     "shield_h": sh, "shield_cut_vs_B0_pct": 100 * (1 - sh / b["B0"]["shield_h"]),
                     "n_shield": float(np.mean([r["total_shield_water"] + r["total_shield_pcm"] for r in rs])),
                     "switches": float(np.mean([r["action_switches"] for r in rs])),
                     "violations": int(sum(r["n_safety_violations"] for r in rs)),
                     "hours_over_limit": float(sum(r["hours_over_limit"] for r in rs)),
                     "B0_E45_kWh": b["B0"]["E45"], "B0_shield_h": b["B0"]["shield_h"], "B1_E45_kWh": b["B1"]["E45"], "B1_shield_h": b["B1"]["shield_h"]})
    for name, data in ((f"gen2_results_{algo}.csv", rows + base), (f"gen2_summary_{algo}.csv", summ)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(g / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    print(f"\n{'run':44s} {'E45':>7s} {'%B0':>6s} {'win%B0':>6s} {'shield h':>8s} {'cut%':>6s} {'switch':>7s} {'viol':>4s}")
    for s in summ:
        print(f"{s['tag']:44s} {s['E45_kWh']:7.1f} {s['pct_of_B0_delivery']:6.1f} {s['pct_of_B0_window']:6.1f} {s['shield_h']:8.0f} "
              f"{s['shield_cut_vs_B0_pct']:6.1f} {s['switches']:7.0f} {s['violations']:4d}")
    selected = {}
    for r, c in combos:
        ok = [s for s in summ if s["regime_id"] == r and s["control_step_min"] == c and s["violations"] == 0 and s["hours_over_limit"] == 0
              and s["pct_of_B0_delivery"] >= 100 * MIN_DELIVERY_FRAC and s["pct_of_B0_window"] >= 100 * MIN_DELIVERY_FRAC]
        b = ref[(r, c)]
        print(f"\nregime {r}, {c}-min control: B0 E45 {b['B0']['E45']:.1f} kWh / shield {b['B0']['shield_h']:.0f} h | "
              f"B1 {b['B1']['E45']:.1f} kWh / {b['B1']['shield_h']:.0f} h")
        if ok:
            best = max(ok, key=lambda s: s["shield_cut_vs_B0_pct"])
            path = run_model_path(g / best["tag"])
            selected[f"r{r}_{c}min"] = {**best, "model": str(path), "sha256": sha256_of_file(path),
                                        "rule": "zero hard violations -> delivery (annual and window) >= 95 % of B0 -> max shield-hour reduction; validation years"}
            print(f"  SELECTED {best['tag']}: {best['pct_of_B0_delivery']:.1f} % of B0 delivery, shield cut {best['shield_cut_vs_B0_pct']:.1f} %")
        else:
            print("  no run meets the delivery >= 95 % of B0 rule with zero violations")
    (g / f"gen2_selected_{algo}.json").write_text(json.dumps(selected, indent=2))
    print(f"\n-> {g / f'gen2_summary_{algo}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="dqn", choices=["dqn", "sac", "qrdqn", "ppo", "maskable", "recurrent"])
    ap.add_argument("--study", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--multipliers", type=float, nargs="+", default=MULTIPLIERS)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--summarise-only", action="store_true")
    add_parallel_args(ap)
    a = ap.parse_args()
    n_jobs, _ = resolve(a)
    if not a.summarise_only:
        runs = plan_runs(a.algo, a.study, a.regime, a.seeds, a.multipliers, a.steps)
        g = gen2_dir(a.results_dir)
        todo = [r for r in runs if not (g / r["tag"] / "run_meta.json").exists()]
        print(f"Gen-2 study {a.study} | {a.algo} | {len(runs)} runs planned, {len(todo)} to train | {a.steps:,} steps | n_jobs {n_jobs}")
        t0 = time.time()
        run_parallel(_train, [dict(results_dir=a.results_dir, **r) for r in todo], n_jobs=n_jobs)
        print(f"training wall time {(time.time() - t0) / 60:.1f} min")
    summarise(a.algo, a.results_dir, n_jobs)


if __name__ == "__main__":
    main()
