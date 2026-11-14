"""
run_all_plots.py — build every plot of the Rajasthan O3 tree in one command, with the same layout as tamilnadu/.

Tamil Nadu's results/plots/ holds, per reward tag, two files:
    results/plots/<tag>/dashboard.html                 (Plotly dashboard from scripts/make_plots.py)
    results/plots/<tag>/tamilnadu_regimes_map.html     (Folium regime map)
This script writes the same two files for Rajasthan (the map is named rajasthan_regimes_map.html):
    results/plots/R2/            regime 1  (the headline regime; Tamil Nadu's R2 is its single regime 0)
    results/plots/R2_r0/         regime 0  (Udaipur)
    results/plots/R2_r2/         regime 2  (Jaipur)
plus the Rajasthan-only plots that already existed:
    results/plots/gen2/gen2_dqn.html, gen2_sac.html    (make_gen2_plots.py)
    results/plots/deck/o3_rj_*.png                     (make_deck_plots.py)

There is no R1 tag: Tamil Nadu's R1 is an archived earlier reward (results/archive_R1_reward), and Rajasthan has only one
reward version, so there is nothing to put under it.

Run from the rajasthan O3 root:
    python scripts/run_all_plots.py                      # everything
    python scripts/run_all_plots.py --refresh            # also rebuild results/phase4_comparison_r*.csv from the saved models first
    python scripts/run_all_plots.py --only dashboards    # dashboards | gen2 | deck
    python scripts/run_all_plots.py --results-dir results/rerun_2026-11-13   # gen2 plots from the retrain run

Reads saved results only; it trains nothing and never touches the locked test.
"""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TAGS = {1: "R2", 0: "R2_r0", 2: "R2_r2"}


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dashboards", "gen2", "deck"], default=None)
    ap.add_argument("--refresh", action="store_true", help="rebuild the phase-4 comparison CSVs from the saved models first")
    ap.add_argument("--results-dir", default=None, help="results root to read (default results/)")
    a = ap.parse_args()
    rd = ["--results-dir", a.results_dir] if a.results_dir else []
    failed = []

    if a.only in (None, "dashboards"):
        for regime, tag in TAGS.items():
            if a.refresh and not run(SCRIPTS / "phase4_reeval.py", "--regime", regime, "--n-jobs", 2, *rd):
                failed.append(f"phase4_reeval r{regime}")
                continue
            if not run(SCRIPTS / "make_plots.py", "--regime", regime, "--tag", tag, *rd):
                failed.append(f"dashboard {tag}")
    if a.only in (None, "gen2"):
        for algo in ("dqn", "sac"):
            if not run(SCRIPTS / "make_gen2_plots.py", "--algo", algo, *rd):
                failed.append(f"gen2 {algo}")
    if a.only in (None, "deck"):
        if not run(SCRIPTS / "make_deck_plots.py"):
            failed.append("deck")

    print("\n" + ("[OK] all plots built under results/plots/" if not failed else "[FAIL] " + ", ".join(failed)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
