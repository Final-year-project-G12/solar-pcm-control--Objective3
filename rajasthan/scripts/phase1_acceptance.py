"""
Phase O3-1 — contract acceptance test, run BEFORE any full training.

The Rajasthan contract (acceptance_test_before_drl_training) lists six checks. Each is mapped to the pytest
tests that prove it, and this script runs exactly those tests and prints PASS / FAIL per check:

  1. every action stays within the flow envelope / temperature limits       -> T4, T6 (limits), action space is discrete
  2. charge/discharge/bypass transitions are physically valid               -> T3 (bypass schedule == zeroed GHI), idle == bypass
  3. the environment conserves energy (< 0.1 % residual, O2 Gate 1)         -> T4
  4. the safety shield cannot be bypassed by the policy                     -> T6
  5. the same seed reproduces the same trajectory                           -> T5
  6. simulator output matches the O2 re-evaluation for identical actions    -> T1 (and T2 split run)

Run from the rajasthan O3 root:
    python scripts/phase1_acceptance.py [--n-jobs N]
Exit code 1 if any check fails. Result written to results/phase1_acceptance.json.
(--n-jobs is passed to pytest-xdist only if installed; otherwise the tests run serially. --n-envs is unused.)
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

from o3.utils.cli import add_parallel_args  # noqa: E402

ENV = "tests/test_environment.py"
BASE = "tests/test_baselines.py"
CHECKS = [
    ("1  every action within flow envelope / temperature limits",
     [f"{ENV}::test_T4_energy_residual_below_gate1", f"{ENV}::test_no_hard_safety_violations_b0",
      f"{ENV}::test_T6_shield_overrides_charge_when_top_node_hot", "tests/test_phase0_audit.py::test_flow_envelope_is_rajasthan",
      f"{BASE}::test_b4_is_reproducible_and_safe"]),
    ("2  charge/discharge/bypass transitions physically valid",
     [f"{ENV}::test_T3_bypass_schedule_equals_zeroed_ghi_year", f"{ENV}::test_idle_and_bypass_are_physically_identical"]),
    ("3  energy conserved (< 0.1 % of collector energy)", [f"{ENV}::test_T4_energy_residual_below_gate1"]),
    ("4  safety shield cannot be bypassed by the policy", [f"{ENV}::test_T6_shield_overrides_charge_when_top_node_hot"]),
    ("5  same seed reproduces the same trajectory", [f"{ENV}::test_T5_same_seed_same_trajectory"]),
    ("6  simulator output equals the O2 re-evaluation for identical actions",
     [f"{ENV}::test_T1_stepped_year_equals_run_case", f"{ENV}::test_T2_two_half_year_calls_equal_one_full_call"]),
    ("+  guards: hash mismatch / test year without lock / SB3 check_env",
     [f"{ENV}::test_T7_contract_hash_mismatch_makes_env_refuse", f"{ENV}::test_T7_test_year_refused_without_lock",
      f"{ENV}::test_T7_lock_is_exclusive", f"{ENV}::test_T8_check_env"]),
]


def main():
    ap = argparse.ArgumentParser()
    add_parallel_args(ap)
    ap.parse_args()
    print("=" * 70)
    print("Phase O3-1 — contract acceptance test (must pass before full training)")
    print("=" * 70)
    results, ok_all = {}, True
    for name, tests in CHECKS:
        cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests]
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        ok_all &= ok
        results[name] = {"passed": ok, "tests": tests}
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(r.stdout[-1500:])
    out = config_O2.RESULTS_DIR / "phase1_acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"all_passed": ok_all, "checks": results}, indent=2))
    print(f"\n{'[OK]   acceptance test PASSED' if ok_all else '[FAIL] acceptance test FAILED'} -> {out}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
