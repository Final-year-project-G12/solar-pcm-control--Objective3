"""
Phase O3-3 — Safe trajectory generator.

Generates transition datasets from B0 and B1 across years, regimes and PCM initial states. No
test-year (2006-2010) data is ever used here (the loader would refuse it without the O3 lock anyway).

Output columns:
    episode_id, regime_id, year, pcm_init, step, obs_0..obs_12, action, reward,
    next_obs_0..next_obs_12, terminated, safety_cost

Parallel (Step 3): one (year, pcm_init, policy) episode = one job = one process. Every job returns its
own rows; rows are written in the deterministic order of the job list, so the file is byte-identical
for any n_jobs. Episode ids are derived from the job (not from uuid4) for the same reason.
"""

import csv
import pathlib

import numpy as np

import config_O2
from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS, VAL_YEARS, TEST_YEARS
from o3.utils.parallel import run_parallel

CONTRACT_PATH = config_O2.CONTRACT_FILE

PCM_INITS = ["solid", "liquid", "partial"]
OBS_DIM = 13


def episode_job(regime_id: int, year: int, pcm_init: str, policy_name: str) -> list[dict]:
    """One demonstration episode -> list of transition rows (module-level for process pools)."""
    from o3.evaluation.rollout import make_baseline
    env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id, fixed_year=year).inner
    pol = make_baseline(policy_name.lower())
    obs, _ = env.reset(year=year, initial_pcm_state=pcm_init)
    pol.reset()
    if hasattr(pol, "attach_env"):
        pol.attach_env(env)
    eid = f"{policy_name}_r{regime_id}_{year}_{pcm_init}"
    rows, done, step = [], False, 0
    while not done:
        action = pol.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        row = {"episode_id": eid, "regime_id": regime_id, "year": year, "pcm_init": pcm_init, "step": step,
               "action": action, "reward": reward, "terminated": int(terminated or truncated),
               "safety_cost": 1 if info.get("shield_override", False) else 0}
        for i in range(OBS_DIM):
            row[f"obs_{i}"] = float(obs[i])
            row[f"next_obs_{i}"] = float(next_obs[i])
        rows.append(row)
        obs, done, step = next_obs, terminated or truncated, step + 1
    return rows


def generate_dataset(out_path: pathlib.Path, regime_id: int = 0, years: list[int] | None = None,
                     pcm_inits: list[str] | None = None, policies: list[str] | None = None,
                     n_jobs: int = 1) -> int:
    """Generate and save a transition dataset; returns the number of transitions.

    years     : training years to cover (default TRAIN_YEARS)
    pcm_inits : initial PCM states (default all three)
    policies  : baseline names, default ["B0", "B1"]
    n_jobs    : worker processes (1 = serial)
    """
    years = years or TRAIN_YEARS
    pcm_inits = pcm_inits or PCM_INITS
    policies = policies or ["B0", "B1"]
    bad = [y for y in years if y in TEST_YEARS]
    if bad:
        raise PermissionError(f"test years {bad} must never be used for dataset generation")

    jobs = [dict(regime_id=regime_id, year=y, pcm_init=pi, policy_name=p)
            for y in years for pi in pcm_inits for p in policies]
    results = run_parallel(episode_job, jobs, n_jobs=n_jobs)

    fieldnames = (["episode_id", "regime_id", "year", "pcm_init", "step", "action", "reward", "terminated", "safety_cost"]
                  + [f"obs_{i}" for i in range(OBS_DIM)] + [f"next_obs_{i}" for i in range(OBS_DIM)])
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j, rows in zip(jobs, results):
            writer.writerows(rows)
            total += len(rows)
            print(f"  r{regime_id} year={j['year']} init={j['pcm_init']:7s} policy={j['policy_name']}  steps={len(rows)}")
    print(f"\n[INFO] Dataset written: {out_path}  ({total} transitions)")
    return total


def write_split_manifest(out_path: pathlib.Path) -> None:
    """Write a machine-readable train/val/test split manifest."""
    import json
    manifest = {
        "train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS,
        "note": "test years (O3-only fresh 2006-2010) must NOT be used for reward tuning, normalisation, or early "
                "stopping; O3 evaluates them once behind results/o3_test.lock. 2024-2025 are O2's used_test_v1 years "
                "and are never used. Plan 8.2 wanted a fresh 2006-2010 test set (not downloaded, see docs/CORRECTIONS.md).",
    }
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Split manifest written: {out_path}")
