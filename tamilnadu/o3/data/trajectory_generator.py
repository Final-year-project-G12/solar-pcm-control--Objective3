"""
Phase O3-3 — Safe trajectory generator.

Generates transition datasets from B0 and B1 policies across
multiple years, regimes, and PCM initial states.
No test-year (2011-2015) data is ever used here.

Output columns:
    episode_id, regime_id, year, pcm_init,
    step, obs_0..obs_10, action, reward,
    next_obs_0..next_obs_10, terminated, safety_cost
"""

import pathlib
import csv
import uuid
import numpy as np

from o3.environment.gym_wrapper import PCMGymEnv, TRAIN_YEARS

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"

PCM_INITS = ["solid", "liquid", "partial"]
OBS_DIM   = 11


def _run_episode(env: PCMGymEnv, policy_fn, year: int, pcm_init: str) -> list[dict]:
    obs, _ = env.inner.reset(year=year, initial_pcm_state=pcm_init)
    episode_id = str(uuid.uuid4())[:8]
    rows = []
    done = False
    step = 0
    while not done:
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, info = env.inner.step(action)
        safety_cost = 1 if info.get("shield_override", False) else 0
        row = {
            "episode_id": episode_id,
            "regime_id":  env.inner.regime_id,
            "year":       year,
            "pcm_init":   pcm_init,
            "step":       step,
            "action":     action,
            "reward":     reward,
            "terminated": int(terminated or truncated),
            "safety_cost": safety_cost,
        }
        for i in range(OBS_DIM):
            row[f"obs_{i}"]      = float(obs[i])
            row[f"next_obs_{i}"] = float(next_obs[i])
        rows.append(row)
        obs  = next_obs
        done = terminated or truncated
        step += 1
    return rows


def generate_dataset(
    out_path: pathlib.Path,
    regime_id: int = 0,
    years: list[int] | None = None,
    pcm_inits: list[str] | None = None,
    policies: dict | None = None,
) -> None:
    """Generate and save a transition dataset.

    Parameters
    ----------
    out_path   : CSV output file path
    regime_id  : 0, 1, or 2
    years      : training years to cover (default TRAIN_YEARS)
    pcm_inits  : initial PCM states (default all three)
    policies   : {name: callable(obs) -> action}  (default B0+B1)
    """
    from o3.baselines.b0_fixed    import FixedChargePolicy
    from o3.baselines.b1_reactive import ReactiveRulePolicy

    years     = years     or TRAIN_YEARS
    pcm_inits = pcm_inits or PCM_INITS
    if policies is None:
        b0 = FixedChargePolicy()
        b1 = ReactiveRulePolicy()
        policies = {
            "B0": lambda obs: b0.act(obs),
            "B1": lambda obs: b1.act(obs),
        }

    env = PCMGymEnv(CONTRACT_PATH, regime_id=regime_id,
                    fixed_year=years[0])

    fieldnames = (
        ["episode_id", "regime_id", "year", "pcm_init", "step",
         "action", "reward", "terminated", "safety_cost"]
        + [f"obs_{i}"      for i in range(OBS_DIM)]
        + [f"next_obs_{i}" for i in range(OBS_DIM)]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for year in years:
            for pcm_init in pcm_inits:
                for pol_name, pol_fn in policies.items():
                    rows = _run_episode(env, pol_fn, year, pcm_init)
                    # tag policy name into episode_id
                    for r in rows:
                        r["episode_id"] = f"{pol_name}_{r['episode_id']}"
                    writer.writerows(rows)
                    total += len(rows)
                    print(f"  r{regime_id} year={year} init={pcm_init:7s} "
                          f"policy={pol_name}  steps={len(rows)}")

    print(f"\n[INFO] Dataset written: {out_path}  ({total} transitions)")


def write_split_manifest(out_path: pathlib.Path) -> None:
    """Write a machine-readable train/val/test split manifest."""
    manifest = {
        "train": TRAIN_YEARS,
        "val":   [2021, 2022, 2023],
        "test":  [2011, 2012, 2013, 2014, 2015],
        "note":  "test years must NOT be used for reward tuning, normalisation, or early stopping",
    }
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Split manifest written: {out_path}")
