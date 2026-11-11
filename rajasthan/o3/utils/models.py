"""
o3/utils/models.py — where trained models live, hashes, and the lexicographic choice rule.

Directory convention (one place; the TN scripts disagreed about it):
    results/<algo_dir>/r<regime>/best_model/best_model.zip     best checkpoint by validation reward (EvalCallback)
    results/<algo_dir>/r<regime>/<algo_dir>_final.zip          final model
    results/<algo_dir>/r<regime>_seed<seed>/...                one directory per seed (phase4_multiseed)
algo_dir: ppo, dqn, recurrent_ppo, qrdqn, maskable_ppo, sac.
"""

import hashlib
import json
import pathlib

import config_O2
from o3.evaluation.rollout import ALGO_DIR


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def model_dir(algo: str, regime_id: int, seed: int | None = None, results_dir=None) -> pathlib.Path:
    base = pathlib.Path(results_dir) if results_dir else config_O2.RESULTS_DIR
    return base / ALGO_DIR[algo] / (f"r{regime_id}" if seed is None else f"r{regime_id}_seed{seed}")


def find_model(algo: str, regime_id: int, seed: int | None = None, results_dir=None) -> pathlib.Path | None:
    """Path of the .zip to load (best checkpoint, else final), or None if nothing is trained."""
    d = model_dir(algo, regime_id, seed, results_dir)
    for p in (d / "best_model" / "best_model.zip", d / f"{ALGO_DIR[algo]}_final.zip"):
        if p.exists():
            return p
    if seed is None:                                   # fall back to any seed directory (first by name)
        for sd in sorted(d.parent.glob(f"r{regime_id}_seed*")):
            for p in (sd / "best_model" / "best_model.zip", sd / f"{ALGO_DIR[algo]}_final.zip"):
                if p.exists():
                    return p
    return None


def lexicographic_key(row: dict):
    """Plan 8.3 selection rule: first zero hard-limit violations, then delivery metric (window gain / window
    energy), then fewer shield hours, then fewer switches. Smaller key is better."""
    gain = row.get("gain_window_kWh", row.get("mean_gain_window_kWh", row.get("E_del45_window_kWh", 0.0)))
    return (row.get("n_safety_violations", 0) > 0, -gain,
            row.get("shield_water_h", 0.0) + row.get("shield_pcm_h", 0.0), row.get("action_switches", 0))
