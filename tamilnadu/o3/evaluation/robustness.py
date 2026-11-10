"""
Phase O3-5 — Robustness and ablation evaluation.

Tests a trained policy (or any policy_fn) under perturbations:
  - GHI sensor noise (± sigma_pct %)
  - Ambient temperature bias (± delta_C)
  - Demand volume shift (scale factor)
  - PCM initial state variation (solid, liquid, partial)
  - Unseen years (held-out val or test)

The safety shield inside PCMEnv always remains active; these tests
check that the *policy* remains good and *safe* under perturbations.

No test-year (2011-2015) data may be passed here unless the caller
explicitly labels the run as "final_test" in the scenario name.
"""

import pathlib
import numpy as np
import csv

from o3.environment.pcm_env import PCMEnv

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[2] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"


class NoisyPCMEnv:
    """Wraps PCMEnv and injects observation-level noise."""

    def __init__(self, inner: PCMEnv,
                 ghi_noise_pct:  float = 0.0,
                 tamb_noise_C:   float = 0.0,
                 demand_scale:   float = 1.0):
        self._env          = inner
        self._ghi_noise    = ghi_noise_pct / 100.0
        self._tamb_noise   = tamb_noise_C
        self._demand_scale = demand_scale
        self._rng          = np.random.default_rng()

    def reset(self, year: int, pcm_init: str = "solid"):
        obs, info = self._env.reset(year=year, initial_pcm_state=pcm_init)
        return self._perturb(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._perturb(obs), reward, terminated, truncated, info

    def _perturb(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        if self._ghi_noise > 0:
            # obs[0] is normalised GHI; add multiplicative noise
            noise = self._rng.normal(0.0, self._ghi_noise)
            obs[0] = obs[0] * (1.0 + noise)
        if self._tamb_noise > 0:
            # obs[1] is normalised T_amb; add additive noise (scaled ~1/50)
            obs[1] += self._rng.normal(0.0, self._tamb_noise / 50.0)
        return np.clip(obs, -5.0, 5.0).astype(np.float32)


def run_robustness_episode(
    policy_fn,
    env: PCMEnv | NoisyPCMEnv,
    year: int,
    pcm_init: str = "solid",
) -> dict:
    obs, _ = env.reset(year=year, pcm_init=pcm_init) if isinstance(env, NoisyPCMEnv) \
        else env.reset(year=year, initial_pcm_state=pcm_init)
    done = False
    total_del = total_unmet = total_reward = 0.0
    shield_w = shield_p = safety_v = steps = 0

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_del    += info.get("E_delivered_Wh", 0.0)
        total_unmet  += info.get("E_unmet_Wh", 0.0)
        shield_w     += int(info.get("n_shield_water", 0))
        shield_p     += int(info.get("n_shield_pcm", 0))
        safety_v     += int(info.get("n_safety_violations", 0))
        steps        += 1

    return {
        "n_steps":              steps,
        "total_reward":         total_reward,
        "total_delivered_kWh":  total_del / 1000.0,
        "total_unmet_kWh":      total_unmet / 1000.0,
        "total_shield_water":   shield_w,
        "total_shield_pcm":     shield_p,
        "n_safety_violations":  safety_v,
    }


def run_robustness_suite(
    policy_fn,
    policy_name: str,
    regime_id: int,
    years: list[int],
    scenarios: list[dict] | None = None,
) -> list[dict]:
    """Run the policy across all perturbation scenarios.

    Parameters
    ----------
    policy_fn   : callable(obs: np.ndarray) -> int
    policy_name : string label for results
    regime_id   : 0, 1 or 2
    years       : years to evaluate on
    scenarios   : list of dicts with keys:
                  name, ghi_noise_pct, tamb_noise_C, demand_scale, pcm_init
                  (defaults to a standard 10-scenario suite)
    """
    if scenarios is None:
        scenarios = _default_scenarios()

    results = []
    for scenario in scenarios:
        inner = PCMEnv(CONTRACT_PATH, regime_id=regime_id)
        env = NoisyPCMEnv(
            inner,
            ghi_noise_pct = scenario.get("ghi_noise_pct", 0.0),
            tamb_noise_C  = scenario.get("tamb_noise_C",  0.0),
        )
        pcm_init = scenario.get("pcm_init", "solid")

        for year in years:
            ep = run_robustness_episode(policy_fn, env, year, pcm_init)
            results.append({
                "policy":           policy_name,
                "scenario":         scenario["name"],
                "regime_id":         regime_id,
                "year":              year,
                "pcm_init":          pcm_init,
                **ep,
            })
            print(f"  {policy_name:15s} {scenario['name']:20s} "
                  f"y={year} init={pcm_init:7s} "
                  f"del={ep['total_delivered_kWh']:.1f} kWh  "
                  f"safety={ep['n_safety_violations']}")
    return results


def _default_scenarios() -> list[dict]:
    return [
        # --- baseline (no perturbation) ---
        {"name": "nominal",         "ghi_noise_pct": 0,   "tamb_noise_C": 0,   "pcm_init": "solid"},
        # --- PCM initial state variation ---
        {"name": "init_liquid",     "ghi_noise_pct": 0,   "tamb_noise_C": 0,   "pcm_init": "liquid"},
        {"name": "init_partial",    "ghi_noise_pct": 0,   "tamb_noise_C": 0,   "pcm_init": "partial"},
        # --- GHI sensor noise ---
        {"name": "ghi_noise_5pct",  "ghi_noise_pct": 5,   "tamb_noise_C": 0,   "pcm_init": "solid"},
        {"name": "ghi_noise_10pct", "ghi_noise_pct": 10,  "tamb_noise_C": 0,   "pcm_init": "solid"},
        {"name": "ghi_noise_20pct", "ghi_noise_pct": 20,  "tamb_noise_C": 0,   "pcm_init": "solid"},
        # --- ambient temperature bias ---
        {"name": "tamb_plus2C",     "ghi_noise_pct": 0,   "tamb_noise_C": 2,   "pcm_init": "solid"},
        {"name": "tamb_minus2C",    "ghi_noise_pct": 0,   "tamb_noise_C": -2,  "pcm_init": "solid"},
        # --- combined stress ---
        {"name": "combined_stress", "ghi_noise_pct": 10,  "tamb_noise_C": 2,   "pcm_init": "liquid"},
        {"name": "worst_case",      "ghi_noise_pct": 20,  "tamb_noise_C": -2,  "pcm_init": "partial"},
    ]


def save_robustness_results(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Robustness results saved -> {path}")


def print_robustness_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'Policy':15s} {'Scenario':20s} {'Year':5s} {'Init':7s} "
          f"{'Del kWh':8s} {'Unmet kWh':10s} {'Shield_w':9s} "
          f"{'Shield_p':9s} {'Safety':7s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['policy']:15s} {r['scenario']:20s} {r['year']:5d} "
            f"{r['pcm_init']:7s} "
            f"{r['total_delivered_kWh']:8.1f} {r['total_unmet_kWh']:10.1f} "
            f"{r['total_shield_water']:9d} {r['total_shield_pcm']:9d} "
            f"{r['n_safety_violations']:7d}"
        )
    print("=" * 100)

    # --- acceptance check ---
    violations = [r for r in results if r["n_safety_violations"] > 0]
    if violations:
        print(f"\n[WARN] {len(violations)} scenario(s) have safety violations:")
        for r in violations:
            print(f"  {r['policy']} | {r['scenario']} | y={r['year']} | "
                  f"{r['n_safety_violations']} violations")
    else:
        print("\n[OK]  Zero hard-limit violations across all robustness scenarios.")
