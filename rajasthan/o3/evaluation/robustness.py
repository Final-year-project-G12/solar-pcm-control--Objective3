"""
Phase O3-5 — Robustness and ablation evaluation.

Tests a trained policy (or any policy_fn) under perturbations:
  - GHI sensor noise (multiplicative, sigma %)                    [observation only]
  - ambient-temperature sensor BIAS (+/- delta C)                  [observation only]
  - demand volume scale (x0.7, x1.3) and stochastic daily demand   [physics: the hot-water draw changes]
  - PCM initial state variation (solid, liquid, partial)
  - unseen years (validation years here; test years only behind the O3 lock)

The safety shield inside the kernel always stays active; these tests check that the *policy* remains
good and *safe* under perturbations.

Corrections to the TN version (docs/CORRECTIONS.md): the TN `tamb_noise_C` was applied only when > 0, so
its "tamb_minus2C" scenario did nothing, and it was scaled by /50 instead of the observation's /25;
`demand_scale` was accepted but never used; the noise RNG was unseeded (results not reproducible).
Here the bias keeps its sign, uses the right scale, demand scenarios change the physics through
PCMEnv(demand_volume_multiplier=...), and every episode's noise is seeded from (seed, year) so the
result does not depend on job order or on how many worker processes are used.

Sensor noise and forecast error are deferred by the plan (sections 8.1, 11.1); the observation-noise
scenarios are kept because the TN suite has them and they are cheap.

No test-year (2006-2010) data may be passed here unless the O3 lock is held (weather_roles enforces it).
"""

import csv
import pathlib

import numpy as np

import config_O2
from o3.environment.pcm_env import PCMEnv
from o3.environment import obs as _obs

CONTRACT_PATH = config_O2.CONTRACT_FILE

_IDX = {"V2": {"ghi": 5, "tamb": 6}, "V3": {"ghi": 5, "tamb": 6}, "V1": {"ghi": 0, "tamb": 1}}


class NoisyPCMEnv:
    """Wraps PCMEnv and injects observation-level noise / bias (the physics is untouched)."""

    def __init__(self, inner: PCMEnv, ghi_noise_pct: float = 0.0, tamb_noise_C: float = 0.0,
                 demand_scale: float = 1.0, seed: int = 0):
        self._env = inner
        self._ghi_noise = ghi_noise_pct / 100.0
        self._tamb_bias = tamb_noise_C            # signed bias, deg C
        self._demand_scale = demand_scale         # recorded; applied when the inner env was built
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._idx = _IDX[inner.obs_version]

    def reset(self, year: int, pcm_init: str = "solid"):
        self._rng = np.random.default_rng([self._seed, int(year)])
        obs, info = self._env.reset(year=year, initial_pcm_state=pcm_init, seed=self._seed + int(year))
        return self._perturb(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._perturb(obs), reward, terminated, truncated, info

    def _perturb(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        if self._ghi_noise > 0:
            obs[self._idx["ghi"]] = obs[self._idx["ghi"]] * (1.0 + self._rng.normal(0.0, self._ghi_noise))
        if self._tamb_bias != 0:
            obs[self._idx["tamb"]] += self._tamb_bias / 25.0          # observation scale of T_amb is (T-15)/25
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    @property
    def regime_id(self):
        return self._env.regime_id

    @property
    def _state(self):
        return self._env._state

    def episode_summary(self):
        return self._env.episode_summary()


def run_robustness_episode(policy_fn, env, year: int, pcm_init: str = "solid") -> dict:
    obs, _ = env.reset(year=year, pcm_init=pcm_init) if isinstance(env, NoisyPCMEnv) \
        else env.reset(year=year, initial_pcm_state=pcm_init)
    done = False
    total_reward = 0.0
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    s = env.episode_summary()
    st = env._state
    return {
        "n_steps": s["n_steps"], "total_reward": total_reward,
        "total_delivered_kWh": s["E_del45_annual_kWh"], "total_unmet_kWh": s["E_unmet_kWh"],
        "total_shield_water": st["n_shield_water"], "total_shield_pcm": st["n_shield_pcm"],
        "n_safety_violations": st["n_safety_violations"],
        "E_del45_window_kWh": s["E_del45_window_kWh"], "shield_water_h": s["shield_water_h"],
        "shield_pcm_h": s["shield_pcm_h"], "hours_over_limit": s["hours_over_limit"],
    }


def _policy_from_spec(spec: tuple):
    """('baseline', 'b1') | ('model', algo, path) -> object with act(obs)/reset()."""
    if spec[0] == "baseline":
        from o3.evaluation.rollout import make_baseline
        return make_baseline(spec[1])
    from o3.evaluation.rollout import ModelPolicy, ALGO_KIND, load_model
    import torch
    torch.set_num_threads(1)
    return ModelPolicy(load_model(spec[1], spec[2]), ALGO_KIND[spec[1]])


def robustness_job(policy_spec: tuple, policy_name: str, regime_id: int, year: int, scenario: dict,
                   contract_path=CONTRACT_PATH, lock_run_id: str | None = None, lock_path: str | None = None) -> dict:
    """One (scenario, year) episode. Module-level so run_parallel can run it one-process-per-job."""
    if lock_run_id is not None:
        from o3.utils import weather_roles
        weather_roles.assume_test_lock(lock_run_id, lock_path)
    pol = _policy_from_spec(policy_spec)
    env_kw = policy_spec[3] if len(policy_spec) > 3 else {}      # Gen-2 runs: their own obs_version / control_step_min
    inner = PCMEnv(contract_path, regime_id=regime_id, year=year,
                   demand_mode=scenario.get("demand_mode", "deterministic"),
                   demand_volume_multiplier=scenario.get("demand_scale", 1.0), **env_kw)
    env = NoisyPCMEnv(inner, ghi_noise_pct=scenario.get("ghi_noise_pct", 0.0), tamb_noise_C=scenario.get("tamb_noise_C", 0.0),
                      demand_scale=scenario.get("demand_scale", 1.0))
    pcm_init = scenario.get("pcm_init", "solid")
    pol.reset()
    if hasattr(pol, "attach_env"):
        pol.attach_env(inner)
    ep = run_robustness_episode(lambda o: pol.act(o), env, year, pcm_init)
    return {"policy": policy_name, "scenario": scenario["name"], "regime_id": regime_id, "year": year,
            "pcm_init": pcm_init, **ep}


def run_robustness_suite(policy_fn=None, policy_name: str = "policy", regime_id: int = 0, years: list[int] | None = None,
                         scenarios: list[dict] | None = None, policy_spec: tuple | None = None, n_jobs: int = 1,
                         verbose: bool = True) -> list[dict]:
    """Run the policy across all perturbation scenarios and years.

    policy_fn   : callable(obs) -> int (serial only)
    policy_spec : ('baseline', 'b1') or ('model', algo, path[, env_kwargs]) — picklable, enables n_jobs > 1
    n_jobs      : worker processes, one (scenario, year) episode per job. n_jobs=1 -> serial.
    """
    years = years or [2021]
    scenarios = scenarios or _default_scenarios()
    if policy_spec is not None:
        from o3.utils.parallel import run_parallel
        jobs = [dict(policy_spec=policy_spec, policy_name=policy_name, regime_id=regime_id, year=y, scenario=sc)
                for sc in scenarios for y in years]
        results = run_parallel(robustness_job, jobs, n_jobs=n_jobs)
    else:
        assert policy_fn is not None, "give policy_fn or policy_spec"
        results = []
        for sc in scenarios:
            inner = PCMEnv(CONTRACT_PATH, regime_id=regime_id, demand_mode=sc.get("demand_mode", "deterministic"),
                           demand_volume_multiplier=sc.get("demand_scale", 1.0))
            env = NoisyPCMEnv(inner, ghi_noise_pct=sc.get("ghi_noise_pct", 0.0), tamb_noise_C=sc.get("tamb_noise_C", 0.0),
                              demand_scale=sc.get("demand_scale", 1.0))
            for y in years:
                ep = run_robustness_episode(policy_fn, env, y, sc.get("pcm_init", "solid"))
                results.append({"policy": policy_name, "scenario": sc["name"], "regime_id": regime_id, "year": y,
                                "pcm_init": sc.get("pcm_init", "solid"), **ep})
    if verbose:
        for r in results:
            print(f"  {r['policy']:15s} {r['scenario']:20s} y={r['year']} init={r['pcm_init']:7s} "
                  f"del={r['total_delivered_kWh']:.1f} kWh  safety={r['n_safety_violations']}")
    return results


def _default_scenarios() -> list[dict]:
    return [
        {"name": "nominal",         "ghi_noise_pct": 0,  "tamb_noise_C": 0,  "pcm_init": "solid"},
        {"name": "init_liquid",     "ghi_noise_pct": 0,  "tamb_noise_C": 0,  "pcm_init": "liquid"},
        {"name": "init_partial",    "ghi_noise_pct": 0,  "tamb_noise_C": 0,  "pcm_init": "partial"},
        {"name": "ghi_noise_5pct",  "ghi_noise_pct": 5,  "tamb_noise_C": 0,  "pcm_init": "solid"},
        {"name": "ghi_noise_10pct", "ghi_noise_pct": 10, "tamb_noise_C": 0,  "pcm_init": "solid"},
        {"name": "ghi_noise_20pct", "ghi_noise_pct": 20, "tamb_noise_C": 0,  "pcm_init": "solid"},
        {"name": "tamb_plus2C",     "ghi_noise_pct": 0,  "tamb_noise_C": 2,  "pcm_init": "solid"},
        {"name": "tamb_minus2C",    "ghi_noise_pct": 0,  "tamb_noise_C": -2, "pcm_init": "solid"},
        {"name": "combined_stress", "ghi_noise_pct": 10, "tamb_noise_C": 2,  "pcm_init": "liquid"},
        {"name": "worst_case",      "ghi_noise_pct": 20, "tamb_noise_C": -2, "pcm_init": "partial"},
        # added for Rajasthan (plan M5: stochastic demand and reset scenarios on validation years)
        {"name": "demand_x0p7",     "demand_scale": 0.7, "pcm_init": "solid"},
        {"name": "demand_x1p3",     "demand_scale": 1.3, "pcm_init": "solid"},
        {"name": "demand_stochastic", "demand_mode": "stochastic", "pcm_init": "solid"},
    ]


def save_robustness_results(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Robustness results saved -> {path}")


def print_robustness_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'Policy':15s} {'Scenario':20s} {'Year':5s} {'Init':7s} {'Del kWh':8s} {'Unmet kWh':10s} "
          f"{'Shield_w':9s} {'Shield_p':9s} {'Safety':7s}")
    print("-" * 100)
    for r in results:
        print(f"{r['policy']:15s} {r['scenario']:20s} {r['year']:5d} {r['pcm_init']:7s} "
              f"{r['total_delivered_kWh']:8.1f} {r['total_unmet_kWh']:10.1f} "
              f"{r['total_shield_water']:9d} {r['total_shield_pcm']:9d} {r['n_safety_violations']:7d}")
    print("=" * 100)
    violations = [r for r in results if r["n_safety_violations"] > 0]
    if violations:
        print(f"\n[WARN] {len(violations)} scenario(s) have safety violations:")
        for r in violations:
            print(f"  {r['policy']} | {r['scenario']} | y={r['year']} | {r['n_safety_violations']} violations")
    else:
        print("\n[OK]  Zero hard-limit violations across all robustness scenarios.")
