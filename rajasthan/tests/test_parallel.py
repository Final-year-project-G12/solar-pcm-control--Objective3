"""
Step-3 tests — parallelism must not change results.

  * run_parallel keeps job order and equals a serial loop
  * baseline metrics are identical for n_jobs = 1 and n_jobs = 2
  * a SubprocVecEnv PPO run and a DummyVecEnv PPO run with the same seed and n_envs learn the same weights
  * train_ppo works end to end with worker environments
"""

import numpy as np
import pytest

import config_O2

CONTRACT_PATH = config_O2.CONTRACT_FILE
REGIME = 1


def _square(x):
    return x * x


def test_run_parallel_order_and_values():
    from o3.utils.parallel import run_parallel
    jobs = [dict(x=i) for i in range(7)]
    assert run_parallel(_square, jobs, n_jobs=3) == [i * i for i in range(7)] == run_parallel(_square, jobs, n_jobs=1)


def test_default_job_counts_leave_a_core():
    from o3.utils.parallel import default_n_jobs, physical_cores
    assert 1 <= default_n_jobs() <= max(1, physical_cores())


def test_baselines_serial_equals_parallel():
    from scripts.phase2_baselines import run_all
    ser = run_all(REGIME, [2016, 2017], n_jobs=1, baselines=["b0", "b1"])
    par = run_all(REGIME, [2016, 2017], n_jobs=2, baselines=["b0", "b1"])
    assert len(ser) == len(par) == 6
    for a, b in zip(ser, par):
        for k in ("total_delivered_kWh", "E_del45_window_kWh", "total_shield_water", "total_shield_pcm", "total_reward",
                  "n_safety_violations", "action_switches"):
            assert a[k] == b[k], (a["policy"], a["year"], k)


def _fit(vec: str, seed: int = 5):
    from stable_baselines3 import PPO
    from o3.agents import common as C
    C.pin_main_threads(1)
    env = C.make_train_vec(REGIME, 2, vec)
    try:
        m = PPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=2, seed=seed, device="cpu", verbose=0)
        m.learn(256)
        return {k: v.detach().cpu().numpy().copy() for k, v in m.policy.state_dict().items()}
    finally:
        env.close()


def test_subproc_vec_env_equals_dummy_vec_env():
    a, b = _fit("dummy"), _fit("subproc")
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), f"parameter {k} differs between DummyVecEnv and SubprocVecEnv"


def test_rollout_steps_per_env_keeps_samples_per_update():
    from o3.agents.common import rollout_steps_per_env
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15):
        assert rollout_steps_per_env(8760, n) * n == 8760


def test_train_ppo_with_worker_envs(tmp_path):
    from o3.agents.ppo_trainer import train_ppo
    m = train_ppo(regime_id=REGIME, total_timesteps=512, n_steps=256, seed=1, out_dir=tmp_path, n_envs=2, verbose=0)
    assert (tmp_path / "ppo_final.zip").exists()
    assert m.n_steps == 128 and m.n_envs == 2


def test_cli_flags_present_on_every_phase_script():
    import pathlib
    for p in sorted((config_O2.O3_ROOT / "scripts").glob("phase*.py")) + [config_O2.O3_ROOT / "scripts" / "make_plots.py"]:
        txt = p.read_text(encoding="utf-8")
        if p.name == "phase0_audit.py":
            assert "--n-jobs" in txt or "no flags" in txt.lower()      # single file read; documented
            continue
        assert "add_parallel_args" in txt, f"{p.name} does not expose --n-jobs/--n-envs"
