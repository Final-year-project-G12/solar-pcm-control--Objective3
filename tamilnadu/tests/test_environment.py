"""
Phase O3-1 tests — PCMEnv step interface.

Critical gate test: running the environment with the fixed-charge policy
(B0) must produce annual delivered energy within 5% of run_case() / run_year()
under the same weather year and initial conditions.

Run with: pytest tests/test_environment.py -v
"""

import pathlib
import numpy as np
import pytest

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"


@pytest.fixture(scope="module")
def env():
    from o3.environment.pcm_env import PCMEnv
    return PCMEnv(CONTRACT_PATH, regime_id=0, year=2016)


def test_reset_returns_obs_shape(env):
    obs, info = env.reset(year=2016)
    assert obs.shape == (11,), f"Expected obs dim 11, got {obs.shape}"
    assert obs.dtype == np.float32


def test_step_returns_valid_types(env):
    obs, _ = env.reset(year=2016)
    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "E_delivered_Wh" in info
    assert "shield_override" in info


def test_episode_runs_to_end(env):
    obs, _ = env.reset(year=2016)
    n_steps = 0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        n_steps += 1
    # Tamil Nadu year should be ~8760 hours (minus leap-day if dropped)
    assert 8700 <= n_steps <= 8760, f"Unexpected episode length {n_steps}"


def test_no_hard_safety_violations_b0(env):
    """B0 with the existing shield should produce zero hard-limit violations."""
    obs, _ = env.reset(year=2016)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
    assert env._state["n_safety_violations"] == 0, (
        f"B0 produced {env._state['n_safety_violations']} hard-limit violations. "
        "The safety shield must prevent all hard-limit crossings."
    )


def test_delivered_energy_positive(env):
    obs, _ = env.reset(year=2016)
    done = False
    total = 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        total += info["E_delivered_Wh"]
    assert total > 1000.0, f"Annual delivered energy {total:.0f} Wh seems too low"


def test_action_space_all_valid(env):
    """All three actions should run without error for at least one step."""
    from o3.environment.pcm_env import CHARGE, IDLE, BYPASS
    for action in [CHARGE, IDLE, BYPASS]:
        obs, _ = env.reset(year=2016)
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2 is not None


def test_obs_range_reasonable(env):
    """Observation values should be in a rough [-3, 3] range after normalisation."""
    obs, _ = env.reset(year=2016)
    done = False
    out_of_range = 0
    total = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        out_of_range += int(np.any(np.abs(obs) > 5.0))
        total += 1
    frac = out_of_range / total if total else 1.0
    # allow up to 2% out-of-range; most hours should be near 0
    assert frac < 0.02, f"{frac*100:.1f}% of observations had |x|>5"


def test_step_vs_runcase_energy_within_5pct():
    """
    Gate test: step-based total vs run_case() must agree within 5%.

    This is the Phase O3-1 exit gate from the implementation plan.
    If this fails, the step() physics has drifted from the O2 simulator.
    """
    pytest.importorskip("src.simulation.run_case",
                        reason="O2 simulator not importable — check sys.path")

    import sys, pathlib
    _O2 = pathlib.Path(__file__).resolve().parents[3].parent / "objective2_design_optimization"
    if str(_O2) not in sys.path:
        sys.path.insert(0, str(_O2))

    from src.simulation.run_case import run_case
    from src.design.schema import DesignVector
    from o3.utils.contract_loader import load_contract, get_regime_design

    c  = load_contract(CONTRACT_PATH)
    rd = get_regime_design(c, regime_id=0)

    design_vec = DesignVector(
        capsule_diameter_m  = rd.get("capsule_outer_diameter_mm", 81.6) / 1000.0,
        n_capsule           = rd["n_capsules"],
        flow_rate_kg_s      = rd["nominal_flow_kg_s"],
        capsule_arrangement = rd["arrangement"],
    )
    o2_result = run_case("tamilnadu", 0, rd["selected_pcm"], design_vec,
                         record_hourly=False)
    assert o2_result["valid"], f"run_case failed: {o2_result['reason']}"
    o2_kWh = o2_result["metrics"]["useful_energy_kWh"]

    from o3.environment.pcm_env import PCMEnv
    env_local = PCMEnv(CONTRACT_PATH, regime_id=0, year=2016)
    obs, _ = env_local.reset(year=2016)
    done = False
    total_Wh = 0.0
    while not done:
        obs, reward, terminated, truncated, info = env_local.step(0)
        done = terminated or truncated
        total_Wh += info["E_delivered_Wh"]

    o3_kWh = total_Wh / 1000.0
    pct_diff = abs(o3_kWh - o2_kWh) / max(o2_kWh, 1e-6) * 100.0
    assert pct_diff < 5.0, (
        f"Step env ({o3_kWh:.1f} kWh) differs from run_case ({o2_kWh:.1f} kWh) "
        f"by {pct_diff:.1f}% — exceeds 5% gate."
    )
