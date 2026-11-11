"""Plan section 6 / Q2 reward form: twin-normalised delivery, dominance-rule w3, swept w2 / w4."""
import json

import numpy as np
import pytest

import config_O2

REGIME = 1


@pytest.fixture(scope="module")
def norm():
    p = config_O2.CONTRACTS_DIR / "reward_normalisation_rajasthan.json"
    if not p.exists():
        pytest.skip("run `python scripts/phase2_baselines.py --reward-norm` first")
    return json.loads(p.read_text())["regimes"][str(REGIME)]


def test_w3_dominates_largest_one_hour_delivery(norm):
    assert norm["w3_dominance"] > norm["max_one_hour_delivery_Wh"] / norm["E_norm_Wh_per_hour"]


def test_q2_reward_matches_its_formula(norm):
    from o3.environment.pcm_env import PCMEnv, BYPASS
    env = PCMEnv(config_O2.CONTRACT_FILE, regime_id=REGIME, year=2016, reward_weights={"mode": "Q2", "w2": 0.5, "w4": 0.25})
    env.reset(year=2016)
    rng = np.random.default_rng(0)
    prev = None
    for _ in range(400):
        a = int(rng.integers(0, 3))
        _, r, _, _, info = env.step(a)
        sh_h = (info["shield_water_s"] + info["shield_pcm_s"]) / 3600.0
        into = int(a == BYPASS and prev != BYPASS)
        exp = info["E_del45_Wh"] / norm["E_norm_Wh_per_hour"] - 0.5 * sh_h - norm["w3_dominance"] * float(info["shield_override"]) - 0.25 * into
        assert r == pytest.approx(exp, abs=1e-9)
        prev = a


def test_twin_earns_about_one_per_hour_on_training_years(norm):
    from o3.evaluation.rollout import evaluate_twin
    r = evaluate_twin(REGIME, 2018)
    assert r["total_delivered_kWh"] * 1000 / 8760 / norm["E_norm_Wh_per_hour"] == pytest.approx(1.0, abs=0.05)


def test_default_reward_mode_is_unchanged():
    from o3.environment.pcm_env import PCMEnv
    assert PCMEnv(config_O2.CONTRACT_FILE, regime_id=REGIME, year=2016).reward_mode == "R2"
