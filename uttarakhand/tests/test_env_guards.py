"""Plan 15.5, tests T5-T8 + O2 snapshot integrity + year-role guards."""
import json
import shutil

import numpy as np
import pytest

from o3 import o2_bridge as ob
from o3 import roles_o3
from o3.stepper import ENABLE, HourlyStepper


def _env(**kw):
    from o3.env import UKPCMEnv
    return UKPCMEnv(regime=1, purpose="train", **kw)


def test_T5_determinism():
    def run(seed):
        env = _env(demand_mode="stochastic", w2=0.3, w4=0.3)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = [obs]
        for _ in range(500):
            obs, r, *_ = env.step(int(rng.integers(2)))
            out.append(np.append(obs, r))
        return np.concatenate(out), env.year, env.scenario
    a, b, c = run(7), run(7), run(8)
    assert a[1:] == b[1:] and np.array_equal(a[0], b[0])
    assert not (np.array_equal(a[0], c[0]) and a[1:] == c[1:])


def test_T6_shield_cannot_be_bypassed():
    w = roles_o3.load_weather(1, 2016, "train")
    case = ob.build_kernel_case(1, w.iloc[:1])
    st = HourlyStepper(case, w)
    noon = int(np.argmax(w["GHI_Wm2"].to_numpy()[:48]))
    st.t = noon
    st.T = np.full_like(case.T0, 74.9)                 # top node well above the 72 C guard band all hour
    rec = st.step(ENABLE)                              # policy asks for the loop to run
    assert rec["loop_available"]
    assert rec["T_top_min"] >= 72.0
    assert rec["shield_water_s"] == pytest.approx(3600.0)
    assert rec["Q_coll_Wh"] < 1e-9                     # the kernel kept the loop off, whatever the policy said


def test_T7_contract_hash_mismatch(tmp_path, monkeypatch):
    bad = tmp_path / "contract.json"
    c = json.loads(ob.CONTRACT_PATH.read_text())
    c["global_limits"]["max_pcm_temp_C"] = 70.0
    bad.write_text(json.dumps(c))
    shutil.copy(ob.CONTRACT_SHA_PATH, tmp_path / "contract.json.sha256")
    monkeypatch.setattr(ob, "CONTRACT_PATH", bad)
    monkeypatch.setattr(ob, "CONTRACT_SHA_PATH", tmp_path / "contract.json.sha256")
    ob.load_contract.cache_clear()
    try:
        with pytest.raises(ob.ContractError):
            ob.verify_contract()
    finally:
        ob.load_contract.cache_clear()


def test_T7_year_roles():
    with pytest.raises(roles_o3.O3LockError):
        roles_o3.load_weather(1, 2008, "test")         # test year without the O3 lock
    with pytest.raises(PermissionError):
        roles_o3.load_weather(1, 2024, "test")         # O2's used held-out year
    with pytest.raises(PermissionError):
        roles_o3.load_weather(1, 2021, "train")        # validation year asked for training
    assert roles_o3.years_for("test") == [2006, 2007, 2008, 2009, 2010]
    assert not set(roles_o3.years_for("test")) & set(range(2016, 2026))


def test_T8_check_env():
    from stable_baselines3.common.env_checker import check_env
    check_env(_env(w2=0.3, w4=0.3), warn=True)


def test_o2_snapshot_unchanged():
    snap = json.loads((ob.O2_ROOT / "O2_SNAPSHOT.json").read_text())
    for rel, h in snap["sha256"].items():
        assert ob.sha256_file(ob.O2_ROOT / rel) == h, rel


def test_attach_existing_lock_guards(tmp_path, monkeypatch):
    """Post-lock attach never creates a lock and refuses a different policy fingerprint."""
    monkeypatch.setattr(roles_o3, "TEST_LOCK_PATH", tmp_path / "o3_test.lock")
    monkeypatch.setattr(roles_o3, "_lock_held", False)
    with pytest.raises(roles_o3.O3LockError):
        roles_o3.attach_existing_lock("abc")                     # no lock yet
    assert not (tmp_path / "o3_test.lock").exists()
    (tmp_path / "o3_test.lock").write_text(json.dumps({"run_id": "x", "fingerprint": "abc"}))
    with pytest.raises(roles_o3.O3LockError):
        roles_o3.attach_existing_lock("different")               # re-trained / other policy set
    assert roles_o3._lock_held is False
    assert roles_o3.attach_existing_lock("abc")["run_id"] == "x"
    assert roles_o3._lock_held is True
