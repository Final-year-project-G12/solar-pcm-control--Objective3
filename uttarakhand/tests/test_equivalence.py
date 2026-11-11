"""Plan 15.5, tests T1-T4: hourly stepping of the O2 kernel reproduces O2's own full-year run."""
import numpy as np
import pytest

from o3 import o2_bridge as ob
from o3 import roles_o3
from o3.stepper import BYPASS, ENABLE, HourlyStepper

REGIME, YEAR = 1, 2016
COLS = [("T_w_C", "T_top"), ("T_w_mean_C", "T_mean"), ("T_pcm_C", "T_pcm"), ("f_melt", "f_melt"),
        ("T_bottom_C", "T_bottom"), ("Edel_target_Wh", "Edel_Wh"), ("Q_collector_Wh", "Q_coll_Wh"),
        ("E_stored_kWh", "E_stored_kWh"), ("E_rec45_kWh", "E_rec_kWh"), ("draw_kg", "draw_kg")]
RTOL = 1e-9          # declared acceptance; the target is exact


@pytest.fixture(scope="module")
def weather():
    return roles_o3.load_weather(REGIME, YEAR, "train")


def _assert_same(hourly, rows):
    for ref_col, key in COLS:
        a = hourly[ref_col].to_numpy()
        b = np.array([r[key] for r in rows])
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=1e-12, err_msg=ref_col)


@pytest.mark.parametrize("scenario", ["fully_solid", "fully_liquid"])
def test_T1_full_year_equivalence(weather, scenario):
    ref = ob.run_case_for(REGIME, weather, scenario)
    st = HourlyStepper(ob.build_kernel_case(REGIME, weather.iloc[:1], scenario), weather)
    rows = [st.step(ENABLE) for _ in range(len(weather))]
    _assert_same(ref["hourly"], rows)
    m = ref["metrics"]
    assert sum(r["shield_water_s"] for r in rows) / 3600 == pytest.approx(m["shield_water_h"], rel=RTOL)
    assert sum(r["shield_pcm_s"] for r in rows) / 3600 == pytest.approx(m["shield_pcm_h"], rel=RTOL)
    assert sum(r["over_limit_s"] for r in rows) / 3600 == pytest.approx(m["hours_over_limit"], abs=1e-12)
    assert np.array_equal(st.T, ref["hourly"].attrs["node_T"][-1])
    assert np.array_equal(st.h, ref["hourly"].attrs["node_h"][-1])


def test_T1_water_only_twin(weather):
    ref = ob.run_case_for(REGIME, weather, water_only=True)
    st = HourlyStepper(ob.build_kernel_case(REGIME, weather.iloc[:1], water_only=True), weather)
    rows = [st.step(ENABLE) for _ in range(len(weather))]
    _assert_same(ref["hourly"], rows)


def test_T2_split_run(weather):
    full = ob.run_case_for(REGIME, weather)["hourly"]
    half = len(weather) // 2
    a = ob.run_case_for(REGIME, weather.iloc[:half].reset_index(drop=True))["hourly"]
    from src.simulation.run_case import run_case
    from src.design.space import geometry_bounds, to_design
    d = ob.chosen_design(REGIME)
    b = run_case(ob.STATE, REGIME, d["pcm"], to_design(d["R_in_m"], d["fraction"], d["flow"], d["arrangement"]),
                 design_bounds=geometry_bounds(), weather=weather.iloc[half:].reset_index(drop=True),
                 initial_state={"T_nodes": a.attrs["node_T"][-1], "h_nodes": a.attrs["node_h"][-1]},
                 tank_model="multinode", n_nodes=10, stack_anchor="center")["hourly"]
    for col, _ in COLS:
        joined = np.concatenate([a[col].to_numpy(), b[col].to_numpy()])
        np.testing.assert_allclose(joined, full[col].to_numpy(), rtol=RTOL, atol=1e-12, err_msg=col)


def test_T3_bypass_mask(weather):
    rng = np.random.default_rng(3)
    sched = rng.random(len(weather)) < 0.3
    masked = weather.copy()
    masked.loc[sched, "GHI_Wm2"] = 0.0
    ref = ob.run_case_for(REGIME, masked)["hourly"]
    st = HourlyStepper(ob.build_kernel_case(REGIME, weather.iloc[:1]), weather)
    rows = [st.step(BYPASS if s else ENABLE) for s in sched]
    _assert_same(ref, rows)
    # the bypass really changed the year (not a vacuous test)
    assert sum(r["Q_coll_Wh"] for r in rows) < 0.95 * ob.run_case_for(REGIME, weather)["hourly"]["Q_collector_Wh"].sum()


def test_T4_energy_balance(weather):
    st = HourlyStepper(ob.build_kernel_case(REGIME, weather.iloc[:1]), weather)
    rng = np.random.default_rng(4)
    for _ in range(len(weather)):
        st.step(int(rng.integers(2)))
    assert st.residual_pct() < 0.1          # O2 Gate 1 threshold
