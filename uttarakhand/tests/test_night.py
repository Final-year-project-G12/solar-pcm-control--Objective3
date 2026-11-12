"""O2's night serve scenario computed by O3 (o3/night.py) equals O2's own run_case night scenarios."""
import pytest

from o3 import o2_bridge as ob
from o3 import roles_o3
from o3.controllers import B0AlwaysOn
from o3.evaluate import episode_metrics, run_episode

RTOL = 1e-9


@pytest.mark.parametrize("water_only", [False, True])
def test_night_serve_equals_o2(water_only):
    w = roles_o3.load_weather(1, 2016, "train")
    ref = ob.run_case_for(1, w, "fully_solid", water_only=water_only, night_day_stride=1)["tes"]
    df, _ = run_episode(B0AlwaysOn(), 1, 2016, "fully_solid", "train", water_only=water_only, night=True)
    m = episode_metrics(df)
    assert m["tau_serve_o2_mean_h"] == pytest.approx(ref["tau_serve_mean_h"], rel=RTOL)
    assert m["tau_serve_o2_p10_h"] == pytest.approx(ref["tau_serve_p10_h"], rel=RTOL, abs=1e-12)
    assert m["serve_frac_o2_mean"] == pytest.approx(ref["serve_frac_45_mean"], rel=RTOL)
    assert m["night_n_days"] == ref["night_n_days"]
