"""M7 demo back end: replay reproduces the evaluator, role guards hold, live parsing / fallback work offline."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from o3 import live
from o3.controllers import B0AlwaysOn, B1Reactive
from o3.evaluate import episode_metrics, run_episode


def test_replay_equals_evaluator():
    """Same code path: FrameEnv on a validation year == the evaluator's own episode, bit for bit."""
    fr = live.replay_frame(1, 2021)
    runs = live.simulate(1, fr, [B0AlwaysOn(), B1Reactive(64, 0.99, 55)])
    for c in (B0AlwaysOn(), B1Reactive(64, 0.99, 55)):
        ref, _ = run_episode(c, 1, 2021, "fully_solid", "validate")
        a, b = episode_metrics(ref), episode_metrics(runs[c.name])
        for k in ("E_del_window_kWh", "overheat_h", "hours_over_limit", "bypass_switches"):
            assert a[k] == b[k], k
    tw, _ = run_episode(B0AlwaysOn(), 1, 2021, "fully_solid", "validate", water_only=True)
    assert episode_metrics(tw)["E_del_window_kWh"] == episode_metrics(runs["water_only_twin"])["E_del_window_kWh"]


@pytest.mark.parametrize("year", [2008, 2024])
def test_replay_refuses_test_and_o2_years(year):
    with pytest.raises(PermissionError):
        live.replay_frame(1, year)


class _Resp:
    status_code = 200

    def __init__(self, js):
        self._js = js

    def raise_for_status(self):
        pass

    def json(self):
        return self._js


class _Sess:
    def __init__(self, js=None, exc=None):
        self.js, self.exc = js, exc

    def get(self, url, params=None, timeout=None):
        if self.exc:
            raise self.exc
        return _Resp(self.js)


def _payload(n=30):
    t = pd.date_range("2026-11-10 01:00", periods=n, freq="h", tz="UTC")       # hour-END stamps
    ghi = [float(max(0, 600 * np.sin((i % 24 - 1) / 24 * 2 * np.pi))) for i in range(n)]
    ghi[5] = None
    return {"latitude": 29.6, "longitude": 79.4, "elevation": 1350, "utc_offset_seconds": 0,
            "hourly": {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in t], "shortwave_radiation": ghi,
                       "temperature_2m": [12.0 + 0.1 * i for i in range(n)]}}


def test_fetch_live_parses_and_fills():
    now = datetime(2026, 11, 11, 4, 20, tzinfo=timezone.utc)                    # last completed hour ends 04:00
    fr, rec = live.fetch_live(1, now_utc=now, session=_Sess(_payload()))
    assert rec["ok"] and rec["raw_json"]["hourly"]["shortwave_radiation"][5] is None   # raw kept unmodified
    assert fr.timestamp_utc.iloc[0] == pd.Timestamp("2026-11-10 00:00", tz="UTC")   # shifted to hour START
    assert fr.timestamp_utc.iloc[-1] == pd.Timestamp("2026-11-11 03:00", tz="UTC")  # no future hours
    assert fr.missing.sum() == 1 and fr.GHI_Wm2.iloc[5] == fr.GHI_Wm2.iloc[4]        # persistence fill
    assert fr.local_hour.iloc[0] == 5 and fr.local_date.iloc[0] == "2026-11-10"       # O2's IST convention
    runs = live.simulate(1, fr, [B0AlwaysOn()])
    assert len(runs["B0"]) == len(fr)


def test_fetch_failure_and_stale_fallback():
    fr, rec = live.fetch_live(1, session=_Sess(exc=ConnectionError("offline")))
    assert not rec["ok"] and fr.empty
    now = datetime(2026, 11, 11, 4, 20, tzinfo=timezone.utc)
    good, _ = live.fetch_live(1, now_utc=now, session=_Sess(_payload()))
    later = datetime(2026, 11, 11, 9, 5, tzinfo=timezone.utc)
    ext, n = live.extend_stale(good, later)
    assert n == 5 and ext.stale_fill.iloc[-n:].all() and not ext.stale_fill.iloc[:-n].any()
    assert (ext.GHI_Wm2.iloc[-n:] == good.GHI_Wm2.iloc[-1]).all()


def test_saved_log_reruns_exactly(tmp_path):
    now = datetime(2026, 11, 11, 4, 20, tzinfo=timezone.utc)
    fr, _ = live.fetch_live(1, now_utc=now, session=_Sess(_payload()))
    p = tmp_path / "w.csv"
    fr.to_csv(p, index=False)
    a = live.simulate(1, fr, [B0AlwaysOn()])["B0"]
    b = live.simulate(1, live.load_weather_log(p), [B0AlwaysOn()])["B0"]
    np.testing.assert_array_equal(a.T_top.to_numpy(), b.T_top.to_numpy())
