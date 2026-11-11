"""
Phase O3-7 — Live weather adapter (software-in-the-loop demo), Rajasthan.

Connects archived / live / stub weather to the frozen simulator and shows the policy's decisions.

    REPLAY  — recorded hourly weather from an ALLOWED role (train / validate; the test years only
              after the O3 lock is held). Reproducible; used for research.
    LIVE    — live weather from Open-Meteo (free, no key). Demo only, no hardware. Needs internet.
    STUB    — deterministic synthetic day (no network; tests / CI).

Every screen / log line carries the label: "Software-in-the-loop simulation. Not field validation."
All thermal state (T_water, T_pcm, f_melt) comes from the simulator, not from sensors.

Differences from the TN adapter (docs/CORRECTIONS.md): TN's runner created its environment but never
fed the weather frame into it (every step re-reset the env). Here the frame IS the hour's weather:
PCMEnv.set_weather_override() injects (GHI, T_amb, hour) and the same env.step() runs the O2 kernel,
so replay of a stored year through the demo reproduces the evaluation numbers (tests/test_demo).
Location -> regime mapping: the user picks one of the three O2 medoid points (plan Q6 option);
no automatic mapping exists.
"""

import csv
import datetime
import json
import pathlib
import time
from enum import Enum

import numpy as np

import config_O2
from o3.environment.pcm_env import PCMEnv, CHARGE, IDLE, BYPASS
from o3.evaluation.rollout import ModelPolicy, ALGO_KIND, load_model
from o3.utils import weather_roles

REPO_ROOT = config_O2.O3_ROOT
CONTRACT_PATH = config_O2.CONTRACT_FILE
BANNER = "Software-in-the-loop simulation. Not field validation."

# O2 medoid points (lat, lon) from the NASA POWER files; the user chooses one of the three.
MEDOIDS = {
    0: {"point": "RJP_0132", "place": "Udaipur", "lat": 24.375, "lon": 74.125},
    1: {"point": "RJP_0192", "place": "Nagaur", "lat": 26.875, "lon": 73.375},
    2: {"point": "RJP_0083", "place": "Jaipur", "lat": 26.625, "lon": 75.875},
}
STALE_AFTER_S = 3 * 3600          # live data older than this -> fall back to persistence


class DemoMode(str, Enum):
    REPLAY = "replay"
    LIVE = "live"
    STUB = "stub"


class WeatherFrame:
    """One hour of weather for the simulator."""
    __slots__ = ["timestamp", "ghi_wm2", "t_amb_C", "local_hour", "source", "quality_ok", "latency_s", "note"]

    def __init__(self, timestamp, ghi, t_amb, local_hour, source="unknown", quality_ok=True, latency_s=0.0, note=""):
        self.timestamp = timestamp
        self.ghi_wm2 = max(0.0, float(ghi))
        self.t_amb_C = float(t_amb)
        self.local_hour = int(local_hour) % 24
        self.source = source
        self.quality_ok = quality_ok
        self.latency_s = latency_s
        self.note = note


# ---------------------------------------------------------------------------
# Weather sources
# ---------------------------------------------------------------------------

def _stub_frame(step: int) -> WeatherFrame:
    """Deterministic synthetic day (sunrise ~6 h, peak 12 h, sunset ~18 h)."""
    hour = step % 24
    ghi = max(0.0, 850.0 * np.sin((hour - 6) / 12.0 * np.pi)) if 6 <= hour <= 18 else 0.0
    t_amb = 29.0 + 7.0 * np.sin((hour - 9) / 24.0 * 2 * np.pi)
    return WeatherFrame(datetime.datetime(2024, 6, 1) + datetime.timedelta(hours=step), ghi, t_amb, hour, "stub")


def fetch_live_frame(regime_id: int, last: WeatherFrame | None = None) -> WeatherFrame:
    """Current-hour weather from Open-Meteo at the chosen medoid; provider, latency and errors are logged.
    The current hour's shortwave radiation is a nowcast value (plan 3.2: the stored-year GHI is an hourly
    mean, a mild look-ahead; live mode uses the provider's value for the hour that has just started)."""
    m = MEDOIDS[regime_id]
    t0 = time.time()
    try:
        import urllib.request
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={m['lat']}&longitude={m['lon']}"
               f"&hourly=shortwave_radiation,temperature_2m&forecast_days=1&timezone=Asia%2FKolkata")
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        now = datetime.datetime.now()
        h = now.hour
        ghi, tamb = data["hourly"]["shortwave_radiation"][h], data["hourly"]["temperature_2m"][h]
        if ghi is None or tamb is None:
            raise ValueError("missing value in provider response")
        return WeatherFrame(now, ghi, tamb, h, "open-meteo", True, time.time() - t0, f"{m['place']} {m['point']}")
    except Exception as e:      # noqa: BLE001
        if last is not None and (datetime.datetime.now() - last.timestamp).total_seconds() < STALE_AFTER_S:
            return WeatherFrame(datetime.datetime.now(), last.ghi_wm2, last.t_amb_C, datetime.datetime.now().hour,
                                "persistence", False, time.time() - t0, f"live fetch failed ({e}); persistence")
        f = _stub_frame(datetime.datetime.now().hour)
        f.quality_ok, f.note = False, f"live fetch failed ({e}); stub fallback"
        return f


def replay_frames(regime_id: int, year: int) -> list[WeatherFrame]:
    """Recorded hourly weather for an allowed year (weather_roles refuses test years without the lock)."""
    w = weather_roles.load_weather(regime_id, year)
    t0 = datetime.datetime(year, 1, 1)
    return [WeatherFrame(t0 + datetime.timedelta(hours=i), w["GHI_Wm2"][i], w["T_amb_C"][i], w["local_hour"][i],
                         f"replay:{year}") for i in range(len(w["GHI_Wm2"]))]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

class LiveDemoRunner:
    """Drive the simulator step-by-step with live, replay or stub weather and a policy.

    algo : ppo | dqn | recurrent | qrdqn | maskable | sac | b0 | b1 | b2   (b* = rule baselines; no model needed)
    """

    ACTION_NAMES = {CHARGE: "CHARGE", IDLE: "IDLE", BYPASS: "BYPASS"}

    def __init__(self, algo: str = "recurrent", regime_id: int = 1, mode: DemoMode = DemoMode.STUB,
                 replay_year: int = 2021, log_dir: pathlib.Path | None = None, model_path=None):
        self.algo, self.regime_id, self.mode, self.replay_year = algo, regime_id, DemoMode(mode), replay_year
        self.log_dir = pathlib.Path(log_dir) if log_dir else (config_O2.RESULTS_DIR / "demo_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._policy = self._load_policy(model_path)
        self._env = PCMEnv(CONTRACT_PATH, regime_id=regime_id, year=weather_roles.VAL_YEARS[0])
        self._frames = replay_frames(regime_id, replay_year) if self.mode == DemoMode.REPLAY else []
        self._step = 0
        self._last_frame: WeatherFrame | None = None
        self._log_rows: list[dict] = []
        self._obs = None

    def _load_policy(self, model_path):
        if self.algo in ("b0", "b1", "b2"):
            from o3.evaluation.rollout import make_baseline
            return make_baseline(self.algo)
        path = model_path or self._find_model()
        if path is None:
            print("[WARN] No trained model found; using B1 reactive fallback.")
            from o3.baselines.b1_reactive import ReactiveRulePolicy
            return ReactiveRulePolicy()
        return ModelPolicy(load_model(self.algo, path), ALGO_KIND[self.algo])

    def _find_model(self):
        from o3.evaluation.rollout import ALGO_DIR
        d = config_O2.RESULTS_DIR / ALGO_DIR[self.algo] / f"r{self.regime_id}"
        for name in ("best_model/best_model.zip", f"{ALGO_DIR[self.algo]}_final.zip"):
            if (d / name).exists():
                return d / name
        return None

    def reset(self) -> np.ndarray:
        self._env.reset(year=weather_roles.VAL_YEARS[0])
        self._policy.reset()
        if hasattr(self._policy, "attach_env"):
            self._policy.attach_env(self._env)
        self._step = 0
        self._last_frame = None
        return self._env.set_weather_override(0.0, 25.0, 0)

    def _get_weather(self) -> WeatherFrame:
        if self.mode == DemoMode.LIVE:
            self._last_frame = fetch_live_frame(self.regime_id, self._last_frame)
            return self._last_frame
        if self.mode == DemoMode.REPLAY:
            return self._frames[self._step % len(self._frames)]
        return _stub_frame(self._step)

    def step_once(self) -> dict:
        wx = self._get_weather()
        obs = self._env.set_weather_override(wx.ghi_wm2, wx.t_amb_C, wx.local_hour)
        action = self._policy.act(obs)
        _, reward, _, _, info = self._env.step(action)
        self._step += 1
        frame = {
            "banner": BANNER, "timestamp": wx.timestamp.isoformat(), "mode": self.mode.value, "source": wx.source,
            "quality_ok": wx.quality_ok, "latency_s": round(wx.latency_s, 3), "note": wx.note, "step": self._step,
            "ghi_wm2": wx.ghi_wm2, "t_amb_C": wx.t_amb_C, "local_hour": wx.local_hour,
            "action": self.ACTION_NAMES[action], "reward": reward,
            "E_del45_Wh": info["E_del45_Wh"], "E_unmet_Wh": info["E_unmet_Wh"], "shield_override": info["shield_override"],
            "safety_violation": info["n_safety_violations"], "T_top_C": info["T_top_C"], "T_bottom_C": info["T_bottom_C"],
            "T_pcm_C": info["T_pcm_C"], "f_melt": info["f_melt"], "state_label": "simulator state (not measured)",
        }
        self._log_rows.append(frame)
        return frame

    def run(self, n_steps: int = 24, interval_s: float = 0.0, verbose: bool = True) -> list[dict]:
        """Run n_steps demo steps (default = one day = 24 h)."""
        self.reset()
        frames = []
        if verbose:
            print(f"*** {BANNER} ***")
        for _ in range(n_steps):
            fr = self.step_once()
            frames.append(fr)
            if verbose:
                sv = "SAFETY!" if fr["safety_violation"] else ""
                sh = "->SHIELD" if fr["shield_override"] else ""
                print(f"  h{fr['step']:4d} | GHI={fr['ghi_wm2']:6.0f} W/m2 | {fr['action']:6s} {sh:8s} "
                      f"| E45={fr['E_del45_Wh']:6.0f} Wh | Ttop={fr['T_top_C']:5.1f} | reward={fr['reward']:8.2f} {sv}")
            if interval_s > 0:
                time.sleep(interval_s)
        self._save_log()
        return frames

    def _save_log(self) -> None:
        if not self._log_rows:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.log_dir / f"demo_log_{self.algo}_r{self.regime_id}_{self.mode.value}_{ts}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._log_rows[0].keys()))
            w.writeheader()
            w.writerows(self._log_rows)
        print(f"\n[INFO] Demo log saved -> {out}")
