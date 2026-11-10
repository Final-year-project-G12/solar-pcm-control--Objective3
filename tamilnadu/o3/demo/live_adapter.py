"""
Phase O3-7 — Live weather adapter (software-in-the-loop demo).

Connects archived/live weather to the frozen simulator and shows the
trained policy's decisions in real-time.

Three modes:
  REPLAY  — historical CSV weather (reproducible, used for research)
  LIVE    — live weather from Open-Meteo API (demo only, no hardware)
  STUB    — deterministic stub (for testing without network)

The demo does NOT claim hardware validation. All thermal state is
simulated. This is a software-in-the-loop demonstration only.
"""

import pathlib
import sys
import time
import datetime
import json
import numpy as np
from enum import Enum

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_O2_ROOT  = REPO_ROOT.parent / "objective2_design_optimization"
for _p in [str(_O2_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTRACT_PATH = REPO_ROOT / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"

# Tamil Nadu representative coordinates (Madurai — regime 0 medoid)
TN_LAT  = 9.93
TN_LON  = 78.12


class DemoMode(str, Enum):
    REPLAY = "replay"
    LIVE   = "live"
    STUB   = "stub"


class WeatherFrame:
    """One hour of weather for the simulator."""
    __slots__ = ["timestamp", "ghi_wm2", "t_amb_C", "t_mains_C",
                 "demand_L", "source", "quality_ok"]

    def __init__(self, timestamp, ghi, t_amb, t_mains=20.0,
                 demand=20.0, source="unknown", quality_ok=True):
        self.timestamp  = timestamp
        self.ghi_wm2    = max(0.0, float(ghi))
        self.t_amb_C    = float(t_amb)
        self.t_mains_C  = float(t_mains)
        self.demand_L   = float(demand)
        self.source     = source
        self.quality_ok = quality_ok


# ---------------------------------------------------------------------------
# Weather sources
# ---------------------------------------------------------------------------

def _stub_frame(hour: int) -> WeatherFrame:
    """Deterministic stub for unit tests / CI."""
    angle = (hour % 24) / 24 * 2 * np.pi
    ghi   = max(0.0, 800 * np.sin(angle - np.pi / 2 + np.pi / 2))
    t_amb = 28.0 + 5.0 * np.sin(angle)
    return WeatherFrame(
        timestamp  = datetime.datetime(2024, 6, 1, hour % 24),
        ghi        = ghi,
        t_amb      = t_amb,
        source     = "stub",
        quality_ok = True,
    )


def _fetch_live_frame() -> WeatherFrame:
    """Fetch current-hour weather from Open-Meteo (free, no API key needed)."""
    try:
        import urllib.request
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={TN_LAT}&longitude={TN_LON}"
            f"&hourly=shortwave_radiation,temperature_2m"
            f"&forecast_days=1&timezone=Asia/Kolkata"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())

        now  = datetime.datetime.now()
        hour = now.hour
        ghi  = data["hourly"]["shortwave_radiation"][hour]
        tamb = data["hourly"]["temperature_2m"][hour]
        return WeatherFrame(
            timestamp  = now,
            ghi        = ghi,
            t_amb      = tamb,
            source     = "open-meteo",
            quality_ok = True,
        )
    except Exception as e:
        print(f"[WARN] Live fetch failed ({e}); using stub fallback.")
        return _stub_frame(datetime.datetime.now().hour)


def _replay_frame(csv_rows: list[dict], step: int) -> WeatherFrame:
    r   = csv_rows[step % len(csv_rows)]
    ts  = datetime.datetime(int(r.get("year", 2016)),
                            int(r.get("month", 1)),
                            int(r.get("day",   1)),
                            int(r.get("hour",  0)))
    return WeatherFrame(
        timestamp  = ts,
        ghi        = float(r.get("ghi_wm2", r.get("GHI", 0))),
        t_amb      = float(r.get("t_amb_C",  r.get("T_amb", 25))),
        t_mains    = float(r.get("t_mains_C", 20)),
        demand_L   = float(r.get("demand_L",  20)),
        source     = "csv_replay",
        quality_ok = True,
    )


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

class LiveDemoRunner:
    """Drive the simulator step-by-step with live or replay weather.

    Parameters
    ----------
    algo      : algorithm name (ppo, dqn, recurrent, qrdqn, maskable)
    regime_id : 0, 1 or 2
    mode      : DemoMode.REPLAY / LIVE / STUB
    replay_csv: path to hourly CSV for REPLAY mode
    log_dir   : where to write the action/state log
    """

    def __init__(
        self,
        algo:       str = "recurrent",
        regime_id:  int = 0,
        mode:       DemoMode = DemoMode.STUB,
        replay_csv: pathlib.Path | None = None,
        log_dir:    pathlib.Path | None = None,
    ):
        self.algo      = algo
        self.regime_id = regime_id
        self.mode      = mode
        self.log_dir   = log_dir or (REPO_ROOT / "results" / "demo_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._model      = self._load_model()
        self._env        = self._make_env()
        self._lstm_state = None
        self._ep_start   = True
        self._step       = 0
        self._log_rows   = []

        # replay CSV
        self._replay_rows: list[dict] = []
        if mode == DemoMode.REPLAY and replay_csv and replay_csv.exists():
            import csv
            with open(replay_csv) as f:
                self._replay_rows = list(csv.DictReader(f))

    # ------------------------------------------------------------------

    def _load_model(self):
        from scripts.phase6_locked_eval import load_model
        try:
            path = load_model(self.algo, self.regime_id)
            if self.algo == "recurrent":
                from sb3_contrib import RecurrentPPO
                return RecurrentPPO.load(str(path))
            elif self.algo == "maskable":
                from sb3_contrib import MaskablePPO
                return MaskablePPO.load(str(path))
            elif self.algo == "qrdqn":
                from sb3_contrib import QRDQN
                return QRDQN.load(str(path))
            elif self.algo == "sac":
                from stable_baselines3 import SAC
                return SAC.load(str(path))
            elif self.algo == "ppo":
                from stable_baselines3 import PPO
                return PPO.load(str(path))
            else:
                from stable_baselines3 import DQN
                return DQN.load(str(path))
        except FileNotFoundError:
            print("[WARN] No trained model found; using B1 reactive fallback.")
            from o3.baselines.b1_reactive import ReactiveRulePolicy
            return ReactiveRulePolicy()

    def _make_env(self):
        from o3.environment.gym_wrapper import PCMGymEnv
        return PCMGymEnv(CONTRACT_PATH, regime_id=self.regime_id,
                         fixed_year=2016)

    # ------------------------------------------------------------------

    def _get_weather(self) -> WeatherFrame:
        if self.mode == DemoMode.STUB:
            return _stub_frame(self._step)
        if self.mode == DemoMode.LIVE:
            return _fetch_live_frame()
        if self._replay_rows:
            return _replay_frame(self._replay_rows, self._step)
        return _stub_frame(self._step)   # fallback

    def _select_action(self, obs: np.ndarray) -> int:
        from o3.baselines.b1_reactive import ReactiveRulePolicy
        if isinstance(self._model, ReactiveRulePolicy):
            return self._model.act(obs)
        if self.algo == "recurrent":
            action, self._lstm_state = self._model.predict(
                obs, state=self._lstm_state,
                episode_start=self._ep_start, deterministic=True,
            )
            self._ep_start = False
            return int(action)
        if self.algo == "maskable":
            masks  = self._env.action_masks()
            action, _ = self._model.predict(obs, action_masks=masks,
                                            deterministic=True)
            return int(action)
        if self.algo == "sac":
            from o3.environment.gym_wrapper import PCMContinuousGymEnv
            action, _ = self._model.predict(obs.astype("float32"),
                                            deterministic=True)
            return PCMContinuousGymEnv._to_discrete(action)
        action, _ = self._model.predict(obs.astype("float32"), deterministic=True)
        return int(action)

    # ------------------------------------------------------------------

    def reset(self, year: int = 2016) -> np.ndarray:
        obs, _          = self._env.reset()
        self._lstm_state = None
        self._ep_start   = True
        self._step       = 0
        return obs

    def step_once(self) -> dict:
        wx  = self._get_weather()
        obs = self._env.inner._state_to_obs() if hasattr(
            self._env.inner, "_state_to_obs") else self._env.reset()[0]

        action    = self._select_action(obs)
        obs2, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1

        ACTION_NAMES = {0: "CHARGE", 1: "IDLE", 2: "BYPASS"}
        frame = {
            "timestamp":          wx.timestamp.isoformat(),
            "mode":               self.mode.value,
            "source":             wx.source,
            "quality_ok":         wx.quality_ok,
            "step":               self._step,
            "ghi_wm2":            wx.ghi_wm2,
            "t_amb_C":            wx.t_amb_C,
            "action":             ACTION_NAMES.get(action, action),
            "reward":             reward,
            "E_delivered_Wh":     info.get("E_delivered_Wh", 0.0),
            "E_unmet_Wh":         info.get("E_unmet_Wh",    0.0),
            "shield_override":    info.get("shield_override", False),
            "safety_violation":   info.get("n_safety_violations", 0),
            "T_water_C":          info.get("T_water_C", None),
            "T_pcm_C":            info.get("T_pcm_C",   None),
            "f_melt":             info.get("f_melt",     None),
        }
        self._log_rows.append(frame)
        return frame

    def run(self, n_steps: int = 24, interval_s: float = 0.0,
            verbose: bool = True) -> list[dict]:
        """Run n_steps demo steps (default = one day = 24 h)."""
        obs = self.reset()
        frames = []
        for _ in range(n_steps):
            frame = self.step_once()
            frames.append(frame)
            if verbose:
                sv = "⚠ SAFETY" if frame["safety_violation"] else ""
                sh = "->SHIELD" if frame["shield_override"] else ""
                print(
                    f"  h{frame['step']:4d} | GHI={frame['ghi_wm2']:6.0f} W/m² "
                    f"| {frame['action']:6s} {sh:8s} "
                    f"| del={frame['E_delivered_Wh']:6.0f} Wh "
                    f"| reward={frame['reward']:7.2f}  {sv}"
                )
            if interval_s > 0:
                time.sleep(interval_s)
        self._save_log()
        return frames

    def _save_log(self) -> None:
        if not self._log_rows:
            return
        import csv
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.log_dir / f"demo_log_{self.algo}_r{self.regime_id}_{ts}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._log_rows[0].keys()))
            w.writeheader()
            w.writerows(self._log_rows)
        print(f"\n[INFO] Demo log saved -> {out}")
