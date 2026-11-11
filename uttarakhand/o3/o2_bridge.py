"""
o3/o2_bridge.py
===============
The ONE place where Objective 3 touches the frozen Objective 2 Uttarakhand code.

* Puts `o2_frozen/` (a byte-for-byte snapshot of Objective---2/objective2-uttarakhand, commit in
  o2_frozen/O2_SNAPSHOT.json) first on sys.path, so `src.*` and `config` resolve to O2's own modules.
* Verifies the contract: its own SHA-256, the config hashes it records, and the selection-log /
  band / held-out hashes. The environment refuses to start if any of them fails.
* Builds a case exactly the way O2's `run_case` does, by CALLING run_case on a one-hour weather frame
  and capturing the arguments it passes to the compiled kernel (`_kernel`) and the collector helper.
  Nothing in O2's setup (geometry, packing layout, capsule masses, wall cp, loss UA, parameter vector)
  is re-derived here, so a change in O2 would show up as a hash failure, not as a silent divergence.

No file under o2_frozen/ is modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

UK_ROOT = Path(__file__).resolve().parents[1]
O2_ROOT = UK_ROOT / "o2_frozen"
CONTRACT_PATH = UK_ROOT / "contracts" / "obj3_environment_contract_uttarakhand_v2.json"
CONTRACT_SHA_PATH = CONTRACT_PATH.with_suffix(".json.sha256")
STATE = "uttarakhand"

if str(O2_ROOT) not in sys.path:
    sys.path.insert(0, str(O2_ROOT))

import src.simulation.tank_model_multinode as _tmm                     # noqa: E402
from src.design.space import geometry_bounds, load_space, to_design    # noqa: E402
from src.io_utils import get_regime, load_system_config                # noqa: E402
from src.simulation.run_case import run_case                           # noqa: E402

RESET_SCENARIOS = {"fully_solid": "solid", "partially_charged": "mid", "fully_liquid": "liquid"}


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_contract() -> dict:
    with open(CONTRACT_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_contract() -> dict:
    """Raise ContractError unless the contract and every input it fingerprints are unchanged.
    Returns a dict of the checks (all True on success)."""
    checks = {}
    recorded = CONTRACT_SHA_PATH.read_text().split()[0] if CONTRACT_SHA_PATH.exists() else None
    checks["contract_sha256"] = recorded is not None and sha256_file(CONTRACT_PATH) == recorded
    c = load_contract()
    for rel, entry in c["config_hashes"].items():
        p = O2_ROOT / (rel if rel.startswith("configs/") else f"configs/{rel}")
        checks[f"config:{rel}"] = p.exists() and sha256_file(p) == entry["sha256"]
    g = c["generated_from"]
    for r, h in g["selection_logs"].items():
        checks[f"selection:{r}"] = sha256_file(O2_ROOT / "results" / f"g10_selection_{r}.json") == h
    checks["wp6_band"] = sha256_file(O2_ROOT / "results" / "wp6_band.csv") == g["wp6_band"]
    checks["heldout_results"] = sha256_file(O2_ROOT / "results" / "g10_heldout_results.csv") == g["heldout_results"]
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ContractError(f"contract / O2 input hash check failed: {bad}")
    if c["global_limits"]["delivery_temp_target_C"] != load_system_config()["delivery"]["target_temp_C"]:
        raise ContractError("delivery target in contract and system config disagree")
    return checks


@lru_cache(maxsize=None)
def selection(regime: int) -> dict:
    with open(O2_ROOT / "results" / f"g10_selection_r{regime}.json", "r", encoding="utf-8") as fh:
        return json.loads(fh.read().replace("NaN", "null"))


def chosen_design(regime: int) -> dict:
    """O2's chosen design for a regime (g10 selection), as the four design variables + PCM."""
    ch = selection(regime)["chosen"]
    return {"pcm": ch["pcm"], "R_in_m": ch["R_in_m"], "fraction": ch["fraction"], "flow": ch["flow"],
            "arrangement": ch["arrangement"], "N": ch["N"],
            "meaningful": bool(selection(regime)["meaningful_on_primary_metric"])}


@dataclass
class KernelCase:
    """Everything the compiled O2 kernel needs, captured from O2's own setup path."""
    regime: int
    water_only: bool
    scenario: str
    m_w: np.ndarray
    n_caps: np.ndarray
    loss_ua: np.ndarray
    P: np.ndarray
    T0: np.ndarray
    h0: np.ndarray
    demand24: np.ndarray
    system_config: dict
    mains_C: float
    target_C: float
    flow_kg_s: float
    pcm: str | None


def _model_kwargs():
    m = load_space()["model"]
    return dict(tank_model=m["tank_model"], n_nodes=m["n_nodes"], stack_anchor=m["stack_anchor"])


def run_case_for(regime: int, weather, scenario: str = "fully_solid", water_only: bool = False,
                 record_hourly: bool = True, demand_kwargs: dict | None = None, night_day_stride: int = 0):
    """O2's run_case on the chosen design (or its same-flow water-only twin), exactly as O2's evaluate.py calls it."""
    d = chosen_design(regime)
    dv = to_design(d["R_in_m"], d["fraction"], d["flow"], d["arrangement"])
    overrides = {"solver": {"initial_pcm_state": RESET_SCENARIOS[scenario]}}
    return run_case(STATE, regime, None if water_only else d["pcm"], dv, design_bounds=geometry_bounds(),
                    weather=weather, record_hourly=record_hourly, system_config_overrides=overrides,
                    compute_tes_metrics=record_hourly, night_day_stride=night_day_stride, **(demand_kwargs or {}),
                    **_model_kwargs())


def build_kernel_case(regime: int, weather_one_hour, scenario: str = "fully_solid",
                      water_only: bool = False) -> KernelCase:
    """Run O2's run_case on a ONE-hour frame and capture the compiled kernel's arguments."""
    cap = {}
    orig_k, orig_c = _tmm._kernel, _tmm._collector_arrays

    def k(*args):
        cap["args"] = args
        return orig_k(*args)

    def c(weather, cfg):
        cap["cfg"] = cfg
        return orig_c(weather, cfg)

    _tmm._kernel, _tmm._collector_arrays = k, c
    try:
        out = run_case_for(regime, weather_one_hour, scenario, water_only, record_hourly=False)
    finally:
        _tmm._kernel, _tmm._collector_arrays = orig_k, orig_c
    if not out["valid"]:
        raise ContractError(f"regime {regime} chosen design rejected by O2 geometry: {out['reason']}")
    (_g, _tamb, _hod, _a, _b, _circ, demand24, m_w, n_caps, loss_ua, P, T0, h0, _use) = cap["args"]
    cfg = cap["cfg"]
    d = chosen_design(regime)
    return KernelCase(regime=regime, water_only=water_only, scenario=scenario, m_w=m_w.copy(),
                      n_caps=n_caps.copy(), loss_ua=loss_ua.copy(), P=P.copy(), T0=T0.copy(), h0=h0.copy(),
                      demand24=demand24.copy(), system_config=cfg, mains_C=float(get_regime(STATE, regime)["T_mains_est_C"]),
                      target_C=float(cfg["delivery"]["target_temp_C"]), flow_kg_s=float(d["flow"]),
                      pcm=None if water_only else d["pcm"])


def kernel():
    """The compiled O2 kernel itself (never a copy)."""
    return _tmm._kernel


def collector_arrays(weather, cfg):
    return _tmm._collector_arrays(weather, cfg)


def hour_of_day_array(weather) -> np.ndarray:
    """Local clock hour per row, through O2's own _demand_hour_of_day (refuses a UTC-only frame)."""
    return np.array([_tmm._demand_hour_of_day(weather.iloc[i], i) for i in range(len(weather))], dtype=np.int64)
