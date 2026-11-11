"""o3/frozen.py -- load the frozen (M4) policies and verify their hashes before any evaluation."""

from __future__ import annotations

import json

from o3 import config as o3c
from o3 import o2_bridge as ob
from o3.controllers import B1Reactive, PPOPolicy


def frozen_dir(regime: int):
    return o3c.RESULTS / "frozen" / f"r{regime}"


def load_frozen(regime: int) -> dict:
    d = frozen_dir(regime)
    fz = json.loads((d / "frozen.json").read_text())
    for name, p in fz["policies"].items():
        if ob.sha256_file(d / p["file"]) != p["sha256"]:
            raise ob.ContractError(f"frozen policy {name} for r{regime} changed since M4")
    if ob.sha256_file(d / "normalisation.json") != fz["normalisation_sha256"]:
        raise ob.ContractError("frozen normalisation changed")
    if ob.sha256_file(o3c.normalisation_path(regime)) != fz["normalisation_sha256"]:
        raise ob.ContractError("results/normalisation file differs from the frozen copy")
    if o3c.config_hash() != fz["o3_config_sha256"]:
        raise ob.ContractError("configs/o3_config.yaml changed since the policy was frozen")
    return fz


def ppo_policies(regime: int) -> list[PPOPolicy]:
    fz = load_frozen(regime)
    d = frozen_dir(regime)
    return [PPOPolicy(d / p["file"], name=f"PPO_{name}") for name, p in fz["policies"].items()]


def primary_policy(regime: int) -> PPOPolicy:
    fz = load_frozen(regime)
    p = fz["policies"][f"seed{fz['primary_seed']}"]
    return PPOPolicy(frozen_dir(regime) / p["file"], name="PPO")


def b1(regime: int) -> B1Reactive:
    return B1Reactive(**json.loads((o3c.RESULTS / f"b1_params_r{regime}.json").read_text())["params"])
