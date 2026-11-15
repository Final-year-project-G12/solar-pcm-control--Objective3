"""
src/simulation/cfd_closure.py
==============================
Named capsule closures for the lumped PCM model (WP2-C, B2).

The lumped capsule has two closure constants: c in R_pcm = r_in / (c k_eff A_in) and the liquid-convection factor F (k_eff = k F once
melt fraction >= 0.5). The frozen values are c = 5, F = 2 (configs/system_config_shared.yaml). `pcm_integration.closure` selects a set:

    absent or "legacy"   -> the config is returned untouched (bit-for-bit the old behaviour; no key is added)
    "cfd_closure_v1"     -> c and F are read from configs/cfd_closure_v1.yaml and written into the existing keys
                            pcm_integration.conduction_coefficient_c and pcm_integration.liquid_convection_enhancement

Both heat-transfer paths (heat_transfer.ua_eff_total_w_k and the multi-node kernel) already read those two keys, so resolving the closure
once here, at config time, gives them identical values. cfd_closure_v1 is UNVALIDATED (see the yaml); it exists to make the closure a switch.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

CLOSURE_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
LEGACY = "legacy"
KNOWN = (LEGACY, "cfd_closure_v1")


def load_closure(name: str) -> dict:
    """Read a named closure file (configs/<name>.yaml). 'legacy' has no file."""
    if name not in KNOWN or name == LEGACY:
        raise ValueError(f"unknown closure {name!r}; choose one of {KNOWN}")
    return yaml.safe_load((CLOSURE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def apply_closure(system_config: dict) -> dict:
    """Return the config with the selected closure applied. A no-op (same object) for the legacy closure."""
    integ = system_config.get("pcm_integration", {})
    name = integ.get("closure", LEGACY)
    if name == LEGACY:
        return system_config
    cl = load_closure(name)
    out = copy.deepcopy(system_config)
    out["pcm_integration"]["conduction_coefficient_c"] = float(cl["conduction_coefficient_c"])
    out["pcm_integration"]["liquid_convection_enhancement"] = float(cl["liquid_convection_enhancement"])
    return out
