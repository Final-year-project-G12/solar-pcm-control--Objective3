"""
src/design/schema.py
=====================
The design vector Objective 2 searches over (D2.2 input). 40-hr scope:
sphere capsules only (configs/design_space.yaml, geometry_engine.capsule_shape).
Arrangement is a searched variable (single-layer, staggered, radial).

x = [capsule_diameter_m, n_capsule, flow_rate_kg_s, capsule_arrangement]

PCM thickness (conduction distance) and PCM volume fraction are DERIVED,
not independently sampled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DesignVector:
    capsule_diameter_m: float
    n_capsule: int
    flow_rate_kg_s: float
    capsule_arrangement: str   # one of "single-layer", "staggered", "radial" —
                                # required, no default: every caller (DOE
                                # generator, optimizer search, manual CLI) must
                                # pass this explicitly so a staggered-only
                                # assumption can't silently creep back in.
    capsule_shape: str = "sphere"        # frozen for 40-hr scope

    def as_dict(self):
        return {
            "capsule_diameter_m": self.capsule_diameter_m,
            "n_capsule": self.n_capsule,
            "flow_rate_kg_s": self.flow_rate_kg_s,
            "capsule_shape": self.capsule_shape,
            "capsule_arrangement": self.capsule_arrangement,
        }
