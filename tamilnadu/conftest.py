"""
pytest conftest — adds O3 root and O2 root to sys.path before any test.

Package naming:
  o3.xxx  — Objective 3 code (this repo, o3/ directory)
  src.xxx — Objective 2 frozen simulator (objective2_design_optimization/src/)

Both roots are on sys.path so imports resolve without conflict.
"""
import pathlib
import sys

_O3_ROOT = pathlib.Path(__file__).parent
_O2_ROOT = _O3_ROOT.parent / "objective2_design_optimization"

for p in [str(_O2_ROOT), str(_O3_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
