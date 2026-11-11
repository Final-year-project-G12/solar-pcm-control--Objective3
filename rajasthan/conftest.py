"""
pytest conftest — puts the O3 root and the O2 Rajasthan root on sys.path before any test.

Package naming:
  o3.xxx  — Objective 3 code (this repo, o3/ directory)
  src.xxx — Objective 2 frozen simulator (OBJECTIVE2/Objective---2/objective2-rajasthan/src/)

Both roots are on sys.path so imports resolve without conflict. Torch threads are pinned to 1
so tests are deterministic and do not fight each other for cores.
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_O3_ROOT = pathlib.Path(__file__).resolve().parent
if str(_O3_ROOT) not in sys.path:
    sys.path.insert(0, str(_O3_ROOT))

import config_O2  # noqa: E402

config_O2.setup_paths()
