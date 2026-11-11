"""Objective 3 (Rajasthan): DRL controller around the frozen Objective-2 simulator.

Importing `o3` also puts the O2 Rajasthan tree on sys.path (see config_O2.py), so worker
processes started by joblib / SubprocVecEnv can `import src...` without any extra setup.
"""
import os as _os
import pathlib as _pl
import sys as _sys

# Stable-Baselines3 tries `torch.utils.tensorboard`, which imports TensorFlow when it is installed. That costs
# ~10 s and ~1 GB per process; with 10 SubprocVecEnv workers it exhausted this laptop's memory
# (WinError 1450 / 1114, "paging file too small"). O3 never logs to tensorboard, so block it (SB3 treats the
# ImportError as "tensorboard not installed"). Opt out with O3_ALLOW_TENSORBOARD=1.
if _os.environ.get("O3_ALLOW_TENSORBOARD") != "1" and "tensorboard" not in _sys.modules:
    _sys.modules["tensorboard"] = None
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ROOT = _pl.Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import config_O2 as _cfg  # noqa: E402

_cfg.setup_paths()
