"""
o3/utils/parallel.py — process-level parallelism that does not change results.

Rules this module enforces (Step 3 of the task, docs/HOW_TO_RUN.md):
  * One process per job. Each job is a pure function of its own arguments (its own seed, regime,
    year, scenario) -> the result is identical whether the job runs serially, in a pool, or in a
    different order. `run_parallel(..., n_jobs=1)` runs the same function in-process.
  * Every worker pins torch / BLAS / numba threads to 1, so N workers use N cores and the numbers
    do not depend on how many threads a library happened to pick.
  * Results come back in the order the jobs were given.

Cores: `physical_cores()` uses psutil when available. `default_n_jobs()` = physical cores - 1
(leave one core for the OS / the main process), minimum 1. `default_n_envs()` is the same number
and is used for SubprocVecEnv.
"""

import os
from typing import Callable, Iterable

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def default_n_jobs() -> int:
    return max(1, physical_cores() - 1)


def default_n_envs() -> int:
    return default_n_jobs()


def pin_threads(n: int = 1) -> None:
    """Pin BLAS/OpenMP/numba threads (env vars; inherited by child processes) and torch if loaded."""
    for v in _THREAD_VARS:
        os.environ[v] = str(n)
    try:
        import sys
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(n)
    except Exception:
        pass


def _call(fn: Callable, kwargs: dict):
    pin_threads(1)
    import o3  # noqa: F401  (puts the O2 tree on sys.path inside the worker)
    return fn(**kwargs)


def run_parallel(fn: Callable, jobs: Iterable[dict], n_jobs: int | None = 1, verbose: int = 0) -> list:
    """Run fn(**job) for every job. n_jobs=1 (or a single job) -> serial in this process.

    `fn` must be a module-level function (picklable). Order of results == order of jobs.
    """
    jobs = list(jobs)
    n_jobs = default_n_jobs() if n_jobs is None or n_jobs <= 0 else int(n_jobs)
    n_jobs = min(n_jobs, len(jobs)) if jobs else 1
    if n_jobs <= 1:
        return [fn(**j) for j in jobs]
    pin_threads(1)                              # children inherit these variables
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(delayed(_call)(fn, j) for j in jobs)
