"""
o3/utils/cli.py — the two parallelism flags every phase script exposes, and the script bootstrap.

    --n-jobs N   worker PROCESSES for independent jobs (seeds, regimes, years, scenarios).
                 1 = serial in this process. 0 / 'auto' = physical cores - 1.
    --n-envs N   environments per on-policy training run (SubprocVecEnv). 'auto' = physical cores - 1.
                 Ignored by scripts / agents that do not train an on-policy agent.

Both are documented in docs/HOW_TO_RUN.md. Defaults are 'auto' so the full-size runs use the laptop;
pass 1 for a serial reference run (results of deterministic jobs are identical either way).
"""

import argparse

from o3.utils.parallel import default_n_jobs, default_n_envs


def _parse(v: str) -> int:
    if str(v).lower() in ("auto", "0"):
        return 0
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0 or 'auto'")
    return n


def add_parallel_args(parser: argparse.ArgumentParser, envs: bool = True) -> None:
    parser.add_argument("--n-jobs", type=_parse, default=0, metavar="N",
                        help="worker processes for independent jobs (1 = serial; 0/auto = physical cores - 1)")
    if envs:
        parser.add_argument("--n-envs", type=_parse, default=0, metavar="N",
                            help="parallel environments for on-policy training (1 = in-process; 0/auto = physical cores - 1)")


def resolve(args) -> tuple[int, int]:
    """(n_jobs, n_envs) with 'auto' replaced by the machine default."""
    n_jobs = default_n_jobs() if getattr(args, "n_jobs", 0) == 0 else args.n_jobs
    n_envs = default_n_envs() if getattr(args, "n_envs", 0) == 0 else getattr(args, "n_envs", 1)
    return n_jobs, n_envs
