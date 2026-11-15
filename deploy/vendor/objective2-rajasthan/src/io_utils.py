"""
src/io_utils.py
================
Shared config/data loaders used by every Phase 1-4 module. Centralized here
so geometry.py, tank_model.py, gates.py etc. never hand-roll their own YAML
or CSV parsing, and so a shared-config change is visible in one place.

Nothing here is state-specific except the `state` argument you pass in.
"""

from pathlib import Path
import functools
import hashlib

import yaml
import pandas as pd

from config import BASE_DIR, CONFIGS_DIR, DATA_DIR

STATES_DIR = CONFIGS_DIR / "states"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_sidecar(results_path: Path, state: str, extra: dict = None) -> Path:
    """Writes `<results_path>.manifest.json` next to a results file, stamped
    with config_hashes(state) plus a UTC timestamp and whatever phase-specific
    `extra` dict is passed in. CSV/parquet files have no header section to
    embed this in, so a sidecar JSON is the equivalent (step 1 of the
    2026-09-20 fix plan: "Write a sha256 of both shared configs and the state
    config into every results file header")."""
    import json
    from datetime import datetime, timezone

    results_path = Path(results_path)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "config_hashes": config_hashes(state),
    }
    if extra:
        manifest.update(extra)
    manifest_path = results_path.with_suffix(results_path.suffix + ".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


def config_hashes(state: str) -> dict:
    """SHA-256 of the two shared configs + the state config, for stamping
    into every results-file header (step 1 of the 2026-09-20 fix plan).
    Lets a results file prove which exact config versions produced it,
    instead of only carrying a hand-written version string that can go
    stale (system_config_shared.yaml gained the safety_shield block on
    2026-09-13 without a version bump — see that file's own history)."""
    system_config_path = CONFIGS_DIR / "system_config_shared.yaml"
    design_space_path = CONFIGS_DIR / "design_space.yaml"
    state_config_path = STATES_DIR / f"{state}.yaml"
    return {
        "system_config_shared.yaml": {
            "version": load_system_config().get("version"),
            "sha256": _sha256_of(system_config_path),
        },
        "design_space.yaml": {
            "version": load_design_bounds().get("version"),
            "sha256": _sha256_of(design_space_path),
        },
        f"states/{state}.yaml": {
            "version": load_state_config(state).get("state_config_version"),
            "sha256": _sha256_of(state_config_path),
        },
    }


def _load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def load_system_config():
    """FROZEN shared config — collector/tank/PCM-integration/pump/safety/solver."""
    return _load_yaml(CONFIGS_DIR / "system_config_shared.yaml")


@functools.lru_cache(maxsize=None)
def load_design_bounds():
    """Bounds dict for the exact geometry check: the `geometry_engine` block of configs/design_space.yaml, with the
    PCM fraction cap and the flow range taken from the approved variables."""
    sp = _load_yaml(CONFIGS_DIR / "design_space.yaml")
    ge, v = sp["geometry_engine"], sp["variables"]
    return {
        "version": sp["version"],
        "capsule_shape": list(ge["capsule_shape"]),
        "capsule_arrangement": list(v["arrangement"]),
        "pcm_thickness_m": dict(ge["pcm_thickness_m"]),
        "capsule_diameter_m": dict(ge["capsule_diameter_m"]),
        "capsule_count": dict(ge["capsule_count"]),
        "pcm_volume_fraction": {"min": ge["pcm_volume_fraction_min"], "max": v["pcm_volume_fraction"]["max"]},
        "flow_rate_kg_s": dict(v["flow_rate_kg_s"]),
        "geometry": {"spacing_min_m": ge["spacing_min_m"], "passage_min_fraction": ge["passage_min_fraction"]},
    }


@functools.lru_cache(maxsize=None)
def load_state_config(state: str):
    """Per-state config (Phase 1 output) — the only state-varying input."""
    return _load_yaml(STATES_DIR / f"{state}.yaml")


# Objective 2 works on Objective 1's MCDM top-3 as written (data/objective1/mcdm_topk_by_cluster.csv).
O1_TOPK_FILE = DATA_DIR / "objective1" / "mcdm_topk_by_cluster.csv"


@functools.lru_cache(maxsize=None)
def o1_top3(state: str = "rajasthan") -> dict:
    """{cluster_id: [pcm names in O1 consensus-rank order]} from Objective 1's frozen mcdm_topk_by_cluster.csv.
    Equal consensus ranks are a genuine O1 tie and are kept in file order; see o1_rank_ties()."""
    df = pd.read_csv(O1_TOPK_FILE)
    out = {}
    for cid, g in df.groupby("cluster_id", sort=True):
        g = g.assign(_order=range(len(g))).sort_values(["consensus_rank", "_order"])
        out[int(cid)] = g["pcm_id"].tolist()
    return out


def o1_rank_ties(state: str = "rajasthan") -> dict:
    """{cluster_id: [names sharing consensus rank 1]} -- more than one name means O1 left rank 1 tied."""
    df = pd.read_csv(O1_TOPK_FILE)
    return {int(c): g.loc[g["consensus_rank"] == 1, "pcm_id"].tolist() for c, g in df.groupby("cluster_id")}


def get_regime(state: str, cluster_id: int) -> dict:
    """One regime's entry from states/<state>.yaml (weather paths, T_mains, PCM shortlist names).
    `pcm_shortlist` is Objective 1's top-3, in consensus-rank order."""
    cfg = load_state_config(state)
    for r in cfg["regimes"]:
        if int(r["cluster_id"]) == int(cluster_id):
            return {**r, "pcm_shortlist": o1_top3(state)[int(cluster_id)]}
    raise KeyError(f"cluster_id {cluster_id} not found in states/{state}.yaml")


def list_regimes(state: str):
    return [r["cluster_id"] for r in load_state_config(state)["regimes"]]


@functools.lru_cache(maxsize=None)
def load_pcm_database(state: str) -> pd.DataFrame:
    cfg = load_state_config(state)
    path = BASE_DIR / cfg["pcm_database_file"]
    return pd.read_csv(path)


def get_pcm_properties(state: str, pcm_name: str) -> dict:
    """Full property record for one PCM by name, from the frozen Objective 1
    database (data/objective1/pcm_database_<state>.csv). Never invents a
    property Objective 1 didn't report."""
    df = load_pcm_database(state)
    row = df[df["name"] == pcm_name]
    if row.empty:
        raise KeyError(f"PCM '{pcm_name}' not found in pcm_database for state={state}")
    return row.iloc[0].to_dict()


def drop_leap_day(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 29 February (local date) so every simulated year is 8760 h.
    A leap year otherwise draws 0.27 % more water and collects 1/365 more sun,
    which confounds year-to-year comparison (O2 G2, 2026-10-26)."""
    if "local_date" in df.columns:
        is_leap_day = df["local_date"].astype(str).str.endswith("-02-29")
    else:
        ts = df["timestamp_utc"]
        is_leap_day = (ts.dt.month == 2) & (ts.dt.day == 29)
    return df.loc[~is_leap_day].reset_index(drop=True)


def load_hourly_weather(state: str, cluster_id: int, drop_29_feb: bool = True) -> pd.DataFrame:
    regime = get_regime(state, cluster_id)
    path = BASE_DIR / regime["weather_hourly"]
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return drop_leap_day(df) if drop_29_feb else df


def load_demand_profile(state: str) -> pd.DataFrame:
    cfg = load_state_config(state)
    path = BASE_DIR / cfg["demand_profile"]["file"]
    return pd.read_csv(path)
