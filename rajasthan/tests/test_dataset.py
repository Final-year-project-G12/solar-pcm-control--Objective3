"""
Phase O3-3 tests — dataset generation (B0/B1 safe trajectories). The file must be identical for any n_jobs.
Run with: pytest tests/test_dataset.py -v
"""

import csv
import json

import pytest

REGIME = 1


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    from o3.data.trajectory_generator import generate_dataset
    p = tmp_path_factory.mktemp("ds") / "ds.csv"
    generate_dataset(p, regime_id=REGIME, years=[2016], pcm_inits=["solid"], policies=["B0"], n_jobs=1)
    return p


def test_dataset_file_created(dataset_path):
    assert dataset_path.exists() and dataset_path.stat().st_size > 0


def test_dataset_no_test_year_leakage(dataset_path):
    with open(dataset_path) as f:
        years = {int(r["year"]) for r in csv.DictReader(f)}
    assert not years & {2006, 2007, 2008, 2009, 2010}


def test_dataset_refuses_test_years(tmp_path):
    from o3.data.trajectory_generator import generate_dataset
    with pytest.raises(PermissionError):
        generate_dataset(tmp_path / "x.csv", regime_id=REGIME, years=[2008], n_jobs=1)


def test_dataset_has_required_columns(dataset_path):
    with open(dataset_path) as f:
        cols = set(next(csv.reader(f)))
    required = {"episode_id", "regime_id", "year", "pcm_init", "step", "action", "reward", "terminated", "safety_cost"}
    required |= {f"obs_{i}" for i in range(13)} | {f"next_obs_{i}" for i in range(13)}
    assert required <= cols


def test_dataset_step_count_full_year(dataset_path):
    with open(dataset_path) as f:
        assert sum(1 for _ in csv.DictReader(f)) == 8760


def test_dataset_identical_for_serial_and_parallel(tmp_path):
    from o3.data.trajectory_generator import generate_dataset
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    kw = dict(regime_id=REGIME, years=[2016, 2017], pcm_inits=["solid"], policies=["B0", "B1"])
    generate_dataset(a, n_jobs=1, **kw)
    generate_dataset(b, n_jobs=2, **kw)
    assert a.read_bytes() == b.read_bytes()


def test_split_manifest_written(tmp_path):
    from o3.data.trajectory_generator import write_split_manifest
    p = tmp_path / "split.json"
    write_split_manifest(p)
    m = json.loads(p.read_text())
    assert m["train"] == [2016, 2017, 2018, 2019, 2020] and m["test"] == [2006, 2007, 2008, 2009, 2010]
