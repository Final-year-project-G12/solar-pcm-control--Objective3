"""
Phase O3-3 tests — dataset generation and split integrity.
"""

import pathlib
import pytest

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / \
    "obj3_environment_contract_tamilnadu_v3.json"
TEST_YEARS    = {2011, 2012, 2013, 2014, 2015}
TRAIN_YEARS   = [2016, 2017]   # use 2 years to keep test fast


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    """Generate a mini dataset (2 years, solid only, B0 policy) into tmp."""
    from o3.data.trajectory_generator import generate_dataset
    from o3.baselines.b0_fixed import FixedChargePolicy

    out = tmp_path_factory.mktemp("data") / "mini_dataset.csv"
    b0  = FixedChargePolicy()
    generate_dataset(
        out_path  = out,
        regime_id = 0,
        years     = [2016],      # one year for speed
        pcm_inits = ["solid"],
        policies  = {"B0": lambda obs: b0.act(obs)},
    )
    return out


def test_dataset_file_created(dataset_path):
    assert dataset_path.exists()
    assert dataset_path.stat().st_size > 0


def test_dataset_no_test_year_leakage(dataset_path):
    import csv
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        leaked = [int(r["year"]) for r in reader
                  if int(r["year"]) in TEST_YEARS]
    assert leaked == [], f"Test years leaked into dataset: {set(leaked)}"


def test_dataset_has_required_columns(dataset_path):
    import csv
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = ["episode_id", "regime_id", "year", "pcm_init", "step",
                 "action", "reward", "terminated", "safety_cost",
                 "obs_0", "next_obs_0"]
    for col in required:
        assert col in row, f"Missing column: {col}"


def test_dataset_step_count_full_year(dataset_path):
    import csv
    with open(dataset_path) as f:
        rows = list(csv.DictReader(f))
    assert 8700 <= len(rows) <= 8760, \
        f"Expected ~8760 rows for one year, got {len(rows)}"


def test_split_manifest_written(tmp_path):
    from o3.data.trajectory_generator import write_split_manifest
    out = tmp_path / "split_manifest.json"
    write_split_manifest(out)
    assert out.exists()
    import json
    manifest = json.loads(out.read_text())
    assert "train" in manifest
    assert "val"   in manifest
    assert "test"  in manifest
    # test years must not appear in train or val
    test_set  = set(manifest["test"])
    train_set = set(manifest["train"])
    val_set   = set(manifest["val"])
    assert test_set.isdisjoint(train_set), "Test years overlap with train!"
    assert test_set.isdisjoint(val_set),   "Test years overlap with val!"
