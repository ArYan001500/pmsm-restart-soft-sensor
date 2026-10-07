import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "data/manifests/profile_splits_literature_v2.json").read_text(encoding="utf-8"))


def test_primary_protocol_partitions_all_profiles():
    p = M["primary_protocol"]
    parts = [set(p["train"]), set(p["validation_V1"]), set(p["validation_V2"]), set(p["test_locked_G"])]
    union = set().union(*parts)
    assert union == set(M["all_profiles"]) and len(M["all_profiles"]) == 69
    assert sum(len(s) for s in parts) == 69  # pairwise disjoint


def test_literature_ids_match_published_protocols():
    p = M["primary_protocol"]
    assert p["validation_V1"] == [4, 67, 71, 78] and p["validation_V2"] == [10, 48, 63]
    assert p["test_locked_G"] == [60, 62, 74]
    s = M["secondary_protocol"]
    assert s["validation"] == [58] and s["test_locked"] == [65, 72]
    assert not set(s["train"]) & {58, 65, 72}


def test_cv_folds_exclude_locked_and_are_disjoint():
    folds = [set(f) for f in M["dev_cv5_over_non_G"]]
    assert not set().union(*folds) & {60, 62, 74}
    assert sum(len(f) for f in folds) == len(set().union(*folds)) == 66


def test_loader_refuses_locked_profiles():
    from pmsm_softsense.realdata import load
    with pytest.raises(PermissionError):
        load([60])
