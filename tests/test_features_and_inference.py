"""Feature computation, config validation and the inference round-trip (no dataset required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires_data

from herg_model import descriptors as descriptor_module
from herg_model import fingerprints as fingerprint_module
from herg_model.config import load_config
from herg_model.predict import compute_blocks_from_smiles, predict_smiles, save_artifact
from herg_model.preprocessing import FeaturePipeline

SMOKE_SMILES = [
    "CN1CCC[C@H]1c2cccnc2",
    "CC(=O)Oc1ccccc1C(=O)O",
    "O=C(Nc1ccccc1)c1ccccc1",
    "Clc1ccc(cc1)C(=O)N1CCN(CC1)C",
]


def test_fingerprint_recomputation_is_deterministic(cfg):
    first = fingerprint_module.recompute_matrix(SMOKE_SMILES, "ecfp", cfg)
    second = fingerprint_module.recompute_matrix(SMOKE_SMILES, "ecfp", cfg)
    assert np.array_equal(first, second)
    assert first.shape == (len(SMOKE_SMILES), 1024)


def test_pattern_fingerprint_uses_requested_size(cfg):
    matrix = fingerprint_module.recompute_matrix(SMOKE_SMILES, "ptfp", cfg)
    assert matrix.shape == (len(SMOKE_SMILES), 1024)


def test_rdkit_descriptor_table_is_stable(cfg):
    frame = descriptor_module.compute_rdkit_descriptors(SMOKE_SMILES, None, use_cache=False, verbose=False)
    assert "MolWt" in frame.columns and "mol_logp" not in frame.columns
    assert "n_basic_n" in frame.columns
    assert frame["MolWt"].iloc[1] == pytest.approx(180.16, abs=0.05)


def test_descriptor_groups_partition_the_feature_set(cfg):
    frame = descriptor_module.compute_rdkit_descriptors(SMOKE_SMILES, None, use_cache=False, verbose=False)
    names = [c for c in frame.columns if c != "SMILES"]
    groups = descriptor_module.assign_groups(names)
    covered = [n for cols in groups.values() for n in cols]
    assert sorted(covered) == sorted(names)
    assert len(covered) == len(set(covered))  # no feature appears in two groups


def test_basic_amine_does_not_use_overlapping_matches():
    """An amide nitrogen must not be counted as a basic amine."""
    from herg_model.chemistry import mol_from_smiles
    from herg_model.descriptors import _extra_features

    features = _extra_features(mol_from_smiles("CC(=O)Nc1ccccc1"))  # acetanilide
    assert features["n_amide"] == 1
    assert features["n_basic_n"] == 0
    assert features["has_basic_amine"] == 0

    features = _extra_features(mol_from_smiles("CCN(CC)CC"))  # triethylamine
    assert features["n_basic_n"] == 1
    assert features["n_tertiary_amine"] == 1


def test_config_validation_rejects_inverted_labels(tmp_path):
    cfg = load_config()
    raw = dict(cfg)
    raw.pop("_config_path", None)
    raw["data"] = {**raw["data"], "label_meanings": {"0": "non_blocker", "1": "blocker"}}
    path = tmp_path / "bad.yaml"
    import yaml

    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_inference_round_trip(tmp_path, cfg):
    """A model trained on 'fingerprint + shipped descriptors' must reload and predict."""
    from sklearn.linear_model import LogisticRegression

    training_smiles = SMOKE_SMILES * 6
    y = np.array([1, 0, 1, 0] * 6)
    names, X = compute_blocks_from_smiles(training_smiles, ["fp:ecfp", "existing_descriptors"], cfg)
    pipeline = FeaturePipeline(names=names, scale=True).fit(X, y)
    model = LogisticRegression(max_iter=1000).fit(pipeline.transform(X), y)
    metadata = {
        "label_meanings": {"0": "blocker (positive class)", "1": "non_blocker"},
        "blocks": ["fp:ecfp", "existing_descriptors"],
        "feature_order": pipeline.kept_names,
        "selected_threshold": 0.5,
    }
    path = save_artifact(tmp_path / "model.joblib", model, pipeline, metadata)
    from herg_model.predict import load_artifact

    artifact = load_artifact(path)
    prediction = predict_smiles(SMOKE_SMILES, artifact, cfg)
    assert list(prediction.columns) == [
        "SMILES", "blocker_probability", "predicted_label", "predicted_class", "threshold"
    ]
    assert len(prediction) == len(SMOKE_SMILES)
    assert prediction["blocker_probability"].between(0, 1).all()
    assert set(prediction["predicted_class"]) <= {"blocker", "non_blocker"}
    # label convention: blocker (positive) is stored as 0, exactly as in train.csv
    blocker_rows = prediction["predicted_class"] == "blocker"
    assert (prediction.loc[blocker_rows, "predicted_label"] == 0).all()


@requires_data
def test_shipped_fingerprints_match_recomputation_on_real_data(cfg):
    from herg_model.data import fingerprint_matrix_from_csv, load_dataset

    df = load_dataset(cfg)[:5]
    for kind in ("ecfp", "fcfp", "ptfp"):
        _, shipped = fingerprint_matrix_from_csv(df, kind, cfg)
        rebuilt = fingerprint_module.recompute_matrix(list(df["SMILES"]), kind, cfg)
        assert np.array_equal(shipped.astype(np.int8), rebuilt)
