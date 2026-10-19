"""Preprocessing must be fitted on the training split only and never leak test statistics."""

from __future__ import annotations

import numpy as np
import pytest

from herg_model.preprocessing import FeaturePipeline, feature_quality_table, high_correlation_pairs


def test_constant_and_duplicate_columns_are_dropped():
    X = np.column_stack([
        np.ones(20),                     # constant -> dropped
        np.arange(20, dtype=float),      # kept
        np.arange(20, dtype=float),      # exact duplicate of the previous column -> dropped
        np.full(20, np.nan),             # all missing -> dropped
    ])
    names = ["const", "keep", "dup", "empty"]
    pipeline = FeaturePipeline(names=names).fit(X)
    assert pipeline.kept_names == ["keep"]
    assert pipeline.dropped["constant"] == ["const"]
    assert pipeline.dropped["all_missing"] == ["empty"]
    assert len(pipeline.dropped["duplicate"]) == 1


def test_imputation_uses_training_medians_only():
    X_train = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
    X_test = np.array([[np.nan, np.nan]])
    pipeline = FeaturePipeline(names=["a", "b"]).fit(X_train)
    assert pipeline.fill_values.tolist() == [3.0, 6.0]
    transformed = pipeline.transform(X_test)
    assert transformed.tolist() == [[3.0, 6.0]]


def test_scaling_statistics_come_from_train_only():
    X_train = np.array([[0.0], [2.0], [4.0]])
    X_test = np.array([[100.0]])
    pipeline = FeaturePipeline(names=["a"], scale=True).fit(X_train)
    train_scaled = pipeline.transform(X_train)
    assert train_scaled.mean() == pytest.approx(0.0, abs=1e-9)
    assert pipeline.transform(X_test)[0, 0] > 10  # test values are not re-centred


def test_infinities_become_nan_then_imputed():
    X = np.array([[np.inf, 1.0], [2.0, 2.0], [3.0, 3.0]])
    pipeline = FeaturePipeline(names=["a", "b"]).fit(X)
    out = pipeline.transform(X)
    assert np.isfinite(out).all()


def test_quality_table_flags_constants():
    X = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
    table = feature_quality_table(X, ["a", "b"]).set_index("feature")
    assert bool(table.loc["a", "is_constant"]) is True
    assert bool(table.loc["b", "is_constant"]) is False


def test_correlation_pairs_detects_perfect_correlation():
    rng = np.random.default_rng(0)
    base = rng.normal(size=50)
    X = np.column_stack([base, base * 2.0, rng.normal(size=50)])
    pairs = high_correlation_pairs(X, ["a", "b", "c"], threshold=0.95)
    assert set(zip(pairs["feature_a"], pairs["feature_b"])) == {("a", "b")}
