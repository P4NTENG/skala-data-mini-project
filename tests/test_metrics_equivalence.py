"""Regression tests for the metric layer.

The metrics are computed from the confusion matrix with numpy (the bootstrap calls them tens of
thousands of times), so they must stay numerically identical to the sklearn definitions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from herg_model.metrics import compute_metrics, select_threshold


def _reference(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


@pytest.mark.parametrize("case", range(30))
def test_compute_metrics_matches_sklearn(case: int) -> None:
    rng = np.random.default_rng(case)
    n = int(rng.integers(40, 600))
    y_true = rng.integers(0, 2, n)
    # rounded scores create ties, which is the realistic case for tree/linear outputs
    scores = np.round(rng.random(n), 2)
    threshold = float(rng.choice([0.3, 0.4, 0.5, 0.6]))

    got = compute_metrics(y_true, scores, threshold)
    expected = _reference(y_true, (scores >= threshold).astype(int))

    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-12), name

    assert got["tn"] + got["fp"] + got["fn"] + got["tp"] == n
    assert got["n_blocker"] == int(y_true.sum())


def test_compute_metrics_positive_class_is_blocker() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_metrics(y_true, scores, 0.5)

    assert metrics["tp"] == 2 and metrics["fp"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["n_blocker"] == 2


def test_compute_metrics_handles_constant_prediction() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    metrics = compute_metrics(y_true, np.zeros(5), 0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["mcc"] == 0.0
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_select_threshold_matches_exhaustive_search() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, 400)
    scores = np.round(rng.random(400), 3)

    threshold, value = select_threshold(y_true, scores, "balanced_accuracy")
    exhaustive = [
        compute_metrics(y_true, scores, candidate)["balanced_accuracy"]
        for candidate in np.unique(np.clip(np.concatenate([np.linspace(0, 1, 201), scores]), 0, 1))
    ]

    assert value == pytest.approx(max(exhaustive), abs=1e-12)
    assert compute_metrics(y_true, scores, threshold)["balanced_accuracy"] == pytest.approx(value)


def test_select_threshold_is_deterministic_with_ties() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    first = select_threshold(y_true, scores, "balanced_accuracy")
    second = select_threshold(y_true, scores, "balanced_accuracy")

    assert first == second
