"""The label semantics are the single most important invariant of this project.

``label = 0`` must always be the positive (hERG blocker) class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from herg_model.data import blocker_target, labels_from_blocker
from herg_model.metrics import compute_metrics, select_threshold


def test_blocker_target_maps_zero_to_positive():
    frame = pd.DataFrame({"label": [0, 1, 0, 1]})
    cfg = {"data": {"label_col": "label"}}
    assert blocker_target(frame, cfg).tolist() == [1, 0, 1, 0]


def test_label_round_trip():
    y = np.array([1, 0, 1, 1, 0])
    assert labels_from_blocker(y).tolist() == [0, 1, 0, 0, 1]


def test_unexpected_label_values_rejected():
    frame = pd.DataFrame({"label": [0, 2]})
    with pytest.raises(ValueError):
        blocker_target(frame, {"data": {"label_col": "label"}})


def test_metrics_treat_blocker_as_positive():
    y = np.array([1, 1, 0, 0])          # two blockers, two non-blockers
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_metrics(y, scores, 0.5)
    assert metrics["tp"] == 2 and metrics["tn"] == 2
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_false_positive_is_a_predicted_blocker():
    y = np.array([0, 0])                # no blockers at all
    scores = np.array([0.9, 0.1])       # first molecule wrongly predicted as blocker
    metrics = compute_metrics(y, scores, 0.5)
    assert metrics["fp"] == 1 and metrics["tp"] == 0
    assert metrics["precision"] == 0.0


def test_threshold_selection_improves_or_keeps_balanced_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    scores = np.clip(0.35 * y + rng.normal(0.3, 0.25, size=400), 0.001, 0.999)
    threshold, value = select_threshold(y, scores, "balanced_accuracy")
    assert 0.0 <= threshold <= 1.0
    assert value >= compute_metrics(y, scores, 0.5)["balanced_accuracy"] - 1e-9
