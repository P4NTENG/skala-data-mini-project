"""Evaluation metrics with an explicit positive class.

Positive class = **hERG blocker**, i.e. ``label == 0`` / ``y_blocker == 1``.
Every function in this module assumes ``y_true`` is the blocker vector and ``scores`` is the
estimated blocker probability, so Precision/Recall/PR-AUC always refer to blockers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "compute_metrics",
    "confusion_counts",
    "select_threshold",
    "bootstrap_metric_ci",
    "cluster_bootstrap_indices",
    "roc_points",
    "pr_points",
    "PRIMARY_METRIC",
]

PRIMARY_METRIC = "balanced_accuracy"


def _safe_auc(fn, y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(fn(y_true, scores))
    except Exception:
        return float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Full metric set for the blocker (positive) class.

    The count-based metrics are derived from the confusion matrix with numpy instead of
    calling one sklearn scorer per metric: ``compute_metrics`` is called tens of thousands
    of times by the bootstrap, and the sklearn call overhead dominated the runtime
    (~10 ms vs ~1 ms here). The values are identical to the sklearn equivalents.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    counts = confusion_counts(y_true, (scores >= threshold).astype(int))
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    pos, neg, n = tp + fn, tn + fp, tp + fp + fn + tn

    recall = tp / pos if pos else 0.0
    specificity = tn / neg if neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if pos and neg:
        balanced_accuracy = 0.5 * (recall + specificity)
    else:  # degenerate resample: sklearn averages over the classes that are present
        balanced_accuracy = float((tp + tn) / n) if n else float("nan")

    result = {
        "threshold": float(threshold),
        "n": int(n),
        "n_blocker": int(pos),
        "n_non_blocker": int(neg),
        "roc_auc": _safe_auc(roc_auc_score, y_true, scores),
        "pr_auc": _safe_auc(average_precision_score, y_true, scores),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float((tp + tn) / n) if n else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "specificity": float(specificity),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
    }
    result.update(counts)
    return result


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = PRIMARY_METRIC,
    grid_size: int = 201,
) -> tuple[float, float]:
    """Threshold maximising ``metric`` on the *validation* split (never the test split).

    Only the distinct score values change the predicted partition, so the search is
    vectorised over those values (plus the all-positive / all-negative endpoints).
    Ties are broken towards the smallest threshold, matching the previous grid search.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) == 0:
        return 0.5, float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)
    # last position of each tie group -> the partition obtained with ``scores >= threshold``
    last = np.searchsorted(-sorted_scores, -sorted_scores, side="right") - 1
    tp = tp_cum[last].astype(float)
    fp = fp_cum[last].astype(float)

    pos = float(sorted_labels.sum())
    neg = float(len(sorted_labels) - pos)
    thresholds = sorted_scores[last]

    if metric == "balanced_accuracy" or metric == "youden":
        recall = tp / pos if pos else np.zeros_like(tp)
        specificity = (neg - fp) / neg if neg else np.zeros_like(tp)
        values = 0.5 * (recall + specificity)
    elif metric == "f1":
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = tp / pos if pos else np.zeros_like(tp)
        values = np.divide(2 * precision * recall, precision + recall,
                           out=np.zeros_like(tp), where=(precision + recall) > 0)
    elif metric == "mcc":
        fn = pos - tp
        tn = neg - fp
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        values = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    elif metric == "accuracy":
        values = (tp + (neg - fp)) / float(len(sorted_labels))
    else:  # unknown threshold metric -> fall back to the generic scorer
        return _select_threshold_reference(y_true, scores, metric, grid_size)

    values = np.nan_to_num(values, nan=-np.inf)
    # prefer the smallest threshold among ties (previous grid search behaviour)
    best_value = float(values.max())
    best_index = int(np.flatnonzero(values >= best_value - 1e-12)[0])
    return float(thresholds[best_index]), best_value


def _select_threshold_reference(
    y_true: np.ndarray, scores: np.ndarray, metric: str, grid_size: int = 201
) -> tuple[float, float]:
    candidates = np.unique(np.clip(np.concatenate([np.linspace(0, 1, grid_size), scores]), 0, 1))
    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        value = compute_metrics(y_true, scores, threshold).get(metric, float("nan"))
        if np.isfinite(value) and value > best_value:
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, best_value


def cluster_bootstrap_indices(
    groups: Sequence,
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[np.ndarray]:
    """Cluster bootstrap: resample *scaffolds* (not rows) with replacement.

    Independence is assumed between scaffolds, not between molecules -- rows sharing a
    scaffold are dependent and are therefore kept together in every resample.
    """
    groups = np.asarray(groups, dtype=object)
    unique_groups = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        draws.append(np.concatenate([members[g] for g in picked]))
    return draws


def bootstrap_metric_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: Sequence | None = None,
    metric: str = PRIMARY_METRIC,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile confidence interval for a threshold-based metric.

    Pass ``groups`` (e.g. scaffolds) for a cluster bootstrap; omit it for a row bootstrap,
    in which case the (unrealistic) row-independence assumption must be stated.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    point = compute_metrics(y_true, scores, threshold).get(metric, float("nan"))

    if groups is None:
        groups = np.arange(len(y_true))
        unit = "row"
    else:
        unit = "scaffold"

    draws = cluster_bootstrap_indices(groups, n_resamples=n_resamples, seed=seed)
    values = []
    for idx in draws:
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        value = compute_metrics(y_boot, scores[idx], threshold).get(metric, float("nan"))
        if np.isfinite(value):
            values.append(value)

    if not values:
        return {"metric": metric, "unit": unit, "point": point, "lo": float("nan"),
                "hi": float("nan"), "n_resamples": 0, "confidence": confidence}

    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "unit": unit,
        "point": float(point),
        "lo": float(np.quantile(values, alpha)),
        "hi": float(np.quantile(values, 1 - alpha)),
        "n_resamples": int(len(values)),
        "confidence": float(confidence),
    }


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(scores, dtype=float))
    return fpr, tpr


def pr_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(
        np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    )
    return recall, precision
