"""Explainability and error analysis.

All analyses in this module describe *associations and model behaviour*. Feature importance
(permutation, impurity, SHAP) is never treated as causal evidence about hERG blockade.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_metrics

__all__ = [
    "permutation_importance_table",
    "shap_importance_table",
    "group_importance_table",
    "misclassification_table",
    "scaffold_performance",
    "class_descriptor_comparison",
    "top_feature_redundancy",
]


def permutation_importance_table(
    model,
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    n_repeats: int = 3,
    seed: int = 42,
    max_samples: float = 0.5,
) -> pd.DataFrame:
    """Permutation importance of the fitted model on the evaluation split."""
    from sklearn.inspection import permutation_importance

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=1, max_samples=max_samples
        )
    return (
        pd.DataFrame(
            {
                "feature": list(names),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
                "source": "permutation",
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_importance_table(
    model,
    X: np.ndarray,
    names: Sequence[str],
    seed: int = 42,
    max_samples: int = 800,
    background_samples: int = 200,
) -> tuple[pd.DataFrame | None, dict]:
    """Mean |SHAP| per feature; ``(None, status)`` when SHAP is unavailable for the model."""
    status: dict = {"available": False, "reason": None, "n_explained": 0}
    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        status["reason"] = f"shap import failed: {exc}"
        return None, status

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_samples, len(X)), replace=False)
    X_explain = X[idx]
    try:
        name = model.__class__.__name__
        if name == "LogisticRegression":
            background = shap.sample(X, min(background_samples, len(X)), random_state=seed)
            explainer = shap.LinearExplainer(model, background, feature_names=list(names))
        elif name in {"RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
                      "HistGradientBoostingClassifier", "LGBMClassifier", "XGBClassifier"}:
            explainer = shap.TreeExplainer(model)
        else:
            status["reason"] = f"no SHAP explainer configured for {name}"
            return None, status
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = explainer.shap_values(X_explain)
    except Exception as exc:
        status["reason"] = f"SHAP explainer failed for {model.__class__.__name__}: {exc}"
        return None, status

    array = np.asarray(values)
    if array.ndim == 3:  # (n, features, classes)
        array = array[:, :, -1]
    elif isinstance(values, list):
        array = np.asarray(values[-1])
    mean_abs = np.abs(array).mean(axis=0)
    if mean_abs.shape[0] != len(names):
        status["reason"] = f"SHAP output shape {mean_abs.shape} does not match {len(names)} features"
        return None, status
    status.update({"available": True, "n_explained": int(len(X_explain))})
    table = (
        pd.DataFrame({"feature": list(names), "importance_mean": mean_abs, "source": "shap"})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return table, status


def group_importance_table(importances: pd.DataFrame, group_of: dict[str, str]) -> pd.DataFrame:
    frame = importances.copy()
    frame["group"] = frame["feature"].map(lambda f: group_of.get(f, "other"))
    return (
        frame.groupby("group", as_index=False)
        .agg(
            n_features=("feature", "size"),
            total_importance=("importance_mean", "sum"),
            mean_importance=("importance_mean", "mean"),
            max_importance=("importance_mean", "max"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )


def misclassification_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    meta: pd.DataFrame,
    extra_features: pd.DataFrame | None = None,
    top_features: Sequence[str] = (),
) -> pd.DataFrame:
    """Every evaluated molecule with its prediction -- errors are never dropped."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    frame = meta.reset_index(drop=True).copy()
    frame["blocker_probability"] = scores
    frame["predicted_blocker"] = predicted
    frame["y_true_blocker"] = y_true
    frame["threshold"] = float(threshold)
    frame["correct"] = predicted == y_true
    frame["error_type"] = np.where(
        frame["correct"], "correct",
        np.where((y_true == 1), "false_negative", "false_positive"),
    )
    if extra_features is not None:
        columns = [c for c in top_features if c in extra_features.columns]
        if columns:
            frame = pd.concat(
                [frame, extra_features.loc[frame.index, columns].reset_index(drop=True)], axis=1
            )
    return frame


def scaffold_performance(
    y_true: np.ndarray,
    scores: np.ndarray,
    scaffolds: Sequence[str],
    threshold: float,
    min_size: int = 2,
) -> pd.DataFrame:
    """Per-scaffold performance distribution (which chemotypes the model fails on)."""
    frame = pd.DataFrame(
        {
            "scaffold": np.asarray(scaffolds, dtype=object),
            "y_true": np.asarray(y_true).astype(int),
            "score": np.asarray(scores, dtype=float),
        }
    )
    frame["predicted"] = (frame["score"] >= threshold).astype(int)
    grouped = frame.groupby("scaffold")
    columns = [
        "scaffold", "n_molecules", "n_blocker", "blocker_fraction", "mean_score",
        "accuracy", "balanced_accuracy", "n_errors", "error_rate",
    ]
    rows = []
    for scaffold, group in grouped:
        if len(group) < min_size:
            continue
        metrics = compute_metrics(group["y_true"].to_numpy(), group["score"].to_numpy(), threshold)
        rows.append(
            {
                "scaffold": scaffold,
                "n_molecules": int(len(group)),
                "n_blocker": int(group["y_true"].sum()),
                "blocker_fraction": float(group["y_true"].mean()),
                "mean_score": float(group["score"].mean()),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"] if group["y_true"].nunique() > 1 else float("nan"),
                "n_errors": int((group["predicted"] != group["y_true"]).sum()),
                "error_rate": float((group["predicted"] != group["y_true"]).mean()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)


def class_descriptor_comparison(
    descriptors: pd.DataFrame,
    y_blocker: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    """Distribution of a descriptor inside the blocker vs non-blocker class."""
    y = np.asarray(y_blocker).astype(int)
    rows = []
    for name in names:
        if name not in descriptors.columns:
            continue
        column = descriptors[name].to_numpy(dtype=float)
        blocker = column[y == 1]
        non_blocker = column[y == 0]
        blocker = blocker[np.isfinite(blocker)]
        non_blocker = non_blocker[np.isfinite(non_blocker)]
        if blocker.size == 0 or non_blocker.size == 0:
            continue
        pooled = np.concatenate([blocker, non_blocker])
        std = pooled.std(ddof=0)
        rows.append(
            {
                "feature": name,
                "n_blocker": int(blocker.size),
                "n_non_blocker": int(non_blocker.size),
                "blocker_mean": float(blocker.mean()),
                "non_blocker_mean": float(non_blocker.mean()),
                "blocker_median": float(np.median(blocker)),
                "non_blocker_median": float(np.median(non_blocker)),
                "standardised_mean_diff": float((blocker.mean() - non_blocker.mean()) / std) if std > 0 else float("nan"),
                "auc_direction": float(_rank_auc(blocker, non_blocker)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.reindex(frame["standardised_mean_diff"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC of a single descriptor as a blocker score (0.5 = no separation)."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def top_feature_redundancy(
    importances: pd.DataFrame,
    X: np.ndarray,
    names: Sequence[str],
    top_k: int = 25,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Correlation among the top-importance features (are we counting the same signal twice?)."""
    index_of = {name: j for j, name in enumerate(names)}
    top = [f for f in importances["feature"].head(top_k) if f in index_of]
    if len(top) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    columns = np.column_stack([X[:, index_of[f]] for f in top])
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(np.nan_to_num(columns, nan=0.0), rowvar=False)
    rows = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            if np.isfinite(corr[a, b]) and abs(corr[a, b]) >= threshold:
                rows.append({"feature_a": top[a], "feature_b": top[b], "pearson_r": float(corr[a, b])})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "pearson_r"])
