"""Feature selection that only ever looks at the training data.

Methods
-------
``extra_trees_importance``
    Fit an ExtraTrees model on the training split and keep the ``k`` highest-importance
    columns. Cheap, model-agnostic and effective for wide fingerprint + descriptor matrices.
``correlation_prune``
    Greedy removal of columns whose |Pearson r| with an already-kept column exceeds a
    threshold (keeps the first column of every correlated cluster).
``none``
    No selection.

Selection is *never* run with target information from the validation or test split, and the
selected column names are stored in the experiment manifest so the final pipeline can be
replayed exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "select_features",
    "importance_ranking",
    "correlation_prune",
    "aggregate_importance_by_group",
]


def _importance(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return np.abs(coef).reshape(-1) if coef.ndim <= 1 else np.abs(coef).mean(axis=0)
    raise AttributeError("model exposes neither feature_importances_ nor coef_")


def importance_ranking(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    seed: int = 42,
    n_estimators: int = 400,
    n_jobs: int = -1,
) -> pd.Series:
    """ExtraTrees-based feature importance, sorted descending."""
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    model.fit(X, y)
    return pd.Series(_importance(model, X, y), index=names).sort_values(ascending=False)


def correlation_prune(
    X: np.ndarray,
    names: list[str],
    threshold: float = 0.95,
) -> tuple[np.ndarray, list[tuple[str, str, float]]]:
    """Return kept column indices and the list of removed ``(dropped, kept, r)`` triples."""
    X = np.asarray(X, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(X), med, X)
    std = filled.std(axis=0, ddof=0)
    keep: list[int] = []
    removed: list[tuple[str, str, float]] = []
    for j in range(filled.shape[1]):
        if std[j] <= 1e-12:
            continue
        placed = False
        for k in keep:
            denom = std[j] * std[k]
            if denom <= 0:
                continue
            r = float(np.dot(filled[:, j] - filled[:, j].mean(), filled[:, k] - filled[:, k].mean()) / (len(filled) * denom))
            if np.isfinite(r) and abs(r) >= threshold:
                removed.append((names[j], names[k], r))
                placed = True
                break
        if not placed:
            keep.append(j)
    return np.asarray(keep, dtype=int), removed


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    cfg,
    seed: int | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Dispatch on ``experiment.feature_selection.method``; returns (indices, names, meta)."""
    method = str(cfg.get_path("experiment.feature_selection.method", "none"))
    if seed is None:
        seed = int(cfg.get_path("project.seed", 42))
    meta: dict = {"method": method, "n_input": len(names)}

    if method == "none":
        indices = np.arange(len(names))
        meta.update({"n_output": len(names), "selected": list(names)})
        return indices, list(names), meta

    if method == "correlation_prune":
        threshold = float(cfg.get_path("experiment.feature_selection.correlation_threshold", 0.95))
        indices, removed = correlation_prune(X, names, threshold)
        meta.update(
            {
                "threshold": threshold,
                "n_output": int(len(indices)),
                "n_removed": len(removed),
                "removed_examples": removed[:50],
            }
        )
        return indices, [names[j] for j in indices], meta

    if method in {"extra_trees_importance", "model_importance"}:
        k = int(cfg.get_path("experiment.feature_selection.n_features", 300))
        ranking = importance_ranking(X, y, names, seed=seed,
                                     n_jobs=int(cfg.get_path("runtime.n_jobs", -1)))
        k = min(k, len(ranking))
        selected = list(ranking.index[:k])
        index_of = {name: j for j, name in enumerate(names)}
        indices = np.asarray([index_of[n] for n in selected], dtype=int)
        meta.update(
            {
                "n_output": int(len(indices)),
                "requested_k": k,
                "top_20": selected[:20],
            }
        )
        return indices, selected, meta

    if method == "permutation_importance":
        k = int(cfg.get_path("experiment.feature_selection.n_features", 300))
        importances = permutation_importance_scores(X, y, names, cfg, seed=seed)
        selected = list(importances.index[: min(k, len(importances))])
        index_of = {name: j for j, name in enumerate(names)}
        indices = np.asarray([index_of[n] for n in selected], dtype=int)
        meta.update({"n_output": int(len(indices)), "top_20": selected[:20]})
        return indices, selected, meta

    raise ValueError(f"unknown feature selection method {method!r}")


def permutation_importance_scores(
    X: np.ndarray, y: np.ndarray, names: list[str], cfg, seed: int = 42
) -> pd.Series:
    """Permutation importance computed on an internal hold-out *inside the training split*."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = ExtraTreesClassifier(
        n_estimators=300, random_state=seed, n_jobs=int(cfg.get_path("runtime.n_jobs", -1)),
        class_weight="balanced",
    )
    model.fit(Xtr, ytr)
    result = permutation_importance(model, Xho, yho, n_repeats=5, random_state=seed, n_jobs=1)
    return pd.Series(result.importances_mean, index=names).sort_values(ascending=False)


def aggregate_importance_by_group(importances: pd.Series, group_of: dict[str, str]) -> pd.DataFrame:
    """Sum / mean importance per descriptor group (``group_of`` maps feature -> group)."""
    frame = pd.DataFrame({"feature": importances.index, "importance": importances.values})
    frame["group"] = frame["feature"].map(lambda f: group_of.get(f, "fingerprint"))
    agg = (
        frame.groupby("group")
        .agg(n_features=("feature", "size"), total_importance=("importance", "sum"),
             mean_importance=("importance", "mean"), max_importance=("importance", "max"))
        .sort_values("total_importance", ascending=False)
        .reset_index()
    )
    return agg
