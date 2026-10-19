"""Model zoo, default hyper-parameters and Optuna search spaces.

* Scaling is decided by ``models.<name>.scaling`` in the config and applied by
  :class:`herg_model.preprocessing.FeaturePipeline` (train-only statistics).
* Tree ensembles never see scaled features (the doc explicitly forbids forcing it).
* Models whose optional dependency is missing are reported as unavailable instead of
  silently dropped.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import numpy as np

__all__ = [
    "MODEL_NAMES",
    "model_requires_scaling",
    "available_models",
    "unavailable_models",
    "default_params",
    "make_model",
    "suggest_params",
]

MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "svm",
)

_IMPORT_GUARDS: dict[str, tuple[str, ...]] = {
    "lightgbm": ("lightgbm",),
    "xgboost": ("xgboost",),
}


def _is_importable(name: str) -> bool:
    modules = _IMPORT_GUARDS.get(name, ())
    for module in modules:
        if module in sys.modules:
            continue
        try:
            __import__(module)
        except Exception:
            return False
    return True


def available_models(names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    candidates = list(names) if names is not None else list(MODEL_NAMES)
    return [name for name in candidates if _is_importable(name)]


def unavailable_models(names: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    candidates = list(names) if names is not None else list(MODEL_NAMES)
    reasons = {}
    for name in candidates:
        if not _is_importable(name):
            reasons[name] = f"missing dependency: {', '.join(_IMPORT_GUARDS.get(name, ()))}"
    return reasons


def model_requires_scaling(name: str, cfg) -> bool:
    return bool(cfg.get_path(f"models.{name}.scaling", name in {"logistic_regression", "svm"}))


def default_params(name: str, cfg, seed: int) -> dict[str, Any]:
    """Sane, reproducible defaults (used as the 'no tuning' reference in every experiment)."""
    class_weight = cfg.get_path(f"models.{name}.class_weight", None)
    if name in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": class_weight,
            "random_state": seed,
            "n_jobs": int(cfg.get_path("runtime.n_jobs", -1)),
        }
    if name == "logistic_regression":
        return {"C": 1.0, "max_iter": 2000, "tol": 1e-4, "class_weight": class_weight,
                "random_state": seed}
    if name == "hist_gradient_boosting":
        return {"learning_rate": 0.06, "max_iter": 300, "max_leaf_nodes": 31, "random_state": seed}
    if name == "lightgbm":
        return {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 1e-2,
            "class_weight": class_weight,
            "random_state": seed,
            "n_jobs": int(cfg.get_path("runtime.n_jobs", -1)),
            "verbose": -1,
        }
    if name == "xgboost":
        return {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": int(cfg.get_path("runtime.n_jobs", -1)),
            "verbosity": 0,
        }
    if name == "svm":
        return {"C": 1.0, "gamma": "scale", "kernel": "rbf", "probability": False,
                "class_weight": class_weight, "random_state": seed}
    raise ValueError(f"unknown model {name!r}")


def make_model(name: str, params: dict[str, Any], cfg, seed: int):
    """Instantiate a fresh estimator (never reuse a fitted model across trials)."""
    merged = dict(default_params(name, cfg, seed))
    merged.update(params or {})
    merged["random_state"] = seed
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        solver = merged.pop("solver", "liblinear")
        return LogisticRegression(solver=solver, **merged)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**merged)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(**merged)
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**merged)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**merged)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**merged)
    if name == "svm":
        from sklearn.svm import SVC

        return SVC(**merged)
    raise ValueError(f"unknown model {name!r}")


def suggest_params(trial, name: str, cfg) -> dict[str, Any]:
    """Optuna search space per model family."""
    class_weight = cfg.get_path(f"models.{name}.class_weight", None)
    weight_choices = [None, "balanced"] if class_weight is not None else [None]

    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", weight_choices),
        }
    if name in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.3]),
            "class_weight": trial.suggest_categorical("class_weight", weight_choices),
        }
    if name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", weight_choices),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1e1, log=True),
            "class_weight": trial.suggest_categorical("class_weight", weight_choices),
        }
    raise ValueError(f"unknown model {name!r}")


def positive_scores(model, X: np.ndarray) -> np.ndarray:
    """Blocker probability (or a monotone score) for the positive class (``label == 0``)."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 2 and classes[1] == 1:
            return proba[:, 1]
        if classes is not None and len(classes) == 2 and classes[0] == 1:
            return proba[:, 0]
        return proba[:, -1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        # map to (0, 1) monotonically; only ranking metrics are meaningful for these models
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("model exposes neither predict_proba nor decision_function")
