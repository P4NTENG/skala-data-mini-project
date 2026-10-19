"""Feature blocks, experiment definitions (E0..E11 + ablations) and the experiment runner.

Fair-comparison rules implemented here
--------------------------------------
* every experiment sees the **identical** split objects (same seed, same scaffolds);
* every experiment/model gets the **same** Optuna budget (``optimization.n_trials[mode]``);
* preprocessing (constant / duplicate removal, imputation, scaling) and feature selection
  are fitted on the training split only;
* the **validation** split drives threshold and model selection; the test split is touched
  once, by ``scripts/train_final_model.py``, after selection is frozen.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import descriptors as descriptor_module
from . import models as model_module
from . import preprocessing as preprocessing_module
from . import feature_selection as selection_module
from .config import Config, budget, mode
from .data import (
    existing_descriptor_matrix,
    fingerprint_matrix_from_csv,
    fingerprint_columns,
)
from .metrics import bootstrap_metric_ci, compute_metrics, select_threshold
from .splitting import Splits

__all__ = [
    "ExperimentSpec",
    "FeatureStore",
    "experiment_definitions",
    "resolve_best_fingerprint",
    "run_configuration",
    "run_suite",
]

FP_BLOCKS = ("fp:ecfp", "fp:fcfp", "fp:ptfp")
DOMAIN_GROUPS = tuple(descriptor_module.RDKIT_GROUP_ORDER)


# --------------------------------------------------------------------------------------
# feature blocks
# --------------------------------------------------------------------------------------
class FeatureStore:
    """Lazily materialises named feature blocks and caches them for the whole run."""

    def __init__(self, df: pd.DataFrame, cfg: Config, verbose: bool = True):
        self.df = df
        self.cfg = cfg
        self.verbose = verbose
        self._cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._rdkit: pd.DataFrame | None = None
        self._mordred: pd.DataFrame | None = None
        self.mordred_status: dict = {}

    # -- blocks --------------------------------------------------------------------------
    def _rdkit_frame(self) -> pd.DataFrame:
        if self._rdkit is None:
            self._rdkit = descriptor_module.compute_rdkit_descriptors(
                list(self.df[str(self.cfg.get_path("data.smiles_col", "SMILES"))]),
                self.cfg,
                use_cache=bool(self.cfg.get_path("runtime.cache", True)),
                verbose=self.verbose,
            )
        return self._rdkit

    def _mordred_frame(self) -> pd.DataFrame | None:
        if self._mordred is None and not self.mordred_status:
            self._mordred, self.mordred_status = descriptor_module.compute_mordred_descriptors(
                list(self.df[str(self.cfg.get_path("data.smiles_col", "SMILES"))]),
                self.cfg,
                use_cache=bool(self.cfg.get_path("runtime.cache", True)),
                verbose=self.verbose,
            )
        return self._mordred

    def rdkit_names(self) -> list[str]:
        return [c for c in self._rdkit_frame().columns if c != "SMILES"]

    def block(self, name: str) -> tuple[list[str], np.ndarray]:
        if name in self._cache:
            return self._cache[name]

        if name.startswith("fp:"):
            kind = name.split(":", 1)[1]
            result = fingerprint_matrix_from_csv(self.df, kind, self.cfg)
        elif name == "existing_descriptors":
            result = existing_descriptor_matrix(self.df, self.cfg)
        elif name == "rdkit_all":
            frame = self._rdkit_frame()
            names = [c for c in frame.columns if c != "SMILES"]
            result = (names, frame[names].to_numpy(dtype=np.float64, copy=True))
        elif name == "mordred_2d":
            frame = self._mordred_frame()
            if frame is None:
                result = ([], np.zeros((len(self.df), 0), dtype=np.float64))
            else:
                names = [c for c in frame.columns if c != "SMILES"]
                result = (names, frame[names].to_numpy(dtype=np.float64, copy=True))
        elif name in DOMAIN_GROUPS:
            frame = self._rdkit_frame()
            groups = descriptor_module.assign_groups(self.rdkit_names())
            names = groups.get(name, [])
            result = (names, frame[names].to_numpy(dtype=np.float64, copy=True))
        elif name == "all_fingerprints":
            names: list[str] = []
            mats: list[np.ndarray] = []
            for block_name in FP_BLOCKS:
                block_names, block_matrix = self.block(block_name)
                names.extend(block_names)
                mats.append(block_matrix)
            result = (names, np.hstack(mats))
        else:
            raise ValueError(f"unknown feature block {name!r}")

        result = (list(result[0]), np.asarray(result[1], dtype=np.float64))
        self._cache[name] = result
        return result

    def matrix(self, blocks: Iterable[str]) -> tuple[list[str], np.ndarray]:
        names: list[str] = []
        mats: list[np.ndarray] = []
        seen: set[str] = set()
        for block_name in blocks:
            block_names, block_matrix = self.block(block_name)
            if not block_names:
                continue
            if len(block_names) != len(set(block_names)):
                # a descriptor can legitimately be listed in two blocks -- keep it once
                keep = [j for j, n in enumerate(block_names) if n not in seen]
                block_names = [block_names[j] for j in keep]
                block_matrix = block_matrix[:, keep]
            seen.update(block_names)
            names.extend(block_names)
            mats.append(block_matrix)
        if not mats:
            return [], np.zeros((len(self.df), 0), dtype=np.float64)
        return names, np.hstack(mats)

    def group_of(self, names: list[str]) -> dict[str, str]:
        """Map every feature to the block it came from (used for grouped importance)."""
        mapping: dict[str, str] = {}
        for kind in ("ecfp", "fcfp", "ptfp"):
            for column in fingerprint_columns(kind, self.cfg, self.df):
                mapping[column] = f"fingerprint_{kind}"
        for column in self.cfg.get_path("data.existing_descriptors", []) or []:
            mapping[column] = "existing_descriptor"
        groups = descriptor_module.assign_groups(self.rdkit_names())
        for group, columns in groups.items():
            for column in columns:
                mapping[column] = group
        frame = self._mordred_frame()
        if frame is not None:
            for column in frame.columns:
                if column == "SMILES":
                    continue
                # do not overwrite an RDKit group: names that exist in both tables are
                # deduplicated by ``matrix()`` (RDKit block wins), so attribute them to RDKit
                mapping.setdefault(column, "mordred_2d")
        return {name: mapping.get(name, "other") for name in names}


# --------------------------------------------------------------------------------------
# experiment definitions
# --------------------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    id: str
    name: str
    description: str
    blocks: list[str]
    models: list[str] | None = None
    feature_selection: bool = False
    group: str = "main"
    hypothesis: str = ""

    def resolved_blocks(self, best_fingerprint: str) -> list[str]:
        resolved = []
        for block in self.blocks:
            if block == "fp:BEST":
                resolved.extend(FP_BLOCKS if best_fingerprint == "combined" else [f"fp:{best_fingerprint}"])
            else:
                resolved.append(block)
        return resolved


def experiment_definitions(cfg: Config, best_fingerprint: str = "ecfp") -> list[ExperimentSpec]:
    """The E0..E11 matrix from the task description plus fingerprint and ablation studies."""
    base = ["fp:BEST", "existing_descriptors"]
    domain = list(DOMAIN_GROUPS[:-1])  # excludes misc_rdkit, which is included via rdkit_all

    specs = [
        ExperimentSpec("E0", "ecfp baseline", "baseline: shipped ECFP bits + shipped descriptors",
                       ["fp:ecfp", "existing_descriptors"]),
        ExperimentSpec("E1", "fcfp baseline", "fingerprint family comparison: FCFP",
                       ["fp:fcfp", "existing_descriptors"]),
        ExperimentSpec("E2", "ptfp baseline", "fingerprint family comparison: PTFP (pattern)",
                       ["fp:ptfp", "existing_descriptors"]),
        ExperimentSpec("E0b", "ecfp only", "ECFP bits without any descriptor", ["fp:ecfp"],
                       models=["lightgbm"], group="fingerprint_only"),
        ExperimentSpec("E1b", "fcfp only", "FCFP bits without any descriptor", ["fp:fcfp"],
                       models=["lightgbm"], group="fingerprint_only"),
        ExperimentSpec("E2b", "ptfp only", "PTFP bits without any descriptor", ["fp:ptfp"],
                       models=["lightgbm"], group="fingerprint_only"),
        ExperimentSpec("E3", "best fingerprint baseline",
                       "validation-selected single fingerprint + shipped descriptors", list(base)),
        ExperimentSpec("E3c", "combined fingerprints",
                       "ECFP+FCFP+PTFP concatenated + shipped descriptors",
                       ["all_fingerprints", "existing_descriptors"], models=["lightgbm"],
                       group="fingerprint_combo"),
        ExperimentSpec("E4", "rdkit descriptors only",
                       "no fingerprint at all: full RDKit 2D descriptor set", ["rdkit_all"]),
        ExperimentSpec("E5", "best fp + physicochemical",
                       "general property descriptors on top of the best fingerprint", base + ["physicochemical"]),
        ExperimentSpec("E6", "best fp + herg domain descriptors",
                       "physicochemical + all six hERG-motivated descriptor groups on top of the best fingerprint",
                       base + domain, hypothesis="hERG domain knowledge outperforms generic properties"),
        ExperimentSpec("E7", "best fp + all rdkit descriptors",
                       "full RDKit 2D descriptor expansion (domain groups + misc)", base + ["rdkit_all"]),
        ExperimentSpec("E8", "best fp + mordred 2d",
                       "Mordred 2D descriptor expansion (unselected)", base + ["mordred_2d"]),
        ExperimentSpec("E8b", "best fp + selected mordred",
                       "Mordred expansion with train-only feature selection", base + ["mordred_2d"],
                       feature_selection=True, group="mordred"),
        ExperimentSpec("E9", "best fp + rdkit + mordred",
                       "RDKit and Mordred descriptors combined", base + ["rdkit_all", "mordred_2d"]),
        ExperimentSpec("E10", "best fp + rdkit + selection",
                       "E7 feature set with train-only feature selection", base + ["rdkit_all"],
                       feature_selection=True),
        ExperimentSpec("E11", "best fp + rdkit + selection (all models)",
                       "identical features to E10, expanded to every available model family",
                       base + ["rdkit_all"], feature_selection=True, group="model_comparison"),
    ]

    for group in domain:
        specs.append(
            ExperimentSpec(
                f"A_{group}",
                f"ablation: + {group}",
                f"best fingerprint baseline plus the single descriptor group '{group}'",
                base + [group],
                models=["lightgbm"],
                group="ablation",
            )
        )
    return specs


def resolve_best_fingerprint(results: pd.DataFrame, metric: str = "tuned_balanced_accuracy") -> str:
    """Pick the fingerprint family with the highest mean validation metric in E0/E1/E2.

    ``metric`` is a column of ``experiment_results``-style frames; the plain metric name
    (``balanced_accuracy``) is also accepted and mapped to the ``tuned_`` variant.
    """
    candidates = {"E0": "ecfp", "E1": "fcfp", "E2": "ptfp"}
    subset = results[results["experiment_id"].isin(candidates.keys())]
    if subset.empty:
        return "ecfp"
    column = metric if metric in subset.columns else f"tuned_{metric}"
    if column not in subset.columns:
        return "ecfp"
    means = subset.groupby("experiment_id")[column].mean()
    return candidates[str(means.idxmax())]


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------
def _tune(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    seed: int,
    n_trials: int,
    timeout: int | None,
    metric: str,
) -> tuple[dict[str, Any], list[dict], float]:
    """Optuna study maximising ``metric`` on the validation split."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=str(cfg.get_path("optimization.direction", "maximize")),
                                sampler=sampler)
    trials: list[dict] = []

    def objective(trial):
        params = model_module.suggest_params(trial, model_name, cfg)
        model = model_module.make_model(model_name, params, cfg, seed)
        model.fit(X_train, y_train)
        scores = model_module.positive_scores(model, X_val)
        value = compute_metrics(y_val, scores, float(cfg.get_path("evaluation.default_threshold", 0.5)))[metric]
        trials.append({"trial": trial.number, "value": float(value), "params": json.dumps(params, default=str)})
        return value

    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return dict(study.best_params), trials, time.time() - started


def run_configuration(
    spec: ExperimentSpec,
    store: FeatureStore,
    splits: Splits,
    y: np.ndarray,
    scaffolds: np.ndarray,
    cfg: Config,
    model_names: list[str] | None = None,
    keep_models: bool = False,
    best_fingerprint: str = "ecfp",
    verbose: bool = False,
) -> tuple[list[dict], dict, dict]:
    """Fit every requested model for one experiment. Returns (rows, models, metadata)."""
    seed = int(splits.seed)
    n_trials = int(budget(cfg, "optimization.n_trials", 6) or 0)
    timeout = budget(cfg, "optimization.timeout_seconds", None)
    tune_metric = str(cfg.get_path("optimization.metric", "balanced_accuracy"))
    default_threshold = float(cfg.get_path("evaluation.default_threshold", 0.5))
    threshold_metric = str(cfg.get_path("evaluation.threshold_metric", "balanced_accuracy"))
    bootstrap_cfg = cfg.get_path("evaluation.bootstrap", {}) or {}

    blocks = spec.resolved_blocks(best_fingerprint)
    names, X = store.matrix(blocks)
    if not names:
        return [], {}, {"skipped": "no features available", "blocks": blocks}

    model_names = model_names or spec.models or model_module.available_models(
        budget(cfg, "optimization.models", None)
    )
    include_mordred = any("mordred" in b for b in blocks)

    rows: list[dict] = []
    fitted: dict[str, Any] = {}
    metadata: dict = {
        "blocks": blocks,
        "n_features_input": len(names),
        "feature_names": names if len(names) <= 400 else None,
        "mordred_included": include_mordred,
        "models_requested": list(model_names),
        "n_trials": n_trials,
        "tune_metric": tune_metric,
        "seed": seed,
        "trials": [],
    }

    train_idx, val_idx = splits.train, splits.val
    y_train, y_val = y[train_idx], y[val_idx]

    for model_name in model_names:
        scaling = model_module.model_requires_scaling(model_name, cfg)
        pipeline = preprocessing_module.FeaturePipeline(names=names, scale=scaling)
        if include_mordred:
            pipeline.missing_rate_limit = float(
                cfg.get_path("descriptors.mordred.max_missing_rate", 1.0)
            )
        X_train = pipeline.fit_transform(X[train_idx], y_train)
        X_val = pipeline.transform(X[val_idx])
        feature_names = list(pipeline.kept_names)

        selection_meta: dict = {"method": "none", "n_output": len(feature_names)}
        if spec.feature_selection:
            sel_idx, selected_names, selection_meta = selection_module.select_features(
                X_train, y_train, feature_names, cfg, seed=seed
            )
            X_train = X_train[:, sel_idx]
            X_val = X_val[:, sel_idx]
            feature_names = list(selected_names)

        common = {
            "experiment_id": spec.id,
            "experiment": spec.name,
            "group": spec.group,
            "split_strategy": splits.strategy,
            "seed": seed,
            "model": model_name,
            "scaling": scaling,
            "n_features_input": len(names),
            "n_features_used": len(feature_names),
            "n_features_dropped": len(names) - len(pipeline.kept_names),
            "feature_selection": selection_meta["method"],
            "n_trials_budget": n_trials,
            "blocks": "+".join(blocks),
            "hypothesis": spec.hypothesis,
        }

        # --- default (untuned) reference -------------------------------------------------
        started = time.time()
        default_model = model_module.make_model(model_name, model_module.default_params(model_name, cfg, seed), cfg, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            default_model.fit(X_train, y_train)
        default_scores = model_module.positive_scores(default_model, X_val)
        default_fit_time = time.time() - started
        default_metrics = compute_metrics(y_val, default_scores, default_threshold)

        # --- hyper-parameter search ------------------------------------------------------
        best_params, trials, tuning_time = ({}, [], 0.0)
        if n_trials > 0 and bool(cfg.get_path("optimization.enabled", True)):
            best_params, trials, tuning_time = _tune(
                model_name, X_train, y_train, X_val, y_val, cfg, seed, n_trials, timeout, tune_metric
            )

        tuned_model = model_module.make_model(model_name, best_params, cfg, seed)
        started = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tuned_model.fit(X_train, y_train)
        best_fit_time = time.time() - started
        val_scores = model_module.positive_scores(tuned_model, X_val)
        tuned_metrics = compute_metrics(y_val, val_scores, default_threshold)

        tuned_threshold, tuned_threshold_value = select_threshold(y_val, val_scores, threshold_metric)
        threshold_metrics = compute_metrics(y_val, val_scores, tuned_threshold)

        ci = {}
        if bootstrap_cfg.get("enabled", True):
            for label, metric, threshold in (
                ("ba_0.5", "balanced_accuracy", default_threshold),
                ("ba_tuned_threshold", "balanced_accuracy", tuned_threshold),
                ("roc_auc", "roc_auc", default_threshold),
                ("pr_auc", "pr_auc", default_threshold),
            ):
                ci[label] = bootstrap_metric_ci(
                    y_val,
                    val_scores,
                    groups=scaffolds[val_idx] if bootstrap_cfg.get("unit", "scaffold") == "scaffold" else None,
                    metric=metric,
                    threshold=threshold,
                    n_resamples=int(bootstrap_cfg.get("n_resamples", 1000)),
                    confidence=float(bootstrap_cfg.get("confidence", 0.95)),
                    seed=seed,
                )

        row = dict(common)
        row.update({
            "default_params": json.dumps(model_module.default_params(model_name, cfg, seed), default=str),
            "best_params": json.dumps(best_params, default=str),
            "n_trials_run": len(trials),
            "tune_time_seconds": round(tuning_time, 2),
            "default_fit_seconds": round(default_fit_time, 2),
            "best_fit_seconds": round(best_fit_time, 2),
            "selected_threshold": float(tuned_threshold),
            "selected_threshold_value": float(tuned_threshold_value),
            "delta_ba_vs_default": float(tuned_metrics["balanced_accuracy"] - default_metrics["balanced_accuracy"]),
            "ci_ba_0.5_lo": ci.get("ba_0.5", {}).get("lo"),
            "ci_ba_0.5_hi": ci.get("ba_0.5", {}).get("hi"),
            "ci_ba_tuned_lo": ci.get("ba_tuned_threshold", {}).get("lo"),
            "ci_ba_tuned_hi": ci.get("ba_tuned_threshold", {}).get("hi"),
            "ci_roc_auc_lo": ci.get("roc_auc", {}).get("lo"),
            "ci_roc_auc_hi": ci.get("roc_auc", {}).get("hi"),
            "ci_pr_auc_lo": ci.get("pr_auc", {}).get("lo"),
            "ci_pr_auc_hi": ci.get("pr_auc", {}).get("hi"),
            "ci_unit": ci.get("ba_0.5", {}).get("unit", "scaffold"),
        })
        for key, value in default_metrics.items():
            row[f"default_{key}"] = value
        for key, value in tuned_metrics.items():
            row[f"tuned_{key}"] = value
        for key, value in threshold_metrics.items():
            row[f"threshold_{key}"] = value
        rows.append(row)
        metadata["trials"].extend({"model": model_name, **trial} for trial in trials)
        if verbose:
            print(
                f"[experiments]   {spec.id:<6} {model_name:<24} "
                f"feat={len(feature_names):<5} BA={threshold_metrics['balanced_accuracy']:.4f} "
                f"ROC={threshold_metrics['roc_auc']:.4f} ({row['tune_time_seconds'] + row['best_fit_seconds']:.0f}s)",
                flush=True,
            )

        if keep_models:
            fitted[model_name] = {
                "model": tuned_model,
                "pipeline": pipeline,
                "feature_names": feature_names,
                "selected_names": feature_names,
                "params": best_params,
                "threshold": float(tuned_threshold),
            }

    metadata["selection"] = {"feature_selection": spec.feature_selection}
    return rows, fitted, metadata


def run_suite(
    specs: list[ExperimentSpec],
    store: FeatureStore,
    splits: Splits,
    y: np.ndarray,
    scaffolds: np.ndarray,
    cfg: Config,
    keep_models_for: tuple[str, ...] = (),
    verbose: bool = True,
    best_fingerprint: str = "ecfp",
    partial_path=None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run every experiment spec on one split; returns (results, models, metadata)."""
    all_rows: list[dict] = []
    all_models: dict[str, dict] = {}
    metadata: dict = {}
    ordered = sorted(specs, key=lambda s: s.id)
    for spec in ordered:
        keep = spec.id in keep_models_for
        if verbose:
            print(f"[experiments] {spec.id:<6} {spec.name}", flush=True)
        rows, models, meta = run_configuration(
            spec, store, splits, y, scaffolds, cfg, keep_models=keep,
            best_fingerprint=best_fingerprint, verbose=verbose,
        )
        all_rows.extend(rows)
        metadata[spec.id] = meta
        if keep:
            all_models[spec.id] = models
        if partial_path is not None and all_rows:
            pd.DataFrame(all_rows).to_csv(partial_path, index=False)
    return pd.DataFrame(all_rows), all_models, metadata
