"""Stage B/D/E -- run the controlled experiment matrix (E0..E11 + ablations) and the
scaffold-seed stability re-runs.

Run:
    python scripts/run_experiments.py                 # primary seed, all experiments
    python scripts/run_experiments.py --only E0 E1 E2 # subset
    python scripts/run_experiments.py --skip-stability

Outputs
-------
* ``reports/experiment_results.csv``     : one row per (experiment, model) with validation metrics
* ``reports/optuna_trials.csv``          : every hyper-parameter trial that was evaluated
* ``reports/stability_results.csv``      : the selected configurations re-run on every scaffold seed
* ``reports/class_imbalance_results.csv``: class weighting vs threshold tuning
* ``artifacts/experiment_manifest.json`` : blocks, budgets, seeds, skipped items
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from _common import ROOT, artifacts_path, base_parser, context, reports_path

from herg_model import data as data_module
from herg_model import experiments as experiments_module
from herg_model import models as model_module
from herg_model import preprocessing as preprocessing_module
from herg_model import splitting as splitting_module
from herg_model.chemistry import load_structure_table
from herg_model.config import budget, environment_info, mode
from herg_model.metrics import compute_metrics, select_threshold


def metrics_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Compact metric view (validation, threshold 0.5 and validation-selected threshold)."""
    keep = [
        "experiment_id", "experiment", "group", "model", "split_strategy", "seed",
        "n_features_input", "n_features_used", "n_features_dropped", "feature_selection",
        "n_trials_budget", "n_trials_run", "best_params", "selected_threshold",
        "tuned_balanced_accuracy", "tuned_roc_auc", "tuned_pr_auc", "tuned_precision",
        "tuned_recall", "tuned_f1", "tuned_mcc", "tuned_accuracy", "tuned_specificity",
        "threshold_balanced_accuracy", "threshold_precision", "threshold_recall",
        "default_balanced_accuracy", "default_roc_auc", "delta_ba_vs_default",
        "ci_ba_0.5_lo", "ci_ba_0.5_hi", "ci_roc_auc_lo", "ci_roc_auc_hi",
        "ci_pr_auc_lo", "ci_pr_auc_hi", "ci_unit", "tune_time_seconds", "best_fit_seconds",
        "blocks", "hypothesis",
    ]
    return results[[c for c in keep if c in results.columns]]


def collect_trials(metadata: dict) -> pd.DataFrame:
    rows = []
    for experiment_id, meta in metadata.items():
        for trial in meta.get("trials", []):
            rows.append({"experiment_id": experiment_id, **trial})
    return pd.DataFrame(rows)


def stability_stage(
    cfg,
    dirs,
    store,
    df,
    y,
    scaffolds,
    specs_by_id,
    results,
    verbose: bool,
) -> pd.DataFrame:
    """Re-run the best configurations on every configured scaffold seed."""
    n_experiments = int(cfg.get_path("experiment.stability.n_experiments", 3) or 0)
    if n_experiments <= 0:
        return pd.DataFrame()
    stability_models = cfg.get_path("experiment.stability.models", ["lightgbm"])
    seeds = list(cfg.get_path("split.seeds", []))
    primary = int(cfg.get_path("split.primary_seed", 42))
    extra_seeds = [s for s in seeds if s != primary]
    if not extra_seeds:
        print("[stability] no extra seeds configured -- skipped")
        return pd.DataFrame()

    ranked = (
        results.groupby("experiment_id")["tuned_balanced_accuracy"].mean().sort_values(ascending=False)
    )
    chosen = list(ranked.index[:n_experiments])
    if "E3" not in chosen:
        chosen.append("E3")
    print(f"[stability] experiments={chosen} seeds={extra_seeds} models={stability_models}")

    rows = []
    for seed in extra_seeds:
        splits = splitting_module.make_splits(scaffolds, y, cfg, strategy="scaffold", seed=seed)
        for experiment_id in chosen:
            spec = specs_by_id.get(experiment_id)
            if spec is None:
                continue
            best_fp = "ecfp"
            if "fp:BEST" in spec.blocks:
                e0e1e2 = results[results["experiment_id"].isin(["E0", "E1", "E2"])]
                if len(e0e1e2):
                    best_fp = experiments_module.resolve_best_fingerprint(e0e1e2)
            rows_run, _, _ = experiments_module.run_configuration(
                spec, store, splits, y, scaffolds, cfg,
                model_names=list(stability_models), keep_models=False, best_fingerprint=best_fp,
                verbose=verbose,
            )
            for row in rows_run:
                row["stability_run"] = True
            rows.extend(rows_run)
            if verbose:
                print(f"[stability] seed={seed} {experiment_id} done", flush=True)
    return pd.DataFrame(rows)


def imbalance_stage(cfg, store, splits, y, scaffolds, results, specs_by_id,
                    verbose=True, best_fingerprint: str = "ecfp") -> pd.DataFrame:
    """Class weighting vs threshold tuning for the best configuration (doc section 10)."""
    if not bool(cfg.get_path("evaluation.class_imbalance.compare_class_weight", True)):
        return pd.DataFrame()
    best = results.sort_values("tuned_balanced_accuracy", ascending=False).iloc[0]
    experiment_id, model_name = best["experiment_id"], best["model"]
    spec = specs_by_id.get(experiment_id)
    if spec is None:
        return pd.DataFrame()

    blocks = spec.resolved_blocks(best_fingerprint)
    names, X = store.matrix(blocks)
    scaling = model_module.model_requires_scaling(model_name, cfg)
    pipeline = preprocessing_module.FeaturePipeline(names=names, scale=scaling)
    X_train = pipeline.fit_transform(X[splits.train], y[splits.train])
    X_val = pipeline.transform(X[splits.val])
    if spec.feature_selection:
        from herg_model import feature_selection as selection_module

        sel_idx, feature_names, _ = selection_module.select_features(
            X_train, y[splits.train], list(pipeline.kept_names), cfg
        )
        X_train, X_val = X_train[:, sel_idx], X_val[:, sel_idx]
    else:
        feature_names = list(pipeline.kept_names)

    params = json.loads(best["best_params"]) if best.get("best_params") else {}
    rows = []
    for weight in (None, "balanced"):
        trial_params = dict(params)
        if "class_weight" in model_module.default_params(model_name, cfg, 0):
            trial_params["class_weight"] = weight
        model = model_module.make_model(model_name, trial_params, cfg, int(splits.seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y[splits.train])
        scores = model_module.positive_scores(model, X_val)
        for threshold_label, threshold in (
            ("0.5", 0.5),
            ("validation-selected", select_threshold(y[splits.val], scores, "balanced_accuracy")[0]),
        ):
            metrics = compute_metrics(y[splits.val], scores, threshold)
            rows.append({
                "experiment_id": experiment_id,
                "model": model_name,
                "class_weight": str(weight),
                "threshold_source": threshold_label,
                "threshold": metrics["threshold"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "n_features": len(feature_names),
            })
        if verbose:
            print(f"[imbalance] class_weight={weight} done", flush=True)
    return pd.DataFrame(rows)


def split_reference_stage(
    cfg, store, y, scaffolds, specs_by_id, ids, best_fingerprint, verbose=True
) -> pd.DataFrame:
    """Reference-only re-run on a **random** split (the doc requires it as a contrast, not a result).

    Same experiments, same budget, same preprocessing; only the partition changes. Random splits
    share scaffolds across splits (see the data quality report), so the numbers are optimistic and
    must not be compared head-to-head with the scaffold numbers.
    """
    primary_seed = int(cfg.get_path("split.primary_seed", 42))
    splits = splitting_module.make_splits(scaffolds, y, cfg, strategy="random", seed=primary_seed)
    rows: list[dict] = []
    for experiment_id in ids:
        spec = specs_by_id.get(experiment_id)
        if spec is None:
            continue
        if verbose:
            print(f"[random-reference] {experiment_id} {spec.name}", flush=True)
        rows_run, _, _ = experiments_module.run_configuration(
            spec, store, splits, y, scaffolds, cfg,
            best_fingerprint=best_fingerprint, verbose=verbose,
        )
        for row in rows_run:
            row["reference"] = "random split (참고용, scaffold 중복 존재)"
        rows.extend(rows_run)
    return pd.DataFrame(rows)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="experiment ids to run")
    parser.add_argument("--models", nargs="*", default=None, help="override model list")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-imbalance", action="store_true")
    parser.add_argument(
        "--random-reference", action="store_true",
        help="only run the random-split reference for --only experiments (default E3)",
    )
    args = parser.parse_args()
    cfg, dirs = context(args)

    df = data_module.load_dataset(cfg)
    y = data_module.blocker_target(df, cfg)
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    structure = load_structure_table(list(df[smiles_col].astype(str)), cfg, verbose=not args.quiet)
    scaffolds = structure["scaffold"].to_numpy()

    primary_seed = int(cfg.get_path("split.primary_seed", 42))
    splits = splitting_module.make_splits(scaffolds, y, cfg, strategy="scaffold", seed=primary_seed)
    splitting_module.assert_no_leakage(scaffolds, splits)

    store = experiments_module.FeatureStore(df, cfg, verbose=not args.quiet)
    all_specs = experiments_module.experiment_definitions(cfg)
    specs_by_id = {spec.id: spec for spec in all_specs}

    if args.random_reference:
        manifest_file = artifacts_path(cfg, "experiment_manifest.json")
        best_fp = "ecfp"
        if manifest_file.exists():
            best_fp = json.loads(manifest_file.read_text(encoding="utf-8")).get("best_fingerprint", "ecfp")
        reference = split_reference_stage(
            cfg, store, y, scaffolds, specs_by_id,
            ids=args.only or ["E3"], best_fingerprint=best_fp, verbose=not args.quiet,
        )
        path = dirs["reports"] / "split_comparison_results.csv"
        reference.to_csv(path, index=False)
        print(f"[random-reference] wrote {path} ({len(reference)} rows)")
        return

    requested = args.only or [spec.id for spec in all_specs]
    models_override = args.models or None

    manifest: dict = {
        "mode": mode(cfg),
        "primary_seed": primary_seed,
        "n_trials": budget(cfg, "optimization.n_trials"),
        "timeout_seconds": budget(cfg, "optimization.timeout_seconds"),
        "tune_metric": cfg.get_path("optimization.metric"),
        "models": models_override or budget(cfg, "optimization.models"),
        "models_not_run": {
            name: f"not in optimization.models for mode={mode(cfg)}"
            for name in model_module.MODEL_NAMES
            if name not in (models_override or budget(cfg, "optimization.models") or [])
        },
        "environment": environment_info(),
        "split": {"sizes": splits.sizes(), "realised_ratios": splits.meta.get("realised_ratios")},
        "mordred": None,
        "experiments": {},
        "skipped": [],
    }

    # ---- phase 1: fingerprint comparison, needed to resolve "best fingerprint" -------------
    partial_dir = ROOT / ".cache" / "herg_model"
    partial_dir.mkdir(parents=True, exist_ok=True)
    phase1_ids = [i for i in ("E0", "E1", "E2", "E0b", "E1b", "E2b") if i in requested]
    started = time.time()
    phase1, models1, meta1 = experiments_module.run_suite(
        [specs_by_id[i] for i in phase1_ids], store, splits, y, scaffolds, cfg,
        verbose=not args.quiet, partial_path=partial_dir / "partial_phase1.csv",
    )
    best_fingerprint = experiments_module.resolve_best_fingerprint(phase1)
    print(f"[run] best fingerprint (validation BA) = {best_fingerprint}")

    phase2_ids = [i for i in requested if i not in phase1_ids]
    phase2, models2, meta2 = experiments_module.run_suite(
        [specs_by_id[i] for i in phase2_ids], store, splits, y, scaffolds, cfg,
        verbose=not args.quiet, best_fingerprint=best_fingerprint,
        partial_path=partial_dir / "partial_phase2.csv",
    )

    results = pd.concat([phase1, phase2], ignore_index=True)
    metadata = {**meta1, **meta2}
    manifest["best_fingerprint"] = best_fingerprint
    manifest["mordred"] = {k: v for k, v in store.mordred_status.items() if k != "warnings"}
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    manifest["experiments"] = {
        key: {k: v for k, v in value.items() if k != "feature_names"} for key, value in metadata.items()
    }

    compact = metrics_rows(results)
    compact.to_csv(dirs["reports"] / "experiment_results.csv", index=False)
    print(f"[run] wrote {dirs['reports'] / 'experiment_results.csv'} ({len(compact)} rows)")

    trials = collect_trials(metadata)
    if len(trials):
        trials.to_csv(dirs["reports"] / "optuna_trials.csv", index=False)
        print(f"[run] wrote optuna_trials.csv ({len(trials)} rows)")

    # ---- stability across scaffold seeds ---------------------------------------------------
    stability = pd.DataFrame()
    if not args.skip_stability:
        stability = stability_stage(
            cfg, dirs, store, df, y, scaffolds, specs_by_id, results, verbose=not args.quiet
        )
        if len(stability):
            stability.to_csv(dirs["reports"] / "stability_results.csv", index=False)
            print(f"[run] wrote stability_results.csv ({len(stability)} rows)")

    # ---- class imbalance ------------------------------------------------------------------
    imbalance = pd.DataFrame()
    if not args.skip_imbalance:
        imbalance = imbalance_stage(
            cfg, store, splits, y, scaffolds, results, specs_by_id, verbose=not args.quiet,
            best_fingerprint=best_fingerprint,
        )
        if len(imbalance):
            imbalance.to_csv(dirs["reports"] / "class_imbalance_results.csv", index=False)
            print(f"[run] wrote class_imbalance_results.csv ({len(imbalance)} rows)")

    # ---- validation curves for the figure stage -------------------------------------------
    curves = {}
    for experiment_id in ("E3", "E7", "E10", "E11"):
        if experiment_id not in specs_by_id or experiment_id not in requested:
            continue
        subset = results[results["experiment_id"] == experiment_id]
        if subset.empty:
            continue
        best_model = subset.sort_values("tuned_balanced_accuracy", ascending=False).iloc[0]["model"]
        spec = specs_by_id[experiment_id]
        blocks = spec.resolved_blocks(best_fingerprint)
        names, X = store.matrix(blocks)
        pipeline = preprocessing_module.FeaturePipeline(
            names=names, scale=model_module.model_requires_scaling(best_model, cfg)
        )
        X_train = pipeline.fit_transform(X[splits.train], y[splits.train])
        X_val = pipeline.transform(X[splits.val])
        if spec.feature_selection:
            from herg_model import feature_selection as selection_module

            sel_idx, _, _ = selection_module.select_features(
                X_train, y[splits.train], list(pipeline.kept_names), cfg
            )
            X_train, X_val = X_train[:, sel_idx], X_val[:, sel_idx]
        params = json.loads(subset.sort_values("tuned_balanced_accuracy", ascending=False).iloc[0]["best_params"])
        model = model_module.make_model(best_model, params, cfg, primary_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y[splits.train])
        scores = model_module.positive_scores(model, X_val)
        from herg_model.metrics import pr_points, roc_points

        fpr, tpr = roc_points(y[splits.val], scores)
        recall, precision = pr_points(y[splits.val], scores)
        metrics = compute_metrics(y[splits.val], scores, 0.5)
        curves[f"{experiment_id}:{best_model}"] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "recall": recall.tolist(), "precision": precision.tolist(),
            "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
        }
    if curves:
        with open(dirs["reports"] / "validation_curves.json", "w", encoding="utf-8") as handle:
            json.dump(curves, handle)

    with open(artifacts_path(cfg, "experiment_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    print(f"[run] wrote {artifacts_path(cfg, 'experiment_manifest.json')}")

    print("\n=== validation ranking (top 12 by balanced accuracy @0.5) ===")
    top = compact.sort_values("tuned_balanced_accuracy", ascending=False).head(12)
    print(top[["experiment_id", "model", "n_features_used", "tuned_balanced_accuracy",
               "tuned_roc_auc", "tuned_pr_auc", "ci_ba_0.5_lo", "ci_ba_0.5_hi"]].to_string(index=False))
    print(f"\n[run] total elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
