"""Stage F -- freeze the validation-selected configuration, evaluate the test split ONCE,
store the artefact and prove that inference works on new SMILES.

Selection protocol (no test leakage):
  1. the (experiment, model) pair with the highest **validation** balanced accuracy is chosen
     from ``reports/experiment_results.csv``;
  2. a model is fitted on the training split only and its validation scores select the threshold;
  3. the same configuration is refitted on train+val and evaluated on test exactly once,
     using the threshold from step 2.

Run:  python scripts/train_final_model.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from _common import artifacts_path, base_parser, context, reports_path

from herg_model import data as data_module
from herg_model import experiments as experiments_module
from herg_model import feature_selection as selection_module
from herg_model import models as model_module
from herg_model import predict as predict_module
from herg_model import preprocessing as preprocessing_module
from herg_model import splitting as splitting_module
from herg_model.chemistry import load_structure_table
from herg_model.config import environment_info, mode, budget
from herg_model.metrics import compute_metrics, pr_points, roc_points, select_threshold


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg, dirs = context(args)
    warnings.simplefilter("ignore")

    results = pd.read_csv(dirs["reports"] / "experiment_results.csv")
    primary_seed = int(cfg.get_path("split.primary_seed", 42))
    best = results.sort_values("tuned_balanced_accuracy", ascending=False).iloc[0]
    experiment_id, model_name = str(best["experiment_id"]), str(best["model"])
    print(f"[final] selected {experiment_id} / {model_name} "
          f"(validation BA {best['tuned_balanced_accuracy']:.4f})")

    df = data_module.load_dataset(cfg)
    y = data_module.blocker_target(df, cfg)
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    structure = load_structure_table(list(df[smiles_col].astype(str)), cfg, verbose=not args.quiet)
    scaffolds = structure["scaffold"].to_numpy()

    splits = splitting_module.make_splits(scaffolds, y, cfg, strategy="scaffold", seed=primary_seed)
    specs_by_id = {spec.id: spec for spec in experiments_module.experiment_definitions(cfg)}
    spec = specs_by_id[experiment_id]
    store = experiments_module.FeatureStore(df, cfg, verbose=not args.quiet)
    manifest = json.loads(artifacts_path(cfg, "experiment_manifest.json").read_text(encoding="utf-8"))
    best_fingerprint = manifest.get("best_fingerprint", "ecfp")
    blocks = spec.resolved_blocks(best_fingerprint)
    names, X = store.matrix(blocks)

    params = json.loads(best["best_params"]) if isinstance(best.get("best_params"), str) else {}
    scaling = model_module.model_requires_scaling(model_name, cfg)

    # ---- step 2: train-only fit -> validation threshold -----------------------------------
    pipeline_train = preprocessing_module.FeaturePipeline(names=names, scale=scaling)
    X_train = pipeline_train.fit_transform(X[splits.train], y[splits.train])
    X_val = pipeline_train.transform(X[splits.val])
    if spec.feature_selection:
        sel_idx, selected, _ = selection_module.select_features(
            X_train, y[splits.train], list(pipeline_train.kept_names), cfg, seed=primary_seed
        )
        X_train, X_val = X_train[:, sel_idx], X_val[:, sel_idx]
        feature_names = list(selected)
    else:
        feature_names = list(pipeline_train.kept_names)

    threshold_model = model_module.make_model(model_name, params, cfg, primary_seed)
    threshold_model.fit(X_train, y[splits.train])
    val_scores = model_module.positive_scores(threshold_model, X_val)
    threshold, threshold_value = select_threshold(y[splits.val], val_scores, "balanced_accuracy")
    val_metrics = compute_metrics(y[splits.val], val_scores, threshold)
    print(f"[final] threshold {threshold:.4f} (validation BA {threshold_value:.4f})")

    # ---- step 3: refit on train+val, evaluate test once -----------------------------------
    train_val = np.concatenate([splits.train, splits.val])
    pipeline = preprocessing_module.FeaturePipeline(names=names, scale=scaling)
    X_train_val = pipeline.fit_transform(X[train_val], y[train_val])
    X_test = pipeline.transform(X[splits.test])
    selection_meta = {"method": "none"}
    selected_indices = None
    if spec.feature_selection:
        selected_indices, selected, selection_meta = selection_module.select_features(
            X_train_val, y[train_val], list(pipeline.kept_names), cfg, seed=primary_seed
        )
        X_train_val, X_test = X_train_val[:, selected_indices], X_test[:, selected_indices]
        feature_names = list(selected)
    else:
        feature_names = list(pipeline.kept_names)

    model = model_module.make_model(model_name, params, cfg, primary_seed)
    model.fit(X_train_val, y[train_val])
    test_scores = model_module.positive_scores(model, X_test)
    test_metrics = compute_metrics(y[splits.test], test_scores, threshold)
    test_metrics_05 = compute_metrics(y[splits.test], test_scores, 0.5)
    print(f"[final] TEST balanced accuracy {test_metrics['balanced_accuracy']:.4f} "
          f"roc_auc {test_metrics['roc_auc']:.4f} pr_auc {test_metrics['pr_auc']:.4f}")

    metrics_path = dirs["reports"] / "final_model_test_metrics.csv"
    pd.DataFrame([{**test_metrics, "experiment_id": experiment_id, "model": model_name,
                   "threshold_source": "validation", "refit": "train+val"}]).to_csv(metrics_path, index=False)

    payload = {
        "experiment_id": experiment_id,
        "model": model_name,
        "threshold": float(threshold),
        "threshold_source": "validation split of a train-only fit",
        "refit": "train+val",
        "metrics": test_metrics,
        "metrics_at_0.5": test_metrics_05,
        "validation_metrics": val_metrics,
        "blocks": blocks,
        "n_features": len(feature_names),
        "feature_selection": selection_meta,
        "split_sizes": splits.sizes(),
        "best_params": params,
        "budget_mode": mode(cfg),
        "n_trials": budget(cfg, "optimization.n_trials"),
    }
    with open(artifacts_path(cfg, "final_model_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    # ---- artefacts ------------------------------------------------------------------------
    training_info = {
        "n_rows_total": int(len(df)),
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "n_test": int(len(splits.test)),
        "refit_on": "train+val",
        "split_strategy": "scaffold (Bemis-Murcko)",
        "primary_seed": primary_seed,
        "n_features": len(feature_names),
        "feature_selection": selection_meta.get("method"),
        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
    }
    metadata = predict_module.build_metadata(
        cfg, training_info, float(threshold), blocks, feature_names,
        extra={
            "experiment_id": experiment_id,
            "model": model_name,
            "hyperparameters": params,
            "scaling": scaling,
            "test_metrics": test_metrics,
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        },
    )
    artifact_path = predict_module.save_artifact(
        artifacts_path(cfg, "best_model.joblib"), model, pipeline, metadata,
        selected_indices=selected_indices,
    )
    with open(artifacts_path(cfg, "model_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f"[final] wrote {artifact_path} and model_metadata.json")

    # ---- test-set error analysis (contains SMILES -> git-ignored) -------------------------
    meta = pd.DataFrame({
        "row_id": splits.test,
        "SMILES": df[smiles_col].astype(str).to_numpy()[splits.test],
        "label": df[str(cfg.get_path("data.label_col", "label"))].to_numpy()[splits.test],
        "y_blocker": y[splits.test],
        "scaffold": scaffolds[splits.test],
        "n_heavy_atoms": structure["n_heavy_atoms"].to_numpy()[splits.test],
    })
    from herg_model.explainability import misclassification_table

    errors = misclassification_table(y[splits.test], test_scores, threshold, meta)
    errors.to_csv(dirs["reports"] / "error_analysis_test.csv", index=False)
    errors[errors["error_type"] != "correct"].to_csv(dirs["reports"] / "misclassified_test.csv", index=False)
    print(f"[final] test errors: {(errors['error_type'] != 'correct').sum()} / {len(errors)}")

    with open(dirs["reports"] / "final_test_curves.json", "w", encoding="utf-8") as handle:
        fpr, tpr = roc_points(y[splits.test], test_scores)
        recall, precision = pr_points(y[splits.test], test_scores)
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "recall": recall.tolist(),
                   "precision": precision.tolist(), "roc_auc": test_metrics["roc_auc"],
                   "pr_auc": test_metrics["pr_auc"]}, handle)

    # ---- inference smoke test -------------------------------------------------------------
    probe = [
        "CN1CCC[C@H]1c2cccnc2",              # in the dataset
        "CC(=O)Oc1ccccc1C(=O)O",             # aspirin, expected non-blocker
        "O=C(Nc1ccccc1)c1ccccc1",            # benzanilide
        "CN(C)CCC1=CNc2ccccc12",             # sumatriptan-like
    ]
    artifact = predict_module.load_artifact(artifact_path)
    prediction = predict_module.predict_smiles(probe, artifact, cfg)
    prediction.to_csv(dirs["reports"] / "inference_examples.csv", index=False)
    print("\n=== inference on new SMILES ===")
    print(prediction.to_string(index=False))
    print(f"\n[final] environment: {json.dumps(environment_info()['packages'], default=str)}")


if __name__ == "__main__":
    main()
