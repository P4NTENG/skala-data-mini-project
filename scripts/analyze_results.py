"""Stage E/F -- ablation, importance, error analysis, figures and the final report.

Reads the artefacts produced by ``run_experiments.py`` (+ optionally ``train_final_model.py``)
and writes:

* ``reports/ablation_results.csv``      -- descriptor-group added effect vs the fingerprint baseline
* ``reports/feature_importance.csv``    -- permutation / SHAP importance of the best configuration
* ``reports/feature_importance_by_group.csv``
* ``reports/error_analysis.csv``        -- every validation molecule with its prediction (SMILES -> git-ignored)
* ``reports/scaffold_performance.csv``
* ``reports/descriptor_class_comparison.csv``
* ``reports/figures/*.png``
* ``reports/final_report.md``

Run:  python scripts/analyze_results.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from _common import artifacts_path, base_parser, context, figures_dir, reports_path

from herg_model import data as data_module
from herg_model import experiments as experiments_module
from herg_model import explainability as explainability_module
from herg_model import feature_selection as selection_module
from herg_model import figures as figures_module
from herg_model import models as model_module
from herg_model import preprocessing as preprocessing_module
from herg_model import splitting as splitting_module
from herg_model.chemistry import load_structure_table
from herg_model.config import budget, environment_info, mode
from herg_model.metrics import compute_metrics, select_threshold
from herg_model.reporting import write_final_report

warnings.simplefilter("ignore")


def build_ablation(results: pd.DataFrame, baseline_id: str = "E3") -> pd.DataFrame:
    """Descriptor-group added effect, measured on the identical split/budget at threshold 0.5."""
    baseline_rows = results[(results["experiment_id"] == baseline_id)]
    if baseline_rows.empty:
        return pd.DataFrame()
    rows = []
    for baseline_model, baseline_group in baseline_rows.groupby("model"):
        baseline_ba = float(baseline_group["tuned_balanced_accuracy"].max())
        baseline_auc = float(baseline_group["tuned_roc_auc"].max())
        for _, row in results[results["group"] == "ablation"].iterrows():
            if row["model"] != baseline_model:
                continue
            rows.append({
                "experiment_id": row["experiment_id"],
                "label": row["experiment"].replace("ablation: + ", "E3 + "),
                "model": row["model"],
                "n_features_used": int(row["n_features_used"]),
                "baseline_balanced_accuracy": baseline_ba,
                "ablated_balanced_accuracy": float(row["tuned_balanced_accuracy"]),
                "delta_balanced_accuracy": float(row["tuned_balanced_accuracy"]) - baseline_ba,
                "baseline_roc_auc": baseline_auc,
                "ablated_roc_auc": float(row["tuned_roc_auc"]),
                "delta_roc_auc": float(row["tuned_roc_auc"]) - baseline_auc,
                "ci_lo": row.get("ci_ba_0.5_lo"),
                "ci_hi": row.get("ci_ba_0.5_hi"),
            })
    return pd.DataFrame(rows).sort_values("delta_balanced_accuracy", ascending=False).reset_index(drop=True)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg, dirs = context(args)
    reports = dirs["reports"]
    figures = figures_dir(cfg)

    results = pd.read_csv(reports / "experiment_results.csv")
    stability = pd.read_csv(reports / "stability_results.csv") if (reports / "stability_results.csv").exists() else pd.DataFrame()
    if len(stability):
        # the primary seed is already in experiment_results.csv -- include it so the
        # stability summary covers all three scaffold seeds (42, 101, 2024)
        extra_ids = set(stability["experiment_id"])
        extra_models = set(stability["model"])
        primary = results[
            results["experiment_id"].isin(extra_ids) & results["model"].isin(extra_models)
        ].copy()
        primary["stability_run"] = False
        stability = pd.concat([primary, stability], ignore_index=True)
    imbalance = pd.read_csv(reports / "class_imbalance_results.csv") if (reports / "class_imbalance_results.csv").exists() else pd.DataFrame()
    split_reference = (
        pd.read_csv(reports / "split_comparison_results.csv")
        if (reports / "split_comparison_results.csv").exists() else pd.DataFrame()
    )
    final_metrics = None
    if (artifacts_path(cfg, "final_model_metrics.json")).exists():
        final_metrics = json.loads(artifacts_path(cfg, "final_model_metrics.json").read_text(encoding="utf-8"))

    # ---- ablation -------------------------------------------------------------------------
    ablation = build_ablation(results)
    if len(ablation):
        ablation.to_csv(reports / "ablation_results.csv", index=False)
        print(f"[analyze] wrote ablation_results.csv ({len(ablation)} rows)")

    # ---- best configuration (validation only) ---------------------------------------------
    best = results.sort_values("tuned_balanced_accuracy", ascending=False).iloc[0]
    experiment_id, model_name = str(best["experiment_id"]), str(best["model"])
    print(f"[analyze] best validation configuration: {experiment_id} / {model_name} "
          f"(BA {best['tuned_balanced_accuracy']:.4f})")

    df = data_module.load_dataset(cfg)
    y = data_module.blocker_target(df, cfg)
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    structure = load_structure_table(list(df[smiles_col].astype(str)), cfg, verbose=False)
    scaffolds = structure["scaffold"].to_numpy()
    primary_seed = int(cfg.get_path("split.primary_seed", 42))
    splits = splitting_module.make_splits(scaffolds, y, cfg, strategy="scaffold", seed=primary_seed)

    specs_by_id = {spec.id: spec for spec in experiments_module.experiment_definitions(cfg)}
    spec = specs_by_id[experiment_id]
    manifest = json.loads(artifacts_path(cfg, "experiment_manifest.json").read_text(encoding="utf-8"))
    best_fingerprint = manifest.get("best_fingerprint", "ecfp")
    blocks = spec.resolved_blocks(best_fingerprint)

    store = experiments_module.FeatureStore(df, cfg, verbose=False)
    names, X = store.matrix(blocks)
    scaling = model_module.model_requires_scaling(model_name, cfg)
    pipeline = preprocessing_module.FeaturePipeline(names=names, scale=scaling)
    X_train = pipeline.fit_transform(X[splits.train], y[splits.train])
    X_val = pipeline.transform(X[splits.val])
    if spec.feature_selection:
        sel_idx, feature_names, _ = selection_module.select_features(
            X_train, y[splits.train], list(pipeline.kept_names), cfg, seed=primary_seed
        )
        X_train, X_val = X_train[:, sel_idx], X_val[:, sel_idx]
    else:
        feature_names = list(pipeline.kept_names)

    params = json.loads(best["best_params"]) if isinstance(best.get("best_params"), str) else {}
    model = model_module.make_model(model_name, params, cfg, primary_seed)
    model.fit(X_train, y[splits.train])
    val_scores = model_module.positive_scores(model, X_val)
    threshold, _ = select_threshold(y[splits.val], val_scores, str(cfg.get_path("evaluation.threshold_metric", "balanced_accuracy")))
    val_metrics = compute_metrics(y[splits.val], val_scores, threshold)
    print(f"[analyze] validation metrics at selected threshold: BA {val_metrics['balanced_accuracy']:.4f}, "
          f"ROC-AUC {val_metrics['roc_auc']:.4f}")

    group_of = store.group_of(names)
    importance = explainability_module.permutation_importance_table(
        model, X_val, y[splits.val], feature_names, n_repeats=3, seed=primary_seed
    )
    importance_path = reports / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    shap_table, shap_status = explainability_module.shap_importance_table(
        model, X_val, feature_names, seed=primary_seed
    )
    if shap_table is not None:
        shap_table.to_csv(reports / "feature_importance_shap.csv", index=False)
        combined = pd.concat([importance, shap_table], ignore_index=True)
    else:
        combined = importance
        print(f"[analyze] SHAP unavailable: {shap_status['reason']}")
    combined.to_csv(reports / "feature_importance_all.csv", index=False)

    group_importance = explainability_module.group_importance_table(importance, group_of)
    group_importance.to_csv(reports / "feature_importance_by_group.csv", index=False)

    redundancy = explainability_module.top_feature_redundancy(importance, X_val, feature_names, top_k=25)

    meta = pd.DataFrame({
        "row_id": splits.val,
        "SMILES": df[smiles_col].astype(str).to_numpy()[splits.val],
        "label": df[str(cfg.get_path("data.label_col", "label"))].to_numpy()[splits.val],
        "y_blocker": y[splits.val],
        "scaffold": scaffolds[splits.val],
        "n_heavy_atoms": structure["n_heavy_atoms"].to_numpy()[splits.val],
        "sa_score": structure["sa_score"].to_numpy()[splits.val],
    })
    top_features = list(importance["feature"].head(12))
    errors = explainability_module.misclassification_table(
        y[splits.val], val_scores, threshold, meta,
        extra_features=None, top_features=top_features,
    )
    errors.to_csv(reports / "error_analysis.csv", index=False)
    errors[errors["error_type"] != "correct"].to_csv(reports / "misclassified_cases.csv", index=False)

    # Scaffold split makes every validation molecule its own Bemis-Murcko scaffold (by
    # construction), so the per-scaffold view uses the element-stripped generic scaffold
    # instead: it groups chemotype families and is the only level with repeat members.
    generic_scaffolds = structure["generic_scaffold"].to_numpy()
    scaffold_table = explainability_module.scaffold_performance(
        y[splits.val], val_scores, generic_scaffolds[splits.val], threshold, min_size=3
    )
    scaffold_table.to_csv(reports / "scaffold_performance.csv", index=False)

    rdkit_frame = store._rdkit_frame()
    comparison = explainability_module.class_descriptor_comparison(
        rdkit_frame[[c for c in top_features if c in rdkit_frame.columns]],
        y,
        [c for c in top_features if c in rdkit_frame.columns],
    )
    comparison.to_csv(reports / "descriptor_class_comparison.csv", index=False)

    # ---- figures --------------------------------------------------------------------------
    figures_module.plot_experiment_comparison(results, figures / "experiment_comparison.png")
    if len(ablation):
        figures_module.plot_ablation(ablation, figures / "ablation.png")
    figures_module.plot_feature_importance(importance, figures / "feature_importance.png")
    figures_module.plot_group_importance(group_importance, figures / "feature_importance_by_group.png")
    figures_module.plot_scaffold_performance(scaffold_table, figures / "scaffold_performance.png")
    if len(stability):
        figures_module.plot_seed_stability(stability, figures / "seed_stability.png")

    confusion = {f"{experiment_id}:{model_name}": np.array([
        [val_metrics["tn"], val_metrics["fp"]],
        [val_metrics["fn"], val_metrics["tp"]],
    ])}
    if (reports / "validation_curves.json").exists():
        curves = json.loads((reports / "validation_curves.json").read_text(encoding="utf-8"))
        figures_module.plot_roc_pr_curves(curves, figures / "roc_pr_validation.png",
                                          title="Validation ROC / PR (best model per experiment)")
    if (reports / "final_test_curves.json").exists():
        payload = json.loads((reports / "final_test_curves.json").read_text(encoding="utf-8"))
        figures_module.plot_roc_pr_curves({"final model (test)": payload}, figures / "roc_pr_test.png",
                                          title="Test ROC / PR (single frozen evaluation)")
        if final_metrics:
            m = final_metrics["metrics"]
            confusion["final model (test)"] = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    figures_module.plot_confusion_matrices(confusion, figures / "confusion_matrices.png")

    if (reports / "data_quality_splits.csv").exists():
        split_frame = results if split_reference.empty else pd.concat([results, split_reference], ignore_index=True)
        figures_module.plot_split_comparison(split_frame, figures / "split_comparison.png")

    # ---- final report ---------------------------------------------------------------------
    specs = [specs_by_id[i] for i in sorted(specs_by_id)]
    spec_table = pd.DataFrame([{
        "id": s.id, "name": s.name, "description": s.description, "group": s.group,
        "blocks": " + ".join(s.resolved_blocks(best_fingerprint)),
    } for s in specs])

    error_summary = (
        errors.groupby("error_type", as_index=False)
        .agg(n=("SMILES", "size"), mean_probability=("blocker_probability", "mean"))
    )
    scaffold_errors = scaffold_table[scaffold_table["n_errors"] > 0].sort_values(
        ["n_errors", "error_rate"], ascending=False
    )

    bootstrap_cfg = cfg.get_path("evaluation.bootstrap", {}) or {}
    ctx = {
        "config_path": cfg["_config_path"],
        "mode": mode(cfg),
        "primary_seed": primary_seed,
        "n_trials": budget(cfg, "optimization.n_trials"),
        "tune_metric": cfg.get_path("optimization.metric"),
        "bootstrap": {"n_resamples": bootstrap_cfg.get("n_resamples", 1000),
                      "unit": bootstrap_cfg.get("unit", "scaffold")},
        "data_summary": data_module.dataset_summary(df, cfg),
        "split_table": pd.read_csv(reports / "data_quality_splits.csv"),
        "leakage": splitting_module.check_leakage(scaffolds, splits),
        "spec_table": spec_table,
        "results": results,
        "ablation": ablation if len(ablation) else pd.DataFrame(columns=["label", "delta_balanced_accuracy"]),
        "stability": stability,
        "best_fingerprint": best_fingerprint,
        "importance_head": importance.head(25),
        "group_importance": group_importance,
        "error_summary": error_summary,
        "scaffold_errors": scaffold_errors,
        "imbalance": imbalance,
        "split_reference": split_reference,
        "final": None,
        "caveats": [],
    }
    if final_metrics:
        ctx["final"] = {
            "experiment_id": final_metrics["experiment_id"],
            "model": final_metrics["model"],
            "threshold": final_metrics["threshold"],
            "metrics": final_metrics["metrics"],
            "metrics_at_0.5": final_metrics["metrics_at_0.5"],
            "refit": final_metrics["refit"],
        }
    else:
        ctx["caveats"].append(
            "test split 평가(`scripts/train_final_model.py`)가 아직 실행되지 않아 test 성능이 보고서에 없다."
        )
    if shap_table is None:
        ctx["caveats"].append(f"SHAP 미실행: {shap_status['reason']}")
    ctx["caveats"].append(
        "ablation과 모델 비교는 기본 임계값 0.5의 validation balanced accuracy 기준이다."
    )
    quick_models = budget(cfg, "optimization.models") or []
    all_models = list(model_module.MODEL_NAMES)
    not_run = [name for name in all_models if name not in quick_models]
    if not_run:
        ctx["caveats"].append(
            f"이 실행은 mode={mode(cfg)}으로 모델 {len(quick_models)}종({', '.join(quick_models)})만 "
            f"탐색했다. {', '.join(not_run)}는 실행하지 않았으므로 '모델 계열 간 비교'는 이 부분집합 "
            f"안에서만 유효하다."
        )
    if final_metrics and final_metrics.get("n_features") != int(best["n_features_used"]):
        ctx["caveats"].append(
            f"최종 모델은 train+val로 전처리를 다시 적합하므로, 학습 split 기준 실험 "
            f"({int(best['n_features_used']):,}개)보다 feature가 "
            f"{final_metrics['n_features'] - int(best['n_features_used']):+d}개 다르다"
            f"(최종 {final_metrics['n_features']:,}개). 상수/중복 열 판정이 재적합 시점에 달라지기 때문이다."
        )
    if len(redundancy):
        ctx["caveats"].append(
            f"상위 25개 중요 feature 중 |r| >= 0.8 쌍이 {len(redundancy)}개 있다 -- 중요도가 중복 계산될 수 있다."
        )

    report_path = reports / "final_report.md"
    write_final_report(report_path, ctx)
    print(f"[analyze] wrote {report_path}")

    summary = {
        "best_configuration": f"{experiment_id}/{model_name}",
        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
        "validation_roc_auc": val_metrics["roc_auc"],
        "n_validation_errors": int((errors["error_type"] != "correct").sum()),
        "n_validation": int(len(errors)),
        "n_scaffolds_with_errors": int(len(scaffold_errors)),
        "shap_available": shap_table is not None,
        "test_evaluated": final_metrics is not None,
        "environment": environment_info()["packages"],
    }
    with open(reports / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "environment"}, indent=2, default=str))


if __name__ == "__main__":
    main()
