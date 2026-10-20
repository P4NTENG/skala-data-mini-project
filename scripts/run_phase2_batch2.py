"""Phase 2 - Batch 2 (Experiments 6..10): 5 Model Families on E6_DRV with Optuna (30 trials/model).

Updated with PURE F1(label=1) objective in Optuna (reflecting the user's intelligence that
final challenge ranking is determined strictly by F1(label=1)).

Models evaluated on ``E6_DRV`` (1,252 features):
- Exp 6  (P2_07_E6_DRV_xgboost)    : XGBoost + Optuna 30 trials (already completed)
- Exp 7  (P2_08_E6_DRV_catboost)   : CatBoost + Optuna 30 trials (F1 objective)
- Exp 8  (P2_09_E6_DRV_hgb)        : HistGradientBoosting + Optuna 30 trials (F1 objective)
- Exp 9  (P2_10_E6_DRV_extratrees) : ExtraTrees + Optuna 30 trials (F1 objective)
- Exp 10 (P2_11_E6_DRV_rf)         : RandomForest + Optuna 30 trials (F1 objective)

STRICT RULE COMPLIANCE:
- Only uses ``data/train.csv`` (8,349 rows).
- NEVER touches ``data/predict_input_chembl_labeled.csv``.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase2_common import (  # noqa: E402
    OOF_CACHE_DIR,
    append_experiment_log,
    compute_oof_and_scaffold_metrics,
    find_best_f1_threshold,
    prepare_folds_and_scaffold,
)
from herg_model.chemistry import load_structure_table  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.experiments import FeatureStore  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def f1_objective_score(y_label1: np.ndarray, p_label1: np.ndarray) -> float:
    """Pure F1(label=1) objective: 0.55 * F1(@0.5) + 0.45 * F1(@opt_thr)."""
    f1_05 = float(f1_score(y_label1, (p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    _, f1_opt = find_best_f1_threshold(y_label1, p_label1)
    return 0.55 * f1_05 + 0.45 * f1_opt


def fit_predict_fold(
    model_family: str,
    params: dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray | None,
    seed: int,
    is_full_refit: bool = False,
) -> tuple[np.ndarray, int]:
    """Fit one model on (X_tr, y_tr) and return (p_label1 on X_va, best_iter)."""
    if model_family == "xgboost":
        from xgboost import XGBClassifier
        p = dict(params)
        n_est = int(p.pop("n_estimators", 700))
        early = None if (y_va is None or is_full_refit) else 35
        model = XGBClassifier(
            n_estimators=n_est,
            tree_method="hist",
            max_bin=63,
            eval_metric="logloss",
            early_stopping_rounds=early,
            random_state=seed,
            n_jobs=8,
            verbosity=0,
            **p,
        )
        if y_va is not None and not is_full_refit:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            best_it = int(getattr(model, "best_iteration", n_est) or n_est) + 1
        else:
            model.fit(X_tr, y_tr, verbose=False)
            best_it = n_est
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, max(50, best_it)

    if model_family == "catboost":
        from catboost import CatBoostClassifier
        p = dict(params)
        n_est = int(p.pop("iterations", 700))
        w_blocker = float(p.pop("w_blocker", 1.0))
        model = CatBoostClassifier(
            iterations=n_est,
            border_count=32,
            class_weights=[1.0, w_blocker],
            thread_count=8,
            random_seed=seed,
            verbose=False,
            **p,
        )
        if y_va is not None and not is_full_refit:
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30, verbose=False)
            best_it = int(model.get_best_iteration() or n_est)
        else:
            model.fit(X_tr, y_tr, verbose=False)
            best_it = n_est
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, max(80, best_it)

    if model_family == "hist_gradient_boosting":
        p = dict(params)
        w_blocker = float(p.pop("w_blocker", 1.0))
        sample_weight = np.where(y_tr == 1, w_blocker, 1.0)
        model = HistGradientBoostingClassifier(
            max_bins=32,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=18,
            random_state=seed,
            **p,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, int(getattr(model, "n_iter_", p.get("max_iter", 300)))

    if model_family == "extra_trees":
        p = dict(params)
        w_blocker = float(p.pop("w_blocker", 1.0))
        cw = {0: 1.0, 1: w_blocker} if p.pop("use_custom_weight", False) else p.get("class_weight", None)
        p["class_weight"] = cw
        model = ExtraTreesClassifier(
            random_state=seed,
            n_jobs=8,
            **p,
        )
        model.fit(X_tr, y_tr)
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, int(p.get("n_estimators", 500))

    if model_family == "random_forest":
        p = dict(params)
        w_blocker = float(p.pop("w_blocker", 1.0))
        cw = {0: 1.0, 1: w_blocker} if p.pop("use_custom_weight", False) else p.get("class_weight", None)
        p["class_weight"] = cw
        model = RandomForestClassifier(
            random_state=seed,
            n_jobs=8,
            **p,
        )
        model.fit(X_tr, y_tr)
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, int(p.get("n_estimators", 500))

    raise ValueError(f"Unknown model_family: {model_family}")


def suggest_model_params(trial: optuna.Trial, model_family: str) -> dict[str, Any]:
    if model_family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 0.85),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 15.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.65, 1.15),
        }

    if model_family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 450, 900, step=150),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.10, log=True),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 15.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.30, 0.75),
            "random_strength": trial.suggest_float("random_strength", 0.1, 3.0, log=True),
            "w_blocker": trial.suggest_float("w_blocker", 0.68, 1.12),
        }

    if model_family == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 200, 450, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.09, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 16, 56),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 8, 40, log=True),
            "max_features": trial.suggest_float("max_features", 0.25, 0.65),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "w_blocker": trial.suggest_float("w_blocker", 0.70, 1.10),
        }

    if model_family in {"extra_trees", "random_forest"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 22, 32, 42]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.08, 0.15]),
            "use_custom_weight": True,
            "w_blocker": trial.suggest_float("w_blocker", 0.65, 1.20),
        }

    raise ValueError(f"Unknown model_family: {model_family}")


def default_model_params(model_family: str) -> dict[str, Any]:
    if model_family == "xgboost":
        return {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "max_depth": 7,
            "min_child_weight": 2.0,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "reg_lambda": 1.0,
            "reg_alpha": 1e-3,
            "scale_pos_weight": 0.85,
        }
    if model_family == "catboost":
        return {
            "iterations": 750,
            "learning_rate": 0.055,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "rsm": 0.50,
            "random_strength": 1.0,
            "w_blocker": 0.88,
        }
    if model_family == "hist_gradient_boosting":
        return {
            "max_iter": 350,
            "learning_rate": 0.045,
            "max_leaf_nodes": 42,
            "min_samples_leaf": 12,
            "max_features": 0.45,
            "l2_regularization": 0.1,
            "w_blocker": 0.85,
        }
    if model_family in {"extra_trees", "random_forest"}:
        return {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "use_custom_weight": True,
            "w_blocker": 0.90,
        }
    raise ValueError(model_family)


def full_5fold_eval(
    model_family: str,
    params: dict[str, Any],
    bundle: dict,
    y_label1: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, np.ndarray, list[int]]:
    oof_p_label1 = np.zeros(len(y_label1), dtype=float)
    best_iters = []
    for f in bundle["folds"]:
        fold = f["fold"]
        p_label1, best_it = fit_predict_fold(
            model_family, params, f["X_tr"], f["y_tr"], f["X_va"], f["y_va"], seed=seed + fold
        )
        oof_p_label1[f["va_idx"]] = p_label1
        best_iters.append(best_it)
    oof_auc = float(roc_auc_score(y_label1, oof_p_label1))
    oof_f1_05 = float(f1_score(y_label1, (oof_p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    return oof_auc, oof_f1_05, oof_p_label1, best_iters


def run_model_optuna_experiment(
    run_id: str,
    model_family: str,
    bundle: dict,
    notes: str,
    y_label1: np.ndarray,
    n_trials: int = 30,
    seed: int = 42,
) -> dict | None:
    if (OOF_CACHE_DIR / f"{run_id}_oof.npy").exists():
        print(f">>> [{run_id}] Already completed in cache, skipping.", flush=True)
        return None

    t0 = time.time()
    print(f"\n>>> [{run_id}] Model={model_family} on {bundle['feature_set_name']} (feat={bundle['n_kept_mean']}), Optuna {n_trials} trials (F1 objective)...", flush=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    screening_folds = bundle["folds"][:1]
    trial_params_store: dict[int, dict[str, Any]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, model_family)
        trial_params_store[trial.number] = dict(params)

        screen_params = dict(params)
        if model_family in {"extra_trees", "random_forest"}:
            screen_params["n_estimators"] = 160
        elif model_family == "catboost":
            screen_params["iterations"] = min(int(screen_params["iterations"]), 350)
        elif model_family == "hist_gradient_boosting":
            screen_params["max_iter"] = min(int(screen_params["max_iter"]), 220)

        scores = []
        for f in screening_folds:
            fold = f["fold"]
            p_label1, _ = fit_predict_fold(
                model_family, screen_params, f["X_tr"], f["y_tr"], f["X_va"], f["y_va"], seed=seed + fold
            )
            y_va_l1 = 1 - f["y_va"]
            f1_obj = f1_objective_score(y_va_l1, p_label1)
            scores.append(f1_obj)
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    candidates = [trial_params_store[t.number] for t in completed[:2]] + [default_model_params(model_family)]

    best_obj = -1.0
    best_oof_p = None
    best_params_full = None
    best_iters_list = None

    for cand in candidates:
        oof_auc, oof_f1_05, oof_p, iters = full_5fold_eval(model_family, cand, bundle, y_label1, seed=seed)
        obj = f1_objective_score(y_label1, oof_p)
        if obj > best_obj:
            best_obj = obj
            best_oof_p = oof_p
            best_params_full = dict(cand)
            best_iters_list = list(iters)

    # Fit on scaffold split using best parameters
    sc = bundle["scaffold"]
    sc_params = dict(best_params_full)
    if model_family == "xgboost":
        sc_params["n_estimators"] = int(np.mean(best_iters_list))
    elif model_family == "catboost":
        sc_params["iterations"] = int(np.mean(best_iters_list))

    X_va_te = np.vstack([sc["X_va"], sc["X_te"]])
    p_va_te, _ = fit_predict_fold(
        model_family, sc_params, sc["X_tr"], sc["y_tr"], X_va_te, None, seed=seed, is_full_refit=True
    )
    n_va = len(sc["X_va"])
    sc_val_p = p_va_te[:n_va]
    sc_te_tronly_p = p_va_te[n_va:]

    sc_te_trva_p, _ = fit_predict_fold(
        model_family, sc_params, sc["X_trva"], sc["y_trva"], sc["X_te_trva"], None, seed=seed, is_full_refit=True
    )

    metrics = compute_oof_and_scaffold_metrics(
        best_oof_p, sc_val_p, sc_te_tronly_p, sc_te_trva_p, y_label1, sc["va"], sc["te"], seed=seed
    )
    elapsed = round(time.time() - t0, 2)

    OOF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_CACHE_DIR / f"{run_id}_oof.npy", best_oof_p)
    np.save(OOF_CACHE_DIR / f"{run_id}_sc_trva.npy", sc_te_trva_p)
    (OOF_CACHE_DIR / f"{run_id}_meta.json").write_text(
        json.dumps({
            "run_id": run_id,
            "model": model_family,
            "feature_set_name": bundle["feature_set_name"],
            "best_params": sc_params,
            "best_iters": best_iters_list,
            "metrics": metrics,
            "elapsed_sec": elapsed,
        }, indent=2),
        encoding="utf-8",
    )

    log_row = {
        "run_id": run_id,
        "model": model_family,
        "features": bundle["feat_desc"],
        "n_features_kept": bundle["n_kept_mean"],
        "cv_scheme": f"Stratified5Fold(seed={seed})+Optuna({n_trials},F1)",
        "oof_f1_label1_05": round(metrics["oof_f1_05"], 4),
        "oof_f1_ci95_lo": round(metrics["oof_f1_ci_lo"], 4),
        "oof_f1_ci95_hi": round(metrics["oof_f1_ci_hi"], 4),
        "oof_f1_label1_opt": round(metrics["oof_f1_opt"], 4),
        "oof_nested_opt_f1": round(metrics["oof_nested_f1"], 4),
        "oof_roc_auc": round(metrics["oof_auc"], 4),
        "oof_accuracy_05": round(metrics["oof_acc_05"], 4),
        "scaffold_f1_label1_tronly_05": round(metrics["sc_f1_tronly_05"], 4),
        "scaffold_f1_label1_tronly_opt": round(metrics["sc_f1_tronly_opt"], 4),
        "scaffold_f1_label1_trva_05": round(metrics["sc_f1_trva_05"], 4),
        "scaffold_roc_auc_tronly": round(metrics["sc_auc_tronly"], 4),
        "elapsed_sec": elapsed,
        "eval_f1_label1": "NOT_EVALUATED_YET",
        "threshold_label1_opt": round(metrics["opt_thr"], 3),
        "notes": f"{notes} | best_params={json.dumps(sc_params)}",
    }
    append_experiment_log(log_row)

    print(
        f"<<< [{run_id}] DONE ({elapsed:.1f}s) | model={model_family} | feat={bundle['n_kept_mean']} | "
        f"OOF F1(1)@0.5={metrics['oof_f1_05']:.4f} (opt={metrics['oof_f1_opt']:.4f}, nested={metrics['oof_nested_f1']:.4f}) | "
        f"OOF AUC={metrics['oof_auc']:.4f} | "
        f"Scaffold F1(1) tronly@0.5={metrics['sc_f1_tronly_05']:.4f} (opt={metrics['sc_f1_tronly_opt']:.4f}, trva@0.5={metrics['sc_f1_trva_05']:.4f}) | "
        f"Scaffold AUC={metrics['sc_auc_tronly']:.4f}",
        flush=True,
    )
    return log_row


def main() -> None:
    cfg = load_config()
    df_train = pd.read_csv(ROOT / "data" / "train.csv")
    y_label1 = df_train["label"].to_numpy(dtype=np.int8)
    y_blocker = (y_label1 == 0).astype(np.int8)

    structure = load_structure_table(list(df_train["SMILES"].astype(str)), cfg, verbose=False)
    scaffolds = structure["scaffold"].to_numpy()
    store = FeatureStore(df_train, cfg, verbose=False)

    print(">>> Preparing shared E6_DRV folds once for all 5 model families...", flush=True)
    bundle_e6_drv = prepare_folds_and_scaffold("E6_DRV", df_train, store, y_blocker, scaffolds, cfg, seed=42)

    experiments = [
        ("P2_07_E6_DRV_xgboost", "xgboost", "Phase 2 Batch 2 Exp 6: XGBoost on E6_DRV, Optuna 30 trials"),
        ("P2_08_E6_DRV_catboost", "catboost", "Phase 2 Batch 2 Exp 7: CatBoost on E6_DRV, Optuna 30 trials (F1 obj)"),
        ("P2_09_E6_DRV_hgb", "hist_gradient_boosting", "Phase 2 Batch 2 Exp 8: HistGradientBoosting on E6_DRV, Optuna 30 trials (F1 obj)"),
        ("P2_10_E6_DRV_extratrees", "extra_trees", "Phase 2 Batch 2 Exp 9: ExtraTrees on E6_DRV, Optuna 30 trials (F1 obj)"),
        ("P2_11_E6_DRV_rf", "random_forest", "Phase 2 Batch 2 Exp 10: RandomForest on E6_DRV, Optuna 30 trials (F1 obj)"),
    ]

    for run_id, model_family, notes in experiments:
        run_model_optuna_experiment(
            run_id=run_id,
            model_family=model_family,
            bundle=bundle_e6_drv,
            notes=notes,
            y_label1=y_label1,
            n_trials=30,
            seed=42,
        )


if __name__ == "__main__":
    main()
