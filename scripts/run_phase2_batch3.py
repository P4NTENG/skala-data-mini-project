"""Phase 2 - Batch 3 (Experiments 11..15): Pure F1(label=1) Optuna optimization across top feature sets & models.

Experiments:
- Exp 11 (P2_12_E6_DRV_lgbm_f1)            : LightGBM on E6_DRV (1,252 feats), Optuna 30 trials (Pure F1 + continuous scale_pos_weight)
- Exp 12 (P2_13_ALL_FP_E6_DRV_lgbm_f1)     : LightGBM on ALL_FP_E6_DRV (3,189 feats), Optuna 30 trials (Pure F1 + continuous scale_pos_weight)
- Exp 13 (P2_14_ALL_FP_E6_DRV_xgboost_f1)  : XGBoost on ALL_FP_E6_DRV (3,189 feats), Optuna 30 trials (Pure F1 + continuous scale_pos_weight)
- Exp 14 (P2_15_ALL_FP_E6_DRV_et_f1)       : ExtraTrees on ALL_FP_E6_DRV (3,189 feats), Optuna 30 trials (Pure F1 + continuous w_blocker)
- Exp 15 (P2_16_ALL_FP_E7_DRV_lgbm_f1)     : LightGBM on ALL_FP_E7_DRV (3,215 feats), Optuna 30 trials (Pure F1 + continuous scale_pos_weight)

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

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

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
    """Pure F1(label=1) objective: 0.65 * F1(@0.5) + 0.35 * F1(@opt_thr)."""
    f1_05 = float(f1_score(y_label1, (p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    _, f1_opt = find_best_f1_threshold(y_label1, p_label1)
    return 0.65 * f1_05 + 0.35 * f1_opt


def fit_predict_batch3(
    model_family: str,
    params: dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray | None,
    seed: int,
    is_full_refit: bool = False,
) -> tuple[np.ndarray, int]:
    if model_family == "lightgbm":
        p = dict(params)
        n_est = int(p.pop("n_estimators", 700))
        model = LGBMClassifier(
            n_estimators=n_est,
            max_bin=63,
            force_col_wise=True,
            subsample_freq=1,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
            **p,
        )
        if y_va is not None and not is_full_refit:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
            )
            best_it = int(getattr(model, "best_iteration_", n_est) or n_est)
        else:
            model.fit(X_tr, y_tr)
            best_it = n_est
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, max(50, best_it)

    if model_family == "xgboost":
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

    if model_family == "extra_trees":
        p = dict(params)
        w_blocker = float(p.pop("w_blocker", 0.85))
        p["class_weight"] = {0: 1.0, 1: w_blocker}
        model = ExtraTreesClassifier(
            random_state=seed,
            n_jobs=8,
            **p,
        )
        model.fit(X_tr, y_tr)
        p_blocker = model.predict_proba(X_va)[:, 1]
        return 1.0 - p_blocker, int(p.get("n_estimators", 450))

    raise ValueError(model_family)


def suggest_batch3_params(trial: optuna.Trial, model_family: str) -> dict[str, Any]:
    if model_family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.075, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 28, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 4, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.85),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.62, 1.05),
        }

    if model_family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.075, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.80),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 15.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.62, 1.05),
        }

    if model_family == "extra_trees":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [300, 400, 450, 500]),
            "max_depth": trial.suggest_categorical("max_depth", [None, 24, 32, 42]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.06, 0.08, 0.12]),
            "w_blocker": trial.suggest_float("w_blocker", 0.65, 1.05),
        }

    raise ValueError(model_family)


def default_batch3_params(model_family: str) -> dict[str, Any]:
    if model_family == "lightgbm":
        return {
            "n_estimators": 700,
            "learning_rate": 0.031,
            "num_leaves": 56,
            "min_child_samples": 6,
            "subsample": 0.80,
            "colsample_bytree": 0.55,
            "reg_lambda": 0.05,
            "reg_alpha": 0.005,
            "scale_pos_weight": 0.80,
        }
    if model_family == "xgboost":
        return {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "max_depth": 7,
            "min_child_weight": 2.0,
            "subsample": 0.80,
            "colsample_bytree": 0.45,
            "reg_lambda": 1.5,
            "reg_alpha": 0.01,
            "scale_pos_weight": 0.78,
        }
    if model_family == "extra_trees":
        return {
            "n_estimators": 450,
            "max_depth": 32,
            "min_samples_leaf": 2,
            "min_samples_split": 5,
            "max_features": 0.08,
            "w_blocker": 0.82,
        }
    raise ValueError(model_family)


def full_5fold_batch3(
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
        p_label1, best_it = fit_predict_batch3(
            model_family, params, f["X_tr"], f["y_tr"], f["X_va"], f["y_va"], seed=seed + fold
        )
        oof_p_label1[f["va_idx"]] = p_label1
        best_iters.append(best_it)
    oof_auc = float(roc_auc_score(y_label1, oof_p_label1))
    oof_f1_05 = float(f1_score(y_label1, (oof_p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    return oof_auc, oof_f1_05, oof_p_label1, best_iters


def run_batch3_experiment(
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
    print(
        f"\n>>> [{run_id}] Model={model_family} on {bundle['feature_set_name']} (feat={bundle['n_kept_mean']}), "
        f"Optuna {n_trials} trials (Pure F1 objective)...",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(default_batch3_params(model_family))

    screening_folds = bundle["folds"][:1]
    trial_params_store: dict[int, dict[str, Any]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_batch3_params(trial, model_family)
        trial_params_store[trial.number] = dict(params)
        screen_params = dict(params)
        if model_family == "extra_trees":
            screen_params["n_estimators"] = 160

        f = screening_folds[0]
        p_label1, _ = fit_predict_batch3(
            model_family, screen_params, f["X_tr"], f["y_tr"], f["X_va"], f["y_va"], seed=seed
        )
        return f1_objective_score(1 - f["y_va"], p_label1)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    candidates = [trial_params_store[t.number] for t in completed[:2]] + [default_batch3_params(model_family)]

    best_obj = -1.0
    best_oof_p = None
    best_params_full = None
    best_iters_list = None

    for cand in candidates:
        oof_auc, oof_f1_05, oof_p, iters = full_5fold_batch3(model_family, cand, bundle, y_label1, seed=seed)
        obj = f1_objective_score(y_label1, oof_p)
        if obj > best_obj:
            best_obj = obj
            best_oof_p = oof_p
            best_params_full = dict(cand)
            best_iters_list = list(iters)

    # Fit on scaffold split using best parameters
    sc = bundle["scaffold"]
    sc_params = dict(best_params_full)
    if model_family in {"lightgbm", "xgboost"}:
        sc_params["n_estimators"] = int(np.mean(best_iters_list))

    X_va_te = np.vstack([sc["X_va"], sc["X_te"]])
    p_va_te, _ = fit_predict_batch3(
        model_family, sc_params, sc["X_tr"], sc["y_tr"], X_va_te, None, seed=seed, is_full_refit=True
    )
    n_va = len(sc["X_va"])
    sc_val_p = p_va_te[:n_va]
    sc_te_tronly_p = p_va_te[n_va:]

    sc_te_trva_p, _ = fit_predict_batch3(
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

    print(">>> Preparing ALL_FP_E6_DRV and ALL_FP_E7_DRV bundles...", flush=True)
    bundle_all_e6_drv = prepare_folds_and_scaffold("ALL_FP_E6_DRV", df_train, store, y_blocker, scaffolds, cfg, seed=42)
    bundle_all_e7_drv = prepare_folds_and_scaffold("ALL_FP_E7_DRV", df_train, store, y_blocker, scaffolds, cfg, seed=42)

    experiments = [
        ("P2_15_ALL_FP_E6_DRV_et_f1", "extra_trees", bundle_all_e6_drv, "Phase 2 Batch 3 Exp 14: ExtraTrees on ALL_FP_E6_DRV, Optuna 30 trials (Pure F1 + w_blocker)"),
        ("P2_16_ALL_FP_E7_DRV_lgbm_f1", "lightgbm", bundle_all_e7_drv, "Phase 2 Batch 3 Exp 15: LightGBM on ALL_FP_E7_DRV, Optuna 30 trials (Pure F1 + scale_pos_weight)"),
    ]

    for run_id, model_family, bundle, notes in experiments:
        run_batch3_experiment(
            run_id=run_id,
            model_family=model_family,
            bundle=bundle,
            notes=notes,
            y_label1=y_label1,
            n_trials=30,
            seed=42,
        )


if __name__ == "__main__":
    main()
