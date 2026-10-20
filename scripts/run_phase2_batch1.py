"""Phase 2 - Batch 1 (Experiments 1..5): Compare all 6 feature sets with LightGBM + Optuna (30 trials).

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

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase2_common import (  # noqa: E402
    LOG_PATH,
    OOF_CACHE_DIR,
    append_experiment_log,
    compute_oof_and_scaffold_metrics,
    prepare_folds_and_scaffold,
)
from herg_model.chemistry import load_structure_table  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.experiments import FeatureStore  # noqa: E402
from herg_model.predict import load_artifact  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def full_5fold_lgbm(
    params: dict,
    bundle: dict,
    y_label1: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, np.ndarray, list[int]]:
    oof_p_label1 = np.zeros(len(y_label1), dtype=float)
    best_iters = []
    for f in bundle["folds"]:
        fold = f["fold"]
        model = LGBMClassifier(
            **params,
            max_bin=63,
            force_col_wise=True,
            random_state=seed + fold,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(
            f["X_tr"],
            f["y_tr"],
            eval_set=[(f["X_va"], f["y_va"])],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        best_it = int(getattr(model, "best_iteration_", params.get("n_estimators", 700)) or params.get("n_estimators", 700))
        best_iters.append(max(50, best_it))
        p_blocker = model.predict_proba(f["X_va"])[:, 1]
        oof_p_label1[f["va_idx"]] = 1.0 - p_blocker

    oof_auc = float(roc_auc_score(y_label1, oof_p_label1))
    oof_f1_05 = float(f1_score(y_label1, (oof_p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    return oof_auc, oof_f1_05, oof_p_label1, best_iters


def run_lgbm_optuna_experiment(
    run_id: str,
    feature_set_name: str,
    notes: str,
    df_train: pd.DataFrame,
    store: FeatureStore,
    y_label1: np.ndarray,
    y_blocker: np.ndarray,
    scaffolds: np.ndarray,
    cfg,
    base_params: dict,
    n_trials: int = 30,
    mordred_top_k: int | None = None,
    seed: int = 42,
) -> dict | None:
    if (OOF_CACHE_DIR / f"{run_id}_oof.npy").exists():
        print(f">>> [{run_id}] Already completed in cache, skipping.")
        return None

    t0 = time.time()
    print(f"\n>>> [{run_id}] Preparing folds for feature_set={feature_set_name} ...", flush=True)
    bundle = prepare_folds_and_scaffold(
        feature_set_name, df_train, store, y_blocker, scaffolds, cfg, seed=seed, mordred_top_k=mordred_top_k
    )
    prep_sec = time.time() - t0
    print(
        f"    Prepared {bundle['n_kept_mean']} kept features (raw {bundle['n_raw']}) in {prep_sec:.1f}s. "
        f"Running Optuna ({n_trials} trials)...",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    default_e6 = {
        "n_estimators": int(base_params.get("n_estimators", 800)),
        "learning_rate": float(base_params.get("learning_rate", 0.03084)),
        "num_leaves": int(base_params.get("num_leaves", 56)),
        "min_child_samples": int(base_params.get("min_child_samples", 5)),
        "subsample": float(base_params.get("subsample", 0.7795)),
        "subsample_freq": 1,
        "colsample_bytree": float(base_params.get("colsample_bytree", 0.5256)),
        "reg_lambda": float(base_params.get("reg_lambda", 0.02585)),
        "reg_alpha": 1e-3,
        "class_weight": None,
    }
    study.enqueue_trial(default_e6)

    # Fast 2-fold screening inside Optuna (3,340 validation rows) with pruning after fold 0,
    # followed by full 5-fold OOF verification on the top candidates.
    screening_folds = bundle["folds"][:2]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 35, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.80),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        scores = []
        for f in screening_folds:
            fold = f["fold"]
            model = LGBMClassifier(
                **params,
                max_bin=63,
                force_col_wise=True,
                random_state=seed + fold,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(
                f["X_tr"],
                f["y_tr"],
                eval_set=[(f["X_va"], f["y_va"])],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            p_label1 = 1.0 - model.predict_proba(f["X_va"])[:, 1]
            y_va_l1 = 1 - f["y_va"]
            f1_f = float(f1_score(y_va_l1, (p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
            auc_f = float(roc_auc_score(y_va_l1, p_label1))
            scores.append(0.5 * f1_f + 0.5 * auc_f)
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Evaluate top 2 completed trials + default_e6 on full 5-fold OOF and pick the best full OOF
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    candidates = [dict(t.params, subsample_freq=1) for t in completed[:2]] + [default_e6]

    best_obj = -1.0
    best_oof_p = None
    best_params_full = None
    best_iters_list = None

    for cand in candidates:
        oof_auc, oof_f1_05, oof_p, iters = full_5fold_lgbm(cand, bundle, y_label1, seed=seed)
        obj = 0.5 * oof_f1_05 + 0.5 * oof_auc
        if obj > best_obj:
            best_obj = obj
            best_oof_p = oof_p
            best_params_full = dict(cand)
            best_iters_list = list(iters)

    # Fit on scaffold split using best parameters
    sc = bundle["scaffold"]
    sc_params = dict(best_params_full)
    sc_params["n_estimators"] = int(np.mean(best_iters_list))
    sc_params["subsample_freq"] = 1

    model_sc_tr = LGBMClassifier(**sc_params, max_bin=63, force_col_wise=True, random_state=seed, n_jobs=4, verbose=-1)
    model_sc_tr.fit(sc["X_tr"], sc["y_tr"])
    sc_val_p = 1.0 - model_sc_tr.predict_proba(sc["X_va"])[:, 1]
    sc_te_tronly_p = 1.0 - model_sc_tr.predict_proba(sc["X_te"])[:, 1]

    model_sc_trva = LGBMClassifier(**sc_params, max_bin=63, force_col_wise=True, random_state=seed, n_jobs=4, verbose=-1)
    model_sc_trva.fit(sc["X_trva"], sc["y_trva"])
    sc_te_trva_p = 1.0 - model_sc_trva.predict_proba(sc["X_te_trva"])[:, 1]

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
            "model": "lightgbm",
            "feature_set_name": feature_set_name,
            "best_params": sc_params,
            "best_iters": best_iters_list,
            "metrics": metrics,
            "elapsed_sec": elapsed,
        }, indent=2),
        encoding="utf-8",
    )

    log_row = {
        "run_id": run_id,
        "model": "lightgbm",
        "features": bundle["feat_desc"],
        "n_features_kept": bundle["n_kept_mean"],
        "cv_scheme": f"Stratified5Fold(seed={seed})+Optuna({n_trials})",
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
        f"<<< [{run_id}] DONE ({elapsed:.1f}s) | feat={bundle['n_kept_mean']} | "
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

    artifact = load_artifact(ROOT / "artifacts" / "best_model.joblib")
    base_params = dict(artifact["metadata"]["hyperparameters"])
    base_params.pop("class_weight", None)

    store = FeatureStore(df_train, cfg, verbose=False)

    experiments = [
        ("P2_02_E3_lgbm", "E3", None, "Phase 2 Batch 1 Exp 1: E3 (FCFP + 4 shipped descriptors), LGBM Optuna 30 trials"),
        ("P2_03_E7_lgbm", "E7", None, "Phase 2 Batch 1 Exp 2: E7 (FCFP + all RDKit descriptors), LGBM Optuna 30 trials"),
        ("P2_04_MORDRED_SEL_lgbm", "MORDRED_RAW_E6", 400, "Phase 2 Batch 1 Exp 3: E6 + Mordred 2D (train-only ExtraTrees top 400), LGBM Optuna 30 trials"),
        ("P2_05_E6_DRV_lgbm", "E6_DRV", None, "Phase 2 Batch 1 Exp 4: E6 + 29 domain-derived features (CAD/amine/FP density), LGBM Optuna 30 trials"),
        ("P2_06_ALL_FP_E6_DRV_lgbm", "ALL_FP_E6_DRV", None, "Phase 2 Batch 1 Exp 5: All 3 FPs (ECFP+FCFP+PTFP) + E6 + 29 domain-derived, LGBM Optuna 30 trials"),
    ]

    for run_id, feat_set, top_k, notes in experiments:
        run_lgbm_optuna_experiment(
            run_id=run_id,
            feature_set_name=feat_set,
            notes=notes,
            df_train=df_train,
            store=store,
            y_label1=y_label1,
            y_blocker=y_blocker,
            scaffolds=scaffolds,
            cfg=cfg,
            base_params=base_params,
            n_trials=30,
            mordred_top_k=top_k,
            seed=42,
        )


if __name__ == "__main__":
    main()
