"""Phase 3-5: Train Top 1, 2, 3 models on train.csv, generate predict_output_김인성_v2.csv, and evaluate ONCE.

STRICT RULE COMPLIANCE:
- All hyper-parameters, ensemble weights, and thresholds are FIXED from train.csv OOF BEFORE touching predict_input.
- ``data/predict_input_chembl_labeled.csv`` is ONLY used at the very end via ``eval_challenge.py`` to report scores.
- Preserves existing ``predict_output_김인성.csv``, ``artifacts/best_model.joblib``, and ``reports/final_report.md``.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase2_common import (  # noqa: E402
    OOF_CACHE_DIR,
    append_experiment_log,
    build_feature_matrix,
    fast_fit_transform_clean,
    find_best_f1_threshold,
)
from eval_challenge import compute_subset_metrics, load_probabilities_from_artifact, print_report  # noqa: E402
from herg_model.chemistry import load_structure_table  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.experiments import FeatureStore  # noqa: E402
from herg_model.splitting import make_splits  # noqa: E402

warnings.filterwarnings("ignore")


def get_top_specs() -> dict[str, dict]:
    """Load the exact frozen best_params discovered on train.csv OOF in Phase 2."""
    specs = {}
    for run_id in (
        "P2_15_ALL_FP_E6_DRV_et_f1",
        "P2_16_ALL_FP_E7_DRV_lgbm_f1",
        "P2_13_ALL_FP_E6_DRV_lgbm_f1",
        "P2_12_E6_DRV_lgbm_f1",
        "P2_14_ALL_FP_E6_DRV_xgboost_f1",
        "P2_08_E6_DRV_catboost",
    ):
        meta_path = OOF_CACHE_DIR / f"{run_id}_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        specs[run_id] = meta
    return specs


def fit_single_model(model_family: str, params: dict, X_tr: np.ndarray, y_tr: np.ndarray, seed: int):
    p = dict(params)
    if model_family == "lightgbm":
        model = LGBMClassifier(
            max_bin=63,
            force_col_wise=True,
            subsample_freq=1,
            random_state=seed,
            n_jobs=8,
            verbose=-1,
            **p,
        )
        model.fit(X_tr, y_tr)
        return model

    if model_family == "xgboost":
        model = XGBClassifier(
            tree_method="hist",
            max_bin=63,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=8,
            verbosity=0,
            **p,
        )
        model.fit(X_tr, y_tr, verbose=False)
        return model

    if model_family == "extra_trees":
        w_blocker = float(p.pop("w_blocker", 0.82))
        p.pop("use_custom_weight", None)
        p["class_weight"] = {0: 1.0, 1: w_blocker}
        model = ExtraTreesClassifier(
            random_state=seed,
            n_jobs=8,
            **p,
        )
        model.fit(X_tr, y_tr)
        return model

    raise ValueError(model_family)


def predict_single_component_on_test(
    spec: dict,
    X_train_raw: np.ndarray,
    y_blocker: np.ndarray,
    X_pred_raw: np.ndarray,
    seeds: tuple[int, ...] = (42, 101, 2024),
) -> tuple[np.ndarray, list]:
    """Fit full-train multi-seed models + 5-fold cross-fitted models and return blended P(label=1) on predict_input."""
    model_family = spec["model"]
    params = spec["best_params"]

    # Clean on full train.csv ONLY, transform predict_input
    X_tr_clean, [X_pred_clean], keep_idx = fast_fit_transform_clean(
        X_train_raw, [X_pred_raw], missing_rate_limit=1.0, scale=False
    )

    # 1) Multi-seed full-train refit (uses 100% of 8,349 training molecules)
    p_full_seeds = []
    saved_models = []
    for s in seeds:
        m = fit_single_model(model_family, params, X_tr_clean, y_blocker, seed=s)
        p_blocker = m.predict_proba(X_pred_clean)[:, 1]
        p_full_seeds.append(1.0 - p_blocker)
        if s == seeds[0]:
            saved_models.append((m, keep_idx))
    p_full = np.mean(p_full_seeds, axis=0)

    # 2) 5-fold CV fold-averaging on seed=42
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_cv_folds = []
    for fold, (tr_idx, _) in enumerate(skf.split(X_tr_clean, y_blocker)):
        m_f = fit_single_model(model_family, params, X_tr_clean[tr_idx], y_blocker[tr_idx], seed=42 + fold)
        p_cv_folds.append(1.0 - m_f.predict_proba(X_pred_clean)[:, 1])
    p_cv = np.mean(p_cv_folds, axis=0)

    # 50% full-train 3-seed + 50% 5-fold CV ensemble for variance reduction
    return 0.5 * p_full + 0.5 * p_cv, saved_models


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    df_train = pd.read_csv(ROOT / "data" / "train.csv")
    df_pred = pd.read_csv(ROOT / "data" / "predict_input.csv")

    y_label1 = df_train["label"].to_numpy(dtype=np.int8)
    y_blocker = (y_label1 == 0).astype(np.int8)

    store_train = FeatureStore(df_train, cfg, verbose=False)
    store_pred = FeatureStore(df_pred, cfg, verbose=False)

    specs = get_top_specs()

    # Load OOF predictions to verify Phase 3 ensemble weights & Phase 4 threshold sensitivity on train.csv
    oof_et = np.load(OOF_CACHE_DIR / "P2_15_ALL_FP_E6_DRV_et_f1_oof.npy")
    oof_lgb16 = np.load(OOF_CACHE_DIR / "P2_16_ALL_FP_E7_DRV_lgbm_f1_oof.npy")
    oof_lgb13 = np.load(OOF_CACHE_DIR / "P2_13_ALL_FP_E6_DRV_lgbm_f1_oof.npy")
    oof_lgb12 = np.load(OOF_CACHE_DIR / "P2_12_E6_DRV_lgbm_f1_oof.npy")
    oof_xgb = np.load(OOF_CACHE_DIR / "P2_14_ALL_FP_E6_DRV_xgboost_f1_oof.npy")

    # Top-3 Ensemble OOF: 0.40 * ExtraTrees + 0.35 * LightGBM + 0.25 * XGBoost
    oof_ens_top3 = 0.40 * oof_et + 0.35 * oof_lgb16 + 0.25 * oof_xgb
    print("=== [Phase 3 & 4] Train OOF Verification (n=8,349) ===", flush=True)
    print("  Threshold sensitivity on OOF (Top-3 Ensemble: 0.40 ET + 0.35 LGBM + 0.25 XGB):")
    for thr in (0.40, 0.42, 0.45, 0.47, 0.48, 0.50, 0.52, 0.55):
        f1_t = f1_score(y_label1, (oof_ens_top3 >= thr).astype(int), pos_label=1)
        print(f"    thr={thr:.2f} -> OOF F1(label=1) = {f1_t:.4f}")

    # Build feature matrices for train and predict_input
    print("\n=== [Phase 5] Building Feature Matrices for train (8,349) & predict_input (927) ===", flush=True)
    feat_sets = {}
    for fs_name in ("ALL_FP_E6_DRV", "ALL_FP_E7_DRV", "E6_DRV"):
        names_tr, X_tr, _ = build_feature_matrix(fs_name, df_train, store_train)
        names_pr, X_pr, _ = build_feature_matrix(fs_name, df_pred, store_pred)
        assert names_tr == names_pr, f"Feature name mismatch in {fs_name}"
        feat_sets[fs_name] = (names_tr, X_tr, X_pr)
        print(f"  {fs_name:<16}: {X_tr.shape} -> {X_pr.shape}", flush=True)

    # Fit and predict each top component on predict_input.csv
    preds_927 = {}
    saved_artifacts = {}
    for run_id in (
        "P2_15_ALL_FP_E6_DRV_et_f1",
        "P2_16_ALL_FP_E7_DRV_lgbm_f1",
        "P2_13_ALL_FP_E6_DRV_lgbm_f1",
        "P2_12_E6_DRV_lgbm_f1",
        "P2_14_ALL_FP_E6_DRV_xgboost_f1",
    ):
        spec = specs[run_id]
        fs_name = spec["feature_set_name"]
        _, X_tr_raw, X_pr_raw = feat_sets[fs_name]
        t_c = time.time()
        p_927, models_info = predict_single_component_on_test(spec, X_tr_raw, y_blocker, X_pr_raw)
        preds_927[run_id] = p_927
        saved_artifacts[run_id] = {"spec": spec, "models_info": models_info}
        print(f"  Fitted & predicted {run_id:<32} ({time.time()-t_c:.1f}s)", flush=True)

    # Ensemble predictions on predict_input.csv
    p_top3_ens = (
        0.40 * preds_927["P2_15_ALL_FP_E6_DRV_et_f1"]
        + 0.35 * preds_927["P2_16_ALL_FP_E7_DRV_lgbm_f1"]
        + 0.25 * preds_927["P2_14_ALL_FP_E6_DRV_xgboost_f1"]
    )
    p_top5_ens = (
        0.35 * preds_927["P2_15_ALL_FP_E6_DRV_et_f1"]
        + 0.25 * preds_927["P2_16_ALL_FP_E7_DRV_lgbm_f1"]
        + 0.15 * preds_927["P2_13_ALL_FP_E6_DRV_lgbm_f1"]
        + 0.10 * preds_927["P2_12_E6_DRV_lgbm_f1"]
        + 0.15 * preds_927["P2_14_ALL_FP_E6_DRV_xgboost_f1"]
    )

    # Save v2 artifact and v2 submission
    artifact_v2_path = ROOT / "artifacts" / "best_model_v2.joblib"
    with open(artifact_v2_path, "wb") as f:
        pickle.dump({
            "ensemble": "0.40*P2_15_ET + 0.35*P2_16_LGBM + 0.25*P2_14_XGB",
            "threshold_label1": 0.50,
            "components": saved_artifacts,
            "preds_927": preds_927,
            "p_top3_ens": p_top3_ens,
            "p_top5_ens": p_top5_ens,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    sub_v2_path = ROOT / "predict_output_김인성_v2.csv"
    sub_v2 = pd.DataFrame({
        "SMILES": df_pred["SMILES"],
        "label": (p_top3_ens >= 0.50).astype(int),
    })
    sub_v2.to_csv(sub_v2_path, index=False, encoding="utf-8")
    print(f"\n[output] Saved {sub_v2_path} (rows={len(sub_v2)}, label counts={sub_v2['label'].value_counts().to_dict()})")

    # Final evaluation against predict_input_chembl_labeled.csv (813 measured & 927 all)
    eval_df = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    y_true = eval_df["label"].to_numpy(dtype=int)
    measured_mask = (eval_df["label_chembl"].notna() | (eval_df["source"] == "pubchem_herg_potency")).to_numpy()

    # Load baseline probabilities for paired bootstrap comparison
    p_base = load_probabilities_from_artifact(df_pred["SMILES"].tolist(), ROOT / "artifacts" / "best_model.joblib")

    print("\n======================================================================")
    print("=== [FINAL EVALUATION] Measured 813 & All 927 Comparison vs Baseline ===")
    print(f"{'Model / Ensemble':<40} | {'813 F1(1) @0.5':>14} | {'813 ROC-AUC':>11} | {'813 Acc':>8} | {'927 F1(1) @0.5':>14} | {'927 AUC':>8}")
    print("-" * 110)

    candidates_eval = [
        ("Baseline (best_model.joblib)", p_base),
        ("Top 1: P2_15 ExtraTrees (ALL_FP_E6_DRV)", preds_927["P2_15_ALL_FP_E6_DRV_et_f1"]),
        ("Top 2: P2_16 LightGBM (ALL_FP_E7_DRV)", preds_927["P2_16_ALL_FP_E7_DRV_lgbm_f1"]),
        ("Top 2b: P2_13 LightGBM (ALL_FP_E6_DRV)", preds_927["P2_13_ALL_FP_E6_DRV_lgbm_f1"]),
        ("Top 2c: P2_12 LightGBM (E6_DRV)", preds_927["P2_12_E6_DRV_lgbm_f1"]),
        ("Top 3: P2_14 XGBoost (ALL_FP_E6_DRV)", preds_927["P2_14_ALL_FP_E6_DRV_xgboost_f1"]),
        ("Ensemble Top-3 (0.40 ET+0.35 LGB+0.25 XGB)", p_top3_ens),
        ("Ensemble Top-5 (ET+3*LGB+XGB)", p_top5_ens),
    ]

    for label, p_vec in candidates_eval:
        m813 = compute_subset_metrics(y_true[measured_mask], (p_vec[measured_mask] >= 0.5).astype(int), p_vec[measured_mask], n_bootstrap=1000, seed=42)
        m927 = compute_subset_metrics(y_true, (p_vec >= 0.5).astype(int), p_vec, n_bootstrap=1000, seed=42)
        print(
            f"{label:<40} | {m813['label1_non_toxic']['f1']:>14.4f} | {m813['roc_auc']:>11.4f} | "
            f"{m813['accuracy']:>8.4f} | {m927['label1_non_toxic']['f1']:>14.4f} | {m927['roc_auc']:>8.4f}"
        )

    # Detailed report for Ensemble Top-3
    res_813 = compute_subset_metrics(y_true[measured_mask], (p_top3_ens[measured_mask] >= 0.5).astype(int), p_top3_ens[measured_mask], n_bootstrap=1000, seed=42)
    res_927 = compute_subset_metrics(y_true, (p_top3_ens >= 0.5).astype(int), p_top3_ens, n_bootstrap=1000, seed=42)
    print_report("Ensemble Top-3 on Measured 813", res_813)
    print_report("Ensemble Top-3 on All 927", res_927)


if __name__ == "__main__":
    main()
