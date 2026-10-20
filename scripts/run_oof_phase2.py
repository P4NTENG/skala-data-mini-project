"""Run Random Stratified 5-Fold OOF + Scaffold Holdout for the baseline model and Phase 2 Exp 1.

STRICT RULE COMPLIANCE:
- Only loads ``data/train.csv`` (8,349 rows).
- NEVER loads or touches ``data/predict_input_chembl_labeled.csv`` or ``data/predict_input.csv``.
- Logs internal CV results to ``reports/experiment_log_v2.csv``.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_oof_phase2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from herg_model.chemistry import bem_murcko_scaffold, mol_from_smiles  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.experiments import DOMAIN_GROUPS, FeatureStore  # noqa: E402
from herg_model.models import positive_scores  # noqa: E402
from herg_model.predict import load_artifact  # noqa: E402
from herg_model.preprocessing import FeaturePipeline  # noqa: E402
from herg_model.splitting import make_splits  # noqa: E402


def find_best_f1_threshold(y_label1: np.ndarray, p_label1: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.10, 0.90, 161)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        score = f1_score(y_label1, (p_label1 >= t).astype(int), pos_label=1, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def bootstrap_ci(
    y_label1: np.ndarray,
    p_label1: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_label1)
    pred = (p_label1 >= threshold).astype(int)
    f1s = np.empty(n_bootstrap, dtype=float)
    aucs = np.empty(n_bootstrap, dtype=float)
    accs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yb = y_label1[idx]
        pb = pred[idx]
        f1s[b] = f1_score(yb, pb, pos_label=1, zero_division=0)
        aucs[b] = roc_auc_score(yb, p_label1[idx])
        accs[b] = accuracy_score(yb, pb)
    return {
        "f1_label1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "roc_auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "f1_samples": f1s,
        "auc_samples": aucs,
    }


def evaluate_predictions(y_label1: np.ndarray, p_label1: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p_label1 >= threshold).astype(int)
    cm = confusion_matrix(y_label1, pred, labels=[0, 1])
    return {
        "threshold_label1": float(threshold),
        "threshold_blocker": float(1.0 - threshold),
        "f1_label1": float(f1_score(y_label1, pred, pos_label=1, zero_division=0)),
        "precision_label1": float(precision_score(y_label1, pred, pos_label=1, zero_division=0)),
        "recall_label1": float(recall_score(y_label1, pred, pos_label=1, zero_division=0)),
        "pr_auc_label1": float(average_precision_score(y_label1, p_label1)),
        "f1_label0": float(f1_score(y_label1, pred, pos_label=0, zero_division=0)),
        "precision_label0": float(precision_score(y_label1, pred, pos_label=0, zero_division=0)),
        "recall_label0": float(recall_score(y_label1, pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_label1, p_label1)),
        "accuracy": float(accuracy_score(y_label1, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_label1, pred)),
        "cm": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
    }


def run_cv_and_scaffold(
    run_id: str,
    description: str,
    blocks: list[str],
    params: dict,
    store: FeatureStore,
    y_label1: np.ndarray,
    y_blocker: np.ndarray,
    scaffolds: list[str],
    cfg,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    names, X = store.matrix(blocks)

    # 1. Stratified 5-Fold OOF (random)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_p_label1 = np.zeros(len(y_label1), dtype=float)
    oof_pred_nested_thr = np.zeros(len(y_label1), dtype=int)
    fold_f1_05 = []
    fold_aucs = []
    fold_n_features = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_blocker)):
        tf = time.time()
        pipe = FeaturePipeline(names=names, scale=False, drop_constant=True, drop_duplicate=True)
        X_tr = pipe.fit_transform(X[tr_idx], y_blocker[tr_idx])
        X_va = pipe.transform(X[va_idx])
        fold_n_features.append(X_tr.shape[1])

        model = LGBMClassifier(**params, random_state=seed + fold, n_jobs=4, verbose=-1)
        model.fit(X_tr, y_blocker[tr_idx])
        p_blocker_va = positive_scores(model, X_va)
        p_label1_va = 1.0 - p_blocker_va
        oof_p_label1[va_idx] = p_label1_va

        f1_v = float(f1_score(y_label1[va_idx], (p_label1_va >= 0.5).astype(int), pos_label=1))
        auc_v = float(roc_auc_score(y_label1[va_idx], p_label1_va))
        fold_f1_05.append(f1_v)
        fold_aucs.append(auc_v)
        print(f"  [{run_id}] fold {fold+1}/5: F1(1)={f1_v:.4f}, AUC={auc_v:.4f} ({time.time()-tf:.1f}s)", flush=True)

    # Nested threshold selection per fold (using the other 4 folds' OOF predictions)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_blocker)):
        t_opt, _ = find_best_f1_threshold(y_label1[tr_idx], oof_p_label1[tr_idx])
        oof_pred_nested_thr[va_idx] = (oof_p_label1[va_idx] >= t_opt).astype(int)

    oof_at_05 = evaluate_predictions(y_label1, oof_p_label1, threshold=0.5)
    global_opt_thr, _ = find_best_f1_threshold(y_label1, oof_p_label1)
    oof_at_opt = evaluate_predictions(y_label1, oof_p_label1, threshold=global_opt_thr)
    nested_opt_f1 = float(f1_score(y_label1, oof_pred_nested_thr, pos_label=1, zero_division=0))
    boot_05 = bootstrap_ci(y_label1, oof_p_label1, threshold=0.5, n_bootstrap=1000, seed=seed)

    # 2. Scaffold Holdout (seed 42: train -> val threshold, refit train+val -> test)
    splits = make_splits(scaffolds, y_blocker, cfg, "scaffold", seed)
    tr, va, te = splits.train, splits.val, splits.test

    pipe_sc_tr = FeaturePipeline(names=names, scale=False, drop_constant=True, drop_duplicate=True)
    X_sc_tr = pipe_sc_tr.fit_transform(X[tr], y_blocker[tr])
    X_sc_va = pipe_sc_tr.transform(X[va])
    model_sc_tr = LGBMClassifier(**params, random_state=seed, n_jobs=4, verbose=-1)
    model_sc_tr.fit(X_sc_tr, y_blocker[tr])
    p_sc_va = 1.0 - positive_scores(model_sc_tr, X_sc_va)
    sc_val_opt_thr, _ = find_best_f1_threshold(y_label1[va], p_sc_va)

    # Evaluate train-only model on scaffold test
    X_sc_te_tronly = pipe_sc_tr.transform(X[te])
    p_sc_te_tronly = 1.0 - positive_scores(model_sc_tr, X_sc_te_tronly)
    sc_test_tronly_05 = evaluate_predictions(y_label1[te], p_sc_te_tronly, threshold=0.5)
    sc_test_tronly_opt = evaluate_predictions(y_label1[te], p_sc_te_tronly, threshold=sc_val_opt_thr)

    # Also refit on train+val (matching best_model.joblib training protocol) and evaluate on scaffold test
    tr_va = np.concatenate([tr, va])
    pipe_sc_trva = FeaturePipeline(names=names, scale=False, drop_constant=True, drop_duplicate=True)
    X_sc_trva = pipe_sc_trva.fit_transform(X[tr_va], y_blocker[tr_va])
    X_sc_te_trva = pipe_sc_trva.transform(X[te])
    model_sc_trva = LGBMClassifier(**params, random_state=seed, n_jobs=4, verbose=-1)
    model_sc_trva.fit(X_sc_trva, y_blocker[tr_va])
    p_sc_te_trva = 1.0 - positive_scores(model_sc_trva, X_sc_te_trva)
    sc_test_trva_05 = evaluate_predictions(y_label1[te], p_sc_te_trva, threshold=0.5)

    elapsed = time.time() - t0
    return {
        "run_id": run_id,
        "description": description,
        "blocks": blocks,
        "n_features_raw": len(names),
        "n_features_kept_mean": int(np.mean(fold_n_features)),
        "oof_p_label1": oof_p_label1,
        "oof_at_05": oof_at_05,
        "oof_at_opt": oof_at_opt,
        "oof_nested_opt_f1": nested_opt_f1,
        "oof_fold_f1_05_mean": float(np.mean(fold_f1_05)),
        "oof_fold_f1_05_std": float(np.std(fold_f1_05, ddof=1)),
        "oof_fold_auc_mean": float(np.mean(fold_aucs)),
        "oof_fold_auc_std": float(np.std(fold_aucs, ddof=1)),
        "boot_05": boot_05,
        "scaffold_test_tronly_05": sc_test_tronly_05,
        "scaffold_test_tronly_opt": sc_test_tronly_opt,
        "scaffold_test_trva_05": sc_test_trva_05,
        "elapsed_sec": round(elapsed, 2),
    }


def print_run_summary(res: dict) -> None:
    m05 = res["oof_at_05"]
    mopt = res["oof_at_opt"]
    b05 = res["boot_05"]
    sc05 = res["scaffold_test_tronly_05"]
    scopt = res["scaffold_test_tronly_opt"]
    sctrva = res["scaffold_test_trva_05"]
    cm = m05["cm"]

    print(f"\n======================================================================")
    print(f"[{res['run_id']}] {res['description']} ({res['elapsed_sec']}s)")
    print(f"  Features: raw={res['n_features_raw']} -> kept≈{res['n_features_kept_mean']}")
    print(f"  [Random 5-Fold OOF @ thr=0.5] (n=8,349)")
    print(f"    F1(label=1) : {m05['f1_label1']:.4f}  (95% CI: [{b05['f1_label1_ci95'][0]:.4f}, {b05['f1_label1_ci95'][1]:.4f}])"
          f"  [fold mean±sd: {res['oof_fold_f1_05_mean']:.4f}±{res['oof_fold_f1_05_std']:.4f}]")
    print(f"    P(label=1)  : {m05['precision_label1']:.4f}  R(label=1): {m05['recall_label1']:.4f}  PR-AUC(1): {m05['pr_auc_label1']:.4f}")
    print(f"    F1(label=0) : {m05['f1_label0']:.4f}  P(label=0): {m05['precision_label0']:.4f}  R(label=0): {m05['recall_label0']:.4f}")
    print(f"    ROC-AUC     : {m05['roc_auc']:.4f}  (95% CI: [{b05['roc_auc_ci95'][0]:.4f}, {b05['roc_auc_ci95'][1]:.4f}])"
          f"  [fold mean±sd: {res['oof_fold_auc_mean']:.4f}±{res['oof_fold_auc_std']:.4f}]")
    print(f"    Accuracy    : {m05['accuracy']:.4f}  (95% CI: [{b05['accuracy_ci95'][0]:.4f}, {b05['accuracy_ci95'][1]:.4f}])"
          f"  Balanced Acc: {m05['balanced_accuracy']:.4f}")
    print(f"    Confusion Matrix (rows=actual 0/1, cols=pred 0/1): [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")
    print(f"  [Random 5-Fold OOF @ Optimal Threshold]")
    print(f"    Global OOF opt thr(p_label1)={mopt['threshold_label1']:.3f} (p_blocker={mopt['threshold_blocker']:.3f}) -> "
          f"OOF F1(1)={mopt['f1_label1']:.4f} (nested out-of-fold thr F1(1)={res['oof_nested_opt_f1']:.4f})")
    print(f"  [Scaffold Holdout Test (n=835, seed=42)]")
    print(f"    train-only @0.5      : F1(1)={sc05['f1_label1']:.4f}, ROC-AUC={sc05['roc_auc']:.4f}, Acc={sc05['accuracy']:.4f}")
    print(f"    train-only @val_thr  : F1(1)={scopt['f1_label1']:.4f} (thr={scopt['threshold_label1']:.3f})")
    print(f"    train+val  @0.5      : F1(1)={sctrva['f1_label1']:.4f}, ROC-AUC={sctrva['roc_auc']:.4f}, Acc={sctrva['accuracy']:.4f}")


def main() -> None:
    cfg = load_config()
    df_train = pd.read_csv(ROOT / "data" / "train.csv")
    y_label1 = df_train["label"].to_numpy(dtype=np.int8)
    y_blocker = (y_label1 == 0).astype(np.int8)

    mols = [mol_from_smiles(s) for s in df_train["SMILES"]]
    scaffolds = [bem_murcko_scaffold(m, True) if m is not None else "INVALID" for m in mols]

    artifact = load_artifact(ROOT / "artifacts" / "best_model.joblib")
    base_params = dict(artifact["metadata"]["hyperparameters"])
    base_params.pop("class_weight", None)

    store = FeatureStore(df_train, cfg, verbose=False)
    domain_blocks = list(DOMAIN_GROUPS[:-1])

    # Run 0: Baseline E6 (fp:fcfp + existing_descriptors + 7 domain groups)
    blocks_e6 = ["fp:fcfp", "existing_descriptors"] + domain_blocks
    res_base = run_cv_and_scaffold(
        run_id="P0_baseline_E6_lgbm",
        description="Baseline E6 LightGBM (fp:fcfp + existing_descriptors + 7 domain groups)",
        blocks=blocks_e6,
        params=base_params,
        store=store,
        y_label1=y_label1,
        y_blocker=y_blocker,
        scaffolds=scaffolds,
        cfg=cfg,
        seed=42,
    )
    print_run_summary(res_base)

    # Run 1 (Phase 2 Exp 1): Expand fingerprint from fp:fcfp to all_fingerprints (ecfp+fcfp+ptfp)
    blocks_p2_1 = ["all_fingerprints", "existing_descriptors"] + domain_blocks
    res_p2_1 = run_cv_and_scaffold(
        run_id="P2_01_all_fp_domain_lgbm",
        description="Phase 2 Exp 1: All 3 Fingerprints (ECFP+FCFP+PTFP) + existing_descriptors + 7 domain groups (same LGBM params)",
        blocks=blocks_p2_1,
        params=base_params,
        store=store,
        y_label1=y_label1,
        y_blocker=y_blocker,
        scaffolds=scaffolds,
        cfg=cfg,
        seed=42,
    )
    print_run_summary(res_p2_1)

    # Paired bootstrap comparison on OOF
    diff_f1 = res_p2_1["boot_05"]["f1_samples"] - res_base["boot_05"]["f1_samples"]
    diff_auc = res_p2_1["boot_05"]["auc_samples"] - res_base["boot_05"]["auc_samples"]
    print("\n======================================================================")
    print("[Comparison: P2_01_all_fp_domain_lgbm minus P0_baseline_E6_lgbm on 5-Fold OOF]")
    print(f"  Delta OOF F1(1) @0.5 : {res_p2_1['oof_at_05']['f1_label1'] - res_base['oof_at_05']['f1_label1']:+.4f} "
          f"(95% paired CI: [{np.percentile(diff_f1, 2.5):+.4f}, {np.percentile(diff_f1, 97.5):+.4f}], "
          f"P(diff<=0)={(diff_f1 <= 0).mean():.3f})")
    print(f"  Delta OOF ROC-AUC    : {res_p2_1['oof_at_05']['roc_auc'] - res_base['oof_at_05']['roc_auc']:+.4f} "
          f"(95% paired CI: [{np.percentile(diff_auc, 2.5):+.4f}, {np.percentile(diff_auc, 97.5):+.4f}], "
          f"P(diff<=0)={(diff_auc <= 0).mean():.3f})")
    print(f"  Delta Scaffold F1(1) : "
          f"train-only @0.5 {res_p2_1['scaffold_test_tronly_05']['f1_label1'] - res_base['scaffold_test_tronly_05']['f1_label1']:+.4f} | "
          f"train+val @0.5 {res_p2_1['scaffold_test_trva_05']['f1_label1'] - res_base['scaffold_test_trva_05']['f1_label1']:+.4f}")

    # Save experiment log
    log_rows = []
    for r, eval_val, note in [
        (res_base, "0.8303 (baseline)", "Checked-in E6 LightGBM architecture on 5-fold OOF & scaffold"),
        (res_p2_1, "NOT_EVALUATED_YET", "Phase 2 Exp 1: replace fp:fcfp with all_fingerprints (ecfp+fcfp+ptfp)"),
    ]:
        log_rows.append({
            "run_id": r["run_id"],
            "model": "lightgbm",
            "features": "+".join(r["blocks"]),
            "n_features_kept": r["n_features_kept_mean"],
            "cv_scheme": "Stratified5Fold(seed=42)",
            "oof_f1_label1_05": round(r["oof_at_05"]["f1_label1"], 4),
            "oof_f1_ci95_lo": round(r["boot_05"]["f1_label1_ci95"][0], 4),
            "oof_f1_ci95_hi": round(r["boot_05"]["f1_label1_ci95"][1], 4),
            "oof_f1_label1_opt": round(r["oof_at_opt"]["f1_label1"], 4),
            "oof_nested_opt_f1": round(r["oof_nested_opt_f1"], 4),
            "oof_roc_auc": round(r["oof_at_05"]["roc_auc"], 4),
            "oof_accuracy_05": round(r["oof_at_05"]["accuracy"], 4),
            "scaffold_f1_label1_tronly_05": round(r["scaffold_test_tronly_05"]["f1_label1"], 4),
            "scaffold_f1_label1_tronly_opt": round(r["scaffold_test_tronly_opt"]["f1_label1"], 4),
            "scaffold_f1_label1_trva_05": round(r["scaffold_test_trva_05"]["f1_label1"], 4),
            "scaffold_roc_auc_tronly": round(r["scaffold_test_tronly_05"]["roc_auc"], 4),
            "eval_f1_label1": eval_val,
            "threshold_label1_opt": round(r["oof_at_opt"]["threshold_label1"], 3),
            "notes": note,
        })

    log_df = pd.DataFrame(log_rows)
    log_path = ROOT / "reports" / "experiment_log_v2.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8")
    print(f"\n[log] wrote {len(log_df)} rows -> {log_path}")


if __name__ == "__main__":
    main()
