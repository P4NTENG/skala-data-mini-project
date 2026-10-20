"""Phase 3-5 Final Execution: OOF-only weight & threshold selection, single evaluation on 813, and submission generation.

Candidate 3 Models (Top 3 single models):
  1. ``P2_13_ALL_FP_E6_DRV_lgbm_f1`` : LightGBM on ALL_FP_E6_DRV (3,189 features)
  2. ``P2_12_E6_DRV_lgbm_f1``        : LightGBM on E6_DRV (1,252 features)
  3. ``P2_16_ALL_FP_E7_DRV_lgbm_f1`` : LightGBM on ALL_FP_E7_DRV (3,220 features)
  (Also reports the 3-family OOF ensemble ``P2_13 LGBM + P2_14 XGB + P2_15 ET`` for completeness.)

Protocol:
  1. Weights (w1, w2, w3) are selected STRICTLY on train.csv 5-fold OOF predictions (n=8,349).
  2. Threshold is selected STRICTLY on train.csv 5-fold OOF predictions (n=8,349).
  3. With weights and threshold frozen, ``predict_output_김인성_v2.csv`` and ``predict_output_김인성.csv``
     (after backing up the original baseline to ``predict_output_김인성_baseline.csv``) are written.
  4. Evaluated ONCE on Measured 813 (and All 927) with 1,000 bootstrap 95% CI and confusion matrix.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase2_common import OOF_CACHE_DIR, append_experiment_log  # noqa: E402
from eval_challenge import compute_subset_metrics, load_probabilities_from_artifact, print_report  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    df_train = pd.read_csv(ROOT / "data" / "train.csv")
    df_pred = pd.read_csv(ROOT / "data" / "predict_input.csv")
    y_oof = df_train["label"].to_numpy(dtype=int)

    # 1. Load OOF predictions (train.csv, n=8,349) for the 3 candidate models
    cand_ids = [
        "P2_13_ALL_FP_E6_DRV_lgbm_f1",
        "P2_12_E6_DRV_lgbm_f1",
        "P2_16_ALL_FP_E7_DRV_lgbm_f1",
    ]
    oofs = [np.load(OOF_CACHE_DIR / f"{cid}_oof.npy") for cid in cand_ids]

    print("=== [Step 1] OOF-Only Ensemble Weight Optimization (n=8,349) ===")
    for cid, p_o in zip(cand_ids, oofs):
        print(
            f"  Candidate {cid:<30}: OOF F1(1)@0.50={f1_score(y_oof, (p_o>=0.5).astype(int)):.4f} | "
            f"OOF ROC-AUC={roc_auc_score(y_oof, p_o):.4f} | OOF LogLoss={log_loss(y_oof, np.clip(p_o, 1e-6, 1-1e-6)):.4f}"
        )

    best_w = None
    best_oof_f1 = -1.0
    best_oof_auc = -1.0
    for w1 in np.linspace(0.05, 0.90, 18):
        for w2 in np.linspace(0.05, 0.90, 18):
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0.05 - 1e-9:
                continue
            p_blend = w1 * oofs[0] + w2 * oofs[1] + w3 * oofs[2]
            f1_val = float(f1_score(y_oof, (p_blend >= 0.5).astype(int), pos_label=1))
            auc_val = float(roc_auc_score(y_oof, p_blend))
            if (round(f1_val, 6), round(auc_val, 6)) > (round(best_oof_f1, 6), round(best_oof_auc, 6)):
                best_oof_f1 = f1_val
                best_oof_auc = auc_val
                best_w = (round(float(w1), 2), round(float(w2), 2), round(float(w3), 2))

    oof_ens = best_w[0] * oofs[0] + best_w[1] * oofs[1] + best_w[2] * oofs[2]
    print(f"\n  -> Selected OOF Weights (w1, w2, w3) = {best_w}")
    print(f"  -> Ensemble OOF F1(1) @0.50 = {best_oof_f1:.4f} | OOF ROC-AUC = {best_oof_auc:.4f}")

    # 2. OOF-Only Threshold Optimization & Sensitivity Table
    print("\n=== [Step 2] OOF-Only Threshold Selection & Sensitivity (n=8,349) ===")
    best_thr = 0.50
    best_thr_f1 = -1.0
    for t in np.linspace(0.40, 0.60, 41):
        t_round = round(float(t), 3)
        f1_t = float(f1_score(y_oof, (oof_ens >= t_round).astype(int), pos_label=1))
        if f1_t > best_thr_f1:
            best_thr_f1 = f1_t
            best_thr = t_round
        if t_round in (0.40, 0.425, 0.45, 0.475, 0.49, 0.50, 0.51, 0.525, 0.55, 0.60):
            acc_t = float(accuracy_score(y_oof, (oof_ens >= t_round).astype(int)))
            print(f"  threshold={t_round:.3f} -> OOF F1(1)={f1_t:.4f}, OOF Acc={acc_t:.4f}")

    # Nested 5-fold threshold verification on OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_pred = np.zeros_like(y_oof)
    fold_thrs = []
    for tr_idx, va_idx in skf.split(oof_ens, 1 - y_oof):
        bt, bf = 0.50, -1.0
        for t in np.linspace(0.40, 0.60, 41):
            ft = f1_score(y_oof[tr_idx], (oof_ens[tr_idx] >= t).astype(int), pos_label=1)
            if ft > bf:
                bf, bt = ft, float(t)
        fold_thrs.append(bt)
        nested_pred[va_idx] = (oof_ens[va_idx] >= bt).astype(int)
    nested_f1 = float(f1_score(y_oof, nested_pred, pos_label=1))
    print(f"\n  -> OOF Selected Threshold = {best_thr:.3f} (OOF F1(1) = {best_thr_f1:.4f}, Nested OOF F1(1) = {nested_f1:.4f}, fold mean thr = {np.mean(fold_thrs):.3f})")

    # 3. Generate Final Predictions on predict_input.csv (927 rows)
    with open(ROOT / "artifacts" / "best_model_v2.joblib", "rb") as f:
        art2 = pickle.load(f)
    pr = art2["preds_927"]

    p_final_927 = best_w[0] * pr[cand_ids[0]] + best_w[1] * pr[cand_ids[1]] + best_w[2] * pr[cand_ids[2]]
    pred_labels_927 = (p_final_927 >= best_thr).astype(int)

    np.save(OOF_CACHE_DIR / "final_phase3_probs_927.npy", p_final_927)
    art2["phase3_final"] = {
        "candidates": cand_ids,
        "weights": best_w,
        "threshold": best_thr,
        "oof_f1": best_thr_f1,
        "oof_auc": best_oof_auc,
        "p_final_927": p_final_927,
    }
    with open(ROOT / "artifacts" / "best_model_v2.joblib", "wb") as f:
        pickle.dump(art2, f, protocol=pickle.HIGHEST_PROTOCOL)

    submission_df = pd.DataFrame({
        "SMILES": df_pred["SMILES"],
        "label": pred_labels_927,
    })

    # Verify strict format requirements
    assert len(submission_df) == 927, f"Expected 927 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == ["SMILES", "label"], f"Invalid columns: {submission_df.columns}"
    assert (submission_df["SMILES"].values == df_pred["SMILES"].values).all(), "SMILES order mismatch!"
    assert set(submission_df["label"].unique()).issubset({0, 1}), "Labels must be integer 0/1!"

    # Backup original predict_output_김인성.csv if not already backed up
    orig_sub_path = ROOT / "predict_output_김인성.csv"
    backup_sub_path = ROOT / "predict_output_김인성_baseline.csv"
    if orig_sub_path.exists() and not backup_sub_path.exists():
        shutil.copy2(orig_sub_path, backup_sub_path)
        print(f"\n[backup] Preserved original baseline submission -> {backup_sub_path.name}")

    v2_sub_path = ROOT / "predict_output_김인성_v2.csv"
    submission_df.to_csv(v2_sub_path, index=False, encoding="utf-8")
    submission_df.to_csv(orig_sub_path, index=False, encoding="utf-8")
    print(f"[output] Wrote {v2_sub_path.name} and {orig_sub_path.name} (927 rows, label counts={submission_df['label'].value_counts().sort_index().to_dict()})")

    # 4. Evaluate ONCE on Measured 813 (and All 927) with 1,000 bootstrap CI & confusion matrix
    eval_df = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    y_true = eval_df["label"].to_numpy(dtype=int)
    measured_mask = (eval_df["label_chembl"].notna() | (eval_df["source"] == "pubchem_herg_potency")).to_numpy()
    chembl_811_mask = eval_df["label_chembl"].notna().to_numpy()

    res_813 = compute_subset_metrics(
        y_true[measured_mask],
        pred_labels_927[measured_mask],
        p_final_927[measured_mask],
        n_bootstrap=1000,
        seed=42,
    )
    res_813["roc_auc_chembl_811"] = float(roc_auc_score(y_true[chembl_811_mask], p_final_927[chembl_811_mask]))
    res_927 = compute_subset_metrics(
        y_true,
        pred_labels_927,
        p_final_927,
        n_bootstrap=1000,
        seed=42,
    )

    print_report("Phase 3 Final Ensemble on Measured 813 (Primary Metric)", res_813)
    print(f"  [Note] ROC-AUC on ChEMBL IC50 direct 811 rows: {res_813['roc_auc_chembl_811']:.4f}")
    print_report("Phase 3 Final Ensemble on All 927 (Reference)", res_927)

    # Paired bootstrap difference vs Baseline (0.8303)
    p_base = load_probabilities_from_artifact(df_pred["SMILES"].tolist(), ROOT / "artifacts" / "best_model.joblib")
    pred_base = (p_base >= 0.5).astype(int)

    rng = np.random.default_rng(42)
    y813 = y_true[measured_mask]
    b813 = pred_base[measured_mask]
    n813 = pred_labels_927[measured_mask]
    pb813 = p_base[measured_mask]
    pn813 = p_final_927[measured_mask]

    diff_f1_813 = np.empty(1000, dtype=float)
    diff_auc_813 = np.empty(1000, dtype=float)
    for b in range(1000):
        idx = rng.integers(0, len(y813), len(y813))
        diff_f1_813[b] = f1_score(y813[idx], n813[idx], pos_label=1) - f1_score(y813[idx], b813[idx], pos_label=1)
        diff_auc_813[b] = roc_auc_score(y813[idx], pn813[idx]) - roc_auc_score(y813[idx], pb813[idx])

    print("\n=== [Step 4] Paired Bootstrap Comparison vs Baseline (0.8303) on Measured 813 (1,000 resamples) ===")
    print(
        f"  Delta F1(label=1) : {res_813['label1_non_toxic']['f1'] - 0.8303:+.4f} "
        f"(boot mean {diff_f1_813.mean():+.4f}, 95% CI [{np.percentile(diff_f1_813, 2.5):+.4f}, {np.percentile(diff_f1_813, 97.5):+.4f}], "
        f"P(diff<=0)={(diff_f1_813 <= 0).mean():.4f})"
    )
    print(
        f"  Delta ROC-AUC     : {res_813['roc_auc'] - 0.8878:+.4f} "
        f"(boot mean {diff_auc_813.mean():+.4f}, 95% CI [{np.percentile(diff_auc_813, 2.5):+.4f}, {np.percentile(diff_auc_813, 97.5):+.4f}], "
        f"P(diff<=0)={(diff_auc_813 <= 0).mean():.4f})"
    )


if __name__ == "__main__":
    main()
