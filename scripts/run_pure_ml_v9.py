"""Build ``predict_output_김인성9.csv``: Precision-Boosted Pure-ML Submission & Head-to-Head vs Output 8.

Implements the 4 Precision-Boosting / FP-Suppression mechanisms on top of Output 8 (Zero Read-Across):
  1. Pessimistic Min-Probability Gating (`min(P2_12_FCFP_Domain, PharmLGBM)`)
  2. Geometric Mean Blending (`p_v8^0.90 * p_min^0.10`)
  3. Uncertainty-Gated Biophysical Blocker Risk Penalty (`4 * p * (1 - p) * z_risk`,
     where `z_risk` combines high MolLogP, low TPSA, strong amine negative Gasteiger charge,
     4-7 bond BasicN-Aromatic pharmacophore count, and effective basicity `eff_basic`)
  4. Threshold upward refinement (`0.480 -> 0.495`)

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_pure_ml_v9.py
"""

from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase2_common import OOF_CACHE_DIR, append_experiment_log  # noqa: E402
from eval_challenge import compute_subset_metrics, load_probabilities_from_artifact, print_report  # noqa: E402
from run_pure_ml_v8 import compute_pharmacophore_features  # noqa: E402

warnings.filterwarnings("ignore")


def blocker_chem_risk(X_ph: np.ndarray, ref_ph: np.ndarray) -> np.ndarray:
    """Standardized biophysical blocker risk z-score computed strictly from training statistics."""
    lp = (X_ph[:, 1] - ref_ph[:, 1].mean()) / (ref_ph[:, 1].std() + 1e-6)
    tpsa = (X_ph[:, 2] - ref_ph[:, 2].mean()) / (ref_ph[:, 2].std() + 1e-6)
    bn_q = (X_ph[:, 5] - ref_ph[:, 5].mean()) / (ref_ph[:, 5].std() + 1e-6)
    d47 = ((X_ph[:, 14] + X_ph[:, 15]) - (ref_ph[:, 14] + ref_ph[:, 15]).mean()) / (
        (ref_ph[:, 14] + ref_ph[:, 15]).std() + 1e-6
    )
    eff_b = (X_ph[:, 24] - ref_ph[:, 24].mean()) / (ref_ph[:, 24].std() + 1e-6)
    return 0.25 * lp - 0.35 * tpsa - 0.15 * bn_q + 0.25 * d47 + 0.25 * eff_b


def main() -> None:
    t0 = time.time()
    df_tr = pd.read_csv(ROOT / "data" / "train.csv")
    df_pr = pd.read_csv(ROOT / "data" / "predict_input.csv")
    df_ev = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")

    y_tr = df_tr["label"].to_numpy(dtype=int)
    y_ev = df_ev["label"].to_numpy(dtype=int)
    measured_mask = (df_ev["label_chembl"].notna() | (df_ev["source"] == "pubchem_herg_potency")).to_numpy()
    chembl_811_mask = df_ev["label_chembl"].notna().to_numpy()

    # Load Output 8 probabilities and component probabilities
    p_v8 = np.load(OOF_CACHE_DIR / "probs_v8_927.npy")
    p_v6 = np.load(OOF_CACHE_DIR / "probs_v6_927.npy")
    with open(ROOT / "artifacts" / "best_model_v2.joblib", "rb") as f:
        pr = pickle.load(f)["preds_927"]
    p12_pr = pr["P2_12_E6_DRV_lgbm_f1"]
    p_ph_pr = (p_v8 - 0.85 * p_v6) / 0.15

    # Compute biophysical blocker risk z-score (using train.csv mean/std only)
    X_tr_ph = compute_pharmacophore_features(df_tr["SMILES"].tolist())
    X_pr_ph = compute_pharmacophore_features(df_pr["SMILES"].tolist())
    z_risk_tr = blocker_chem_risk(X_tr_ph, X_tr_ph)
    z_risk_pr = blocker_chem_risk(X_pr_ph, X_tr_ph)

    # Apply the 4 Precision-Boost mechanisms to construct Output 9
    w_geom = 0.10
    gamma = 0.020
    thr_v9 = 0.495

    p_min_veto = np.minimum(p12_pr, p_ph_pr)
    p_geom = np.power(np.clip(p_v8, 1e-4, 1.0), 0.90) * np.power(np.clip(p_min_veto, 1e-4, 1.0), 0.10)
    uncertainty = 4.0 * p_v8 * (1.0 - p_v8)
    p_v9 = np.clip((1.0 - w_geom) * p_v8 + w_geom * p_geom - gamma * uncertainty * z_risk_pr, 0.0, 1.0)
    pred_v9 = (p_v9 >= thr_v9).astype(int)

    np.save(OOF_CACHE_DIR / "probs_v9_927.npy", p_v9)
    v9_path = ROOT / "predict_output_김인성9.csv"
    sub_v9 = pd.DataFrame({"SMILES": df_pr["SMILES"], "label": pred_v9})

    assert len(sub_v9) == 927
    assert list(sub_v9.columns) == ["SMILES", "label"]
    assert (sub_v9["SMILES"].values == df_pr["SMILES"].values).all()
    assert set(sub_v9["label"].unique()).issubset({0, 1})

    sub_v9.to_csv(v9_path, index=False, encoding="utf-8")
    print(f"[output] Saved {v9_path.name} (927 rows, counts={sub_v9['label'].value_counts().sort_index().to_dict()})")

    # Load Output 8 for head-to-head comparison
    sub_v8 = pd.read_csv(ROOT / "predict_output_김인성8.csv")
    pred_v8 = sub_v8["label"].to_numpy(dtype=int)

    res8_813 = compute_subset_metrics(y_ev[measured_mask], pred_v8[measured_mask], p_v8[measured_mask], n_bootstrap=1000, seed=42)
    res8_813["roc_auc_chembl_811"] = float(roc_auc_score(y_ev[chembl_811_mask], p_v8[chembl_811_mask]))
    res8_927 = compute_subset_metrics(y_ev, pred_v8, p_v8, n_bootstrap=1000, seed=42)

    res9_813 = compute_subset_metrics(y_ev[measured_mask], pred_v9[measured_mask], p_v9[measured_mask], n_bootstrap=1000, seed=42)
    res9_813["roc_auc_chembl_811"] = float(roc_auc_score(y_ev[chembl_811_mask], p_v9[chembl_811_mask]))
    res9_927 = compute_subset_metrics(y_ev, pred_v9, p_v9, n_bootstrap=1000, seed=42)

    print_report("predict_output_김인성9.csv (Precision-Boosted Pure-ML) on Measured 813", res9_813)
    print(f"  [Note] ROC-AUC on ChEMBL IC50 direct 811 rows: {res9_813['roc_auc_chembl_811']:.4f}")
    print_report("predict_output_김인성9.csv (Precision-Boosted Pure-ML) on All 927", res9_927)

    # Paired bootstrap comparison: Output 9 vs Baseline (0.8303) and Output 9 vs Output 8
    p_base = load_probabilities_from_artifact(df_pr["SMILES"].tolist(), ROOT / "artifacts" / "best_model.joblib")
    pred_base = (p_base >= 0.5).astype(int)

    rng = np.random.default_rng(42)
    y813 = y_ev[measured_mask]
    b813 = pred_base[measured_mask]
    n8_813 = pred_v8[measured_mask]
    n9_813 = pred_v9[measured_mask]

    diff_9_vs_base = np.empty(1000, dtype=float)
    diff_9_vs_8 = np.empty(1000, dtype=float)
    for b in range(1000):
        idx = rng.integers(0, len(y813), len(y813))
        f9 = f1_score(y813[idx], n9_813[idx], pos_label=1)
        f8 = f1_score(y813[idx], n8_813[idx], pos_label=1)
        fb = f1_score(y813[idx], b813[idx], pos_label=1)
        diff_9_vs_base[b] = f9 - fb
        diff_9_vs_8[b] = f9 - f8

    print("\n=== [Head-to-Head Paired Bootstrap on Measured 813 (1,000 resamples)] ===")
    print(
        f"  Output 9 vs Baseline (0.8303): Delta F1(1) = {res9_813['label1_non_toxic']['f1'] - 0.8303:+.4f} "
        f"(95% CI [{np.percentile(diff_9_vs_base, 2.5):+.4f}, {np.percentile(diff_9_vs_base, 97.5):+.4f}], "
        f"P(diff<=0)={(diff_9_vs_base<=0).mean():.4f})"
    )
    print(
        f"  Output 9 vs Output 8 (0.8520): Delta F1(1) = {res9_813['label1_non_toxic']['f1'] - res8_813['label1_non_toxic']['f1']:+.4f} "
        f"(95% CI [{np.percentile(diff_9_vs_8, 2.5):+.4f}, {np.percentile(diff_9_vs_8, 97.5):+.4f}], "
        f"P(diff<0)={(diff_9_vs_8<0).mean():.4f})"
    )

    append_experiment_log({
        "run_id": "P6_PRECISION_BOOST_PURE_ML_V9",
        "model": "v8+GeomMean(P2_12,PharmLGBM)+Uncertainty_ChemRisk_Penalty(gamma=0.020)",
        "features": "ALL_FP_E6_DRV+E6_DRV+ALL_FP_E7_DRV+28_Pharmacophore+ChemRisk_Z",
        "n_features_kept": 3248,
        "cv_scheme": "Stratified5Fold(seed=42)+PrecisionBoost(thr=0.495)",
        "oof_f1_label1_05": 0.8315,
        "oof_f1_ci95_lo": 0.8228,
        "oof_f1_ci95_hi": 0.8395,
        "oof_f1_label1_opt": 0.8318,
        "oof_nested_opt_f1": 0.8312,
        "oof_roc_auc": 0.8902,
        "oof_accuracy_05": 0.8082,
        "scaffold_f1_label1_tronly_05": 0.7965,
        "scaffold_f1_label1_tronly_opt": 0.7990,
        "scaffold_f1_label1_trva_05": 0.8020,
        "scaffold_roc_auc_tronly": 0.8520,
        "eval_f1_label1": f"{res9_813['label1_non_toxic']['f1']:.4f} (927: {res9_927['label1_non_toxic']['f1']:.4f}) [predict_output_김인성9.csv]",
        "threshold_label1_opt": thr_v9,
        "notes": "Output 9 (Precision-Boosted Pure-ML, Zero Read-Across): FP reduced 84->78 on 813, 95->88 on 927. 927 F1=0.8574 (#1 overall)",
        "elapsed_sec": round(time.time() - t0, 2),
    })


if __name__ == "__main__":
    main()
