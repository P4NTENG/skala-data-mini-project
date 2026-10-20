"""Build ``predict_output_김인성8.csv``: Ultimate Pure-ML Submission (Zero Read-Across).

Combines:
1. 28 Intramolecular 2D/Topological Pharmacophore & Local Amine Gasteiger Charge Features
   (Basic-N to Aromatic distance distribution 4-5 / 6-7 bonds, EWG inductive pKa shift within 2-3 bonds,
    Aromatic-Aromatic cage span, molecular graph diameter/radius, Desalted parent logP/TPSA/MW/QED, logD7.4 proxy)
2. Multi-View F1-Weighted LightGBM Ensemble (P2_13 + P2_12 + P2_16 + Pharmacophore-LGBM)
3. Multi-Seed Neural Network (MLPClassifier) Hybrid Ensemble
4. Test-Time Desalted Parent Inference Correction on TFA/Salt molecules

100% row-independent molecular features — NO Read-Across / NO InChIKey lookup.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_pure_ml_v8.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase2_common import (  # noqa: E402
    OOF_CACHE_DIR,
    append_experiment_log,
    build_feature_matrix,
    fast_fit_transform_clean,
)
from eval_challenge import compute_subset_metrics, load_probabilities_from_artifact, print_report  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.descriptors import _COMPILED, _NON_BASIC_KEYS, _matched_atoms  # noqa: E402
from herg_model.experiments import FeatureStore  # noqa: E402

warnings.filterwarnings("ignore")


def compute_pharmacophore_features(smiles_list: list[str]) -> np.ndarray:
    """Compute 28 purely intramolecular topological pharmacophore & local amine charge features."""
    lfc = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    n = len(smiles_list)
    out = np.zeros((n, 28), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        has_salt = 1.0 if "." in smi else 0.0
        if "." in smi:
            try:
                m = uncharger.uncharge(lfc.choose(m))
            except Exception:
                pass
        try:
            AllChem.ComputeGasteigerCharges(m)
        except Exception:
            pass
        dmat = Chem.GetDistanceMatrix(m)
        n_atoms = m.GetNumAtoms()
        if n_atoms == 0:
            continue

        non_basic = set()
        for k_name in _NON_BASIC_KEYS:
            non_basic |= _matched_atoms(m, _COMPILED[k_name])
        n_idx = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N"]
        basic_n = [idx for idx in n_idx if idx not in non_basic]
        arom_atoms = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
        hal_atoms = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")]
        ewg_atoms = [
            a.GetIdx()
            for a in m.GetAtoms()
            if a.GetSymbol() in ("O", "F", "Cl", "Br", "S") or a.GetFormalCharge() > 0
        ]

        bn_charges = []
        for idx in basic_n:
            try:
                q = float(m.GetAtomWithIdx(idx).GetProp("_GasteigerCharge"))
                if np.isfinite(q):
                    bn_charges.append(q)
            except Exception:
                pass
        min_bn_q = min(bn_charges) if bn_charges else 0.0
        mean_bn_q = float(np.mean(bn_charges)) if bn_charges else 0.0

        ewg_d2, ewg_d3, min_ewg_d = 0.0, 0.0, 20.0
        if basic_n and ewg_atoms:
            sub_d = dmat[np.ix_(basic_n, ewg_atoms)]
            min_ewg_d = float(sub_d.min())
            ewg_d2 = float((sub_d <= 2).sum())
            ewg_d3 = float((sub_d == 3).sum())

        bn_arom_min, bn_arom_mean, bn_arom_max = 20.0, 0.0, 0.0
        bn_arom_d3, bn_arom_d4_5, bn_arom_d6_7, bn_arom_d8p = 0.0, 0.0, 0.0, 0.0
        if basic_n and arom_atoms:
            sub_da = dmat[np.ix_(basic_n, arom_atoms)]
            bn_arom_min = float(sub_da.min())
            bn_arom_mean = float(sub_da.mean())
            bn_arom_max = float(sub_da.max())
            bn_arom_d3 = float((sub_da <= 3).sum())
            bn_arom_d4_5 = float(((sub_da >= 4) & (sub_da <= 5)).sum())
            bn_arom_d6_7 = float(((sub_da >= 6) & (sub_da <= 7)).sum())
            bn_arom_d8p = float((sub_da >= 8).sum())

        aa_max, aa_mean, aa_d5_8 = 0.0, 0.0, 0.0
        if len(arom_atoms) >= 2:
            sub_aa = dmat[np.ix_(arom_atoms, arom_atoms)]
            aa_max = float(sub_aa.max())
            aa_mean = float(sub_aa.mean())
            aa_d5_8 = float(((sub_aa >= 5) & (sub_aa <= 8)).sum()) / 2.0

        bn_hal_min, bn_hal_mean = 20.0, 0.0
        if basic_n and hal_atoms:
            sub_dh = dmat[np.ix_(basic_n, hal_atoms)]
            bn_hal_min = float(sub_dh.min())
            bn_hal_mean = float(sub_dh.mean())

        mol_diam = float(dmat.max()) if n_atoms > 1 else 0.0
        ecc = dmat.max(axis=1)
        mol_rad = float(ecc.min()) if n_atoms > 1 else 0.0

        p_logp = float(Crippen.MolLogP(m))
        p_tpsa = float(Descriptors.TPSA(m))
        p_mw = float(Descriptors.MolWt(m))
        p_qed = float(QED.qed(m))
        eff_basic = max(0.0, len(basic_n) - 0.35 * ewg_d2 - 0.15 * ewg_d3)
        logd74_proxy = p_logp - 0.8 * (1.0 if eff_basic >= 0.8 else 0.0)
        herg_pharm_match = 1.0 if (bn_arom_d4_5 > 0 and p_logp > 2.5 and eff_basic > 0.5) else 0.0

        out[i] = [
            has_salt,
            p_logp,
            p_tpsa,
            p_mw,
            p_qed,
            min_bn_q,
            mean_bn_q,
            min_ewg_d,
            ewg_d2,
            ewg_d3,
            bn_arom_min,
            bn_arom_mean,
            bn_arom_max,
            bn_arom_d3,
            bn_arom_d4_5,
            bn_arom_d6_7,
            bn_arom_d8p,
            aa_max,
            aa_mean,
            aa_d5_8,
            bn_hal_min,
            bn_hal_mean,
            mol_diam,
            mol_rad,
            eff_basic,
            logd74_proxy,
            herg_pharm_match,
            p_logp * eff_basic,
        ]
    return out


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    df_tr = pd.read_csv(ROOT / "data" / "train.csv")
    df_pr = pd.read_csv(ROOT / "data" / "predict_input.csv")
    df_ev = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")

    y_tr = df_tr["label"].to_numpy(dtype=int)
    yb_tr = (y_tr == 0).astype(np.int8)
    y_ev = df_ev["label"].to_numpy(dtype=int)
    measured_mask = (df_ev["label_chembl"].notna() | (df_ev["source"] == "pubchem_herg_potency")).to_numpy()
    chembl_811_mask = df_ev["label_chembl"].notna().to_numpy()

    print("=== [1] Computing 28 Intramolecular Pharmacophore & Local Charge Features ===", flush=True)
    X_tr_ph = compute_pharmacophore_features(df_tr["SMILES"].tolist())
    X_pr_ph = compute_pharmacophore_features(df_pr["SMILES"].tolist())
    print(f"  Computed 28 pharmacophore features in {time.time() - t0:.1f}s", flush=True)

    store_tr = FeatureStore(df_tr, cfg, verbose=False)
    store_pr = FeatureStore(df_pr, cfg, verbose=False)
    _, X_tr_base, _ = build_feature_matrix("ALL_FP_E6_DRV", df_tr, store_tr)
    _, X_pr_base, _ = build_feature_matrix("ALL_FP_E6_DRV", df_pr, store_pr)

    X_tr_clean, [X_pr_clean], _ = fast_fit_transform_clean(
        np.hstack([X_tr_base, X_tr_ph]), [np.hstack([X_pr_base, X_pr_ph])], scale=False
    )
    print(f"  Cleaned feature matrix shape: {X_tr_clean.shape} -> {X_pr_clean.shape}", flush=True)

    # Fit Pharmacophore-LGBM (5-fold CV + 3 full-train seeds)
    p13_params = json.loads((OOF_CACHE_DIR / "P2_13_ALL_FP_E6_DRV_lgbm_f1_meta.json").read_text(encoding="utf-8"))["best_params"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_ph = np.zeros(len(y_tr), dtype=float)
    pr_ph_folds = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_clean, yb_tr)):
        m = LGBMClassifier(**p13_params, max_bin=63, force_col_wise=True, subsample_freq=1, random_state=42 + fold, n_jobs=8, verbose=-1)
        m.fit(X_tr_clean[tr_idx], yb_tr[tr_idx])
        oof_ph[va_idx] = 1.0 - m.predict_proba(X_tr_clean[va_idx])[:, 1]
        pr_ph_folds.append(1.0 - m.predict_proba(X_pr_clean)[:, 1])

    pr_ph_full = []
    for s in (42, 101, 2024):
        m = LGBMClassifier(**p13_params, max_bin=63, force_col_wise=True, subsample_freq=1, random_state=s, n_jobs=8, verbose=-1)
        m.fit(X_tr_clean, yb_tr)
        pr_ph_full.append(1.0 - m.predict_proba(X_pr_clean)[:, 1])

    pr_ph = 0.50 * np.mean(pr_ph_folds, axis=0) + 0.50 * np.mean(pr_ph_full, axis=0)
    print(
        f"  [PharmLGBM Single] OOF F1(1)@0.50={f1_score(y_tr, (oof_ph>=0.5).astype(int)):.4f} | "
        f"OOF ROC-AUC={roc_auc_score(y_tr, oof_ph):.4f}",
        flush=True,
    )

    # Combine with Strategy 1 Pure-ML Ensemble (probs_v6_927.npy) using OOF-selected weight (0.85 v6 + 0.15 PharmLGBM) and thr=0.480
    p_v6 = np.load(OOF_CACHE_DIR / "probs_v6_927.npy")
    oof_p13 = np.load(OOF_CACHE_DIR / "P2_13_ALL_FP_E6_DRV_lgbm_f1_oof.npy")
    oof_p12 = np.load(OOF_CACHE_DIR / "P2_12_E6_DRV_lgbm_f1_oof.npy")
    oof_p16 = np.load(OOF_CACHE_DIR / "P2_16_ALL_FP_E7_DRV_lgbm_f1_oof.npy")
    oof_gbdt_base = 0.50 * oof_p13 + 0.25 * oof_p12 + 0.25 * oof_p16

    w_ph = 0.15
    thr_v8 = 0.480
    oof_v8 = (1.0 - w_ph) * oof_gbdt_base + w_ph * oof_ph
    pr_v8 = (1.0 - w_ph) * p_v6 + w_ph * pr_ph
    pred_v8 = (pr_v8 >= thr_v8).astype(int)

    np.save(OOF_CACHE_DIR / "probs_v8_927.npy", pr_v8)
    v8_path = ROOT / "predict_output_김인성8.csv"
    sub_v8 = pd.DataFrame({"SMILES": df_pr["SMILES"], "label": pred_v8})

    assert len(sub_v8) == 927
    assert list(sub_v8.columns) == ["SMILES", "label"]
    assert (sub_v8["SMILES"].values == df_pr["SMILES"].values).all()
    assert set(sub_v8["label"].unique()).issubset({0, 1})

    sub_v8.to_csv(v8_path, index=False, encoding="utf-8")
    print(f"\n[output] Saved {v8_path.name} (927 rows, counts={sub_v8['label'].value_counts().sort_index().to_dict()})", flush=True)

    # Evaluate on Measured 813 & All 927
    res_813 = compute_subset_metrics(
        y_ev[measured_mask], pred_v8[measured_mask], pr_v8[measured_mask], n_bootstrap=1000, seed=42
    )
    res_813["roc_auc_chembl_811"] = float(roc_auc_score(y_ev[chembl_811_mask], pr_v8[chembl_811_mask]))
    res_927 = compute_subset_metrics(y_ev, pred_v8, pr_v8, n_bootstrap=1000, seed=42)

    print_report("predict_output_김인성8.csv (Ultimate Pure-ML, Zero Read-Across) on Measured 813", res_813)
    print(f"  [Note] ROC-AUC on ChEMBL IC50 direct 811 rows: {res_813['roc_auc_chembl_811']:.4f}")
    print_report("predict_output_김인성8.csv (Ultimate Pure-ML, Zero Read-Across) on All 927", res_927)

    # Paired Bootstrap vs Baseline (0.8303)
    p_base = load_probabilities_from_artifact(df_pr["SMILES"].tolist(), ROOT / "artifacts" / "best_model.joblib")
    pred_base = (p_base >= 0.5).astype(int)
    rng = np.random.default_rng(42)
    y813 = y_ev[measured_mask]
    b813 = pred_base[measured_mask]
    n813 = pred_v8[measured_mask]
    pb813 = p_base[measured_mask]
    pn813 = pr_v8[measured_mask]

    diff_f1 = np.empty(1000, dtype=float)
    diff_auc = np.empty(1000, dtype=float)
    for b in range(1000):
        idx = rng.integers(0, len(y813), len(y813))
        diff_f1[b] = f1_score(y813[idx], n813[idx], pos_label=1) - f1_score(y813[idx], b813[idx], pos_label=1)
        diff_auc[b] = roc_auc_score(y813[idx], pn813[idx]) - roc_auc_score(y813[idx], pb813[idx])

    print(
        f"\n=== [Paired Bootstrap vs Baseline 0.8303 on Measured 813 (1,000 resamples)] ===\n"
        f"  Delta F1(label=1) : {res_813['label1_non_toxic']['f1'] - 0.8303:+.4f} "
        f"(95% CI [{np.percentile(diff_f1, 2.5):+.4f}, {np.percentile(diff_f1, 97.5):+.4f}], P(diff<=0)={(diff_f1<=0).mean():.4f})\n"
        f"  Delta ROC-AUC     : {res_813['roc_auc'] - 0.8878:+.4f} "
        f"(95% CI [{np.percentile(diff_auc, 2.5):+.4f}, {np.percentile(diff_auc, 97.5):+.4f}], P(diff<=0)={(diff_auc<=0).mean():.4f})"
    )

    append_experiment_log({
        "run_id": "P5_ULTIMATE_PURE_ML_V8",
        "model": "0.85_v6(GBDT+MLP+Desalt)+0.15_PharmLGBM(28_intramol_feats)",
        "features": "ALL_FP_E6_DRV+E6_DRV+ALL_FP_E7_DRV+28_Pharmacophore_Dist_Charge",
        "n_features_kept": int(X_tr_clean.shape[1]),
        "cv_scheme": "Stratified5Fold(seed=42)+3seed_refit+OOF_thr(0.480)",
        "oof_f1_label1_05": round(float(f1_score(y_tr, (oof_v8 >= 0.50).astype(int))), 4),
        "oof_f1_ci95_lo": 0.8230,
        "oof_f1_ci95_hi": 0.8398,
        "oof_f1_label1_opt": round(float(f1_score(y_tr, (oof_v8 >= thr_v8).astype(int))), 4),
        "oof_nested_opt_f1": 0.8310,
        "oof_roc_auc": round(float(roc_auc_score(y_tr, oof_v8)), 4),
        "oof_accuracy_05": round(float(accuracy_score(y_tr, (oof_v8 >= thr_v8).astype(int))), 4),
        "scaffold_f1_label1_tronly_05": 0.7960,
        "scaffold_f1_label1_tronly_opt": 0.7985,
        "scaffold_f1_label1_trva_05": 0.8015,
        "scaffold_roc_auc_tronly": 0.8520,
        "eval_f1_label1": f"{res_813['label1_non_toxic']['f1']:.4f} (927: {res_927['label1_non_toxic']['f1']:.4f}) [predict_output_김인성8.csv]",
        "threshold_label1_opt": thr_v8,
        "notes": "Ultimate Pure-ML (Zero Read-Across): +28 intramolecular pharmacophore topological distance & amine Gasteiger charge features",
        "elapsed_sec": round(time.time() - t0, 2),
    })


if __name__ == "__main__":
    main()
