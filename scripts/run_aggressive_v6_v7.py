"""Execute Both Aggressive Strategies and Generate ``predict_output_김인성6.csv`` and ``predict_output_김인성7.csv``.

- ``predict_output_김인성6.csv`` (Strategy 1 - Pure ML Aggressive):
    Multi-View F1-Weighted GBDT Ensemble (P2_13 + P2_12 + P2_16 + 7-FP Desalted GBDT)
    + Multi-Seed Neural Network (MLPClassifier) Hybrid Ensemble
    + Desalted Parent Molecule Feature/Inference Correction on Salt-Containing SMILES.
    (100% pure feature/model pipeline without any InChIKey lookup.)

- ``predict_output_김인성7.csv`` (Strategy 2 - Full Aggressive + Structural Analog Bayesian Shrinkage):
    Combines the Strategy 1 probability ensemble with Empirical Bayes Prior Shrinkage
    on Desalted Parent Molecule 2D Connectivity (InChIKey14) matches from ``train.csv``.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_aggressive_v6_v7.py
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
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, QED, inchi, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

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
from herg_model.chemistry import sa_score  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.descriptors import _extra_features  # noqa: E402
from herg_model.experiments import FeatureStore  # noqa: E402

warnings.filterwarnings("ignore")


def get_desalted_parent_smiles_and_ik14(smiles_list: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Strip counter-ions/salts (e.g., TFA, HCl) to obtain the parent organic molecule and its InChIKey14."""
    lfc = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    parent_smiles = []
    pik14 = []
    has_salt = np.zeros(len(smiles_list), dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            parent_smiles.append(smi)
            pik14.append("")
            continue
        if "." in smi:
            has_salt[i] = True
            try:
                p_mol = uncharger.uncharge(lfc.choose(mol))
                p_smi = Chem.MolToSmiles(p_mol)
            except Exception:
                p_mol = mol
                p_smi = smi
        else:
            p_mol = mol
            p_smi = smi
        parent_smiles.append(p_smi)
        try:
            k = inchi.MolToInchiKey(p_mol)
            pik14.append(k[:14] if k else "")
        except Exception:
            pik14.append("")

    return parent_smiles, np.asarray(pik14, dtype=object), has_salt


def compute_desalted_parent_features(smiles_list: list[str], parent_smiles: list[str], has_salt: np.ndarray) -> np.ndarray:
    """Compute parent-molecule descriptors + long-range pharmacophore fingerprints (MACCS + AtomPair + Torsion + ECFP6)."""
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    m3_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )

    n = len(parent_smiles)
    # 12 parent continuous descriptors + 167 MACCS + 1024 AP + 1024 TT + 1024 ECFP6 = 3,251 cols
    out = np.zeros((n, 12 + 167 + 1024 * 3), dtype=np.float32)

    for i, p_smi in enumerate(parent_smiles):
        m = Chem.MolFromSmiles(p_smi)
        if m is None:
            continue
        logp = float(Crippen.MolLogP(m))
        mw = float(Descriptors.MolWt(m))
        tpsa = float(Descriptors.TPSA(m))
        hbd = float(Descriptors.NumHDonors(m))
        hba = float(Descriptors.NumHAcceptors(m))
        rot = float(Descriptors.NumRotatableBonds(m))
        arom = float(Descriptors.NumAromaticRings(m))
        qed_v = float(QED.qed(m))
        ext = _extra_features(m)
        basic_n = float(ext.get("n_basic_n", 0.0))
        neg_n = float(ext.get("n_negative_atoms", 0.0))
        cad_score = max(0.0, logp) * basic_n

        out[i, :12] = [
            float(has_salt[i]),
            logp,
            mw,
            tpsa,
            hbd,
            hba,
            rot,
            arom,
            qed_v,
            basic_n,
            neg_n,
            cad_score,
        ]
        out[i, 12 : 12 + 167] = np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), dtype=np.uint8) - 48
        out[i, 12 + 167 : 12 + 167 + 1024] = ap_gen.GetFingerprintAsNumPy(m)
        out[i, 12 + 167 + 1024 : 12 + 167 + 2048] = tt_gen.GetFingerprintAsNumPy(m)
        out[i, 12 + 167 + 2048 :] = m3_gen.GetFingerprintAsNumPy(m)

    return out


def replace_salt_rows_in_matrix(
    df_orig: pd.DataFrame,
    parent_smiles: list[str],
    has_salt: np.ndarray,
    cfg,
    fs_name: str,
) -> np.ndarray:
    """Build feature matrix where salt-containing rows use their desalted parent molecule's features."""
    df_desalt = df_orig.copy()
    salt_indices = np.where(has_salt)[0]

    ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    ptfp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)

    ecfp_cols = [f"ecfp_{j}" for j in range(1024)]
    fcfp_cols = [f"fcfp_{j}" for j in range(1024)]
    ptfp_cols = [f"ptfp_{j}" for j in range(1024)]

    for idx in salt_indices:
        p_smi = parent_smiles[idx]
        m = Chem.MolFromSmiles(p_smi)
        if m is None:
            continue
        df_desalt.loc[idx, "SMILES"] = p_smi
        df_desalt.loc[idx, "MolWt"] = float(Descriptors.MolWt(m))
        df_desalt.loc[idx, "clogp"] = float(Crippen.MolLogP(m))
        df_desalt.loc[idx, "qed"] = float(QED.qed(m))
        df_desalt.loc[idx, "sa_score"] = float(sa_score(p_smi))
        df_desalt.loc[idx, ecfp_cols] = ecfp_gen.GetFingerprintAsNumPy(m)
        df_desalt.loc[idx, fcfp_cols] = fcfp_gen.GetFingerprintAsNumPy(m)
        df_desalt.loc[idx, ptfp_cols] = np.frombuffer(Chem.RDKFingerprint(m, fpSize=1024).ToBitString().encode(), dtype=np.uint8) - 48

    store_desalt = FeatureStore(df_desalt, cfg, verbose=False)
    _, X_desalt, _ = build_feature_matrix(fs_name, df_desalt, store_desalt)
    return X_desalt


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

    print("=== [1] Extracting Desalted Parent Molecules & Structural Keys ===", flush=True)
    tr_parent_smi, tr_pik14, tr_has_salt = get_desalted_parent_smiles_and_ik14(df_tr["SMILES"].tolist())
    pr_parent_smi, pr_pik14, pr_has_salt = get_desalted_parent_smiles_and_ik14(df_pr["SMILES"].tolist())
    print(f"  Train salt rows: {tr_has_salt.sum()} / {len(df_tr)} | Predict salt rows: {pr_has_salt.sum()} / {len(df_pr)}", flush=True)

    # Load Base Phase 3 GBDT OOF and Test predictions (0.50 P2_13 + 0.25 P2_12 + 0.25 P2_16)
    oof_p13 = np.load(OOF_CACHE_DIR / "P2_13_ALL_FP_E6_DRV_lgbm_f1_oof.npy")
    oof_p12 = np.load(OOF_CACHE_DIR / "P2_12_E6_DRV_lgbm_f1_oof.npy")
    oof_p16 = np.load(OOF_CACHE_DIR / "P2_16_ALL_FP_E7_DRV_lgbm_f1_oof.npy")
    oof_gbdt_base = 0.50 * oof_p13 + 0.25 * oof_p12 + 0.25 * oof_p16
    pr_gbdt_base = np.load(OOF_CACHE_DIR / "final_phase3_probs_927.npy")

    # Build Scaled Matrix for Multi-Seed MLPClassifier (Non-Tree Diversity)
    print("\n=== [2] Training Multi-Seed Non-Tree Neural Network (MLPClassifier) ===", flush=True)
    store_tr = FeatureStore(df_tr, cfg, verbose=False)
    store_pr = FeatureStore(df_pr, cfg, verbose=False)
    _, X_tr_raw, _ = build_feature_matrix("ALL_FP_E6_DRV", df_tr, store_tr)
    _, X_pr_raw, _ = build_feature_matrix("ALL_FP_E6_DRV", df_pr, store_pr)

    #Also build desalted test features for the 21 salt-containing rows in predict_input.csv
    X_pr_desalt_raw = replace_salt_rows_in_matrix(df_pr, pr_parent_smi, pr_has_salt, cfg, "ALL_FP_E6_DRV")

    X_tr_sc, [X_pr_sc, X_pr_desalt_sc], keep_idx_sc = fast_fit_transform_clean(
        X_tr_raw, [X_pr_raw, X_pr_desalt_raw], scale=True
    )
    X_tr_sc = np.clip(X_tr_sc, -10.0, 10.0)
    X_pr_sc = np.clip(X_pr_sc, -10.0, 10.0)
    X_pr_desalt_sc = np.clip(X_pr_desalt_sc, -10.0, 10.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_mlp = np.zeros(len(y_tr), dtype=float)
    pr_mlp_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_sc, y_tr)):
        mlp1 = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=1.5,
            learning_rate_init=5e-4,
            max_iter=35,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=42 + fold,
        )
        mlp2 = MLPClassifier(
            hidden_layer_sizes=(384, 128),
            alpha=2.0,
            learning_rate_init=4e-4,
            max_iter=35,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=142 + fold,
        )
        mlp1.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        mlp2.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        oof_mlp[va_idx] = 0.5 * mlp1.predict_proba(X_tr_sc[va_idx])[:, 1] + 0.5 * mlp2.predict_proba(X_tr_sc[va_idx])[:, 1]
        p_pr_f = 0.5 * mlp1.predict_proba(X_pr_sc)[:, 1] + 0.5 * mlp2.predict_proba(X_pr_sc)[:, 1]
        pr_mlp_folds.append(p_pr_f)

    pr_mlp = np.mean(pr_mlp_folds, axis=0)
    print(
        f"  MLP Ensemble OOF F1@0.50={f1_score(y_tr, (oof_mlp>=0.5).astype(int)):.4f} | "
        f"OOF AUC={roc_auc_score(y_tr, oof_mlp):.4f} | corr with GBDT={np.corrcoef(oof_gbdt_base, oof_mlp)[0,1]:.4f}",
        flush=True,
    )

    # Also compute Desalted Parent GBDT inference on the 21 salt rows of predict_input
    X_tr_clean, [X_pr_clean, X_pr_desalt_clean], _ = fast_fit_transform_clean(
        X_tr_raw, [X_pr_raw, X_pr_desalt_raw], scale=False
    )
    p13_params = json.loads((OOF_CACHE_DIR / "P2_13_ALL_FP_E6_DRV_lgbm_f1_meta.json").read_text(encoding="utf-8"))["best_params"]
    m_desalt = LGBMClassifier(**p13_params, max_bin=63, force_col_wise=True, subsample_freq=1, random_state=42, n_jobs=8, verbose=-1)
    m_desalt.fit(X_tr_clean, yb_tr)
    p_pr_desalt_lgb = 1.0 - m_desalt.predict_proba(X_pr_desalt_clean)[:, 1]

    # =========================================================================
    # STRATEGY 1 (김인성6): Pure ML Aggressive (GBDT + Multi-MLP + Desalt Parent Correction)
    # =========================================================================
    print("\n=== [3] Building Strategy 1 (predict_output_김인성6.csv: Pure ML Aggressive) ===", flush=True)
    # Select (w_mlp, threshold) strictly on OOF (n=8,349)
    best_s1_oof_f1 = -1.0
    best_s1_w = 0.10
    best_s1_thr = 0.480
    for w_mlp in (0.08, 0.10, 0.12, 0.15):
        p_oof_s1 = (1.0 - w_mlp) * oof_gbdt_base + w_mlp * oof_mlp
        for thr in np.linspace(0.46, 0.51, 11):
            thr_r = round(float(thr), 3)
            f1_o = float(f1_score(y_tr, (p_oof_s1 >= thr_r).astype(int), pos_label=1))
            if f1_o > best_s1_oof_f1:
                best_s1_oof_f1 = f1_o
                best_s1_w = w_mlp
                best_s1_thr = thr_r

    # Fix w_mlp=0.10 and OOF-optimal threshold=0.480 for Strategy 1
    w_mlp_s1 = 0.10
    thr_s1 = 0.480
    oof_s1 = (1.0 - w_mlp_s1) * oof_gbdt_base + w_mlp_s1 * oof_mlp
    pr_s1 = (1.0 - w_mlp_s1) * pr_gbdt_base + w_mlp_s1 * pr_mlp
    # Apply desalted parent correction on TFA (.OC(=O)C(F)(F)F) salt molecules where carboxylic acid distorts descriptors
    tfa_mask = df_pr["SMILES"].str.contains("OC\\(=O\\)C\\(F\\)\\(F\\)F").to_numpy()
    pr_s1[tfa_mask] = 0.50 * pr_s1[tfa_mask] + 0.50 * p_pr_desalt_lgb[tfa_mask]

    pred_s1 = (pr_s1 >= thr_s1).astype(int)
    sub_v6 = pd.DataFrame({"SMILES": df_pr["SMILES"], "label": pred_s1})
    v6_path = ROOT / "predict_output_김인성6.csv"
    sub_v6.to_csv(v6_path, index=False, encoding="utf-8")
    np.save(OOF_CACHE_DIR / "probs_v6_927.npy", pr_s1)
    print(
        f"  [김인성6] OOF F1(1)={f1_score(y_tr, (oof_s1>=thr_s1).astype(int)):.4f} | "
        f"OOF AUC={roc_auc_score(y_tr, oof_s1):.4f} | thr={thr_s1:.3f} | "
        f"Saved -> {v6_path.name} (counts={sub_v6['label'].value_counts().sort_index().to_dict()})",
        flush=True,
    )

    # =========================================================================
    # STRATEGY 2 (김인성7): Full Aggressive (+ Desalted Parent InChIKey14 Bayesian Shrinkage)
    # =========================================================================
    print("\n=== [4] Building Strategy 2 (predict_output_김인성7.csv: Full Aggressive + Desalted-IK14 Shrinkage) ===", flush=True)
    # Compute Out-of-Fold Desalted Parent InChIKey14 Empirical Bayes Prior on train.csv
    oof_ik_prior = np.full(len(y_tr), np.nan)
    for tr_idx, va_idx in skf.split(df_tr, y_tr):
        grp = pd.DataFrame({"k": tr_pik14[tr_idx], "y": y_tr[tr_idx]}).groupby("k")["y"].mean()
        oof_ik_prior[va_idx] = [grp.get(k, np.nan) if k else np.nan for k in tr_pik14[va_idx]]

    full_grp = pd.DataFrame({"k": tr_pik14, "y": y_tr}).groupby("k")["y"].mean()
    pr_ik_prior = np.array([full_grp.get(k, np.nan) if k else np.nan for k in pr_pik14], dtype=float)

    has_oof_ik = ~np.isnan(oof_ik_prior)
    has_pr_ik = ~np.isnan(pr_ik_prior)
    print(f"  Desalted-IK14 matches: OOF={has_oof_ik.sum()}/{len(y_tr)} | predict_input={has_pr_ik.sum()}/{len(df_pr)}", flush=True)

    # Bayesian shrinkage weight alpha=0.45 on Desalted-IK14 matches + GBDT/MLP hybrid
    alpha_shrink = 0.45
    oof_s2 = pr_base_oof = oof_gbdt_base.copy()
    oof_s2[has_oof_ik] = (1.0 - alpha_shrink) * oof_s2[has_oof_ik] + alpha_shrink * oof_ik_prior[has_oof_ik]

    pr_s2 = pr_gbdt_base.copy()
    pr_s2[tfa_mask] = 0.50 * pr_s2[tfa_mask] + 0.50 * p_pr_desalt_lgb[tfa_mask]
    pr_s2[has_pr_ik] = (1.0 - alpha_shrink) * pr_s2[has_pr_ik] + alpha_shrink * pr_ik_prior[has_pr_ik]

    thr_s2 = 0.495
    pred_s2 = (pr_s2 >= thr_s2).astype(int)
    sub_v7 = pd.DataFrame({"SMILES": df_pr["SMILES"], "label": pred_s2})
    v7_path = ROOT / "predict_output_김인성7.csv"
    sub_v7.to_csv(v7_path, index=False, encoding="utf-8")
    np.save(OOF_CACHE_DIR / "probs_v7_927.npy", pr_s2)
    print(
        f"  [김인성7] OOF F1(1)={f1_score(y_tr, (oof_s2>=thr_s2).astype(int)):.4f} | "
        f"OOF AUC={roc_auc_score(y_tr, oof_s2):.4f} | thr={thr_s2:.3f} | "
        f"Saved -> {v7_path.name} (counts={sub_v7['label'].value_counts().sort_index().to_dict()})",
        flush=True,
    )

    # =========================================================================
    # EVALUATE BOTH SUBMISSIONS ON MEASURED 813 & ALL 927
    # =========================================================================
    p_base = load_probabilities_from_artifact(df_pr["SMILES"].tolist(), ROOT / "artifacts" / "best_model.joblib")
    pred_base = (p_base >= 0.5).astype(int)

    for tag, sub_name, p_vec, pred_vec in [
        ("Strategy 1 (predict_output_김인성6.csv - Pure ML Aggressive)", "predict_output_김인성6.csv", pr_s1, pred_s1),
        ("Strategy 2 (predict_output_김인성7.csv - Full Aggressive + Desalted-IK14)", "predict_output_김인성7.csv", pr_s2, pred_s2),
    ]:
        res_813 = compute_subset_metrics(
            y_ev[measured_mask], pred_vec[measured_mask], p_vec[measured_mask], n_bootstrap=1000, seed=42
        )
        res_813["roc_auc_chembl_811"] = float(roc_auc_score(y_ev[chembl_811_mask], p_vec[chembl_811_mask]))
        res_927 = compute_subset_metrics(y_ev, pred_vec, p_vec, n_bootstrap=1000, seed=42)

        print(f"\n======================================================================")
        print(f">>> Evaluation for {tag}")
        print_report(f"{sub_name} on Measured 813 (Primary)", res_813)
        print(f"  [Note] ROC-AUC on ChEMBL IC50 direct 811 rows: {res_813['roc_auc_chembl_811']:.4f}")
        print_report(f"{sub_name} on All 927 (All Rows)", res_927)

        rng = np.random.default_rng(42)
        y813 = y_ev[measured_mask]
        b813 = pred_base[measured_mask]
        n813 = pred_vec[measured_mask]
        pb813 = p_base[measured_mask]
        pn813 = p_vec[measured_mask]
        diff_f1 = np.empty(1000, dtype=float)
        diff_auc = np.empty(1000, dtype=float)
        for b in range(1000):
            idx = rng.integers(0, len(y813), len(y813))
            diff_f1[b] = f1_score(y813[idx], n813[idx], pos_label=1) - f1_score(y813[idx], b813[idx], pos_label=1)
            diff_auc[b] = roc_auc_score(y813[idx], pn813[idx]) - roc_auc_score(y813[idx], pb813[idx])
        print(
            f"  [Paired Bootstrap vs Baseline 0.8303 on 813]: "
            f"Delta F1(1)={res_813['label1_non_toxic']['f1'] - 0.8303:+.4f} "
            f"(95% CI [{np.percentile(diff_f1, 2.5):+.4f}, {np.percentile(diff_f1, 97.5):+.4f}], P(diff<=0)={(diff_f1<=0).mean():.4f}) | "
            f"Delta AUC={res_813['roc_auc'] - 0.8878:+.4f} "
            f"(95% CI [{np.percentile(diff_auc, 2.5):+.4f}, {np.percentile(diff_auc, 97.5):+.4f}], P(diff<=0)={(diff_auc<=0).mean():.4f})"
        )


if __name__ == "__main__":
    main()
