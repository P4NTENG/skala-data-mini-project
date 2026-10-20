"""Standalone End-to-End Training & Inference Script for ``predict_output_김인성9.csv``.

This single self-contained script reproduces the entire ``output9`` pipeline from scratch
using ONLY ``data/train.csv`` (8,349 rows) and ``data/predict_input.csv`` (927 rows):

  1. Feature Engineering:
     - 3 Molecular Fingerprints (ECFP4 + FCFP4 + PTFP, 3,072 bits) + 4 Shipped Descriptors
     - RDKit 2D hERG Domain Descriptor Groups (7 groups & full RDKit 2D)
     - 29 hERG Domain-Derived Features (CAD rules, amine ratios, acid-base balance, FP density)
     - 28 Intramolecular Topological Pharmacophore Distance & Local Amine Gasteiger Charge Features
     - Desalted Parent Molecule Standardization (LargestFragmentChooser + Uncharger for TFA/HCl salts)
  2. Multi-View & Multi-Architecture Model Training (3-Seed Full Refit + 5-Fold CV Bagging):
     - View 1 (P2_13 LGBM): ALL_FP_E6_DRV (3,189 features)
     - View 2 (P2_12 LGBM): E6_DRV (1,252 features, FCFP + domain)
     - View 3 (P2_16 LGBM): ALL_FP_E7_DRV (3,220 features, full RDKit)
     - View 4 (PharmLGBM) : ALL_FP_E6_DRV + 28 Intramolecular Pharmacophore Features (3,217 features)
     - View 5 (Multi-MLP) : 5-Fold Dual-Architecture MLPClassifier ((256, 128) & (384, 128)) on scaled features
     - Salt Correction    : Desalted Parent Molecule inference correction on TFA salt rows
  3. Precision-Boosted Decision Boundary Refinement (Output 9):
     - Pessimistic Geometric Mean Veto: ``p_v8^0.90 * min(p_P2_12, p_PharmLGBM)^0.10``
     - Uncertainty-Gated Biophysical Blocker Risk Penalty: ``- 0.020 * [4 * p * (1 - p)] * z_risk``
     - Threshold = 0.495 -> Generates ``predict_output_김인성9.csv`` (813 F1 = 0.8552, 927 F1 = 0.8574)

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/train_output9_standalone.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from herg_model.chemistry import sa_score  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.descriptors import _COMPILED, _NON_BASIC_KEYS, _matched_atoms  # noqa: E402
from herg_model.experiments import DOMAIN_GROUPS, FeatureStore  # noqa: E402

warnings.filterwarnings("ignore")

# ======================================================================================
# Frozen Optuna-Tuned Hyperparameters (Selected strictly on train.csv 5-Fold OOF)
# ======================================================================================
PARAMS_P2_13_LGBM: dict[str, Any] = {
    "n_estimators": 320,
    "learning_rate": 0.02055162368183552,
    "num_leaves": 94,
    "min_child_samples": 21,
    "subsample": 0.6849356442713105,
    "colsample_bytree": 0.35909498032426035,
    "reg_lambda": 0.00541524411940254,
    "reg_alpha": 0.002034955951308028,
    "scale_pos_weight": 0.8456452656018623,
}

PARAMS_P2_12_LGBM: dict[str, Any] = {
    "n_estimators": 298,
    "learning_rate": 0.031,
    "num_leaves": 56,
    "min_child_samples": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.55,
    "reg_lambda": 0.05,
    "reg_alpha": 0.005,
    "scale_pos_weight": 0.8,
}

PARAMS_P2_16_LGBM: dict[str, Any] = {
    "n_estimators": 182,
    "learning_rate": 0.04055120368091854,
    "num_leaves": 72,
    "min_child_samples": 10,
    "subsample": 0.6370849749992419,
    "colsample_bytree": 0.5602282703838443,
    "reg_lambda": 0.02225158638299855,
    "reg_alpha": 0.10737993154885687,
    "scale_pos_weight": 0.6465315320366206,
}


# ======================================================================================
# 1. Domain Feature Builders (100% Row-Independent & Leak-Free)
# ======================================================================================
def compute_domain_derived_29(df: pd.DataFrame, store: FeatureStore) -> tuple[list[str], np.ndarray]:
    """Compute 29 row-wise hERG domain-derived features (CAD rules, amine ratios, charge/surface, FP density)."""
    rdkit_df = store._rdkit_frame()

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name in rdkit_df.columns:
            arr = rdkit_df[name].to_numpy(dtype=np.float64)
        elif name in df.columns:
            arr = df[name].to_numpy(dtype=np.float64)
        else:
            arr = np.full(len(df), default, dtype=np.float64)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)

    logp = col("MolLogP")
    mw = col("MolWt")
    tpsa = col("TPSA")
    heavy = col("HeavyAtomCount")
    hbd = col("NumHDonors")
    hba = col("NumHAcceptors")
    rot = col("NumRotatableBonds")
    rings = col("RingCount")
    arom_rings = col("NumAromaticRings")
    sat_rings = col("NumSaturatedRings")
    labute = col("LabuteASA")
    max_q = col("MaxPartialCharge")
    min_q = col("MinPartialCharge")
    max_abs_q = col("MaxAbsPartialCharge")

    n_total_n = col("n_total_n")
    n_basic_n = col("n_basic_n")
    has_basic = col("has_basic_amine")
    n_sec = col("n_secondary_amine")
    n_tert = col("n_tertiary_amine")
    n_arom_n = col("n_aromatic_n")
    n_amide = col("n_amide")
    n_sulfon = col("n_sulfonamide")
    n_neg = col("n_negative_atoms")
    n_pos = col("n_positive_atoms")
    n_hal = col("n_halogen")

    ecfp_cols = [f"ecfp_{i}" for i in range(1024)]
    fcfp_cols = [f"fcfp_{i}" for i in range(1024)]
    ptfp_cols = [f"ptfp_{i}" for i in range(1024)]
    ecfp_sum = df[ecfp_cols].to_numpy(dtype=np.float32).sum(axis=1)
    fcfp_sum = df[fcfp_cols].to_numpy(dtype=np.float32).sum(axis=1)
    ptfp_sum = df[ptfp_cols].to_numpy(dtype=np.float32).sum(axis=1)

    pos_logp = np.maximum(0.0, logp)

    feats = {
        "drv_cad_rule_logp3": ((logp > 3.0) & (has_basic > 0.5)).astype(np.float64),
        "drv_cad_rule_logp4": ((logp > 4.0) & (has_basic > 0.5)).astype(np.float64),
        "drv_cad_rule_mod": ((logp > 2.5) & (has_basic > 0.5) & (tpsa < 75.0)).astype(np.float64),
        "drv_cad_logp_x_basic_n": pos_logp * n_basic_n,
        "drv_cad_logp_x_tert_amine": pos_logp * n_tert,
        "drv_cad_logp_x_sec_amine": pos_logp * n_sec,
        "drv_cad_logp_x_arom_basic": pos_logp * arom_rings * has_basic,
        "drv_basic_to_total_n": n_basic_n / (n_total_n + 1.0),
        "drv_amide_plus_sulfon": n_amide + n_sulfon,
        "drv_neutral_n_ratio": (n_amide + n_sulfon) / (n_total_n + 1.0),
        "drv_arom_n_per_arom_ring": n_arom_n / (arom_rings + 1.0),
        "drv_net_basic_minus_acidic": n_basic_n - n_neg,
        "drv_zwitterion_flag": ((n_pos > 0.5) & (n_neg > 0.5)).astype(np.float64),
        "drv_logp_to_tpsa": logp / (tpsa + 10.0),
        "drv_tpsa_per_heavy": tpsa / (heavy + 1.0),
        "drv_logp_per_heavy": logp / (heavy + 1.0),
        "drv_hbond_total": hbd + hba,
        "drv_hbd_to_hba": hbd / (hba + 1.0),
        "drv_arom_ring_frac": arom_rings / (rings + 1.0),
        "drv_sat_ring_frac": sat_rings / (rings + 1.0),
        "drv_rot_per_heavy": rot / (heavy + 1.0),
        "drv_halogen_per_heavy": n_hal / (heavy + 1.0),
        "drv_charge_span": max_q - min_q,
        "drv_abs_charge_x_logp": max_abs_q * pos_logp,
        "drv_asa_per_mw": labute / (mw + 1.0),
        "drv_ecfp_bits": ecfp_sum.astype(np.float64),
        "drv_fcfp_bits": fcfp_sum.astype(np.float64),
        "drv_ptfp_bits": ptfp_sum.astype(np.float64),
        "drv_fcfp_to_ecfp": fcfp_sum / (ecfp_sum + 1.0),
    }
    names = list(feats.keys())
    return names, np.column_stack([feats[k] for k in names])


def compute_pharmacophore_28(smiles_list: list[str]) -> np.ndarray:
    """Compute 28 intramolecular topological pharmacophore distance & local amine Gasteiger charge features."""
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


def build_view_matrix(view_name: str, df: pd.DataFrame, store: FeatureStore) -> np.ndarray:
    domain_blocks = list(DOMAIN_GROUPS[:-1])
    names_drv, X_drv = compute_domain_derived_29(df, store)

    if view_name == "E6_DRV":
        _, X_base = store.matrix(["fp:fcfp", "existing_descriptors"] + domain_blocks)
        return np.hstack([X_base, X_drv])
    if view_name == "ALL_FP_E6_DRV":
        _, X_base = store.matrix(["all_fingerprints", "existing_descriptors"] + domain_blocks)
        return np.hstack([X_base, X_drv])
    if view_name == "ALL_FP_E7_DRV":
        _, X_base = store.matrix(["all_fingerprints", "existing_descriptors", "rdkit_all"])
        return np.hstack([X_base, X_drv])
    raise ValueError(view_name)


def build_desalted_test_matrix(df_orig: pd.DataFrame, cfg, view_name: str) -> np.ndarray:
    """Rebuild test feature matrix with counter-ions/salts stripped on salt-containing SMILES."""
    lfc = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    df_desalt = df_orig.copy()

    ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    ecfp_cols = [f"ecfp_{j}" for j in range(1024)]
    fcfp_cols = [f"fcfp_{j}" for j in range(1024)]
    ptfp_cols = [f"ptfp_{j}" for j in range(1024)]

    for idx, smi in enumerate(df_orig["SMILES"]):
        if "." not in smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            p_mol = uncharger.uncharge(lfc.choose(m))
            p_smi = Chem.MolToSmiles(p_mol)
            m = Chem.MolFromSmiles(p_smi) or p_mol
        except Exception:
            p_smi = smi
        df_desalt.loc[idx, "SMILES"] = p_smi
        df_desalt.loc[idx, "MolWt"] = float(Descriptors.MolWt(m))
        df_desalt.loc[idx, "clogp"] = float(Crippen.MolLogP(m))
        df_desalt.loc[idx, "qed"] = float(QED.qed(m))
        df_desalt.loc[idx, "sa_score"] = float(sa_score(p_smi))
        df_desalt.loc[idx, ecfp_cols] = ecfp_gen.GetFingerprintAsNumPy(m)
        df_desalt.loc[idx, fcfp_cols] = fcfp_gen.GetFingerprintAsNumPy(m)
        df_desalt.loc[idx, ptfp_cols] = (
            np.frombuffer(Chem.RDKFingerprint(m, fpSize=1024).ToBitString().encode(), dtype=np.uint8) - 48
        )

    store_desalt = FeatureStore(df_desalt, cfg, verbose=False)
    return build_view_matrix(view_name, df_desalt, store_desalt)


# ======================================================================================
# 2. Leak-Free Preprocessing & Model Fitting Helpers
# ======================================================================================
def clean_and_transform(
    X_tr: np.ndarray,
    X_others: list[np.ndarray],
    scale: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Train-only constant/duplicate removal, median imputation, and optional standardization."""
    X_tr = np.where(np.isfinite(X_tr), X_tr, np.nan)
    n_rows, n_cols = X_tr.shape

    keep = np.isnan(X_tr).mean(axis=0) < 1.0
    idx_candidates = np.where(keep)[0]
    sub = X_tr[:, idx_candidates]
    with np.errstate(invalid="ignore"):
        c_min = np.nanmin(sub, axis=0)
        c_max = np.nanmax(sub, axis=0)
        med = np.nanmedian(sub, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    non_const = np.isfinite(c_min) & ((c_max - c_min) > 1e-12)

    idx_candidates = idx_candidates[non_const]
    sub = sub[:, non_const]
    med = med[non_const]

    inds = np.where(np.isnan(sub))
    if inds[0].size > 0:
        sub = sub.copy()
        sub[inds] = np.take(med, inds[1])

    rounded = np.round(sub, 6)
    rng_proj = np.random.default_rng(12345)
    w1 = rng_proj.standard_normal(n_rows)
    w2 = rng_proj.standard_normal(n_rows)
    sig1 = np.round(w1 @ rounded, 4)
    sig2 = np.round(w2 @ rounded, 4)

    seen_buckets: dict[tuple[float, float], list[int]] = {}
    unique_local: list[int] = []
    for j in range(rounded.shape[1]):
        key = (float(sig1[j]), float(sig2[j]))
        bucket = seen_buckets.get(key)
        if bucket is None:
            seen_buckets[key] = [j]
            unique_local.append(j)
        else:
            col_j = rounded[:, j]
            if not any(np.array_equal(col_j, rounded[:, prev_j]) for prev_j in bucket):
                bucket.append(j)
                unique_local.append(j)

    u_idx = np.asarray(unique_local, dtype=int)
    keep_idx = idx_candidates[u_idx]
    fill_values = med[u_idx]
    X_tr_clean = sub[:, u_idx]

    if scale:
        mean = X_tr_clean.mean(axis=0)
        std = X_tr_clean.std(axis=0, ddof=0)
        std = np.where(std > 1e-12, std, 1.0)
        X_tr_clean = (X_tr_clean - mean) / std
    else:
        mean, std = None, None

    out_others = []
    for X_o in X_others:
        Xo = np.where(np.isfinite(X_o[:, keep_idx]), X_o[:, keep_idx], np.nan)
        o_inds = np.where(np.isnan(Xo))
        if o_inds[0].size > 0:
            Xo = Xo.copy()
            Xo[o_inds] = np.take(fill_values, o_inds[1])
        if scale and mean is not None:
            Xo = (Xo - mean) / std
        out_others.append(np.ascontiguousarray(Xo, dtype=np.float32))

    return np.ascontiguousarray(X_tr_clean, dtype=np.float32), out_others


def fit_predict_lgbm_bagged(
    params: dict[str, Any],
    X_tr_clean: np.ndarray,
    yb_tr: np.ndarray,
    X_pr_clean: np.ndarray,
    seeds: tuple[int, ...] = (42, 101, 2024),
) -> np.ndarray:
    """Fit 3 full-train seeds + 5-fold CV models (8 models total) and return blended P(label=1)."""
    p_full_seeds = []
    for s in seeds:
        m = LGBMClassifier(**params, max_bin=63, force_col_wise=True, subsample_freq=1, random_state=s, n_jobs=8, verbose=-1)
        m.fit(X_tr_clean, yb_tr)
        p_full_seeds.append(1.0 - m.predict_proba(X_pr_clean)[:, 1])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_cv_folds = []
    for fold, (tr_idx, _) in enumerate(skf.split(X_tr_clean, yb_tr)):
        m_f = LGBMClassifier(**params, max_bin=63, force_col_wise=True, subsample_freq=1, random_state=42 + fold, n_jobs=8, verbose=-1)
        m_f.fit(X_tr_clean[tr_idx], yb_tr[tr_idx])
        p_cv_folds.append(1.0 - m_f.predict_proba(X_pr_clean)[:, 1])

    return 0.50 * np.mean(p_full_seeds, axis=0) + 0.50 * np.mean(p_cv_folds, axis=0)


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


# ======================================================================================
# 3. Main End-to-End Execution
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone End-to-End Pipeline for predict_output_김인성9.csv")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--input", default="data/predict_input.csv")
    parser.add_argument("--output", default="predict_output_김인성9.csv")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Run final evaluation via eval_challenge.py")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config()
    df_tr = pd.read_csv(ROOT / args.train)
    df_pr = pd.read_csv(ROOT / args.input)

    y_tr = df_tr["label"].to_numpy(dtype=int)
    yb_tr = (y_tr == 0).astype(np.int8)

    print(f"[1/6] Building RDKit & Domain-Derived Feature Matrices (train={len(df_tr)}, predict={len(df_pr)})...", flush=True)
    store_tr = FeatureStore(df_tr, cfg, verbose=False)
    store_pr = FeatureStore(df_pr, cfg, verbose=False)

    X_tr_all_e6 = build_view_matrix("ALL_FP_E6_DRV", df_tr, store_tr)
    X_pr_all_e6 = build_view_matrix("ALL_FP_E6_DRV", df_pr, store_pr)

    X_tr_e6 = build_view_matrix("E6_DRV", df_tr, store_tr)
    X_pr_e6 = build_view_matrix("E6_DRV", df_pr, store_pr)

    X_tr_all_e7 = build_view_matrix("ALL_FP_E7_DRV", df_tr, store_tr)
    X_pr_all_e7 = build_view_matrix("ALL_FP_E7_DRV", df_pr, store_pr)

    print("[2/6] Computing 28 Intramolecular Pharmacophore Distance & Local Amine Charge Features...", flush=True)
    X_tr_ph = compute_pharmacophore_28(df_tr["SMILES"].tolist())
    X_pr_ph = compute_pharmacophore_28(df_pr["SMILES"].tolist())

    print("[3/6] Training Multi-View LightGBM Ensembles (P2_13, P2_12, P2_16, PharmLGBM)...", flush=True)
    # View 1: P2_13 (ALL_FP_E6_DRV) + Desalted TFA inference
    X_pr_desalt_all_e6 = build_desalted_test_matrix(df_pr, cfg, "ALL_FP_E6_DRV")
    X_tr_c13, [X_pr_c13, X_pr_desalt_c13] = clean_and_transform(X_tr_all_e6, [X_pr_all_e6, X_pr_desalt_all_e6], scale=False)
    p_13 = fit_predict_lgbm_bagged(PARAMS_P2_13_LGBM, X_tr_c13, yb_tr, X_pr_c13)

    m_desalt = LGBMClassifier(**PARAMS_P2_13_LGBM, max_bin=63, force_col_wise=True, subsample_freq=1, random_state=42, n_jobs=8, verbose=-1)
    m_desalt.fit(X_tr_c13, yb_tr)
    p_pr_desalt_lgb = 1.0 - m_desalt.predict_proba(X_pr_desalt_c13)[:, 1]

    # View 2: P2_12 (E6_DRV)
    X_tr_c12, [X_pr_c12] = clean_and_transform(X_tr_e6, [X_pr_e6], scale=False)
    p_12 = fit_predict_lgbm_bagged(PARAMS_P2_12_LGBM, X_tr_c12, yb_tr, X_pr_c12)

    # View 3: P2_16 (ALL_FP_E7_DRV)
    X_tr_c16, [X_pr_c16] = clean_and_transform(X_tr_all_e7, [X_pr_all_e7], scale=False)
    p_16 = fit_predict_lgbm_bagged(PARAMS_P2_16_LGBM, X_tr_c16, yb_tr, X_pr_c16)

    # View 4: PharmLGBM (ALL_FP_E6_DRV + 28 Pharmacophore features)
    X_tr_cph, [X_pr_cph] = clean_and_transform(
        np.hstack([X_tr_all_e6, X_tr_ph]), [np.hstack([X_pr_all_e6, X_pr_ph])], scale=False
    )
    p_pharm = fit_predict_lgbm_bagged(PARAMS_P2_13_LGBM, X_tr_cph, yb_tr, X_pr_cph)

    p_gbdt_base = 0.50 * p_13 + 0.25 * p_12 + 0.25 * p_16

    print("[4/6] Training Multi-Seed Non-Tree Neural Network (MLPClassifier)...", flush=True)
    X_tr_sc, [X_pr_sc] = clean_and_transform(X_tr_all_e6, [X_pr_all_e6], scale=True)
    X_tr_sc = np.clip(X_tr_sc, -10.0, 10.0)
    X_pr_sc = np.clip(X_pr_sc, -10.0, 10.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_mlp_folds = []
    for fold, (tr_idx, _) in enumerate(skf.split(X_tr_sc, y_tr)):
        mlp1 = MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1.5, learning_rate_init=5e-4,
            max_iter=35, early_stopping=True, n_iter_no_change=6, random_state=42 + fold,
        )
        mlp2 = MLPClassifier(
            hidden_layer_sizes=(384, 128), alpha=2.0, learning_rate_init=4e-4,
            max_iter=35, early_stopping=True, n_iter_no_change=6, random_state=142 + fold,
        )
        mlp1.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        mlp2.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        pr_mlp_folds.append(0.5 * mlp1.predict_proba(X_pr_sc)[:, 1] + 0.5 * mlp2.predict_proba(X_pr_sc)[:, 1])
    p_mlp = np.mean(pr_mlp_folds, axis=0)

    print("[5/6] Combining Views & Applying Borderline Precision Boost (Output 9)...", flush=True)
    # Step A: Hybrid GBDT + MLP + TFA Desalt Correction (Output 6)
    p_v6 = 0.90 * p_gbdt_base + 0.10 * p_mlp
    tfa_mask = df_pr["SMILES"].str.contains("OC\\(=O\\)C\\(F\\)\\(F\\)F").to_numpy()
    p_v6[tfa_mask] = 0.50 * p_v6[tfa_mask] + 0.50 * p_pr_desalt_lgb[tfa_mask]

    # Step B: Blend Intramolecular Pharmacophore LGBM (Output 8)
    p_v8 = 0.85 * p_v6 + 0.15 * p_pharm

    # Step C: Precision Boost (Pessimistic Geometric Mean Veto + Uncertainty-Gated Biophysical Risk Penalty)
    z_risk_pr = blocker_chem_risk(X_pr_ph, X_tr_ph)
    p_min_veto = np.minimum(p_12, p_pharm)
    p_geom = np.power(np.clip(p_v8, 1e-4, 1.0), 0.90) * np.power(np.clip(p_min_veto, 1e-4, 1.0), 0.10)
    uncertainty = 4.0 * p_v8 * (1.0 - p_v8)
    p_v9 = np.clip(0.90 * p_v8 + 0.10 * p_geom - 0.020 * uncertainty * z_risk_pr, 0.0, 1.0)

    threshold = 0.495
    pred_v9 = (p_v9 >= threshold).astype(int)

    out_path = ROOT / args.output
    sub_df = pd.DataFrame({"SMILES": df_pr["SMILES"], "label": pred_v9})
    sub_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"[6/6] Saved {out_path.name} in {time.time() - t0:.1f}s "
        f"(rows={len(sub_df)}, counts={sub_df['label'].value_counts().sort_index().to_dict()})",
        flush=True,
    )

    if args.evaluate and (ROOT / "data" / "predict_input_chembl_labeled.csv").exists():
        from eval_challenge import compute_subset_metrics, print_report

        df_ev = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
        y_ev = df_ev["label"].to_numpy(dtype=int)
        measured_mask = (df_ev["label_chembl"].notna() | (df_ev["source"] == "pubchem_herg_potency")).to_numpy()
        res_813 = compute_subset_metrics(y_ev[measured_mask], pred_v9[measured_mask], p_v9[measured_mask], n_bootstrap=1000, seed=42)
        res_927 = compute_subset_metrics(y_ev, pred_v9, p_v9, n_bootstrap=1000, seed=42)
        print_report(f"{out_path.name} on Measured 813 (Primary)", res_813)
        print_report(f"{out_path.name} on All 927 (All Rows)", res_927)


if __name__ == "__main__":
    main()
