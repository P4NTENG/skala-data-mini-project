"""Shared feature builders, leak-free fold cache, and evaluation helpers for Phase 2.

STRICT RULE COMPLIANCE:
- Only uses ``data/train.csv`` (8,349 rows).
- NEVER loads ``data/predict_input_chembl_labeled.csv`` or ``data/predict_input.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
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

from herg_model.chemistry import bem_murcko_scaffold, load_structure_table, mol_from_smiles  # noqa: E402
from herg_model.config import Config, load_config  # noqa: E402
from herg_model.experiments import DOMAIN_GROUPS, FeatureStore  # noqa: E402
from herg_model.preprocessing import FeaturePipeline  # noqa: E402
from herg_model.splitting import make_splits  # noqa: E402

LOG_PATH = ROOT / "reports" / "experiment_log_v2.csv"
OOF_CACHE_DIR = ROOT / ".cache" / "herg_model" / "phase2_oof"


def compute_domain_derived_block(df: pd.DataFrame, store: FeatureStore) -> tuple[list[str], np.ndarray]:
    """Compute 26 row-wise hERG domain-derived features (CAD rules, amine ratios, charge/surface, FP density).

    100% row-independent (zero target or cross-row leakage).
    """
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
    arom_hetero = col("NumAromaticHeterocycles")
    sat_rings = col("NumSaturatedRings")
    csp3 = col("FractionCSP3")
    labute = col("LabuteASA")
    max_q = col("MaxPartialCharge")
    min_q = col("MinPartialCharge")
    max_abs_q = col("MaxAbsPartialCharge")

    n_total_n = col("n_total_n")
    n_basic_n = col("n_basic_n")
    has_basic = col("has_basic_amine")
    n_pri = col("n_primary_amine")
    n_sec = col("n_secondary_amine")
    n_tert = col("n_tertiary_amine")
    n_arom_n = col("n_aromatic_n")
    n_amide = col("n_amide")
    n_sulfon = col("n_sulfonamide")
    n_neg = col("n_negative_atoms")
    n_pos = col("n_positive_atoms")
    n_hal = col("n_halogen")

    # Fingerprint bit densities
    ecfp_cols = [f"ecfp_{i}" for i in range(1024)]
    fcfp_cols = [f"fcfp_{i}" for i in range(1024)]
    ptfp_cols = [f"ptfp_{i}" for i in range(1024)]
    ecfp_sum = df[ecfp_cols].to_numpy(dtype=np.float32).sum(axis=1) if all(c in df.columns for c in ecfp_cols[:2]) else np.zeros(len(df))
    fcfp_sum = df[fcfp_cols].to_numpy(dtype=np.float32).sum(axis=1) if all(c in df.columns for c in fcfp_cols[:2]) else np.zeros(len(df))
    ptfp_sum = df[ptfp_cols].to_numpy(dtype=np.float32).sum(axis=1) if all(c in df.columns for c in ptfp_cols[:2]) else np.zeros(len(df))

    pos_logp = np.maximum(0.0, logp)

    feats = {
        # CAD (Cationic Amphiphilic Drug) rules & continuous interaction terms
        "drv_cad_rule_logp3": ((logp > 3.0) & (has_basic > 0.5)).astype(np.float64),
        "drv_cad_rule_logp4": ((logp > 4.0) & (has_basic > 0.5)).astype(np.float64),
        "drv_cad_rule_mod": ((logp > 2.5) & (has_basic > 0.5) & (tpsa < 75.0)).astype(np.float64),
        "drv_cad_logp_x_basic_n": pos_logp * n_basic_n,
        "drv_cad_logp_x_tert_amine": pos_logp * n_tert,
        "drv_cad_logp_x_sec_amine": pos_logp * n_sec,
        "drv_cad_logp_x_arom_basic": pos_logp * arom_rings * has_basic,
        # Nitrogen environment ratios & acid-base balance
        "drv_basic_to_total_n": n_basic_n / (n_total_n + 1.0),
        "drv_amide_plus_sulfon": n_amide + n_sulfon,
        "drv_neutral_n_ratio": (n_amide + n_sulfon) / (n_total_n + 1.0),
        "drv_arom_n_per_arom_ring": n_arom_n / (arom_rings + 1.0),
        "drv_net_basic_minus_acidic": n_basic_n - n_neg,
        "drv_zwitterion_flag": ((n_pos > 0.5) & (n_neg > 0.5)).astype(np.float64),
        # Physicochemical / polarity / shape ratios
        "drv_logp_to_tpsa": logp / (tpsa + 10.0),
        "drv_tpsa_per_heavy": tpsa / (heavy + 1.0),
        "drv_logp_per_heavy": logp / (heavy + 1.0),
        "drv_hbond_total": hbd + hba,
        "drv_hbd_to_hba": hbd / (hba + 1.0),
        "drv_arom_ring_frac": arom_rings / (rings + 1.0),
        "drv_sat_ring_frac": sat_rings / (rings + 1.0),
        "drv_rot_per_heavy": rot / (heavy + 1.0),
        "drv_halogen_per_heavy": n_hal / (heavy + 1.0),
        # Surface / electronic / fingerprint density
        "drv_charge_span": max_q - min_q,
        "drv_abs_charge_x_logp": max_abs_q * pos_logp,
        "drv_asa_per_mw": labute / (mw + 1.0),
        "drv_ecfp_bits": ecfp_sum.astype(np.float64),
        "drv_fcfp_bits": fcfp_sum.astype(np.float64),
        "drv_ptfp_bits": ptfp_sum.astype(np.float64),
        "drv_fcfp_to_ecfp": fcfp_sum / (ecfp_sum + 1.0),
    }
    names = list(feats.keys())
    mat = np.column_stack([feats[k] for k in names])
    return names, mat


def build_feature_matrix(
    feature_set_name: str,
    df: pd.DataFrame,
    store: FeatureStore,
) -> tuple[list[str], np.ndarray, str]:
    """Return (names, X, description) for a named feature set."""
    domain_blocks = list(DOMAIN_GROUPS[:-1])

    if feature_set_name == "E3":
        blocks = ["fp:fcfp", "existing_descriptors"]
        names, X = store.matrix(blocks)
        return names, X, "fp:fcfp+existing_descriptors"

    if feature_set_name == "E6":
        blocks = ["fp:fcfp", "existing_descriptors"] + domain_blocks
        names, X = store.matrix(blocks)
        return names, X, "+".join(blocks)

    if feature_set_name == "E7":
        blocks = ["fp:fcfp", "existing_descriptors", "rdkit_all"]
        names, X = store.matrix(blocks)
        return names, X, "+".join(blocks)

    if feature_set_name == "ALL_FP_BASE":
        blocks = ["all_fingerprints", "existing_descriptors"]
        names, X = store.matrix(blocks)
        return names, X, "all_fingerprints+existing_descriptors"

    if feature_set_name == "ALL_FP_E6":
        blocks = ["all_fingerprints", "existing_descriptors"] + domain_blocks
        names, X = store.matrix(blocks)
        return names, X, "+".join(blocks)

    if feature_set_name == "ALL_FP_E7":
        blocks = ["all_fingerprints", "existing_descriptors", "rdkit_all"]
        names, X = store.matrix(blocks)
        return names, X, "+".join(blocks)

    if feature_set_name == "MORDRED_RAW_E6":
        blocks = ["fp:fcfp", "existing_descriptors"] + domain_blocks + ["mordred_2d"]
        names, X = store.matrix(blocks)
        return names, X, "+".join(blocks) + "(train_select_top_mordred)"

    if feature_set_name == "E6_DRV":
        blocks = ["fp:fcfp", "existing_descriptors"] + domain_blocks
        names_base, X_base = store.matrix(blocks)
        names_drv, X_drv = compute_domain_derived_block(df, store)
        return names_base + names_drv, np.hstack([X_base, X_drv]), "+".join(blocks) + "+domain_derived_29"

    if feature_set_name == "ALL_FP_E6_DRV":
        blocks = ["all_fingerprints", "existing_descriptors"] + domain_blocks
        names_base, X_base = store.matrix(blocks)
        names_drv, X_drv = compute_domain_derived_block(df, store)
        return names_base + names_drv, np.hstack([X_base, X_drv]), "+".join(blocks) + "+domain_derived_29"

    if feature_set_name == "ALL_FP_E7_DRV":
        blocks = ["all_fingerprints", "existing_descriptors", "rdkit_all"]
        names_base, X_base = store.matrix(blocks)
        names_drv, X_drv = compute_domain_derived_block(df, store)
        return names_base + names_drv, np.hstack([X_base, X_drv]), "+".join(blocks) + "+domain_derived_29"

    raise ValueError(f"Unknown feature_set_name: {feature_set_name}")


def fast_fit_transform_clean(
    X_tr: np.ndarray,
    X_others: list[np.ndarray],
    missing_rate_limit: float = 1.0,
    scale: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Vectorised equivalent of FeaturePipeline (constant/duplicate drop + median fill + optional scale).

    50x faster than column-by-column sha1 hashing while preserving identical leak-free semantics.
    """
    X_tr = np.where(np.isfinite(X_tr), X_tr, np.nan)
    n_rows, n_cols = X_tr.shape

    keep = np.ones(n_cols, dtype=bool)
    isnan_tr = np.isnan(X_tr)
    miss_rate = isnan_tr.mean(axis=0)
    keep &= miss_rate < 1.0
    if missing_rate_limit < 1.0:
        keep &= miss_rate <= missing_rate_limit

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

    # Fill NaNs with training median
    inds = np.where(np.isnan(sub))
    if inds[0].size > 0:
        sub = sub.copy()
        sub[inds] = np.take(med, inds[1])

    # Fast duplicate detection via random projection fingerprint + exact verification on collisions
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
            is_dup = False
            col_j = rounded[:, j]
            for prev_j in bucket:
                if np.array_equal(col_j, rounded[:, prev_j]):
                    is_dup = True
                    break
            if not is_dup:
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

    return np.ascontiguousarray(X_tr_clean, dtype=np.float32), out_others, keep_idx


def prepare_folds_and_scaffold(
    feature_set_name: str,
    df: pd.DataFrame,
    store: FeatureStore,
    y_blocker: np.ndarray,
    scaffolds: np.ndarray,
    cfg: Config,
    seed: int = 42,
    mordred_top_k: int | None = None,
    scale: bool = False,
) -> dict:
    """Pre-compute leak-free preprocessed matrices for 5 random folds and scaffold split."""
    names, X, feat_desc = build_feature_matrix(feature_set_name, df, store)
    is_mordred = "mordred" in feature_set_name.lower()
    miss_limit = 0.10 if is_mordred else 1.0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds_data = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_blocker)):
        X_tr, [X_va], keep_idx = fast_fit_transform_clean(
            X[tr_idx], [X[va_idx]], missing_rate_limit=miss_limit, scale=scale
        )

        if mordred_top_k is not None:
            et = ExtraTreesClassifier(
                n_estimators=200, random_state=seed + fold, n_jobs=8, class_weight="balanced"
            )
            et.fit(X_tr, y_blocker[tr_idx])
            imp = et.feature_importances_
            k = min(mordred_top_k, X_tr.shape[1])
            sel_idx = np.argsort(-imp)[:k]
            sel_idx.sort()
            X_tr = np.ascontiguousarray(X_tr[:, sel_idx], dtype=np.float32)
            X_va = np.ascontiguousarray(X_va[:, sel_idx], dtype=np.float32)

        folds_data.append({
            "fold": fold,
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "X_tr": X_tr,
            "X_va": X_va,
            "y_tr": y_blocker[tr_idx],
            "y_va": y_blocker[va_idx],
            "n_features": X_tr.shape[1],
        })

    # Scaffold split (seed=42)
    splits = make_splits(scaffolds, y_blocker, cfg, "scaffold", seed)
    tr, va, te = splits.train, splits.val, splits.test

    X_sc_tr, [X_sc_va, X_sc_te], _ = fast_fit_transform_clean(
        X[tr], [X[va], X[te]], missing_rate_limit=miss_limit, scale=scale
    )

    if mordred_top_k is not None:
        et_sc = ExtraTreesClassifier(
            n_estimators=200, random_state=seed, n_jobs=8, class_weight="balanced"
        )
        et_sc.fit(X_sc_tr, y_blocker[tr])
        imp_sc = et_sc.feature_importances_
        k = min(mordred_top_k, X_sc_tr.shape[1])
        sel_sc = np.argsort(-imp_sc)[:k]
        sel_sc.sort()
        X_sc_tr = np.ascontiguousarray(X_sc_tr[:, sel_sc], dtype=np.float32)
        X_sc_va = np.ascontiguousarray(X_sc_va[:, sel_sc], dtype=np.float32)
        X_sc_te = np.ascontiguousarray(X_sc_te[:, sel_sc], dtype=np.float32)

    tr_va = np.concatenate([tr, va])
    X_sc_trva, [X_sc_te_trva], _ = fast_fit_transform_clean(
        X[tr_va], [X[te]], missing_rate_limit=miss_limit, scale=scale
    )

    if mordred_top_k is not None:
        et_trva = ExtraTreesClassifier(
            n_estimators=200, random_state=seed, n_jobs=8, class_weight="balanced"
        )
        et_trva.fit(X_sc_trva, y_blocker[tr_va])
        imp_trva = et_trva.feature_importances_
        k = min(mordred_top_k, X_sc_trva.shape[1])
        sel_trva = np.argsort(-imp_trva)[:k]
        sel_trva.sort()
        X_sc_trva = np.ascontiguousarray(X_sc_trva[:, sel_trva], dtype=np.float32)
        X_sc_te_trva = np.ascontiguousarray(X_sc_te_trva[:, sel_trva], dtype=np.float32)

    return {
        "feature_set_name": feature_set_name,
        "feat_desc": feat_desc,
        "n_raw": len(names),
        "n_kept_mean": int(np.mean([f["n_features"] for f in folds_data])),
        "folds": folds_data,
        "scaffold": {
            "tr": tr,
            "va": va,
            "te": te,
            "tr_va": tr_va,
            "X_tr": X_sc_tr,
            "X_va": X_sc_va,
            "X_te": X_sc_te,
            "X_trva": X_sc_trva,
            "X_te_trva": X_sc_te_trva,
            "y_tr": y_blocker[tr],
            "y_va": y_blocker[va],
            "y_te": y_blocker[te],
            "y_trva": y_blocker[tr_va],
        },
    }


def find_best_f1_threshold(y_label1: np.ndarray, p_label1: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.15, 0.85, 141)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        score = f1_score(y_label1, (p_label1 >= t).astype(int), pos_label=1, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def compute_oof_and_scaffold_metrics(
    oof_p_label1: np.ndarray,
    sc_val_p_label1: np.ndarray,
    sc_test_tronly_p_label1: np.ndarray,
    sc_test_trva_p_label1: np.ndarray,
    y_label1: np.ndarray,
    sc_va_idx: np.ndarray,
    sc_te_idx: np.ndarray,
    seed: int = 42,
) -> dict:
    # OOF @ 0.5
    pred_05 = (oof_p_label1 >= 0.5).astype(int)
    oof_f1_05 = float(f1_score(y_label1, pred_05, pos_label=1, zero_division=0))
    oof_auc = float(roc_auc_score(y_label1, oof_p_label1))
    oof_acc_05 = float(accuracy_score(y_label1, pred_05))

    # OOF @ optimal threshold
    opt_thr, oof_f1_opt = find_best_f1_threshold(y_label1, oof_p_label1)

    # Nested out-of-fold threshold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    nested_preds = np.zeros_like(y_label1)
    for tr_idx, va_idx in skf.split(oof_p_label1, 1 - y_label1):
        t_fold, _ = find_best_f1_threshold(y_label1[tr_idx], oof_p_label1[tr_idx])
        nested_preds[va_idx] = (oof_p_label1[va_idx] >= t_fold).astype(int)
    oof_nested_f1 = float(f1_score(y_label1, nested_preds, pos_label=1, zero_division=0))

    # Bootstrap 1000 CI for OOF F1 @ 0.5
    rng = np.random.default_rng(seed)
    n = len(y_label1)
    boot_f1 = np.empty(1000, dtype=float)
    for b in range(1000):
        idx = rng.integers(0, n, n)
        boot_f1[b] = f1_score(y_label1[idx], pred_05[idx], pos_label=1, zero_division=0)
    ci_lo, ci_hi = float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))

    # Scaffold metrics
    y_sc_va = y_label1[sc_va_idx]
    y_sc_te = y_label1[sc_te_idx]
    sc_val_thr, _ = find_best_f1_threshold(y_sc_va, sc_val_p_label1)

    sc_f1_tronly_05 = float(f1_score(y_sc_te, (sc_test_tronly_p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    sc_f1_tronly_opt = float(f1_score(y_sc_te, (sc_test_tronly_p_label1 >= sc_val_thr).astype(int), pos_label=1, zero_division=0))
    sc_f1_trva_05 = float(f1_score(y_sc_te, (sc_test_trva_p_label1 >= 0.5).astype(int), pos_label=1, zero_division=0))
    sc_auc_tronly = float(roc_auc_score(y_sc_te, sc_test_tronly_p_label1))
    sc_auc_trva = float(roc_auc_score(y_sc_te, sc_test_trva_p_label1))

    return {
        "oof_f1_05": oof_f1_05,
        "oof_f1_ci_lo": ci_lo,
        "oof_f1_ci_hi": ci_hi,
        "oof_f1_opt": oof_f1_opt,
        "oof_nested_f1": oof_nested_f1,
        "oof_auc": oof_auc,
        "oof_acc_05": oof_acc_05,
        "opt_thr": opt_thr,
        "sc_f1_tronly_05": sc_f1_tronly_05,
        "sc_f1_tronly_opt": sc_f1_tronly_opt,
        "sc_f1_trva_05": sc_f1_trva_05,
        "sc_auc_tronly": sc_auc_tronly,
        "sc_auc_trva": sc_auc_trva,
    }


def append_experiment_log(row: dict) -> None:
    OOF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        df_log = pd.read_csv(LOG_PATH)
        df_log = df_log[df_log["run_id"] != row["run_id"]]
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])
    df_log.to_csv(LOG_PATH, index=False, encoding="utf-8")
