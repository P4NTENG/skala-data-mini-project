"""Evaluate challenge predictions against the reconstructed ChEMBL evaluation set.

WARNING: ``data/predict_input_chembl_labeled.csv`` MUST ONLY BE USED FOR FINAL EVALUATION.
Never use its labels for training, feature selection, early stopping, or threshold tuning.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_challenge.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_challenge.py \
        --submission predict_output_김인성.csv \
        --artifact artifacts/best_model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def compute_subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_label1: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute metrics for both label=1 (challenge positive) and label=0 (blocker)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # actual 0 (blocker), actual 1 (non-toxic)
    c00, c01, c10, c11 = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    ba = float(balanced_accuracy_score(y_true, y_pred))

    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    roc_auc = float("nan")
    pr_auc_1 = float("nan")
    pr_auc_0 = float("nan")
    if p_label1 is not None:
        p_label1 = np.asarray(p_label1, dtype=float)
        roc_auc = float(roc_auc_score(y_true, p_label1))
        pr_auc_1 = float(average_precision_score(y_true, p_label1))
        pr_auc_0 = float(average_precision_score(1 - y_true, 1.0 - p_label1))

    # Bootstrap 95% CI
    rng = np.random.default_rng(seed)
    boot_f1_1 = np.empty(n_bootstrap, dtype=float)
    boot_f1_0 = np.empty(n_bootstrap, dtype=float)
    boot_acc = np.empty(n_bootstrap, dtype=float)
    boot_auc = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        pb = y_pred[idx]
        boot_f1_1[b] = f1_score(yb, pb, pos_label=1, zero_division=0)
        boot_f1_0[b] = f1_score(yb, pb, pos_label=0, zero_division=0)
        boot_acc[b] = accuracy_score(yb, pb)
        if p_label1 is not None and len(np.unique(yb)) == 2:
            boot_auc[b] = roc_auc_score(yb, p_label1[idx])
        else:
            boot_auc[b] = np.nan

    def _ci(arr: np.ndarray) -> dict:
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
        return {
            "mean": float(valid.mean()),
            "ci95": [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))],
        }

    return {
        "n": int(n),
        "n_label1_true": int((y_true == 1).sum()),
        "n_label0_true": int((y_true == 0).sum()),
        "n_label1_pred": int((y_pred == 1).sum()),
        "n_label0_pred": int((y_pred == 0).sum()),
        "accuracy": acc,
        "balanced_accuracy": ba,
        "roc_auc": roc_auc,
        "label1_non_toxic": {
            "f1": f1_1,
            "precision": prec_1,
            "recall": rec_1,
            "pr_auc": pr_auc_1,
        },
        "label0_blocker": {
            "f1": f1_0,
            "precision": prec_0,
            "recall": rec_0,
            "pr_auc": pr_auc_0,
        },
        "confusion_matrix_actual_rows_0_1_pred_cols_0_1": [
            [c00, c01],
            [c10, c11],
        ],
        "confusion_counts_label1_pos": {
            "tn_blocker": c00,
            "fp_non_toxic": c01,
            "fn_non_toxic": c10,
            "tp_non_toxic": c11,
        },
        "bootstrap_1000_ci95": {
            "f1_label1": _ci(boot_f1_1),
            "f1_label0": _ci(boot_f1_0),
            "accuracy": _ci(boot_acc),
            "roc_auc": _ci(boot_auc),
        },
    }


def load_probabilities_from_artifact(smiles: list[str], artifact_path: Path) -> np.ndarray:
    """Return P(label=1, non-toxic) for each SMILES using a saved model artifact."""
    from herg_model.config import load_config
    from herg_model.models import positive_scores
    from herg_model.predict import compute_blocks_from_smiles, load_artifact

    cfg = load_config()
    artifact = load_artifact(artifact_path)
    names, X = compute_blocks_from_smiles(smiles, artifact["blocks"], cfg, verbose=False)
    if names != list(artifact["feature_names"]):
        raise RuntimeError("Rebuilt feature names do not match artifact feature_names")
    X_trans = artifact["pipeline"].transform(X)
    selected = artifact.get("selected_indices")
    if selected is not None:
        X_trans = X_trans[:, np.asarray(selected, dtype=int)]
    p_blocker = positive_scores(artifact["model"], X_trans)
    return 1.0 - p_blocker


def print_report(title: str, res: dict) -> None:
    cm = res["confusion_matrix_actual_rows_0_1_pred_cols_0_1"]
    b_f1_1 = res["bootstrap_1000_ci95"]["f1_label1"]
    b_f1_0 = res["bootstrap_1000_ci95"]["f1_label0"]
    b_auc = res["bootstrap_1000_ci95"]["roc_auc"]
    b_acc = res["bootstrap_1000_ci95"]["accuracy"]

    print(f"\n=== {title} (n={res['n']}, true 1={res['n_label1_true']}, true 0={res['n_label0_true']}) ===")
    print(f"  [Primary] F1(label=1) : {res['label1_non_toxic']['f1']:.4f}  "
          f"(95% CI: [{b_f1_1['ci95'][0]:.4f}, {b_f1_1['ci95'][1]:.4f}])")
    print(f"            P(label=1)  : {res['label1_non_toxic']['precision']:.4f}  "
          f"R(label=1): {res['label1_non_toxic']['recall']:.4f}  "
          f"PR-AUC(1): {res['label1_non_toxic']['pr_auc']:.4f}")
    print(f"  [Blocker] F1(label=0) : {res['label0_blocker']['f1']:.4f}  "
          f"(95% CI: [{b_f1_0['ci95'][0]:.4f}, {b_f1_0['ci95'][1]:.4f}])")
    print(f"            P(label=0)  : {res['label0_blocker']['precision']:.4f}  "
          f"R(label=0): {res['label0_blocker']['recall']:.4f}  "
          f"PR-AUC(0): {res['label0_blocker']['pr_auc']:.4f}")
    print(f"  [Overall] ROC-AUC     : {res['roc_auc']:.4f}  "
          f"(95% CI: [{b_auc['ci95'][0]:.4f}, {b_auc['ci95'][1]:.4f}])")
    print(f"            Accuracy    : {res['accuracy']:.4f}  "
          f"(95% CI: [{b_acc['ci95'][0]:.4f}, {b_acc['ci95'][1]:.4f}])  "
          f"Balanced Acc: {res['balanced_accuracy']:.4f}")
    print("  [Confusion Matrix] rows=actual (0:blocker, 1:non-toxic), cols=pred (0, 1):")
    print(f"            pred=0   pred=1")
    print(f"    act=0   {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"    act=1   {cm[1][0]:6d}   {cm[1][1]:6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hERG challenge submission on 813 measured & 927 all rows.")
    parser.add_argument("--submission", default="predict_output_김인성.csv",
                        help="Path to submission CSV (SMILES,label) or probability file")
    parser.add_argument("--eval-labels", default="data/predict_input_chembl_labeled.csv",
                        help="Path to evaluation label CSV (for evaluation ONLY)")
    parser.add_argument("--artifact", default="artifacts/best_model.joblib",
                        help="Optional model artifact to compute probabilities for AUC (set 'none' to skip)")
    parser.add_argument("--probs-file", default=None,
                        help="Optional CSV/NPY containing predicted probabilities (p_label1 or p_blocker)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None, help="Optional path to save JSON evaluation report")
    args = parser.parse_args()

    eval_df = pd.read_csv(ROOT / args.eval_labels)
    sub_df = pd.read_csv(ROOT / args.submission)

    if len(sub_df) != len(eval_df):
        raise ValueError(f"Row count mismatch: submission={len(sub_df)} vs eval={len(eval_df)}")
    if not (sub_df["SMILES"].values == eval_df["SMILES"].values).all():
        raise ValueError("SMILES order in submission does not match evaluation file!")

    y_true = eval_df["label"].to_numpy(dtype=int)
    y_pred = sub_df["label"].to_numpy(dtype=int)

    # Obtain continuous probabilities for AUC if available
    p_label1 = None
    if args.probs_file:
        p_path = ROOT / args.probs_file
        if p_path.suffix == ".npy":
            p_label1 = np.load(p_path)
        else:
            p_df = pd.read_csv(p_path)
            if "p_label1" in p_df.columns:
                p_label1 = p_df["p_label1"].to_numpy(dtype=float)
            elif "p_non_toxic" in p_df.columns:
                p_label1 = p_df["p_non_toxic"].to_numpy(dtype=float)
            elif "p_blocker" in p_df.columns:
                p_label1 = 1.0 - p_df["p_blocker"].to_numpy(dtype=float)
            elif "blocker_probability" in p_df.columns:
                p_label1 = 1.0 - p_df["blocker_probability"].to_numpy(dtype=float)
    elif "p_label1" in sub_df.columns:
        p_label1 = sub_df["p_label1"].to_numpy(dtype=float)
    elif "p_non_toxic" in sub_df.columns:
        p_label1 = sub_df["p_non_toxic"].to_numpy(dtype=float)
    elif "p_blocker" in sub_df.columns:
        p_label1 = 1.0 - sub_df["p_blocker"].to_numpy(dtype=float)
    elif args.artifact and args.artifact.lower() != "none":
        art_path = ROOT / args.artifact
        if art_path.exists():
            p_label1 = load_probabilities_from_artifact(sub_df["SMILES"].tolist(), art_path)

    measured_mask = (eval_df["label_chembl"].notna() | (eval_df["source"] == "pubchem_herg_potency")).to_numpy()
    chembl_811_mask = eval_df["label_chembl"].notna().to_numpy()

    res_measured = compute_subset_metrics(
        y_true[measured_mask],
        y_pred[measured_mask],
        p_label1[measured_mask] if p_label1 is not None else None,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    if p_label1 is not None:
        res_measured["roc_auc_chembl_811"] = float(
            roc_auc_score(y_true[chembl_811_mask], p_label1[chembl_811_mask])
        )
    res_all = compute_subset_metrics(
        y_true,
        y_pred,
        p_label1,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print(f"[eval_challenge] submission: {args.submission}")
    print(f"[eval_challenge] artifact  : {args.artifact if p_label1 is not None else 'None (labels only)'}")
    print_report("Measured 813 (Primary Metric Subset)", res_measured)
    if "roc_auc_chembl_811" in res_measured:
        print(f"  [Note] ROC-AUC on ChEMBL IC50 direct 811 rows: {res_measured['roc_auc_chembl_811']:.4f}")
    print_report("All 927 (Includes 114 Low-Confidence Rows)", res_all)

    if args.json_out:
        out_path = ROOT / args.json_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "submission": str(args.submission),
            "measured_813": res_measured,
            "all_927": res_all,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[eval_challenge] saved JSON report -> {out_path}")


if __name__ == "__main__":
    main()
