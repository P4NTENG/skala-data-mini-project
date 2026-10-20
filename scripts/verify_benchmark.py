"""Verify the checked-in final model and run a same-split head-to-head benchmark.

What this script checks
-----------------------
1. Reproduces the Bemis-Murcko scaffold split (seed 42) used by ``reports/final_report.md``.
2. Re-evaluates ``artifacts/best_model.joblib`` on that test split and compares the result with
   the numbers reported in §7.
3. Trains two reference models on **train only** and evaluates them on the same val/test rows:
   * ``eda_lr``  -- the EDA notebook baseline (all 3 fingerprints + 4 shipped descriptors, LR).
   * ``e6_lgbm`` -- the final model's feature blocks and hyper-parameters, retrained on train.
   Both thresholds are selected on validation, once for F1(label=1) and once for blocker BA.

The competition objective is F1 with ``label = 1`` (non-toxic) as the positive class, so every
table reports both orientations explicitly.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from herg_model.chemistry import bem_murcko_scaffold, mol_from_smiles  # noqa: E402
from herg_model.config import load_config  # noqa: E402
from herg_model.models import positive_scores  # noqa: E402
from herg_model.predict import compute_blocks_from_smiles, load_artifact  # noqa: E402
from herg_model.preprocessing import FeaturePipeline  # noqa: E402
from herg_model.splitting import make_splits  # noqa: E402

DESC = ["MolWt", "clogp", "sa_score", "qed"]
FP_COLS = [f"{p}_{i}" for p in ("ecfp", "fcfp", "ptfp") for i in range(1024)]
OUT = ROOT / "reports" / "verification_benchmark.json"


def binary_metrics(y_non_toxic: np.ndarray, p_non_toxic: np.ndarray, threshold: float) -> dict:
    """Metrics in both orientations. ``y_non_toxic = 1`` means label 1 (non-toxic)."""
    pred_non_toxic = (p_non_toxic >= threshold).astype(int)
    blocker = 1 - y_non_toxic
    return {
        "threshold": float(threshold),
        # ROC-AUC is invariant to class relabelling
        "roc_auc": float(roc_auc_score(y_non_toxic, p_non_toxic)),
        # competition orientation: positive = label 1 (non-toxic)
        "f1_label1": float(f1_score(y_non_toxic, pred_non_toxic, pos_label=1, zero_division=0)),
        "precision_label1": float(precision_score(y_non_toxic, pred_non_toxic, pos_label=1, zero_division=0)),
        "recall_label1": float(recall_score(y_non_toxic, pred_non_toxic, pos_label=1, zero_division=0)),
        "pr_auc_label1": float(average_precision_score(y_non_toxic, p_non_toxic)),
        "specificity_blocker": float(recall_score(y_non_toxic, pred_non_toxic, pos_label=1, zero_division=0)),
        # report orientation: positive = label 0 (blocker)
        "balanced_accuracy": float(balanced_accuracy_score(y_non_toxic, pred_non_toxic)),
        "f1_blocker": float(f1_score(blocker, 1 - pred_non_toxic, pos_label=1, zero_division=0)),
        "precision_blocker": float(precision_score(blocker, 1 - pred_non_toxic, pos_label=1, zero_division=0)),
        "recall_blocker": float(recall_score(blocker, 1 - pred_non_toxic, pos_label=1, zero_division=0)),
        "pr_auc_blocker": float(average_precision_score(blocker, 1 - p_non_toxic)),
        "accuracy": float(accuracy_score(y_non_toxic, pred_non_toxic)),
        "n": int(len(y_non_toxic)),
        "n_blocker": int(blocker.sum()),
    }


def best_threshold(y_non_toxic: np.ndarray, p_non_toxic: np.ndarray, objective: str) -> float:
    grid = np.linspace(0.01, 0.99, 197)
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        if objective == "f1_label1":
            value = f1_score(y_non_toxic, (p_non_toxic >= t).astype(int), pos_label=1, zero_division=0)
        elif objective == "blocker_ba":
            value = balanced_accuracy_score(y_non_toxic, (p_non_toxic >= t).astype(int))
        else:
            raise ValueError(objective)
        if value > best_v:
            best_t, best_v = float(t), float(value)
    return best_t


def bootstrap_f1(
    y: np.ndarray, predictions: dict[str, np.ndarray], n_resamples: int = 1000, seed: int = 0
) -> dict:
    """Percentile CI for F1(label=1) and for paired differences vs the first model."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(predictions)
    scores = {k: np.empty(n_resamples) for k in names}
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        for k in names:
            scores[k][b] = f1_score(y[idx], predictions[k][idx], pos_label=1, zero_division=0)
    out = {
        k: {
            "mean": float(scores[k].mean()),
            "ci95": [float(np.percentile(scores[k], 2.5)), float(np.percentile(scores[k], 97.5))],
        }
        for k in names
    }
    for k in names[1:]:
        diff = scores[k] - scores[names[0]]
        out[f"{k}_minus_{names[0]}"] = {
            "mean": float(diff.mean()),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "p_le_0": float((diff <= 0).mean()),
        }
    return out


def main() -> None:
    t0 = time.time()
    cfg = load_config()

    # ------------------------------------------------------------------ data
    df = pd.read_csv(ROOT / "data" / "train.csv")
    y = df["label"].to_numpy(dtype=np.int8)          # 1 = non-toxic
    yb = (y == 0).astype(np.int8)                    # 1 = blocker (report positive class)
    print(f"[data] rows={len(df)} label1={(y == 1).sum()} label0={(y == 0).sum()}")

    mols = [mol_from_smiles(s) for s in df["SMILES"]]
    scaffolds = [bem_murcko_scaffold(m, True) if m is not None else "INVALID" for m in mols]
    splits = make_splits(scaffolds, yb, cfg, "scaffold", int(cfg.get_path("split.primary_seed", 42)))
    split_info = {}
    for name, idx in splits.as_dict().items():
        split_info[name] = {
            "n_rows": int(len(idx)),
            "n_blocker": int(yb[idx].sum()),
            "n_unique_scaffolds": int(len(set(np.asarray(scaffolds, dtype=object)[idx]))),
        }
    print("[split]", json.dumps(split_info))

    tr, va, te = splits.train, splits.val, splits.test

    # ------------------------------------------------------- checked-in model
    artifact = load_artifact(ROOT / "artifacts" / "best_model.joblib")
    blocks = list(artifact["blocks"])
    print(f"[artifact] {artifact['metadata'].get('experiment_id')}/{artifact['metadata'].get('model')} "
          f"blocks={blocks}")
    names, X_blocks = compute_blocks_from_smiles(df["SMILES"].tolist(), blocks, cfg, verbose=False)
    if names != list(artifact["feature_names"]):
        raise RuntimeError("rebuilt feature names do not match the artifact")
    X_art = artifact["pipeline"].transform(X_blocks)
    artifact_test = positive_scores(artifact["model"], X_art[te])
    artifact_val = positive_scores(artifact["model"], X_art[va])

    # the stored threshold applies to P(blocker); the helper takes P(non-toxic)
    artifact_results = {
        "reported_threshold": binary_metrics(
            y[te], 1 - artifact_test, 1.0 - float(artifact["threshold"])
        ),
        "threshold_0.5": binary_metrics(y[te], 1 - artifact_test, 0.5),
    }
    print("[artifact] test @reported threshold:",
          {k: round(v, 4) for k, v in artifact_results["reported_threshold"].items()
           if isinstance(v, float)})
    print("[artifact] test @0.5:", {k: round(v, 4) for k, v in artifact_results["threshold_0.5"].items()
                                    if isinstance(v, float)})

    # ---------------------------------------------------------- EDA reference
    X_fp = sparse.csr_matrix(df[FP_COLS].to_numpy(dtype=np.int8))
    scaler = StandardScaler().fit(df.loc[tr, DESC].to_numpy(dtype=np.float64))
    X_desc = scaler.transform(df[DESC].to_numpy(dtype=np.float64))
    X_eda = sparse.hstack([sparse.csr_matrix(X_desc), X_fp], format="csr")

    eda = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0, random_state=42)
    eda.fit(X_eda[tr], y[tr])
    eda_val = eda.predict_proba(X_eda[va])[:, 1]   # P(non-toxic)
    eda_test = eda.predict_proba(X_eda[te])[:, 1]
    print(f"[eda_lr] fit done ({time.time() - t0:.1f}s)")

    # ---------------------------------------------- E6 retrained on train only
    pipe = FeaturePipeline(names=names, scale=False, drop_constant=True, drop_duplicate=True)
    pipe.fit(X_blocks[tr], yb[tr])
    X_e6_tr = pipe.transform(X_blocks[tr])
    X_e6_va = pipe.transform(X_blocks[va])
    X_e6_te = pipe.transform(X_blocks[te])

    from lightgbm import LGBMClassifier

    params = dict(artifact["metadata"]["hyperparameters"])
    params.pop("class_weight", None)
    lgbm = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(X_e6_tr, yb[tr])
    e6_val_blocker = positive_scores(lgbm, X_e6_va)   # P(blocker)
    e6_test_blocker = positive_scores(lgbm, X_e6_te)
    e6_val = 1 - e6_val_blocker
    e6_test = 1 - e6_test_blocker
    print(f"[e6_lgbm] fit done ({time.time() - t0:.1f}s), features={X_e6_tr.shape[1]}")

    # -------------------------------------------------- threshold + evaluation
    models = {"eda_lr": (eda_val, eda_test), "e6_lgbm": (e6_val, e6_test)}
    results: dict = {}
    for name, (p_val, p_test) in models.items():
        t_f1 = best_threshold(y[va], p_val, "f1_label1")
        t_ba = best_threshold(y[va], p_val, "blocker_ba")
        results[name] = {
            "val_at_0.5": binary_metrics(y[va], p_val, 0.5),
            "test_at_0.5": binary_metrics(y[te], p_test, 0.5),
            "threshold_selected_f1_label1": t_f1,
            "test_at_f1_threshold": binary_metrics(y[te], p_test, t_f1),
            "threshold_selected_blocker_ba": t_ba,
            "test_at_ba_threshold": binary_metrics(y[te], p_test, t_ba),
        }

    # E6 validation metrics at the artifact hyper-parameters (sanity vs report)
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data": {"n_rows": int(len(df)), "n_label1": int((y == 1).sum()), "n_label0": int((y == 0).sum())},
        "split": split_info,
        "artifact": {
            "experiment_id": artifact["metadata"].get("experiment_id"),
            "model": artifact["metadata"].get("model"),
            "n_features_pipeline": int(X_art.shape[1]),
            "threshold": float(artifact["threshold"]),
            # NOTE: the artifact was refit on train+val, so val scores are in-sample and are not
            # comparable to the report's held-out validation table.
            "in_sample_val_auc_only": float(roc_auc_score(y[va], artifact_val)),
            **artifact_results,
        },
        "reference_models": results,
    }

    # ------------------------------------------------ random split (leaderboard proxy)
    # The hidden challenge split is not available; a stratified random split is the closest
    # in-distribution proxy and is reported separately from the scaffold result.
    rnd = make_splits(scaffolds, yb, cfg, "random", 42)
    rtr, rva, rte = rnd.train, rnd.val, rnd.test

    eda_r = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0, random_state=42)
    eda_r.fit(X_eda[rtr], y[rtr])
    p_eda_rva = eda_r.predict_proba(X_eda[rva])[:, 1]
    p_eda_rte = eda_r.predict_proba(X_eda[rte])[:, 1]

    pipe_r = FeaturePipeline(names=names, scale=False, drop_constant=True, drop_duplicate=True)
    pipe_r.fit(X_blocks[rtr], yb[rtr])
    lgbm_r = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_r.fit(pipe_r.transform(X_blocks[rtr]), yb[rtr])
    p_e6_rva = 1 - positive_scores(lgbm_r, pipe_r.transform(X_blocks[rva]))
    p_e6_rte = 1 - positive_scores(lgbm_r, pipe_r.transform(X_blocks[rte]))

    random_results = {}
    for name, (p_val, p_test) in {
        "eda_lr": (p_eda_rva, p_eda_rte),
        "e6_lgbm": (p_e6_rva, p_e6_rte),
    }.items():
        t_f1 = best_threshold(y[rva], p_val, "f1_label1")
        random_results[name] = {
            "n_train": int(len(rtr)),
            "n_val": int(len(rva)),
            "n_test": int(len(rte)),
            "threshold_selected_f1_label1": t_f1,
            "test_at_f1_threshold": binary_metrics(y[rte], p_test, t_f1),
            "test_at_0.5": binary_metrics(y[rte], p_test, 0.5),
        }
    report["random_split_reference"] = random_results
    print("\n== random split (seed 42, in-distribution proxy) ==")
    print(f"{'model':<16}{'thr':>6}{'F1(1)':>8}{'P(1)':>7}{'R(1)':>7}{'BA':>7}{'F1(0)':>7}{'R(0)':>7}{'AUC':>8}")
    for name in ("eda_lr", "e6_lgbm"):
        m = random_results[name]["test_at_f1_threshold"]
        print(f"{name:<16}{m['threshold']:>6.3f}{m['f1_label1']:>8.4f}{m['precision_label1']:>7.4f}"
              f"{m['recall_label1']:>7.4f}{m['balanced_accuracy']:>7.4f}{m['f1_blocker']:>7.4f}"
              f"{m['recall_blocker']:>7.4f}{m['roc_auc']:>8.4f}")

    preds = {
        "eda_lr": (eda_test >= results["eda_lr"]["threshold_selected_f1_label1"]).astype(int),
        "e6_lgbm": (e6_test >= results["e6_lgbm"]["threshold_selected_f1_label1"]).astype(int),
    }
    # the checked-in model at its own threshold, for a three-way view
    preds["checked_in"] = (1 - artifact_test >= 0.5).astype(int)
    report["test_f1_label1_bootstrap"] = bootstrap_f1(y[te], preds)

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ print
    header = f"{'model':<16}{'thr':>6}{'F1(1)':>8}{'P(1)':>7}{'R(1)':>7}{'BA':>7}{'F1(0)':>7}{'R(0)':>7}{'AUC':>8}"

    print("\n== validation @0.5 (reference for the report's 0.8467 / 0.7799) ==")
    print(f"{'model':<16}{'AUC':>8}{'BA':>8}{'F1(1)':>8}{'F1(0)':>8}")
    for name in ("eda_lr", "e6_lgbm"):
        m = results[name]["val_at_0.5"]
        print(f"{name:<16}{m['roc_auc']:>8.4f}{m['balanced_accuracy']:>8.4f}"
              f"{m['f1_label1']:>8.4f}{m['f1_blocker']:>8.4f}")

    print("\n== test (threshold chosen on validation for F1 label=1) ==")
    print(header)
    for name in ("eda_lr", "e6_lgbm"):
        m = results[name]["test_at_f1_threshold"]
        print(f"{name:<16}{m['threshold']:>6.3f}{m['f1_label1']:>8.4f}{m['precision_label1']:>7.4f}"
              f"{m['recall_label1']:>7.4f}{m['balanced_accuracy']:>7.4f}{m['f1_blocker']:>7.4f}"
              f"{m['recall_blocker']:>7.4f}{m['roc_auc']:>8.4f}")
    m = artifact_results["reported_threshold"]
    print(f"{'checked_in':<16}{m['threshold']:>6.3f}{m['f1_label1']:>8.4f}{m['precision_label1']:>7.4f}"
          f"{m['recall_label1']:>7.4f}{m['balanced_accuracy']:>7.4f}{m['f1_blocker']:>7.4f}"
          f"{m['recall_blocker']:>7.4f}{m['roc_auc']:>8.4f}")

    print("\n== test (threshold chosen on validation for blocker BA) ==")
    print(header)
    for name in ("eda_lr", "e6_lgbm"):
        m = results[name]["test_at_ba_threshold"]
        print(f"{name:<16}{m['threshold']:>6.3f}{m['f1_label1']:>8.4f}{m['precision_label1']:>7.4f}"
              f"{m['recall_label1']:>7.4f}{m['balanced_accuracy']:>7.4f}{m['f1_blocker']:>7.4f}"
              f"{m['recall_blocker']:>7.4f}{m['roc_auc']:>8.4f}")

    b = report["test_f1_label1_bootstrap"]
    print("\n== F1(label=1) bootstrap 95% CI (1000 resamples, test) ==")
    for k in ("eda_lr", "e6_lgbm", "checked_in"):
        print(f"  {k:<12} {b[k]['mean']:.4f}  [{b[k]['ci95'][0]:.4f}, {b[k]['ci95'][1]:.4f}]")
    for k in ("e6_lgbm_minus_eda_lr", "checked_in_minus_eda_lr"):
        print(f"  {k:<24} {b[k]['mean']:+.4f}  [{b[k]['ci95'][0]:+.4f}, {b[k]['ci95'][1]:+.4f}]"
              f"  P(diff<=0)={b[k]['p_le_0']:.3f}")
    print(f"\n[done] {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
