"""Standardization ablation for the current E6 baseline.

Compares, with everything else fixed (same rows, same folds, same model, same features):

    raw                -> no structural standardization
    strip              -> keep the largest fragment (salt / mixture removal)
    strip_neutral      -> largest fragment + charge neutralization
    tautomer           -> canonical tautomer
    strip_neutral_taut -> all three

Features are **recomputed from the transformed SMILES** (shipped fingerprint columns are
computed from the original SMILES and would silently ignore the transformation).

Outputs
-------
reports/standardization_ablation.json
reports/standardization_ablation.md

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/standardization_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from herg_model.chemistry import (  # noqa: E402
    bem_murcko_scaffold,
    canonical_tautomer,
    mol_from_smiles,
    neutralize,
    strip_salts,
)
from herg_model.config import load_config  # noqa: E402
from herg_model.models import positive_scores  # noqa: E402
from herg_model.predict import compute_blocks_from_smiles  # noqa: E402
from herg_model.preprocessing import FeaturePipeline  # noqa: E402
from herg_model.splitting import make_splits  # noqa: E402

BLOCKS = [
    "fp:fcfp",
    "existing_descriptors",
    "physicochemical",
    "lipophilicity",
    "polarity_hbond",
    "basic_amine",
    "charge",
    "aromaticity_shape",
    "surface_electronic",
]

# same LightGBM configuration as the checked-in E6 artifact
PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.030834348179355788,
    "num_leaves": 56,
    "min_child_samples": 5,
    "subsample": 0.9849549260809971,
    "subsample_freq": 1,
    "colsample_bytree": 0.8827098485602951,
    "reg_lambda": 0.0070689749506246055,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def identity(mol):
    return mol


def strip_neutral(mol):
    return neutralize(strip_salts(mol))


def strip_neutral_taut(mol):
    return canonical_tautomer(neutralize(strip_salts(mol)))


TRANSFORMS = {
    "raw": identity,
    "strip": strip_salts,
    "strip_neutral": strip_neutral,
    "tautomer": canonical_tautomer,
    "strip_neutral_taut": strip_neutral_taut,
}


def apply_transform(smiles: list[str], fn) -> tuple[list[str], int]:
    out, failed = [], 0
    for text in smiles:
        mol = mol_from_smiles(text)
        if mol is None:
            out.append(text)
            failed += 1
            continue
        try:
            transformed = fn(mol)
            if transformed is None:
                raise ValueError("empty transform")
            out.append(_to_smiles(transformed))
        except Exception:  # noqa: BLE001
            out.append(text)
            failed += 1
    return out, failed


def _to_smiles(mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = [frame.index.name or "variant", *map(str, frame.columns)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    integer_columns = {"rank", "features", "unique_train_smiles"}
    for index, row in frame.iterrows():
        cells = [str(index)]
        for column, value in zip(frame.columns, row.values):
            if column in integer_columns:
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_threshold(y_non_toxic: np.ndarray, p_non_toxic: np.ndarray) -> float:
    grid = np.linspace(0.01, 0.99, 197)
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        value = f1_score(y_non_toxic, (p_non_toxic >= t).astype(int), pos_label=1, zero_division=0)
        if value > best_v:
            best_t, best_v = float(t), float(value)
    return best_t


def metrics(y_non_toxic: np.ndarray, p_non_toxic: np.ndarray, threshold: float) -> dict:
    pred = (p_non_toxic >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1_label1": float(f1_score(y_non_toxic, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_non_toxic, pred)),
        "roc_auc": float(roc_auc_score(y_non_toxic, p_non_toxic)),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(TRANSFORMS), choices=list(TRANSFORMS))
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    predict = pd.read_csv(ROOT / "data" / "predict_input.csv", usecols=["SMILES"])
    reference = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    measured = (reference["label_chembl"].notna()) | (
        reference["source"] == "pubchem_herg_potency"
    )
    y = train["label"].to_numpy(dtype=np.int8)          # 1 = non-toxic (competition positive)
    yb = (y == 0).astype(np.int8)                       # 1 = blocker (model target)

    mols = [mol_from_smiles(s) for s in train["SMILES"]]
    scaffolds = [bem_murcko_scaffold(m, True) if m is not None else "INVALID" for m in mols]
    split = make_splits(scaffolds, yb, cfg, "scaffold", 42)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, yb))

    json_path = ROOT / "reports" / "standardization_ablation.json"
    results: dict[str, dict] = {}
    if json_path.exists():
        try:
            results = json.loads(json_path.read_text(encoding="utf-8")).get("results", {})
        except Exception:  # noqa: BLE001
            results = {}

    for name in args.variants:
        if name in results:
            print(f"[{name}] already in {json_path.name}, skipping")
            continue
        fn = TRANSFORMS[name]
        t_var = time.time()
        train_smiles, fail_train = apply_transform(train["SMILES"].tolist(), fn)
        predict_smiles, fail_predict = apply_transform(predict["SMILES"].tolist(), fn)
        names, X = compute_blocks_from_smiles(train_smiles, BLOCKS, cfg)
        _, Xp = compute_blocks_from_smiles(predict_smiles, BLOCKS, cfg)
        print(f"[{name}] features={X.shape[1]} train_fail={fail_train} predict_fail={fail_predict} "
              f"({time.time() - t_var:.1f}s)")

        # 5-fold OOF
        oof = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in folds:
            pipe = FeaturePipeline(names=names, scale=False)
            pipe.fit(X[tr_idx], yb[tr_idx])
            model = LGBMClassifier(**PARAMS)
            model.fit(pipe.transform(X[tr_idx]), yb[tr_idx])
            oof[va_idx] = 1.0 - positive_scores(model, pipe.transform(X[va_idx]))
        oof_opt = best_threshold(y, oof)

        # scaffold: train -> val (threshold) -> test (once)
        pipe_sc = FeaturePipeline(names=names, scale=False)
        pipe_sc.fit(X[split.train], yb[split.train])
        model_sc = LGBMClassifier(**PARAMS)
        model_sc.fit(pipe_sc.transform(X[split.train]), yb[split.train])
        p_val = 1.0 - positive_scores(model_sc, pipe_sc.transform(X[split.val]))
        p_test = 1.0 - positive_scores(model_sc, pipe_sc.transform(X[split.test]))
        t_sc = best_threshold(y[split.val], p_val)

        # full train -> predict_input
        pipe_full = FeaturePipeline(names=names, scale=False)
        pipe_full.fit(X, yb)
        model_full = LGBMClassifier(**PARAMS)
        model_full.fit(pipe_full.transform(X), yb)
        p_pred = 1.0 - positive_scores(model_full, pipe_full.transform(Xp))

        results[name] = {
            "n_features_raw": int(X.shape[1]),
            "n_features_kept": int(pipe_full.transform(X[:1]).shape[1]),
            "n_transform_fail_train": int(fail_train),
            "n_transform_fail_predict": int(fail_predict),
            "n_unique_train_smiles": int(len(set(train_smiles))),
            "n_unique_predict_smiles": int(len(set(predict_smiles))),
            "oof_at_0.5": metrics(y, oof, 0.5),
            "oof_at_opt": metrics(y, oof, oof_opt),
            "oof_threshold": oof_opt,
            "scaffold_val_threshold": t_sc,
            "scaffold_test_at_0.5": metrics(y[split.test], p_test, 0.5),
            "scaffold_test_at_val_threshold": metrics(y[split.test], p_test, t_sc),
            "predict813_at_0.5": metrics(
                reference.loc[measured, "label"].to_numpy(), p_pred[measured.to_numpy()], 0.5
            ),
            "predict813_at_oof_threshold": metrics(
                reference.loc[measured, "label"].to_numpy(), p_pred[measured.to_numpy()], oof_opt
            ),
            "predict927_at_0.5": metrics(reference["label"].to_numpy(), p_pred, 0.5),
            "elapsed_sec": round(time.time() - t_var, 1),
        }
        print(f"  OOF F1@0.5={results[name]['oof_at_0.5']['f1_label1']:.4f} "
              f"OOF F1@opt={results[name]['oof_at_opt']['f1_label1']:.4f} "
              f"test F1@0.5={results[name]['scaffold_test_at_0.5']['f1_label1']:.4f} "
              f"813 F1@0.5={results[name]['predict813_at_0.5']['f1_label1']:.4f} "
              f"({results[name]['elapsed_sec']}s)")

    out = {"generated_at": pd.Timestamp.now("UTC").isoformat(), "results": results}
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for name, r in results.items():
        rows.append(
            {
                "variant": name,
                "features": r["n_features_kept"],
                "oof_f1@0.5": r["oof_at_0.5"]["f1_label1"],
                "oof_f1@opt": r["oof_at_opt"]["f1_label1"],
                "oof_auc": r["oof_at_0.5"]["roc_auc"],
                "scaffold_test_f1@0.5": r["scaffold_test_at_0.5"]["f1_label1"],
                "scaffold_test_f1@val": r["scaffold_test_at_val_threshold"]["f1_label1"],
                "predict813_f1@0.5": r["predict813_at_0.5"]["f1_label1"],
                "predict813_f1@oof": r["predict813_at_oof_threshold"]["f1_label1"],
                "predict813_auc": r["predict813_at_0.5"]["roc_auc"],
                "unique_train_smiles": r["n_unique_train_smiles"],
            }
        )
    table = pd.DataFrame(rows).set_index("variant")
    # the challenge ranks on F1(label=1) only -> sort by the held-out 813 F1 at the submission threshold
    table = table.sort_values("predict813_f1@0.5", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table = table.round(4)
    md_path = ROOT / "reports" / "standardization_ablation.md"
    md_path.write_text(
        "# Standardization ablation (E6 baseline, fixed model/CV)\n\n"
        f"Generated: {out['generated_at']}\n\n"
        + frame_to_markdown(table)
        + "\n",
        encoding="utf-8",
    )
    print("\n" + table.to_string())
    print(f"\n[done] {time.time() - t0:.1f}s -> {json_path}")


if __name__ == "__main__":
    main()
