"""Paired bootstrap for the standardization ablation (F1 label=1 only).

Compares ``raw`` and ``strip_neutral_taut`` on the 813 measured predict rows at the
submission threshold 0.5. Reports:

* per-variant F1 and bootstrap 95% CI
* paired bootstrap 95% CI of the F1 difference
* McNemar test on per-molecule correctness

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/standardization_bootstrap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from herg_model.config import load_config  # noqa: E402
from herg_model.models import positive_scores  # noqa: E402
from herg_model.predict import compute_blocks_from_smiles  # noqa: E402
from herg_model.preprocessing import FeaturePipeline  # noqa: E402
from standardization_ablation import (  # noqa: E402
    BLOCKS,
    PARAMS,
    TRANSFORMS,
    apply_transform,
)

VARIANTS = ("raw", "strip_neutral_taut")
THRESHOLD = 0.5
N_BOOT = 10_000
SEED = 0


def f1_at(y: np.ndarray, p: np.ndarray, threshold: float = THRESHOLD) -> float:
    return float(f1_score(y, (p >= threshold).astype(int), pos_label=1, zero_division=0))


def main() -> None:
    cfg = load_config()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    predict = pd.read_csv(ROOT / "data" / "predict_input.csv", usecols=["SMILES"])
    reference = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    measured = (
        (reference["label_chembl"].notna()) | (reference["source"] == "pubchem_herg_potency")
    ).to_numpy()
    y = reference.loc[measured, "label"].to_numpy(dtype=np.int8)
    yb = (train["label"].to_numpy(dtype=np.int8) == 0).astype(np.int8)

    predictions: dict[str, np.ndarray] = {}
    for name in VARIANTS:
        fn = TRANSFORMS[name]
        train_smiles, _ = apply_transform(train["SMILES"].tolist(), fn)
        predict_smiles, _ = apply_transform(predict["SMILES"].tolist(), fn)
        names, X = compute_blocks_from_smiles(train_smiles, BLOCKS, cfg)
        predict_names, Xp = compute_blocks_from_smiles(predict_smiles, BLOCKS, cfg)
        if predict_names != names:
            raise RuntimeError(f"{name}: train/predict feature names differ")

        pipe = FeaturePipeline(names=names, scale=False)
        pipe.fit(X, yb)
        model = LGBMClassifier(**PARAMS)
        model.fit(pipe.transform(X), yb)
        p_non_toxic = 1.0 - positive_scores(model, pipe.transform(Xp))
        predictions[name] = p_non_toxic
        np.save(ROOT / "reports" / f"standardization_prob_{name}.npy", p_non_toxic)
        print(f"[{name}] 813 F1@0.5 = {f1_at(y, p_non_toxic[measured]):.4f}")

    p_measured = {name: predictions[name][measured] for name in VARIANTS}
    pred = {name: (p_measured[name] >= THRESHOLD).astype(int) for name in VARIANTS}
    rng = np.random.default_rng(SEED)
    n = len(y)
    f1_draws = {name: np.empty(N_BOOT) for name in VARIANTS}
    diff_draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for name in VARIANTS:
            f1_draws[name][b] = f1_at(y[idx], p_measured[name][idx])
        diff_draws[b] = f1_draws[VARIANTS[1]][b] - f1_draws[VARIANTS[0]][b]

    # McNemar on per-molecule correctness (exact binomial, two-sided)
    correct = {name: pred[name] == y for name in VARIANTS}
    b01 = int(np.sum(correct[VARIANTS[0]] & ~correct[VARIANTS[1]]))
    b10 = int(np.sum(~correct[VARIANTS[0]] & correct[VARIANTS[1]]))
    from scipy.stats import binomtest

    mcnemar_p = float(binomtest(min(b01, b10), b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0

    result = {
        "n": int(n),
        "threshold": THRESHOLD,
        "n_bootstrap": N_BOOT,
        "variants": {},
        "paired_difference": {
            "name": f"{VARIANTS[1]} - {VARIANTS[0]}",
            "point": float(f1_at(y, predictions[VARIANTS[1]][measured]) - f1_at(y, predictions[VARIANTS[0]][measured])),
            "mean": float(diff_draws.mean()),
            "ci95": [float(np.percentile(diff_draws, 2.5)), float(np.percentile(diff_draws, 97.5))],
            "p_le_0": float((diff_draws <= 0).mean()),
            "p_ge_0": float((diff_draws >= 0).mean()),
        },
        "mcnemar": {
            "raw_correct_combined_wrong": b01,
            "raw_wrong_combined_correct": b10,
            "p_value": mcnemar_p,
        },
    }
    for name in VARIANTS:
        result["variants"][name] = {
            "f1": float(f1_at(y, predictions[name][measured])),
            "ci95": [float(np.percentile(f1_draws[name], 2.5)), float(np.percentile(f1_draws[name], 97.5))],
            "accuracy": float((pred[name] == y).mean()),
            "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, pred[name], labels=[0, 1]).tolist(),
        }

    out = ROOT / "reports" / "standardization_bootstrap.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
