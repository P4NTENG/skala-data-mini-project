"""Label-optimized blend search on the 813 measured rows.

The user explicitly asked to use the labelled predict set for the final decision, so this
script searches blend weights + threshold directly on the 813 rows. To keep the overfitting
risk visible it also reports a nested 5-fold estimate (weights fitted on 4/5 of the 813,
scored on the held-out 1/5).

Output: predict_output_김인성6.csv (label-optimized candidate) + reports/phase4_label_blend.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from herg_model.models import positive_scores  # noqa: E402
from phase4_fp_views import DESC, PARAMS, fp_matrix  # noqa: E402

VIEWS = ["maccs", "atom_pair", "torsion", "ecfp_count", "fcfp_count"]


def f1(y, p, t):
    return float(f1_score(y, (p >= t).astype(int), pos_label=1, zero_division=0))


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    pred = pd.read_csv(ROOT / "data" / "predict_input.csv")
    ref = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    measured = ((ref["label_chembl"].notna()) | (ref["source"] == "pubchem_herg_potency")).to_numpy()
    y_all = ref["label"].to_numpy(dtype=np.int8)
    y = y_all[measured]
    yb = (train["label"].to_numpy(dtype=np.int8) == 0).astype(np.int8)

    with open(ROOT / "artifacts" / "best_model_v2.joblib", "rb") as handle:
        art = pickle.load(handle)
    columns = [np.asarray(art["phase3_final"]["p_final_927"])]
    names = ["v4"]

    for kind in VIEWS:
        X = np.hstack([fp_matrix(train["SMILES"].tolist(), kind), train[DESC].to_numpy(dtype=np.float32)])
        Xp = np.hstack([fp_matrix(pred["SMILES"].tolist(), kind), pred[DESC].to_numpy(dtype=np.float32)])
        model = LGBMClassifier(**PARAMS)
        model.fit(X, yb)
        columns.append(1.0 - positive_scores(model, Xp))
        names.append(kind)
        print("fitted", kind, flush=True)

    Z = np.column_stack(columns)          # 927 x 6
    Zm = Z[measured]                      # 813 x 6

    v4 = Zm[:, 0]
    print(f"v4 813 F1@0.5 = {f1(y, v4, 0.5):.4f}")

    rng = np.random.default_rng(0)

    def search(rows: np.ndarray, labels: np.ndarray, n_iter: int = 30000):
        best = (-1.0, None, None)
        for _ in range(n_iter):
            w = rng.dirichlet(np.full(rows.shape[1], 0.5))
            p = rows @ w
            for t in (0.47, 0.48, 0.49, 0.495, 0.50, 0.505, 0.51, 0.52, 0.53):
                value = f1(labels, p, t)
                if value > best[0]:
                    best = (value, w, t)
        return best

    # honest nested estimate: fit weights/threshold on 4/5 of the 813, score on 1/5
    nested_pred = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(Zm, y):
        score, w, t = search(Zm[tr], y[tr], n_iter=4000)
        nested_pred[va] = (Zm[va] @ w >= t).astype(int)
    nested_f1 = f1(y, nested_pred, 0.5)  # nested_pred already hard labels
    print(f"nested 813 F1 (weights fitted out-of-fold) = {nested_f1:.4f}")

    # final: fit on all 813
    score, w, t = search(Zm, y, n_iter=60000)
    p_best = Z @ w
    labels = (p_best >= t).astype(int)
    print("weights:", dict(zip(names, np.round(w, 3))), "threshold", t)
    print(f"in-sample 813 F1 = {score:.4f} | 927 F1 = {f1(y_all, p_best, t):.4f}")
    print("labels:", pd.Series(labels).value_counts().sort_index().to_dict())

    out = ROOT / "predict_output_김인성6.csv"
    pd.DataFrame({"SMILES": pred["SMILES"], "label": labels}).to_csv(out, index=False, encoding="utf-8")
    print("[output]", out)

    report = {
        "names": names,
        "weights": {k: float(v) for k, v in zip(names, w)},
        "threshold": float(t),
        "v4_813_f1": f1(y, v4, 0.5),
        "nested_813_f1": nested_f1,
        "in_sample_813_f1": score,
        "f927": f1(y_all, p_best, t),
        "nested_pred_f1": nested_f1,
    }
    (ROOT / "reports" / "phase4_label_blend.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
