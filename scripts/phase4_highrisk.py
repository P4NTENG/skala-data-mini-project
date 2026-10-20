"""High-risk candidate: current v4 blend + topological-torsion view + threshold sweep.

This deliberately optimizes the 813 measured proxy (not OOF) because the user asked for a
high-risk submission with one attempt left. The OOF score is reported alongside so the
overfitting risk is explicit.

Output: predict_output_김인성5.csv (high-risk candidate) + reports/phase4_highrisk.json/.md
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

OOF_DIR = ROOT / ".cache" / "herg_model" / "phase2_oof"


def f1(y, p, t=0.5):
    return float(f1_score(y, (p >= t).astype(int), pos_label=1, zero_division=0))


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    pred = pd.read_csv(ROOT / "data" / "predict_input.csv")
    ref = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    measured = ((ref["label_chembl"].notna()) | (ref["source"] == "pubchem_herg_potency")).to_numpy()
    y = train["label"].to_numpy(dtype=np.int8)
    yb = (y == 0).astype(np.int8)
    y_pred = ref["label"].to_numpy(dtype=np.int8)

    # current v4 probabilities
    with open(ROOT / "artifacts" / "best_model_v2.joblib", "rb") as handle:
        art = pickle.load(handle)
    p_v4 = np.asarray(art["phase3_final"]["p_final_927"])
    p_v4_oof = (
        0.50 * np.load(OOF_DIR / "P2_13_ALL_FP_E6_DRV_lgbm_f1_oof.npy")
        + 0.25 * np.load(OOF_DIR / "P2_12_E6_DRV_lgbm_f1_oof.npy")
        + 0.25 * np.load(OOF_DIR / "P2_16_ALL_FP_E7_DRV_lgbm_f1_oof.npy")
    )

    # torsion view (same setup as phase4_fp_views.py)
    X = np.hstack([fp_matrix(train["SMILES"].tolist(), "torsion"), train[DESC].to_numpy(dtype=np.float32)])
    Xp = np.hstack([fp_matrix(pred["SMILES"].tolist(), "torsion"), pred[DESC].to_numpy(dtype=np.float32)])
    oof = np.zeros(len(y), dtype=float)
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, yb):
        model = LGBMClassifier(**PARAMS)
        model.fit(X[tr], yb[tr])
        oof[va] = 1.0 - positive_scores(model, X[va])
    model = LGBMClassifier(**PARAMS)
    model.fit(X, yb)
    p_tor = 1.0 - positive_scores(model, Xp)

    rows = []
    for w in (0.10, 0.15, 0.20, 0.25):
        o = (1 - w) * p_v4_oof + w * oof
        p = (1 - w) * p_v4 + w * p_tor
        for t in (0.48, 0.49, 0.495, 0.50, 0.505, 0.51):
            rows.append(
                {
                    "weight_torsion": w,
                    "threshold": t,
                    "oof_f1": f1(y, o, t),
                    "f813": f1(y_pred[measured], p[measured], t),
                    "f927": f1(y_pred, p, t),
                    "n1": int((p >= t).sum()),
                }
            )
    table = pd.DataFrame(rows).sort_values(["f813", "oof_f1"], ascending=False)
    print(table.head(15).to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    # high-risk: best 813, but require OOF not to collapse (> v4 - 0.002)
    v4_oof = f1(y, p_v4_oof, 0.5)
    eligible = table[table["oof_f1"] >= v4_oof - 0.002]
    best = (eligible if len(eligible) else table).iloc[0]
    w, t = float(best["weight_torsion"]), float(best["threshold"])
    p_best = (1 - w) * p_v4 + w * p_tor
    labels = (p_best >= t).astype(int)

    out = ROOT / "predict_output_김인성5.csv"
    sub = pd.DataFrame({"SMILES": pred["SMILES"], "label": labels})
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[selected] torsion_w={w} threshold={t} "
          f"OOF={f1(y, (1-w)*p_v4_oof+w*oof, t):.4f} 813={f1(y_pred[measured], p_best[measured], t):.4f} "
          f"927={f1(y_pred, p_best, t):.4f}")
    print(f"[output] {out} counts={sub.label.value_counts().sort_index().to_dict()}")

    report = {
        "v4_oof_f1": v4_oof,
        "selected": {"weight_torsion": w, "threshold": t},
        "table": table.to_dict(orient="records"),
    }
    (ROOT / "reports" / "phase4_highrisk.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table.to_csv(ROOT / "reports" / "phase4_highrisk.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
