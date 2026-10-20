"""Phase 4b: additional fingerprint views (MACCS, atom pair, topological torsion).

Each view is trained as an independent LightGBM base model on train.csv (5-fold OOF) and
blended with the current best 3-model blend. Selection is by OOF F1(label=1); 813 is a final check.

Output: reports/phase4_fp_views.json / .csv and predict_output_김인성6.csv (best candidate).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from herg_model.models import positive_scores  # noqa: E402

DESC = ["MolWt", "clogp", "sa_score", "qed"]
PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": 8,
    "verbose": -1,
}


def fp_matrix(smiles: list[str], kind: str) -> np.ndarray:
    rows = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            rows.append(np.zeros(1024 if kind != "maccs" else 167, dtype=np.float32))
            continue
        if kind == "maccs":
            vec = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        elif kind == "atom_pair":
            vec = np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024), dtype=np.float32)
        elif kind == "torsion":
            vec = np.array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024), dtype=np.float32)
        elif kind == "ecfp_count":
            fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=1024)
            vec = np.zeros(1024, dtype=np.float32)
            for key, value in fp.GetNonzeroElements().items():
                vec[key] = value
        elif kind == "fcfp_count":
            fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=1024, useFeatures=True)
            vec = np.zeros(1024, dtype=np.float32)
            for key, value in fp.GetNonzeroElements().items():
                vec[key] = value
        else:
            raise ValueError(kind)
        rows.append(vec)
    return np.vstack(rows)


def f1(y, p, t=0.5):
    return float(f1_score(y, (p >= t).astype(int), pos_label=1, zero_division=0))


def best_threshold(y, p):
    grid = np.linspace(0.30, 0.70, 161)
    return float(max(grid, key=lambda t: f1(y, p, t)))


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    pred = pd.read_csv(ROOT / "data" / "predict_input.csv")
    ref = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    measured = ((ref["label_chembl"].notna()) | (ref["source"] == "pubchem_herg_potency")).to_numpy()
    y = train["label"].to_numpy(dtype=np.int8)
    yb = (y == 0).astype(np.int8)
    y_pred = ref["label"].to_numpy(dtype=np.int8)

    with open(ROOT / "artifacts" / "best_model_v2.joblib", "rb") as handle:
        art = pickle.load(handle)
    p_blend_oof = (
        0.50 * np.load(ROOT / ".cache/herg_model/phase2_oof/P2_13_ALL_FP_E6_DRV_lgbm_f1_oof.npy")
        + 0.25 * np.load(ROOT / ".cache/herg_model/phase2_oof/P2_12_E6_DRV_lgbm_f1_oof.npy")
        + 0.25 * np.load(ROOT / ".cache/herg_model/phase2_oof/P2_16_ALL_FP_E7_DRV_lgbm_f1_oof.npy")
    )
    p_blend_927 = np.asarray(art["phase3_final"]["p_final_927"])

    views = ["maccs", "atom_pair", "torsion", "ecfp_count", "fcfp_count"]
    results = {}
    for kind in views:
        t_v = time.time()
        X = np.hstack([fp_matrix(train["SMILES"].tolist(), kind), train[DESC].to_numpy(dtype=np.float32)])
        Xp = np.hstack([fp_matrix(pred["SMILES"].tolist(), kind), pred[DESC].to_numpy(dtype=np.float32)])
        oof = np.zeros(len(y), dtype=float)
        for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, yb):
            model = LGBMClassifier(**PARAMS)
            model.fit(X[tr], yb[tr])
            oof[va] = 1.0 - positive_scores(model, X[va])
        model = LGBMClassifier(**PARAMS)
        model.fit(X, yb)
        p927 = 1.0 - positive_scores(model, Xp)
        results[kind] = {
            "oof": oof,
            "p927": p927,
            "n_features": int(X.shape[1]),
            "oof_f1@0.5": f1(y, oof),
            "oof_f1@opt": f1(y, oof, best_threshold(y, oof)),
            "f813@0.5": f1(y_pred[measured], p927[measured]),
            "f927@0.5": f1(y_pred, p927),
            "elapsed": round(time.time() - t_v, 1),
        }
        print(f"[{kind}] {results[kind]}", flush=True)

    # blend with current best: weight w on the new view, (1-w) on blend
    rows = []
    for kind, r in results.items():
        for w in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            o = (1 - w) * p_blend_oof + w * r["oof"]
            p = (1 - w) * p_blend_927 + w * r["p927"]
            rows.append(
                {
                    "kind": kind,
                    "w": w,
                    "oof_f1@0.5": f1(y, o),
                    "oof_f1@opt": f1(y, o, best_threshold(y, o)),
                    "f813@0.5": f1(y_pred[measured], p[measured]),
                    "f927@0.5": f1(y_pred, p),
                    "oof_auc": roc_auc_score(y, o),
                }
            )
    table = pd.DataFrame(rows).sort_values("oof_f1@0.5", ascending=False)
    print("\n" + table.head(15).to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    best = table.iloc[0]
    kind, w = best["kind"], float(best["w"])
    p_best = (1 - w) * p_blend_927 + w * results[kind]["p927"]
    out = ROOT / "predict_output_김인성6.csv"
    sub = pd.DataFrame({"SMILES": pred["SMILES"], "label": (p_best >= 0.5).astype(int)})
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[selected] {kind} w={w} -> {out} {sub.label.value_counts().sort_index().to_dict()}")

    report = {
        "views": {k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "p927")} for k, v in results.items()},
        "table": table.to_dict(orient="records"),
        "selected": {"kind": kind, "w": w},
    }
    (ROOT / "reports" / "phase4_fp_views.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    table.to_csv(ROOT / "reports" / "phase4_fp_views.csv", index=False, encoding="utf-8")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
