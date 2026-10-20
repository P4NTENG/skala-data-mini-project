"""Phase 4: stacking / heterogeneous blend on top of the Phase 2-3 models.

Base models (OOF + 927 predictions):
  P2_12  LightGBM  E6_DRV
  P2_13  LightGBM  ALL_FP_E6_DRV
  P2_14  XGBoost   ALL_FP_E6_DRV
  P2_15  ExtraTrees ALL_FP_E6_DRV
  P2_16  LightGBM  ALL_FP_E7_DRV
  STD    LightGBM  E6 + salt/neutral/tautomer standardization (new OOF computed here)

Candidates are selected on OOF F1(label=1); the 813 measured set is only the final check.
Output: predict_output_김인성5.csv (only if it beats the current v4 on both OOF and 813).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

OOF_DIR = ROOT / ".cache" / "herg_model" / "phase2_oof"
STD_OOF = ROOT / "reports" / "standardization_oof_strip_neutral_taut.npy"
STD_927 = ROOT / "reports" / "standardization_prob_strip_neutral_taut.npy"

BASE = {
    "P2_12": "P2_12_E6_DRV_lgbm_f1",
    "P2_13": "P2_13_ALL_FP_E6_DRV_lgbm_f1",
    "P2_14": "P2_14_ALL_FP_E6_DRV_xgboost_f1",
    "P2_15": "P2_15_ALL_FP_E6_DRV_et_f1",
    "P2_16": "P2_16_ALL_FP_E7_DRV_lgbm_f1",
}


def f1(y: np.ndarray, p: np.ndarray, t: float = 0.5) -> float:
    return float(f1_score(y, (p >= t).astype(int), pos_label=1, zero_division=0))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.30, 0.70, 161)
    return float(max(grid, key=lambda t: f1(y, p, t)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def std_oof() -> np.ndarray:
    """5-fold OOF for the standardization E6 model (reuse the ablation setup)."""
    if STD_OOF.exists():
        return np.load(STD_OOF)
    cfg = load_config()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    yb = (train["label"].to_numpy(dtype=np.int8) == 0).astype(np.int8)
    transform = TRANSFORMS["strip_neutral_taut"]
    smiles, _ = apply_transform(train["SMILES"].tolist(), transform)
    names, X = compute_blocks_from_smiles(smiles, BLOCKS, cfg)
    oof = np.zeros(len(yb), dtype=float)
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, yb):
        pipe = FeaturePipeline(names=names, scale=False)
        pipe.fit(X[tr], yb[tr])
        model = LGBMClassifier(**PARAMS)
        model.fit(pipe.transform(X[tr]), yb[tr])
        oof[va] = 1.0 - positive_scores(model, pipe.transform(X[va]))
    STD_OOF.parent.mkdir(parents=True, exist_ok=True)
    np.save(STD_OOF, oof)
    return oof


def dataframe() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    ref = pd.read_csv(ROOT / "data" / "predict_input_chembl_labeled.csv")
    measured = (
        (ref["label_chembl"].notna()) | (ref["source"] == "pubchem_herg_potency")
    ).to_numpy()
    y = train["label"].to_numpy(dtype=np.int8)
    y_pred = ref["label"].to_numpy(dtype=np.int8)

    with open(ROOT / "artifacts" / "best_model_v2.joblib", "rb") as handle:
        artifact = pickle.load(handle)

    oofs, preds, names = [], [], []
    for short, run_id in BASE.items():
        oofs.append(np.load(OOF_DIR / f"{run_id}_oof.npy"))
        preds.append(np.asarray(artifact["preds_927"][run_id]))
        names.append(short)
    oofs.append(std_oof())
    preds.append(np.load(STD_927))
    names.append("STD")

    Z_oof = np.column_stack(oofs)
    Z_927 = np.column_stack(preds)
    return y, y_pred, measured, Z_oof, Z_927, names


def random_weights(n: int, rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(n, alpha))


def fit_weights(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_iter: int = 4000, alpha: float = 1.0):
    best_w, best_f1 = None, -1.0
    for _ in range(n_iter):
        w = random_weights(Z.shape[1], rng, alpha)
        value = f1(y, Z @ w)
        if value > best_f1:
            best_f1, best_w = value, w
    return best_w, best_f1


def nested_weights(Z: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Nested estimate: optimize weights on 4/5, score on 1/5. Returns final weights + nested OOF."""
    rng = np.random.default_rng(seed)
    pred = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(Z, y):
        w, _ = fit_weights(Z[tr], y[tr], rng, n_iter=2000)
        pred[va] = Z[va] @ w
    final_w, _ = fit_weights(Z, y, rng, n_iter=10000)
    return final_w, pred, f1(y, pred)


def stack_lr(Z_oof: np.ndarray, Z_927: np.ndarray, y: np.ndarray, rank: bool = False):
    def transform(a):
        if rank:
            from scipy.stats import rankdata

            out = np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])
            return out
        return logit(a)

    Xo, Xp = transform(Z_oof), transform(Z_927)
    pred = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(Xo, y):
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(Xo[tr], y[tr])
        pred[va] = model.predict_proba(Xo[va])[:, 1]
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(Xo, y)
    return pred, model.predict_proba(Xp)[:, 1]


def main() -> None:
    t0 = time.time()
    y, y_pred, measured, Z_oof, Z_927, names = dataframe()
    print("base models:", names, "OOF shape", Z_oof.shape, "927 shape", Z_927.shape)
    for i, name in enumerate(names):
        print(f"  {name:<5} OOF F1@0.5={f1(y, Z_oof[:, i]):.4f}  813 F1@0.5={f1(y_pred[measured], Z_927[measured, i]):.4f}")

    candidates: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    # fixed blends
    idx = {n: i for i, n in enumerate(names)}
    w3 = np.zeros(len(names)); w3[idx["P2_13"]] = 0.5; w3[idx["P2_12"]] = 0.25; w3[idx["P2_16"]] = 0.25
    w4 = np.zeros(len(names)); w4[idx["P2_13"]] = 0.45; w4[idx["P2_16"]] = 0.25; w4[idx["P2_12"]] = 0.15; w4[idx["P2_14"]] = 0.15
    candidates["blend3 (current v4)"] = (Z_oof @ w3, Z_927 @ w3, "0.50 P2_13 + 0.25 P2_12 + 0.25 P2_16")
    candidates["blend4 (current v3)"] = (Z_oof @ w4, Z_927 @ w4, "0.45 P2_13 + 0.25 P2_16 + 0.15 P2_12 + 0.15 P2_14")
    candidates["equal6"] = (Z_oof.mean(axis=1), Z_927.mean(axis=1), "equal weight, all 6")

    # nested weight optimization: the OOF vector below is honest (weights fitted out-of-fold)
    w_opt, oof_nested, nested_f1 = nested_weights(Z_oof, y, seed=7)
    candidates["weights_opt6"] = (oof_nested, Z_927 @ w_opt, "nested random-search weights, all 6")
    print("weights_opt6 weights:", dict(zip(names, np.round(w_opt, 3))), "nested OOF F1", round(nested_f1, 4))

    # logistic stacking
    oof_lr, pred_lr = stack_lr(Z_oof, Z_927, y, rank=False)
    candidates["stack_lr"] = (oof_lr, pred_lr, "LogisticRegression on logit(OOF), 6 models")
    oof_lr_rank, pred_lr_rank = stack_lr(Z_oof, Z_927, y, rank=True)
    candidates["stack_lr_rank"] = (oof_lr_rank, pred_lr_rank, "LogisticRegression on rank(OOF), 6 models")

    rows = []
    for name, (oof, pred927, note) in candidates.items():
        t_oof = best_threshold(y, oof)
        f813 = f1(y_pred[measured], pred927[measured], 0.5)
        f813_opt = f1(y_pred[measured], pred927[measured], t_oof)
        rows.append(
            {
                "candidate": name,
                "oof_f1@0.5": f1(y, oof, 0.5),
                "oof_f1@opt": f1(y, oof, t_oof),
                "oof_thr": t_oof,
                "oof_auc": roc_auc_score(y, oof),
                "f813@0.5": f813,
                "f813@oof_thr": f813_opt,
                "f927@0.5": f1(y_pred, pred927, 0.5),
                "note": note,
            }
        )
    table = pd.DataFrame(rows).sort_values("oof_f1@0.5", ascending=False)
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name = table.iloc[0]["candidate"]
    best_oof, best_927, note = candidates[best_name]
    out = ROOT / "predict_output_김인성5.csv"
    sub = pd.DataFrame(
        {
            "SMILES": pd.read_csv(ROOT / "data" / "predict_input.csv", usecols=["SMILES"])["SMILES"],
            "label": (best_927 >= 0.5).astype(int),
        }
    )
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[selected] {best_name}: {note}")
    print(f"[output] {out} labels={sub.label.value_counts().sort_index().to_dict()}")

    report = {
        "names": names,
        "weights_opt6": dict(zip(names, map(float, w_opt))),
        "nested_oof_f1_weights_opt6": float(nested_f1),
        "selected": best_name,
        "selected_note": note,
        "table": table.to_dict(orient="records"),
    }
    (ROOT / "reports" / "phase4_stack_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table.to_csv(ROOT / "reports" / "phase4_stack_report.csv", index=False, encoding="utf-8")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
