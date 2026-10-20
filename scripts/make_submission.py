"""Write a competition submission (``SMILES,label``) from the checked-in model.

The positive class of the model is ``label = 0`` (blocker); the competition scores F1 with
``label = 1`` (non-toxic) as positive. A molecule is therefore submitted as 1 when the
predicted blocker probability is below the threshold.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/make_submission.py \
        --author 김인성 --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from herg_model.config import load_config  # noqa: E402
from herg_model.models import positive_scores  # noqa: E402
from herg_model.predict import compute_blocks_from_smiles, load_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="김인성")
    parser.add_argument("--input", default="data/predict_input.csv")
    parser.add_argument("--artifact", default="artifacts/best_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="blocker-probability threshold (label 1 when p_blocker < threshold)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config()
    artifact = load_artifact(ROOT / args.artifact)
    frame = pd.read_csv(ROOT / args.input, usecols=["SMILES"])

    names, matrix = compute_blocks_from_smiles(frame["SMILES"].tolist(), artifact["blocks"], cfg)
    if names != list(artifact["feature_names"]):
        raise RuntimeError("rebuilt feature names do not match the artifact")
    transformed = artifact["pipeline"].transform(matrix)
    blocker_probability = positive_scores(artifact["model"], transformed)

    submission = pd.DataFrame(
        {
            "SMILES": frame["SMILES"],
            "label": (blocker_probability < args.threshold).astype(int),
        }
    )
    output = Path(args.output) if args.output else ROOT / f"predict_output_{args.author}.csv"
    submission.to_csv(output, index=False, encoding="utf-8")

    print(f"[submission] {output}")
    print(f"  rows: {len(submission)}")
    print(f"  label counts: {submission['label'].value_counts().sort_index().to_dict()}")
    print(f"  threshold (blocker prob): {args.threshold}")
    print(f"  columns: {submission.columns.tolist()}")


if __name__ == "__main__":
    main()
