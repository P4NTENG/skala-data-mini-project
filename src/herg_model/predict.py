"""Final-model persistence and inference on new SMILES.

The saved pipeline reproduces the **exact** feature construction order used during training:
fingerprints are regenerated from SMILES with the verified generators, shipped descriptors are
recomputed with the same RDKit calls, and descriptor blocks are rebuilt with the same column
order. ``predict_smiles`` therefore accepts arbitrary molecules, not only rows of train.csv.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit.Chem import Crippen, Descriptors, QED

from . import descriptors as descriptor_module
from . import fingerprints as fingerprint_module
from .chemistry import mol_from_smiles, sa_score
from .config import Config, environment_info
from .data import fingerprint_columns
from .experiments import DOMAIN_GROUPS, FP_BLOCKS

__all__ = [
    "compute_blocks_from_smiles",
    "save_artifact",
    "load_artifact",
    "predict_smiles",
    "build_metadata",
    "SHIPPED_DESCRIPTOR_CALLABLES",
]

#: how each shipped descriptor column is recomputed from SMILES
SHIPPED_DESCRIPTOR_CALLABLES = {
    "MolWt": lambda mol, smi: Descriptors.MolWt(mol),
    "clogp": lambda mol, smi: Crippen.MolLogP(mol),
    "qed": lambda mol, smi: QED.qed(mol),
    "sa_score": lambda mol, smi: sa_score(smi),
}


def _shipped_descriptors(smiles: list[str], cfg: Config) -> tuple[list[str], np.ndarray]:
    names = [n for n in (cfg.get_path("data.existing_descriptors", []) or []) if n in SHIPPED_DESCRIPTOR_CALLABLES]
    out = np.full((len(smiles), len(names)), np.nan, dtype=float)
    for i, smi in enumerate(smiles):
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        for j, name in enumerate(names):
            try:
                out[i, j] = float(SHIPPED_DESCRIPTOR_CALLABLES[name](mol, smi))
            except Exception:
                out[i, j] = np.nan
    return names, out


def compute_blocks_from_smiles(
    smiles: Sequence[str],
    blocks: Sequence[str],
    cfg: Config,
    verbose: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Rebuild a feature matrix for arbitrary SMILES with training-identical column order."""
    smiles = list(smiles)
    names: list[str] = []
    mats: list[np.ndarray] = []
    seen: set[str] = set()

    rdkit_frame = None
    mordred_frame = None

    for block in blocks:
        if block.startswith("fp:"):
            kind = block.split(":", 1)[1]
            block_names = fingerprint_columns(kind, cfg)
            block_matrix = fingerprint_module.recompute_matrix(smiles, kind, cfg).astype(float)
        elif block == "all_fingerprints":
            block_names = []
            parts = []
            for fp_block in FP_BLOCKS:
                sub_names, sub_matrix = compute_blocks_from_smiles(smiles, [fp_block], cfg, verbose)
                block_names.extend(sub_names)
                parts.append(sub_matrix)
            block_matrix = np.hstack(parts) if parts else np.zeros((len(smiles), 0))
        elif block == "existing_descriptors":
            block_names, block_matrix = _shipped_descriptors(smiles, cfg)
        elif block in {"rdkit_all", *DOMAIN_GROUPS}:
            if rdkit_frame is None:
                rdkit_frame = descriptor_module.compute_rdkit_descriptors(
                    smiles, cfg, use_cache=True, verbose=verbose
                )
            all_names = [c for c in rdkit_frame.columns if c != "SMILES"]
            if block == "rdkit_all":
                block_names = all_names
            else:
                block_names = descriptor_module.assign_groups(all_names).get(block, [])
            block_matrix = rdkit_frame[block_names].to_numpy(dtype=float)
        elif block == "mordred_2d":
            if mordred_frame is None:
                mordred_frame, _ = descriptor_module.compute_mordred_descriptors(
                    smiles, cfg, use_cache=True, verbose=verbose
                )
            if mordred_frame is None:
                block_names, block_matrix = [], np.zeros((len(smiles), 0))
            else:
                block_names = [c for c in mordred_frame.columns if c != "SMILES"]
                block_matrix = mordred_frame[block_names].to_numpy(dtype=float)
        else:
            raise ValueError(f"unknown feature block {block!r}")

        if not block_names:
            continue
        if len(block_names) != len(set(block_names)):
            keep = [j for j, n in enumerate(block_names) if n not in seen]
            block_names = [block_names[j] for j in keep]
            block_matrix = block_matrix[:, keep]
        seen.update(block_names)
        names.extend(block_names)
        mats.append(np.asarray(block_matrix, dtype=float))

    if not mats:
        return [], np.zeros((len(smiles), 0), dtype=float)
    return names, np.hstack(mats)


def build_metadata(
    cfg: Config,
    training_info: dict,
    threshold: float,
    blocks: Sequence[str],
    feature_names: Sequence[str],
    extra: dict | None = None,
) -> dict:
    """Everything needed to replay the model: label semantics, feature order, versions, split."""
    metadata = {
        "label_meanings": {"0": "blocker (positive class)", "1": "non_blocker"},
        "positive_class": "blocker",
        "prediction_endpoint": "hERG channel blocking activity (NOT clinical QT prolongation)",
        "blocks": list(blocks),
        "feature_order": list(feature_names),
        "n_features": len(feature_names),
        "selected_threshold": float(threshold),
        "descriptor_config": cfg.get_path("descriptors"),
        "fingerprint_config": cfg.get_path("data.fingerprints"),
        "split_config": cfg.get_path("split"),
        "label_rule_note": "labels are treated as given; approx. 'IC50 < 10 uM' rule is not assumed for all rows",
        "inference_note": (
            "shipped descriptors are recomputed from SMILES at inference time; the recomputation "
            "matches the stored columns for all but 1 qed row and 10 sa_score rows "
            "(see reports/data_quality_report.md, section 9)"
        ),
        "training_data": training_info,
        "environment": environment_info(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_artifact(
    path: str | Path,
    model,
    pipeline,
    metadata: dict,
    selected_indices=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "pipeline": pipeline,
        # raw, block-level feature order produced by compute_blocks_from_smiles
        "feature_names": list(getattr(pipeline, "names", metadata["feature_order"])),
        # order of the columns the *model* was fitted on (post-selection)
        "model_feature_order": list(metadata["feature_order"]),
        "selected_indices": None if selected_indices is None else list(map(int, selected_indices)),
        "blocks": metadata["blocks"],
        "threshold": metadata["selected_threshold"],
        "metadata": metadata,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_artifact(path: str | Path, cfg: Config | None = None) -> dict:
    with open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    if cfg is not None:
        payload["cfg"] = cfg
    return payload


def predict_smiles(
    smiles: Sequence[str],
    artifact: dict,
    cfg: Config,
    verbose: bool = False,
) -> pd.DataFrame:
    """Predict blocker probability and class for arbitrary SMILES.

    Returns the blocker probability, the predicted label in the **original** convention
    (0 = blocker) and the predicted probability of the positive (blocker) class.
    """
    blocks = artifact["blocks"]
    names, X = compute_blocks_from_smiles(smiles, blocks, cfg, verbose=verbose)
    expected = list(artifact["feature_names"])
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(
            f"feature mismatch between rebuilt matrix ({len(names)}) and stored model "
            f"({len(expected)}); missing={missing[:5]} unexpected={extra[:5]}"
        )
    pipeline = artifact["pipeline"]
    transformed = pipeline.transform(X)
    selected = artifact.get("selected_indices")
    if selected is not None:
        transformed = transformed[:, np.asarray(selected, dtype=int)]
    model = artifact["model"]
    from .models import positive_scores

    probability = positive_scores(model, transformed)
    threshold = float(artifact["threshold"])
    predicted_blocker = (probability >= threshold).astype(int)
    return pd.DataFrame(
        {
            "SMILES": list(smiles),
            "blocker_probability": probability,
            "predicted_label": np.where(predicted_blocker == 1, 0, 1),
            "predicted_class": np.where(predicted_blocker == 1, "blocker", "non_blocker"),
            "threshold": threshold,
        }
    )
