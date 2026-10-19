"""Dataset loading, label semantics and structural integrity checks.

Label semantics are fixed and must never be inverted:

* ``label = 0`` -> hERG blocker -> positive class (``y_blocker == 1``)
* ``label = 1`` -> non-blocker -> negative class
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, as_config, resolve

__all__ = [
    "load_dataset",
    "blocker_target",
    "labels_from_blocker",
    "fingerprint_columns",
    "existing_descriptor_names",
    "existing_descriptor_matrix",
    "fingerprint_matrix_from_csv",
    "dataset_summary",
    "dataset_index",
]

FP_KINDS = ("ecfp", "fcfp", "ptfp")


def _read_csv(path: Path, cfg: Config) -> pd.DataFrame:
    fp_cols: list[str] = []
    dtypes: dict[str, Any] = {}
    for kind in FP_KINDS:
        prefix = cfg.get_path(f"data.fingerprints.{kind}.prefix", f"{kind}_")
        size = int(cfg.get_path(f"data.fingerprints.{kind}.size", 1024))
        cols = [f"{prefix}{i}" for i in range(size)]
        fp_cols.extend(cols)
        dtypes.update({c: "int8" for c in cols})

    header = pd.read_csv(path, nrows=0)
    available_fp = [c for c in fp_cols if c in header.columns]
    label_col = str(cfg.get_path("data.label_col", "label"))
    for name in cfg.get_path("data.existing_descriptors", []) or []:
        if name in header.columns:
            dtypes[name] = "float32"
    dtypes[label_col] = "int8"

    missing = sorted(set(fp_cols) - set(available_fp))
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} expected fingerprint columns, e.g. {missing[:5]}"
        )
    return pd.read_csv(path, dtype=dtypes)


def load_dataset(cfg: Config, use_cache: bool = True) -> pd.DataFrame:
    """Load ``data/train.csv`` (unmodified) as a typed DataFrame."""
    path = resolve(cfg.get_path("paths.data", "data/train.csv"))
    if not path.exists():
        raise FileNotFoundError(f"training data not found: {path}")

    cache_path = resolve(cfg.get_path("paths.cache", ".cache/herg_model")) / "dataset.pkl"
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("path") == str(path) and payload.get("mtime") == path.stat().st_mtime:
                return payload["df"]
        except Exception:  # pragma: no cover - cache is best effort
            pass

    df = _read_csv(path, cfg)
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    df[smiles_col] = df[smiles_col].astype(str)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump({"path": str(path), "mtime": path.stat().st_mtime, "df": df}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return df


def blocker_target(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """``y_blocker = (label == 0).astype(int)`` -- 1 for the positive (blocker) class."""
    cfg = as_config(cfg)
    label_col = str(cfg.get_path("data.label_col", "label"))
    labels = df[label_col].to_numpy()
    unknown = set(np.unique(labels)) - {0, 1}
    if unknown:
        raise ValueError(f"unexpected label values: {sorted(unknown)}")
    return (labels == 0).astype(np.int8)


def labels_from_blocker(y_blocker: np.ndarray) -> np.ndarray:
    """Map the positive-class vector back to the original label convention."""
    return np.where(np.asarray(y_blocker) == 1, 0, 1).astype(np.int8)


def fingerprint_columns(kind: str, cfg: Config, df: pd.DataFrame | None = None) -> list[str]:
    prefix = cfg.get_path(f"data.fingerprints.{kind}.prefix", f"{kind}_")
    size = int(cfg.get_path(f"data.fingerprints.{kind}.size", 1024))
    cols = [f"{prefix}{i}" for i in range(size)]
    if df is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"fingerprint '{kind}' is missing {len(missing)} columns")
    return cols


def existing_descriptor_names(cfg: Config, df: pd.DataFrame) -> list[str]:
    """Descriptor columns shipped with the CSV (verified reproductions of RDKit values)."""
    names = [n for n in (cfg.get_path("data.existing_descriptors", []) or []) if n in df.columns]
    return names


def existing_descriptor_matrix(df: pd.DataFrame, cfg: Config) -> tuple[list[str], np.ndarray]:
    names = existing_descriptor_names(cfg, df)
    if not names:
        return [], np.zeros((len(df), 0), dtype=np.float32)
    return names, df[names].to_numpy(dtype=np.float32, copy=True)


def fingerprint_matrix_from_csv(
    df: pd.DataFrame, kind: str, cfg: Config
) -> tuple[list[str], np.ndarray]:
    """Fingerprint block as it is shipped in the CSV (bit vectors, int8 -> int8)."""
    cols = fingerprint_columns(kind, cfg, df)
    return cols, df[cols].to_numpy(dtype=np.int8, copy=True)


def dataset_index(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Row-level bookkeeping that preserves the original CSV row order."""
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    label_col = str(cfg.get_path("data.label_col", "label"))
    return pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "SMILES": df[smiles_col].astype(str).to_numpy(),
            "label": df[label_col].to_numpy(),
            "y_blocker": blocker_target(df, cfg),
        }
    )


def dataset_summary(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Structural quality report used by ``reports/data_quality_report.md``."""
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    label_col = str(cfg.get_path("data.label_col", "label"))
    from rdkit import Chem  # local import keeps pandas-only users happy

    smiles = df[smiles_col].astype(str)
    parse_ok = np.array([Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool)

    fp_summary: dict[str, Any] = {}
    for kind in FP_KINDS:
        cols = fingerprint_columns(kind, cfg, df)
        block = df[cols].to_numpy(dtype=np.int16, copy=False)
        nunique = np.array([len(np.unique(block[:, j])) for j in range(block.shape[1])])
        row_sums = block.sum(axis=1)
        fp_summary[kind] = {
            "n_bits": len(cols),
            "constant_bits": int((nunique <= 1).sum()),
            "bits_on_fraction": float(block.mean()),
            "mean_bits_per_molecule": float(row_sums.mean()),
            "min_bits_per_molecule": int(row_sums.min()),
            "max_bits_per_molecule": int(row_sums.max()),
            "duplicate_bit_columns": int(len(cols) - len({tuple(block[:, j]) for j in range(block.shape[1])})),
        }

    descriptors = {}
    for name in existing_descriptor_names(cfg, df):
        series = df[name]
        descriptors[name] = {
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }

    label_counts = df[label_col].value_counts().sort_index().to_dict()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": {
            "smiles": smiles_col,
            "label": label_col,
            "fingerprints": {k: len(fingerprint_columns(k, cfg, df)) for k in FP_KINDS},
            "existing_descriptors": existing_descriptor_names(cfg, df),
        },
        "label_counts": {int(k): int(v) for k, v in label_counts.items()},
        "label_fraction_blocker": float((df[label_col] == 0).mean()),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_smiles": int(smiles.duplicated().sum()),
        "unique_smiles": int(smiles.nunique()),
        "rdkit_parse_success": int(parse_ok.sum()),
        "rdkit_parse_failures": [str(s) for s in smiles[~parse_ok][:10]],
        "fingerprints": fp_summary,
        "existing_descriptors": descriptors,
    }
