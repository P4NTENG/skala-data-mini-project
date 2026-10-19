"""Fingerprint generation and verification against the shipped CSV columns.

Verified on ``data/train.csv`` (2026-10-20):

===========  ====================================================================
CSV block    reproducible definition
===========  ====================================================================
``ecfp_*``   ``MorganGenerator(radius=2, fpSize=1024)``
``fcfp_*``   ``MorganGenerator(radius=2, fpSize=1024, atomInvariantsGenerator=GetMorganFeatureAtomInvGen())``
``ptfp_*``   ``Chem.PatternFingerprint(mol, fpSize=1024)`` ("PTFP" = pattern topological fingerprint)
===========  ====================================================================

Bit ``i`` of the CSV block equals bit ``i`` of ``fp.ToBitString()``. The shipped blocks are
therefore *not* regenerated for the main path -- they are used as-is, and the agreement rate
is reported (see ``reports/data_quality_report.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rfg

from .chemistry import mol_from_smiles
from .config import Config

__all__ = [
    "build_generator",
    "fingerprint_bits",
    "recompute_matrix",
    "verify_fingerprints",
    "GENERATOR_FACTORIES",
]

GENERATOR_FACTORIES = {
    "morgan": lambda params: rfg.GetMorganGenerator(**params),
    "morgan_features": lambda params: rfg.GetMorganGenerator(
        **params, atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen()
    ),
}


def build_generator(kind: str, cfg: Config):
    """Return a callable ``mol -> bit vector`` for the requested fingerprint block."""
    generator_name = str(cfg.get_path(f"data.fingerprints.{kind}.generator"))
    params = dict(cfg.get_path(f"data.fingerprints.{kind}.params", {}) or {})

    if generator_name == "pattern":
        fp_size = int(params.get("fpSize", 1024))

        def _pattern(mol):
            return Chem.PatternFingerprint(mol, fpSize=fp_size)

        return _pattern

    factory = GENERATOR_FACTORIES.get(generator_name)
    if factory is None:
        raise ValueError(f"unknown fingerprint generator {generator_name!r}")
    generator = factory(params)
    return generator.GetFingerprint


def fingerprint_bits(mol: Chem.Mol, generator, size: int) -> np.ndarray:
    fp = generator(mol)
    bits = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
    return bits[:size].astype(np.int8)


def recompute_matrix(smiles: list[str], kind: str, cfg: Config) -> np.ndarray:
    """Re-derive a fingerprint block from SMILES (identical column order to the CSV)."""
    size = int(cfg.get_path(f"data.fingerprints.{kind}.size", 1024))
    generator = build_generator(kind, cfg)
    out = np.zeros((len(smiles), size), dtype=np.int8)
    for i, smi in enumerate(smiles):
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        out[i] = fingerprint_bits(mol, generator, size)
    return out


def verify_fingerprints(
    df: pd.DataFrame,
    cfg: Config,
    sample: int | None = None,
    seed: int | None = None,
) -> dict:
    """Compare shipped fingerprint blocks with freshly generated ones.

    ``sample <= 0`` (or ``None`` with ``data.fingerprint_verify_sample == 0``) checks every row.
    """
    from .data import fingerprint_columns

    if sample is None:
        sample = int(cfg.get_path("data.fingerprint_verify_sample", 200))
    if seed is None:
        seed = int(cfg.get_path("project.seed", 42))

    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    n_rows = len(df)
    if sample and 0 < sample < n_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_rows, size=sample, replace=False))
    else:
        idx = np.arange(n_rows)

    smiles = df[smiles_col].astype(str).to_numpy()[idx]
    report: dict = {"n_checked": int(len(idx)), "checked_all_rows": bool(len(idx) == n_rows), "blocks": {}}
    for kind in cfg.get_path("data.fingerprints", {}).keys():
        cols = fingerprint_columns(kind, cfg, df)
        shipped = df[cols].to_numpy(dtype=np.int8, copy=False)[idx]
        rebuilt = recompute_matrix(list(smiles), kind, cfg)
        equal = shipped == rebuilt
        per_row = equal.all(axis=1)
        report["blocks"][kind] = {
            "generator": cfg.get_path(f"data.fingerprints.{kind}.generator"),
            "params": dict(cfg.get_path(f"data.fingerprints.{kind}.params", {}) or {}),
            "n_bits": int(shipped.shape[1]),
            "bit_agreement": float(equal.mean()),
            "row_exact_rate": float(per_row.mean()),
            "n_rows_exact": int(per_row.sum()),
            "n_rows_checked": int(len(idx)),
            "exact": bool(per_row.all()),
            "mismatch_row_ids": [int(i) for i in idx[~per_row][:10]],
            "mismatch_bit_columns": int((~equal).any(axis=0).sum()),
        }
    return report
