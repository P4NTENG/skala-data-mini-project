"""Molecule-level utilities: parsing, structure keys, Bemis-Murcko scaffolds.

Everything downstream (descriptors, fingerprints, splits, error analysis) is keyed on
the *original* SMILES string shipped in ``data/train.csv`` so that results can always be
traced back to their source row.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")
# parse / valence problems are captured explicitly instead of being logged row by row
RDLogger.DisableLog("rdApp.error")

if os.path.join(RDConfig.RDContribDir, "SA_Score") not in sys.path:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402  (RDKit contrib, shipped with the wheel)

__all__ = [
    "mol_from_smiles",
    "canonical_smiles",
    "inchikey",
    "connectivity_key",
    "bem_murcko_scaffold",
    "generic_scaffold",
    "sa_score",
    "strip_salts",
    "neutralize",
    "canonical_tautomer",
    "molecule_table",
    "structure_duplicate_report",
]

_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=200_000)
def mol_from_smiles(smiles: str, sanitize: bool = True) -> Chem.Mol | None:
    """Parse a SMILES string (memoised -- parsing dominates the runtime)."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""


def inchikey(mol: Chem.Mol) -> str:
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - exotic structures
        return ""


def connectivity_key(key: str) -> str:
    """Stereochemistry-blind skeleton key (first InChIKey block)."""
    return key.split("-")[0] if key else ""


def bem_murcko_scaffold(mol: Chem.Mol, generic_for_acyclic: bool = True) -> str:
    """Bemis-Murcko scaffold SMILES.

    Acyclic molecules have an empty Murcko scaffold; grouping them all together would
    create one enormous, meaningless split group, so they fall back to the generic
    (element-stripped) scaffold which still prevents structural leakage.
    """
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:  # pragma: no cover
        scaffold = ""
    if scaffold:
        return scaffold
    if generic_for_acyclic:
        try:
            return "generic:" + generic_scaffold(mol)
        except Exception:  # pragma: no cover
            pass
    return "single:" + canonical_smiles(mol)


def generic_scaffold(mol: Chem.Mol) -> str:
    """Element-stripped scaffold; falls back to the plain Murcko scaffold on failure."""
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=MurckoScaffold.MakeScaffoldGeneric(mol))
    except Exception:
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:  # pragma: no cover
            return ""


@lru_cache(maxsize=200_000)
def sa_score(smiles: str) -> float:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:  # pragma: no cover
        return float("nan")


def strip_salts(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest organic fragment (sensitivity analysis only)."""
    if mol is None:
        return mol
    return rdMolStandardize.LargestFragmentChooser().choose(mol)


def neutralize(mol: Chem.Mol) -> Chem.Mol:
    """Charge normalisation (sensitivity analysis only)."""
    if mol is None:
        return mol
    return _UNCHARGER.uncharge(mol)


def canonical_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Tautomer canonicalisation (sensitivity analysis only)."""
    if mol is None:
        return mol
    try:
        return _TAUTOMER_ENUM.Canonicalize(mol)
    except Exception:  # pragma: no cover
        return mol


def molecule_table(smiles: Sequence[str], generic_for_acyclic: bool = True,
                   include_tautomer_check: bool = True) -> pd.DataFrame:
    """Per-row structural bookkeeping used for quality checks and grouping."""
    rows = []
    for row_id, smiles in enumerate(smiles):
        mol = mol_from_smiles(smiles)
        has_stereo = False
        if mol is not None:
            has_stereo = Chem.MolToSmiles(mol) != Chem.MolToSmiles(mol, isomericSmiles=False)
        record = {
            "row_id": row_id,
            "SMILES": smiles,
            "parsed": mol is not None,
            "canonical_smiles": canonical_smiles(mol) if mol is not None else "",
            "inchikey": inchikey(mol) if mol is not None else "",
            "n_heavy_atoms": mol.GetNumHeavyAtoms() if mol is not None else np.nan,
            "n_fragments": len(Chem.GetMolFrags(mol)) if mol is not None else np.nan,
            "has_stereo": bool(has_stereo),
            "n_stereo_centers": len(
                Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
            ) if mol is not None else np.nan,
            "scaffold": bem_murcko_scaffold(mol, generic_for_acyclic),
            "generic_scaffold": generic_scaffold(mol) if mol is not None else "",
            "sa_score": sa_score(smiles),
            "formal_charge": Chem.GetFormalCharge(mol) if mol is not None else np.nan,
            "n_rings": mol.GetRingInfo().NumRings() if mol is not None else np.nan,
        }
        if include_tautomer_check and mol is not None:
            record["tautomer_smiles"] = canonical_smiles(canonical_tautomer(mol))
            record["tautomer_changed"] = record["tautomer_smiles"] != record["canonical_smiles"]
        rows.append(record)
    table = pd.DataFrame(rows)
    table["connectivity_key"] = table["inchikey"].str.split("-").str[0]
    return table


def structure_duplicate_report(table: pd.DataFrame, labels: np.ndarray) -> dict[str, pd.DataFrame | int]:
    """Duplicate-structure diagnostics in three granularities.

    ``canonical`` (full stereo) / ``connectivity`` (stereo-blind) / ``normalized``
    (neutralised + largest fragment) -- conflicting labels are reported, never dropped.
    """
    work = table.copy()
    work["label"] = np.asarray(labels)

    normalized = []
    for smiles in work["SMILES"]:
        mol = mol_from_smiles(smiles)
        if mol is None:
            normalized.append("")
            continue
        normalized.append(canonical_smiles(neutralize(strip_salts(mol))))
    work["normalized_key"] = normalized

    reports: dict[str, pd.DataFrame | int] = {}
    for column in ("canonical_smiles", "connectivity_key", "normalized_key", "scaffold"):
        grouped = work.groupby(column, dropna=False)
        sizes = grouped.size().rename("n_members")
        label_nunique = grouped["label"].nunique().rename("n_labels")
        conflicting = pd.concat([sizes, label_nunique], axis=1)
        conflicting = conflicting[(conflicting["n_members"] > 1)]
        work[f"is_{column}_duplicate"] = work[column].duplicated(keep=False)
        reports[column] = conflicting.reset_index().rename(columns={column: "key"})
        reports[f"n_duplicate_members_{column}"] = int(work[f"is_{column}_duplicate"].sum())
    reports["_table"] = work
    return reports


def group_sizes(values: Iterable[str]) -> pd.Series:
    return pd.Series(list(values)).value_counts()


def load_structure_table(
    smiles: Sequence[str],
    cfg=None,
    use_cache: bool = True,
    generic_for_acyclic: bool = True,
    include_tautomer_check: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cached :func:`molecule_table` -- scaffold / InChIKey bookkeeping is expensive."""
    import pickle

    from .config import environment_info, resolve, stable_hash

    cache_path = None
    if cfg is not None and use_cache:
        versions = environment_info()["packages"]
        payload = {
            "kind": "structure_table",
            "n": len(smiles),
            "smiles_hash": stable_hash(list(smiles)),
            "rdkit": versions.get("rdkit"),
            "generic": generic_for_acyclic,
            "tautomer": include_tautomer_check,
        }
        cache_dir = resolve(cfg.get_path("paths.cache", ".cache/herg_model"))
        cache_path = cache_dir / f"structure_table_{stable_hash(payload)}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as handle:
                    return pickle.load(handle)
            except Exception:  # pragma: no cover
                pass

    table = molecule_table(
        list(smiles),
        generic_for_acyclic=generic_for_acyclic,
        include_tautomer_check=include_tautomer_check,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"[chemistry] structure table: {len(table)} rows, "
              f"{table['scaffold'].nunique()} unique scaffolds")
    return table
