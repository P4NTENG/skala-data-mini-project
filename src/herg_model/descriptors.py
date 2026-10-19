"""Descriptor computation: RDKit (2D) descriptor groups + optional Mordred 2D expansion.

Design notes
------------
* Every descriptor is computed **from the SMILES string** so the pipeline is reproducible.
* Descriptors are partitioned into explicit, non-overlapping groups (see ``GROUP_OF``).
  The partition is asserted to cover every computed descriptor; anything unassigned ends up
  in ``misc_rdkit`` and is listed in ``reports/descriptor_report.md`` for review.
* ``rdkit_all`` is the union of every RDKit-derived group, so ``E4 (rdkit only)`` and
  ``E7 (fingerprint + rdkit_all)`` are strict supersets of the domain-group experiments.
* Caches are keyed by the SMILES list, the descriptor configuration and the library
  versions, and are stored outside git (``.cache/herg_model``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .chemistry import mol_from_smiles
from .config import Config, resolve, stable_hash

__all__ = [
    "compute_rdkit_descriptors",
    "compute_mordred_descriptors",
    "assign_groups",
    "group_names",
    "RDKIT_GROUP_ORDER",
    "BASIC_AMINE_SMARTS",
]

# --------------------------------------------------------------------------------------
# group definition
# --------------------------------------------------------------------------------------
RDKIT_GROUP_ORDER = (
    "physicochemical",
    "lipophilicity",
    "polarity_hbond",
    "basic_amine",
    "charge",
    "aromaticity_shape",
    "surface_electronic",
    "misc_rdkit",
)

_DOMAIN_GROUPS = RDKIT_GROUP_ORDER[:-1]

_PHYSICOCHEMICAL = {
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "HeavyAtomCount", "NumValenceElectrons",
    "NumRadicalElectrons", "qed", "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters",
    "n_C", "n_N", "n_O", "n_S", "n_P", "n_F", "n_Cl", "n_Br", "n_I", "n_halogen", "n_hetero",
}
_LIPOPHILICITY = {"MolLogP", "MolMR"}
_POLARITY = {"TPSA", "NumHDonors", "NumHAcceptors", "NHOHCount", "NOCount", "NumHeteroatoms"}
_CHARGE = {
    "FormalCharge", "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
    "MinAbsPartialCharge", "n_positive_atoms", "n_negative_atoms", "n_charged_atoms",
}
_BASIC_AMINE = {
    "NumAmideBonds", "n_total_n", "n_basic_n", "n_nonbasic_n", "n_primary_amine",
    "n_secondary_amine", "n_tertiary_amine", "n_quaternary_n", "n_aromatic_n",
    "n_pyrrole_n", "n_pyridine_n", "n_amide", "n_sulfonamide", "n_nitro", "n_nitrile",
    "n_imine_oxime", "n_azo", "n_n_oxide", "frac_basic_n", "has_basic_amine",
    "fr_NH0", "fr_NH1", "fr_NH2", "fr_Ar_N", "fr_Ar_NH", "fr_Nhpyrrole", "fr_amide",
    "fr_amidine", "fr_guanido", "fr_aniline", "fr_quatN", "fr_priamide", "fr_prisulfonamd",
    "fr_sulfonamd", "fr_urea", "fr_imidazole", "fr_pyridine", "fr_morpholine",
    "fr_piperdine", "fr_piperzine", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_N_O",
    "fr_alkyl_carbamate", "fr_lactam", "fr_imide", "fr_azo", "fr_azide", "fr_diazo",
    "fr_hdrzine", "fr_hdrzone", "fr_isocyan", "fr_isothiocyan", "fr_nitrile", "fr_nitro",
    "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", "fr_thiocyan", "fr_tetrazole",
    "fr_oxazole", "fr_thiazole", "fr_oxime",
}
_AROMATICITY_SHAPE = {
    "NumAromaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    "NumAliphaticRings", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
    "NumSaturatedRings", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "RingCount", "NumHeterocycles", "NumRotatableBonds", "FractionCSP3",
    "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha", "BalabanJ", "BertzCT",
    "NumBridgeheadAtoms", "NumSpiroAtoms", "Phi", "SPS", "Ipc", "AvgIpc", "fr_bicyclic",
    "fr_benzene", "fr_furan", "fr_thiophene", "fr_benzodiazepine", "fr_barbitur",
    "fr_dihydropyridine",
}
_SURFACE_ELECTRONIC = {
    "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
    "LabuteASA", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
}


def _expand(prefix: str, count: int) -> set[str]:
    return {f"{prefix}{i}" for i in range(1, count + 1)}


_LIPOPHILICITY |= _expand("SlogP_VSA", 12) | _expand("SMR_VSA", 10)
_POLARITY |= _expand("PEOE_VSA", 14)
_AROMATICITY_SHAPE |= {"Chi0", "Chi1"} | {f"Chi{i}{s}" for i in (0, 1, 2, 3, 4) for s in ("n", "v")}
_SURFACE_ELECTRONIC |= _expand("EState_VSA", 11) | _expand("VSA_EState", 10)
_SURFACE_ELECTRONIC |= {
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
}

_EXPLICIT: dict[str, str] = {}
for _names, _group in (
    (_PHYSICOCHEMICAL, "physicochemical"),
    (_LIPOPHILICITY, "lipophilicity"),
    (_POLARITY, "polarity_hbond"),
    (_CHARGE, "charge"),
    (_BASIC_AMINE, "basic_amine"),
    (_AROMATICITY_SHAPE, "aromaticity_shape"),
    (_SURFACE_ELECTRONIC, "surface_electronic"),
):
    for _name in _names:
        _EXPLICIT[_name] = _group


def assign_groups(names: Iterable[str]) -> dict[str, list[str]]:
    """Partition descriptor names into the documented groups (order preserved)."""
    groups: dict[str, list[str]] = {group: [] for group in RDKIT_GROUP_ORDER}
    for name in names:
        groups[_EXPLICIT.get(name, "misc_rdkit")].append(name)
    return groups


def group_names(names: Iterable[str]) -> dict[str, list[str]]:
    groups = assign_groups(names)
    groups["rdkit_all"] = [n for n in names]
    return groups


# --------------------------------------------------------------------------------------
# hERG-motivated SMARTS environments
# --------------------------------------------------------------------------------------
#: environments in which a nitrogen is NOT considered a basic (protonatable) amine
NON_BASIC_NITROGEN_SMARTS: dict[str, str] = {
    "amide": "[NX3][CX3]=[OX1]",
    "sulfonamide_or_sulfonamide_like": "[NX3][SX4](=[OX1])(=[OX1])",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "pyrrole_type_aromatic_n": "[nX3,nX2H1]",
    "pyridine_type_aromatic_n": "[nX2]",
    "nitrile": "[NX1]#[CX2]",
    "azo": "[NX2]=[NX2]",
    "imine_oxime": "[CX3]=[NX2,NX3+]",
    "n_oxide": "[N+][O-]",
    "azide": "[NX2-,NX3+]=[NX2+]=[NX1-]",
    "nitroso": "[NX2]=[OX1]",
    "carbamate_or_urea": "[NX3][CX3](=O)[OX2,NX3]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "quaternary_ammonium": "[NX4+]",
}
#: a *basic* nitrogen is a neutral sp3/sp2 nitrogen that is not part of any of the above
BASIC_AMINE_SMARTS: dict[str, str] = dict(NON_BASIC_NITROGEN_SMARTS)
BASIC_AMINE_SMARTS.update(
    {
        "primary_amine": "[NX3;H2;!$(N[CX3]=[OX1,SX1,NX2]);!$(N[!#6;!#1]);!$([N+]);!a]",
        "secondary_amine": "[NX3;H1;!$(N[CX3]=[OX1,SX1,NX2]);!$(N[!#6;!#1]);!$([N+]);!a]",
        "tertiary_amine": "[NX3;H0;!$(N[CX3]=[OX1,SX1,NX2]);!$(N[!#6;!#1]);!$([N+]);!a]",
        "quaternary_ammonium_": "[NX4+]",
        "aniline_type": "[NX3][c]",
    }
)

_COMPILED = {name: Chem.MolFromSmarts(smarts) for name, smarts in BASIC_AMINE_SMARTS.items()}
_NON_BASIC_KEYS = tuple(NON_BASIC_NITROGEN_SMARTS.keys())


def _matched_atoms(mol: Chem.Mol, pattern: Chem.Mol) -> set[int]:
    """Union of the *first* query atom of every substructure match (overlap-safe counting)."""
    if pattern is None:
        return set()
    return {match[0] for match in mol.GetSubstructMatches(pattern)}


def _extra_features(mol: Chem.Mol) -> dict[str, float]:
    """Derived features: atom composition, basic-amine environments, charge descriptors."""
    if mol is None:
        return {}
    out: dict[str, float] = {}

    counts = {"C": 0, "N": 0, "O": 0, "S": 0, "P": 0, "F": 0, "Cl": 0, "Br": 0, "I": 0}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in counts:
            counts[symbol] += 1
    for symbol, value in counts.items():
        out[f"n_{symbol}"] = float(value)
    out["n_halogen"] = float(counts["F"] + counts["Cl"] + counts["Br"] + counts["I"])
    out["n_hetero"] = float(rdMolDescriptors.CalcNumHeteroatoms(mol))

    # --- nitrogen environments (overlap-safe) -------------------------------------------
    n_atoms = mol.GetNumAtoms()
    matched: dict[str, set[int]] = {}
    for name in _NON_BASIC_KEYS:
        matched[name] = _matched_atoms(mol, _COMPILED[name])
    if n_atoms:
        matched["primary_amine"] = _matched_atoms(mol, _COMPILED["primary_amine"])
        matched["secondary_amine"] = _matched_atoms(mol, _COMPILED["secondary_amine"])
        matched["tertiary_amine"] = _matched_atoms(mol, _COMPILED["tertiary_amine"])
        matched["aniline_type"] = _matched_atoms(mol, _COMPILED["aniline_type"])

    nitrogen_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    non_basic = set().union(*(matched[name] for name in _NON_BASIC_KEYS)) if _NON_BASIC_KEYS else set()
    non_basic &= set(nitrogen_atoms)
    basic_atoms = set(nitrogen_atoms) - non_basic

    out["n_total_n"] = float(len(nitrogen_atoms))
    out["n_nonbasic_n"] = float(len(non_basic))
    out["n_basic_n"] = float(len(basic_atoms))
    out["frac_basic_n"] = float(len(basic_atoms) / len(nitrogen_atoms)) if nitrogen_atoms else 0.0
    out["has_basic_amine"] = float(len(basic_atoms) > 0)
    out["n_amide"] = float(len(matched["amide"] & set(nitrogen_atoms)))
    out["n_sulfonamide"] = float(len(matched["sulfonamide_or_sulfonamide_like"] & set(nitrogen_atoms)))
    out["n_nitro"] = float(len(matched["nitro"] & set(nitrogen_atoms)))
    out["n_nitrile"] = float(len(matched["nitrile"] & set(nitrogen_atoms)))
    out["n_imine_oxime"] = float(len(matched["imine_oxime"] & set(nitrogen_atoms)))
    out["n_azo"] = float(len(matched["azo"] & set(nitrogen_atoms)))
    out["n_n_oxide"] = float(len(matched["n_oxide"] & set(nitrogen_atoms)))
    out["n_pyrrole_n"] = float(len(matched["pyrrole_type_aromatic_n"] & set(nitrogen_atoms)))
    out["n_pyridine_n"] = float(len(matched["pyridine_type_aromatic_n"] & set(nitrogen_atoms)))
    out["n_quaternary_n"] = float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() > 0))

    # basic sp3 amine counting (primary/secondary/tertiary restricted to basic nitrogen)
    for label in ("primary_amine", "secondary_amine", "tertiary_amine"):
        out[f"n_{label}"] = float(len(matched.get(label, set()) & basic_atoms))

    out["n_aromatic_n"] = float(
        sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic())
    )

    # --- charge --------------------------------------------------------------------------
    out["FormalCharge"] = float(Chem.GetFormalCharge(mol))
    positive = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0)
    negative = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0)
    out["n_positive_atoms"] = float(positive)
    out["n_negative_atoms"] = float(negative)
    out["n_charged_atoms"] = float(positive + negative)
    return out


def compute_rdkit_descriptors(
    smiles: Sequence[str],
    cfg: Config | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full 2D RDKit descriptor table (+ derived hERG-motivated features) for ``smiles``.

    ``Inf`` values are replaced by ``NaN`` (``missingVal=np.nan`` is passed to RDKit) and the
    raw failures are recorded by the caller.
    """
    from .config import environment_info

    versions = environment_info()["packages"]
    key_payload = {
        "kind": "rdkit_descriptors",
        "n": len(smiles),
        "smiles_hash": stable_hash(list(smiles)),
        "rdkit": versions.get("rdkit"),
        "config": cfg.get_path("descriptors.rdkit") if cfg is not None else None,
    }
    cache_path = _cache_path(cfg, "rdkit_descriptors", key_payload)
    if use_cache and cache_path is not None and cache_path.exists():
        try:
            return pd.read_pickle(cache_path)
        except Exception:  # pragma: no cover
            pass

    records: list[dict[str, float]] = []
    failures: list[str] = []
    for smi in smiles:
        mol = mol_from_smiles(smi)
        if mol is None:
            failures.append(smi)
            records.append({})
            continue
        values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
        values.update(_extra_features(mol))
        records.append(values)

    frame = pd.DataFrame.from_records(records)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "SMILES", list(smiles))
    frame.attrs["parse_failures"] = failures

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)
    if verbose:
        print(f"[descriptors] RDKit: {frame.shape[0]} rows x {frame.shape[1] - 1} descriptors "
              f"({len(failures)} parse failures)")
    return frame


def compute_mordred_descriptors(
    smiles: Sequence[str],
    cfg: Config,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame | None, dict]:
    """Mordred 2D descriptor table, or ``(None, status)`` when Mordred is unavailable."""
    status: dict = {"available": False, "reason": None, "n_descriptors": 0, "n_failed": 0}
    try:
        from mordred import Calculator, descriptors as mordred_descriptors
    except Exception as exc:  # pragma: no cover - optional dependency
        status["reason"] = f"mordred import failed: {exc.__class__.__name__}: {exc}"
        return None, status

    from .config import environment_info

    versions = environment_info()["packages"]
    mordred_cfg = dict(cfg.get_path("descriptors.mordred", {}) or {})
    key_payload = {
        "kind": "mordred",
        "n": len(smiles),
        "smiles_hash": stable_hash(list(smiles)),
        "mordred": versions.get("mordredcommunity") or versions.get("mordred"),
        "ignore_3d": mordred_cfg.get("ignore_3d", True),
    }
    cache_path = _cache_path(cfg, "mordred", key_payload)
    if use_cache and cache_path is not None and cache_path.exists():
        try:
            frame = pd.read_pickle(cache_path)
            status.update({"available": True, "n_descriptors": frame.shape[1] - 1, "cached": True})
            return frame, status
        except Exception:  # pragma: no cover
            pass

    ignore_3d = bool(mordred_cfg.get("ignore_3d", True))
    try:
        calc = Calculator(mordred_descriptors, ignore_3D=ignore_3d)
    except Exception as exc:
        status["reason"] = f"Calculator construction failed: {exc}"
        return None, status

    mols = [mol_from_smiles(s) for s in smiles]
    status["n_failed"] = int(sum(1 for m in mols if m is None))
    nproc = int(mordred_cfg.get("nproc", 0) or 0)
    frame = None
    for attempt, kwargs in enumerate(({"nproc": nproc or None}, {})):
        try:
            result = calc.pandas(mols, quiet=True, **{k: v for k, v in kwargs.items() if v})
            frame = result.map(_to_float).astype("float32")
            break
        except Exception as exc:  # pragma: no cover - platform dependent
            status.setdefault("warnings", []).append(f"pandas(nproc={kwargs}) failed: {exc}")
    if frame is None:
        status["reason"] = "Mordred calculation failed in every mode; see 'warnings'"
        return None, status

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "SMILES", list(smiles))
    status.update({"available": True, "n_descriptors": frame.shape[1] - 1, "cached": False})
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)
    if verbose:
        print(f"[descriptors] Mordred: {frame.shape[0]} rows x {frame.shape[1] - 1} descriptors")
    return frame, status


def _to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _cache_path(cfg: Config | None, kind: str, payload: dict) -> Path | None:
    if cfg is None:
        return None
    cache_dir = resolve(cfg.get_path("paths.cache", ".cache/herg_model"))
    return Path(cache_dir) / f"{kind}_{stable_hash(payload)}.pkl"
