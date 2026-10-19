#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hERG 독성 라벨 검증 파이프라인
================================

`data/train.csv` 의 고정 라벨 정의

    label = 0 : 독성 (hERG 차단 활성이 있을 것으로 예상)
    label = 1 : 비독성 (hERG 차단 활성이 낮을 것으로 예상)

를 공개 실험 데이터와 화합물별로 대조해 검증한다.

사용하는 외부 실험 데이터
    - ChEMBL target CHEMBL240 (hERG / KCNH2, Homo sapiens) activity (assay_type B/F/T 분리)
    - PubChem BioAssay AID588834 (Tox21/NCGC qHTS hERG thallium flux, U2OS)
    - 논문 보충자료 시도 (Krishna et al. 2022, Biology 11(2):209, PMC8869358)
    - 다중 종말점 스캔: 매칭된 화합물의 ChEMBL 전체 target activity

산출물
    output/matched_compounds.csv   구조 매칭 및 실험값 대조 (long format)
    output/compound_level.csv      화합물 단위 집계
    output/label_conflicts.csv     불일치 사례 + 원인 분석
    output/excluded_cases.csv      분석에서 제외한 사례와 사유
    output/match_summary.json      매칭 통계
    output/metrics.csv             라벨 방향 검증 지표
    output/endpoint_scan.csv       종말점 탐색 결과
    figures/*.png

사용법
    .venv/Scripts/python.exe herg_validation/validation.py --stage all
    (stage: catalog | fetch | match | stats | endpoints | figures)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import requests

# ---------------------------------------------------------------- 경로 / 상수

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_CSV = ROOT / "data" / "train.csv"
CACHE = HERE / "cache"
OUT = HERE / "output"
FIG = HERE / "figures"
for _p in (CACHE, OUT, FIG):
    _p.mkdir(parents=True, exist_ok=True)

CHM_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_HERG_TARGET = "CHEMBL240"
PUBCHEM_AIDS = [588834]  # Tox21/NCGC qHTS hERG thallium flux
UA = {"User-Agent": "skala-herg-validation/1.0 (research; contact: repo owner)"}

# 라벨 정의 (지시문에서 고정; 임의 변경 금지)
LABEL_TOXIC = 0      # 독성  -> hERG blocker 기대
LABEL_NONTOXIC = 1   # 비독성 -> hERG 차단 활성 낮음 기대

PRIMARY_THRESHOLD_UM = 10.0
THRESHOLDS_UM = [1.0, 3.0, 10.0, 30.0, 100.0]
POTENCY_TYPES = {
    "ic50", "ac50", "ec50", "ki", "kd", "ic90", "ic80", "ic20", "ic10",
    "potency", "inhibition", "ip", "ac90", "ac80", "ec90", "ec80", "cc50",
}
INHIBITION_TYPES = {"inhibition", "ip", "% inhibition", "percent inhibition"}
UNIT_TO_UM = {
    "nm": 1e-3, "nanomolar": 1e-3,
    "um": 1.0, "µm": 1.0, "micromolar": 1.0, "μm": 1.0, "molar": 1e6 * 1e-6 * 1e0,
    "mm": 1e3, "millimolar": 1e3,
    "pm": 1e-6, "picomolar": 1e-6,
    "m": 1e6, "molarity": 1e6,
    "ug/ml": np.nan, "ug.ml-1": np.nan,
}

LOG_PATH = OUT / "run_log.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- HTTP 유틸

def _request(url: str, params: dict | None = None, *, retries: int = 4,
             timeout: int = 120, stream: bool = False) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout, stream=stream)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 2 ** attempt
            log(f"  ! HTTP 재시도 {attempt + 1}/{retries} {url} :: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"HTTP 실패: {url} :: {last}")


def http_json(url: str, params: dict | None = None) -> dict:
    return _request(url, params).json()


def http_text(url: str, params: dict | None = None, timeout: int = 120) -> str:
    return _request(url, params, timeout=timeout).text


# ---------------------------------------------------------------- 원자료 캐시

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- RDKit 구조 키

_RDKIT_READY = False


def _init_rdkit():
    global _RDKIT_READY
    if _RDKIT_READY:
        return
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_READY = True


TAUTOMER_ENUM = None
UNCHARGER = None
LARGEST_FRAG = None


def _std_tools():
    global TAUTOMER_ENUM, UNCHARGER, LARGEST_FRAG
    if TAUTOMER_ENUM is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
        UNCHARGER = rdMolStandardize.Uncharger()
        LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()
    return TAUTOMER_ENUM, UNCHARGER, LARGEST_FRAG


def mol_keys(smiles: str) -> dict | None:
    """SMILES -> 다양한 정규화 수준의 구조 키.

    반환 키
        canonical_isomeric : RDKit canonical SMILES (입체화학 유지)
        canonical_flat     : 입체화학 제거 canonical SMILES
        inchikey           : 표준 InChIKey (입체화학 포함)
        connectivity_key   : InChIKey 첫 블록 (연결성만)
        parent_smiles      : 최대 fragment (탈염/용매화물 제거)
        parent_inchikey    : 탈염 후 InChIKey
        neutral_smiles     : 전하 중성화 후 SMILES
        tautomer_smiles    : tautomer canonical + 탈염 + 중성화
        norm_inchikey      : 위 정규화 구조의 InChIKey (최종 정규화 키)
        has_salt          : '.' 포함 여부
        has_stereo        : 입체화학 표기 여부
    """
    _init_rdkit()
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    taut_enum, uncharger, largest = _std_tools()

    out: dict = {"parse_ok": True}
    out["canonical_isomeric"] = Chem.MolToSmiles(mol, isomericSmiles=True)
    out["canonical_flat"] = Chem.MolToSmiles(mol, isomericSmiles=False)
    out["has_salt"] = "." in smiles
    out["has_stereo"] = ("@" in smiles) or ("/" in smiles) or ("\\" in smiles)

    try:
        out["inchikey"] = Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        out["inchikey"] = ""
    if not out["inchikey"]:
        out["inchikey"] = ""
    out["connectivity_key"] = out["inchikey"].split("-")[0] if out["inchikey"] else ""

    try:
        frag = largest.choose(mol)
        out["parent_smiles"] = Chem.MolToSmiles(frag, isomericSmiles=True)
        out["parent_inchikey"] = Chem.MolToInchiKey(frag) or ""
        neutral = uncharger.uncharge(frag)
        out["neutral_smiles"] = Chem.MolToSmiles(neutral, isomericSmiles=True)
        taut = taut_enum.Canonicalize(neutral)
        out["tautomer_smiles"] = Chem.MolToSmiles(taut, isomericSmiles=True)
        out["norm_inchikey"] = Chem.MolToInchiKey(taut) or ""
    except Exception:  # noqa: BLE001
        out.setdefault("parent_smiles", out["canonical_isomeric"])
        out.setdefault("parent_inchikey", out["inchikey"])
        out.setdefault("neutral_smiles", out["canonical_isomeric"])
        out.setdefault("tautomer_smiles", out["canonical_isomeric"])
        out.setdefault("norm_inchikey", out["inchikey"])
    out["norm_connectivity_key"] = out["norm_inchikey"].split("-")[0] if out["norm_inchikey"] else ""
    return out


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    _init_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def murcko_scaffold(smiles: str) -> str:
    _init_rdkit()
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- 0. catalog

def load_dataset(columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = ["SMILES", "MolWt", "clogp", "sa_score", "qed", "label"]
    df = pd.read_csv(DATA_CSV, usecols=columns, low_memory=False)
    df = df.reset_index(drop=False).rename(columns={"index": "dataset_row_index"})
    return df


def stage_catalog() -> None:
    log("== stage catalog ==")
    df = load_dataset()
    cat = {"n_rows": int(len(df)), "n_columns_total": 3078,
           "n_unique_smiles": int(df["SMILES"].nunique()),
           "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
           "n_missing": {c: int(df[c].isna().sum()) for c in df.columns}}

    keys = []
    for s in df["SMILES"]:
        k = mol_keys(s)
        keys.append(k)
    n_parse_fail = sum(1 for k in keys if k is None)
    cat["n_parse_fail"] = int(n_parse_fail)
    cat["n_with_salt_dot"] = int(sum(1 for k in keys if k and k["has_salt"]))
    cat["n_with_stereo"] = int(sum(1 for k in keys if k and k["has_stereo"]))

    kdf = pd.DataFrame([k or {} for k in keys])
    kdf.insert(0, "dataset_row_index", df["dataset_row_index"].values)
    kdf.insert(1, "original_smiles", df["SMILES"].values)
    kdf.insert(2, "dataset_label", df["label"].values)
    kdf.to_csv(OUT / "dataset_keys.csv", index=False)

    # 정규화 수준별 키 변화량 (매칭 결과에 미치는 영향 추적용)
    cat["n_canonical_changed_vs_original"] = int((kdf["canonical_isomeric"] != kdf["original_smiles"]).sum())
    cat["n_salt_stripped"] = int((kdf["parent_smiles"] != kdf["canonical_isomeric"]).sum())
    cat["n_tautomer_changed"] = int((kdf["tautomer_smiles"] != kdf["neutral_smiles"]).sum())
    cat["n_unique_inchikey"] = int(kdf["inchikey"].replace("", np.nan).nunique())
    cat["n_unique_connectivity_key"] = int(kdf["connectivity_key"].replace("", np.nan).nunique())
    cat["n_unique_norm_inchikey"] = int(kdf["norm_inchikey"].replace("", np.nan).nunique())

    with open(OUT / "catalog_summary.json", "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=2)
    log(json.dumps(cat, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- 1. fetch

def fetch_chembl_activities() -> Path:
    """hERG target CHEMBL240 activity 전량 (long format jsonl)."""
    path = CACHE / f"chembl_activity_{CHEMBL_HERG_TARGET}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        log(f"  캐시 사용: {path.name} ({len(read_jsonl(path))} rows)")
        return path
    only = ",".join([
        "activity_id", "molecule_chembl_id", "molecule_pref_name", "parent_molecule_chembl_id",
        "canonical_smiles", "standard_type", "standard_relation", "standard_value",
        "standard_units", "standard_upper_value", "pchembl_value", "assay_chembl_id",
        "assay_type", "assay_description", "document_chembl_id", "document_year",
        "target_chembl_id", "target_pref_name", "target_organism", "src_id",
        "data_validity_comment", "potential_duplicate", "bao_endpoint", "record_id",
    ])
    offset, total = 0, None
    while True:
        j = http_json(f"{CHM_BASE}/activity.json", {
            "target_chembl_id": CHEMBL_HERG_TARGET, "limit": 1000, "offset": offset,
            "format": "json", "only": only})
        rows = j.get("activities", [])
        total = j["page_meta"]["total_count"] if total is None else total
        append_jsonl(path, rows)
        offset += 1000
        log(f"  ChEMBL hERG activity {offset}/{total}")
        if offset >= total or not rows:
            break
    return path


def fetch_chembl_assays() -> Path:
    """hERG 관련 assay 메타데이터 (assay_type, cell, description 등)."""
    path = CACHE / f"chembl_assays_{CHEMBL_HERG_TARGET}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        log(f"  캐시 사용: {path.name} ({len(read_jsonl(path))} rows)")
        return path
    offset, total = 0, None
    while True:
        j = http_json(f"{CHM_BASE}/assay.json", {
            "target_chembl_id": CHEMBL_HERG_TARGET, "limit": 1000, "offset": offset,
            "format": "json"})
        rows = j.get("assays", [])
        total = j["page_meta"]["total_count"] if total is None else total
        append_jsonl(path, rows)
        offset += 1000
        log(f"  ChEMBL hERG assay {offset}/{total}")
        if offset >= total or not rows:
            break
    return path


def fetch_pubchem_assay(aid: int) -> Path:
    """PubChem BioAssay CSV (SMILES, activity outcome, potency, curve fit 포함)."""
    path = CACHE / f"pubchem_aid{aid}.csv"
    if path.exists() and path.stat().st_size > 0:
        log(f"  캐시 사용: {path.name}")
        return path
    txt = http_text(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV",
                    {"response_type": "display"}).strip()
    path.write_text(txt, encoding="utf-8")
    log(f"  PubChem AID{aid} 저장 ({len(txt)} bytes)")
    return path


def fetch_pubchem_cid_props(cids: list[str]) -> Path:
    """PubChem CID -> InChIKey / SMILES."""
    path = CACHE / "pubchem_cid_props.jsonl"
    done: set[int] = set()
    for r in read_jsonl(path):
        try:
            done.add(int(r["CID"]))
        except (KeyError, TypeError, ValueError):
            continue
    todo = [c for c in cids if int(c) not in done]
    if not todo:
        log(f"  캐시 사용: {path.name} ({len(done)} cids)")
        return path
    log(f"  PubChem CID property 조회: {len(todo)}건")
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        j = http_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                      + ",".join(chunk)
                      + "/property/InChIKey,SMILES,ConnectivitySMILES,MolecularFormula/JSON")
        props = j.get("PropertyTable", {}).get("Properties", [])
        append_jsonl(path, props)
        if (i // 100) % 5 == 0:
            log(f"    cid props {i + len(chunk)}/{len(todo)}")
        time.sleep(0.21)
    return path


def try_supplements() -> None:
    """논문 보충자료 다운로드 시도 (실패 시 사유 기록)."""
    attempts = [
        ("MDPI s1 (Biology 11(2):209)", "https://www.mdpi.com/2079-7737/11/2/209/s1"),
        ("MDPI article s1", "https://www.mdpi.com/article/10.3390/biology11020209/s1"),
        ("EuropePMC supplementaryFiles", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8869358/supplementaryFiles"),
        ("PMC bin s001", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8869358/bin/biology-11-00209-s001.zip"),
        ("PMC instance s001", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8869358/bin/biology-11-00209-s001.zip"),
        ("PMC OA service", "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC8869358"),
    ]
    results = []
    sup_dir = CACHE / "supplements"
    sup_dir.mkdir(exist_ok=True)
    for name, url in attempts:
        rec = {"name": name, "url": url, "status": None, "bytes": 0, "saved": ""}
        try:
            r = _request(url, retries=2, timeout=90, stream=True)
            content = r.content
            rec["status"] = r.status_code
            rec["bytes"] = len(content)
            if content[:2] == b"PK" and len(content) > 10000:
                fn = sup_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".zip")
                fn.write_bytes(content)
                rec["saved"] = str(fn.relative_to(ROOT))
                log(f"  보충자료 확보: {name} -> {fn.name} ({len(content)} bytes)")
            elif len(content) > 2000 and b"<html" not in content[:2000].lower():
                fn = sup_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".dat")
                fn.write_bytes(content)
                rec["saved"] = str(fn.relative_to(ROOT))
                log(f"  보충자료(비zip) 확보: {name} ({len(content)} bytes)")
        except Exception as exc:  # noqa: BLE001
            rec["status"] = f"ERROR: {type(exc).__name__}"
            log(f"  보충자료 실패: {name} :: {exc}")
        results.append(rec)
        time.sleep(0.5)
    (CACHE / "supplement_attempts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_fetch() -> None:
    log("== stage fetch ==")
    fetch_chembl_activities()
    fetch_chembl_assays()
    for aid in PUBCHEM_AIDS:
        csv_path = fetch_pubchem_assay(aid)
        # CSV의 CID 목록 확보
        rows = list(pd.read_csv(csv_path, skiprows=[1, 2], low_memory=False,
                                encoding="utf-8", encoding_errors="replace").itertuples(index=False))
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        if "PUBCHEM_CID" in cols:
            cids = sorted({str(int(getattr(r, "PUBCHEM_CID"))) for r in rows
                           if pd.notna(getattr(r, "PUBCHEM_CID"))})
            log(f"  AID{aid}: CID {len(cids)}개")
            fetch_pubchem_cid_props(cids)
    try_supplements()
    log("== stage fetch 완료 ==")


# ---------------------------------------------------------------- 2. match

def _classify_record(standard_type: str, relation: str, value, units: str,
                     pchembl, threshold_um: float, strict: bool = False) -> dict:
    """실험 레코드 1건 -> 파생 클래스.

    strict=True 이면 'blocker' 판정을 `value < threshold` 로 계산한다.
    (대상 데이터셋이 정확히 10 uM 인 값을 비독성으로 인코딩한 정황이 있어
     경계 규칙 민감도 분석에 사용한다.)

    반환
        measure_family : potency | percent_inhibition | other
        value_um       : µM 환산값 (가능한 경우)
        censored       : '>' | '<' | None  (검열 측정값 표시)
        derived_class  : blocker | non_blocker | ambiguous_censored | ambiguous | unknown
    """
    st = (standard_type or "").strip().lower()
    rel = (relation or "=").strip()
    rel = {"≥": ">", "≤": "<", "~": "="}.get(rel, rel)
    units_l = (units or "").strip().lower()
    out = {"measure_family": "other", "value_um": np.nan, "censored": None,
           "derived_class": "unknown"}

    try:
        v = float(value) if value is not None and value != "" else np.nan
    except (TypeError, ValueError):
        v = np.nan

    # percent inhibition 계열
    if st in INHIBITION_TYPES and units_l in ("%", "percent", ""):
        out["measure_family"] = "percent_inhibition"
        if np.isfinite(v):
            out["derived_class"] = "blocker" if v >= 50.0 else "non_blocker"
        return out

    if st not in POTENCY_TYPES:
        return out
    if not np.isfinite(v):
        return out

    factor = UNIT_TO_UM.get(units_l, np.nan)
    if not np.isfinite(factor):
        if units_l in ("%", "percent"):
            out["measure_family"] = "percent_inhibition"
            out["derived_class"] = "blocker" if v >= 50.0 else "non_blocker"
            return out
        return out

    out["measure_family"] = "potency"
    v_um = v * factor
    out["value_um"] = v_um

    if rel == "=":
        out["derived_class"] = "blocker" if (v_um < threshold_um if strict else v_um <= threshold_um) else "non_blocker"
    elif rel in (">", ">="):
        out["censored"] = ">"
        if v_um >= threshold_um:
            out["derived_class"] = "non_blocker"      # 실제값 > v_um >= threshold
        else:
            out["derived_class"] = "ambiguous_censored"
    elif rel in ("<", "<="):
        out["censored"] = "<"
        if v_um <= threshold_um:
            out["derived_class"] = "blocker"          # 실제값 < v_um <= threshold
        else:
            out["derived_class"] = "ambiguous_censored"
    elif rel in ("~",):
        out["derived_class"] = "blocker" if (v_um <= threshold_um) else "non_blocker"
    else:
        out["derived_class"] = "ambiguous"
    return out


def _assay_group(assay_type: str, description: str) -> str:
    """assay_type + 설명 텍스트 -> 측정 방식 그룹 (합치지 않기 위한 분리)."""
    d = (description or "").lower()
    if re.search(r"patch|electrophysiolog|voltage clamp|manual patch|qpacth|whole.cell", d):
        return "patch_clamp_electrophysiology"
    if re.search(r"thallium|fluxor|flux", d):
        return "thallium_flux"
    if re.search(r"rubidium|rb\+", d):
        return "rubidium_flux"
    if re.search(r"binding|radioligand|displacement|\[3h\]|\[125i\]|dofetilide", d):
        return "binding"
    if re.search(r"membrane potential|fluorescent|dye", d):
        return "membrane_potential_fluorescence"
    if (assay_type or "").upper() == "B":
        return "binding"
    if (assay_type or "").upper() == "F":
        return "functional_other"
    if (assay_type or "").upper() == "T":
        return "toxicity_other"
    return "other"


def build_external_records() -> pd.DataFrame:
    """ChEMBL hERG + PubChem AID588834 + 논문 보충자료 S2 -> long format 실험 레코드 테이블."""
    recs: list[dict] = []

    # ---- ChEMBL hERG activity
    acts = read_jsonl(CACHE / f"chembl_activity_{CHEMBL_HERG_TARGET}.jsonl")
    log(f"  ChEMBL hERG activity {len(acts)}건 정리")
    for a in acts:
        smi = a.get("canonical_smiles") or ""
        recs.append({
            "source": "ChEMBL",
            "source_record_id": f"activity:{a.get('activity_id')}",
            "external_compound_id": a.get("molecule_chembl_id") or "",
            "compound_name": a.get("molecule_pref_name") or "",
            "record_smiles": smi,
            "assay_id": a.get("assay_chembl_id") or "",
            "assay_type": a.get("assay_type") or "",
            "assay_group": _assay_group(a.get("assay_type"), a.get("assay_description")),
            "assay_description": a.get("assay_description") or "",
            "standard_type": a.get("standard_type") or "",
            "standard_relation": a.get("standard_relation") or "",
            "standard_value": a.get("standard_value"),
            "standard_units": a.get("standard_units") or "",
            "pchembl_value": a.get("pchembl_value"),
            "document_id": a.get("document_chembl_id") or "",
            "document_year": a.get("document_year"),
            "target_chembl_id": a.get("target_chembl_id") or CHEMBL_HERG_TARGET,
            "target_pref_name": a.get("target_pref_name") or "",
            "target_organism": a.get("target_organism") or "",
            "data_validity_comment": a.get("data_validity_comment") or "",
            "potential_duplicate": a.get("potential_duplicate"),
            "source_url": f"https://www.ebi.ac.uk/chembl/activity/{a.get('activity_id')}",
        })

    # ---- PubChem AID588834
    for aid in PUBCHEM_AIDS:
        p = CACHE / f"pubchem_aid{aid}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, skiprows=[1, 2, 3, 4, 5], low_memory=False,
                         encoding="utf-8", encoding_errors="replace")
        cid_props = {r["CID"]: r for r in read_jsonl(CACHE / "pubchem_cid_props.jsonl")}
        for row in df.to_dict("records"):
            cid = row.get("PUBCHEM_CID")
            cid = str(int(cid)) if pd.notna(cid) else ""
            pr = cid_props.get(int(cid)) if cid else None
            potency = row.get("Potency")
            efficacy = row.get("Efficacy")
            outcome = row.get("PUBCHEM_ACTIVITY_OUTCOME") or ""
            if pd.notna(potency):
                stype, srel, sval, sunit, family = "IC50", "=", float(potency), "uM", "potency"
            elif pd.notna(efficacy):
                # 단일 농도 %inhibition (IC50 로 환산하지 않음)
                stype, srel, sval, sunit, family = "Inhibition", "=", float(efficacy), "%", "percent"
            else:
                stype, srel, sval, sunit, family = "Outcome", "", np.nan, "", "outcome"
            recs.append({
                "source": f"PubChem AID{aid}",
                "source_record_id": f"sid:{row.get('PUBCHEM_SID')}",
                "external_compound_id": cid,
                "compound_name": "",
                "record_smiles": (pr or {}).get("SMILES", "") or (row.get("PUBCHEM_EXT_DATASOURCE_SMILES") or ""),
                "assay_id": f"AID{aid}",
                "assay_type": "qHTS",
                "assay_group": "thallium_flux",
                "assay_description": "qHTS Assay for Small Molecule Inhibitors of the Human hERG Channel Activity (U2OS, FluxOR)",
                "standard_type": stype,
                "standard_relation": srel,
                "standard_value": sval,
                "standard_units": sunit,
                "pchembl_value": np.nan,
                "document_id": f"AID{aid}",
                "document_year": 2016,
                "target_chembl_id": CHEMBL_HERG_TARGET,
                "target_pref_name": "hERG (PubChem AID588834)",
                "target_organism": "Homo sapiens",
                "data_validity_comment": "",
                "potential_duplicate": 0,
                "reported_activity": outcome,
                "efficacy_percent": efficacy,
                "pubchem_phenotype": row.get("Phenotype"),
                "source_url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                "inchikey": (pr or {}).get("InChIKey", ""),
                "_family_hint": family,
            })

    # ---- 논문 보충자료 S2 (Tox21 hERG qHTS, Krishna et al. 2022)
    s2 = load_supplement_s2()
    if s2 is not None:
        n_s2 = 0
        for r in s2.to_dict("records"):
            smi = r["SMILES"]
            if not isinstance(smi, str) or not smi.strip() or smi.strip() == "N/A":
                continue
            nl = r["neglogIC50"]
            # 보충자료 S2의 값은 표준 pIC50 (mol/L 기준)이다.
            #   예: pIC50 4.0  -> IC50 = 100 uM (inactive 상한 농도)
            #       pIC50 7.12 -> IC50 = 0.076 uM (vanoxerine 문헌값과 일치)
            # 따라서 IC50[uM] = 10**(6 - pIC50).
            v_um = 10 ** (6 - nl) if isinstance(nl, (int, float)) and math.isfinite(nl) else np.nan
            aff = r["Aff"]
            recs.append({
                "source": "Tox21 (Biology 2022 S2)",
                "source_record_id": f"S2:{r['CASRN']}",
                "external_compound_id": str(r["CASRN"]),
                "compound_name": r["PREFERRED_NAME"] if isinstance(r["PREFERRED_NAME"], str) else "",
                "record_smiles": smi,
                "assay_id": "Tox21_hERG_qHTS_S2",
                "assay_type": "F",
                "assay_group": "thallium_flux",
                "assay_description": "Tox21 qHTS thallium flux hERG (U2OS), 14-point CR curve",
                "standard_type": "IC50" if np.isfinite(v_um) else "Unknown",
                "standard_relation": "=" if np.isfinite(v_um) else "",
                "standard_value": v_um,
                "standard_units": "uM",
                "pchembl_value": nl if isinstance(nl, (int, float)) else np.nan,
                "document_id": "PMC8869358-S2",
                "document_year": 2022,
                "target_chembl_id": CHEMBL_HERG_TARGET,
                "target_pref_name": "hERG (Tox21 thallium flux)",
                "target_organism": "Homo sapiens",
                "data_validity_comment": "",
                "potential_duplicate": 0,
                "reported_activity": "Active" if aff == 1 else ("Inactive" if aff == 0 else ""),
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8869358/"
                              "#sec2-biology-11-00209 (Supplementary S2)",
                "inchikey": "",
            })
            n_s2 += 1
        log(f"  보충자료 S2 레코드 {n_s2}건")

    df = pd.DataFrame(recs)
    log(f"  외부 레코드 {len(df)}건 (ChEMBL/PubChem/보충자료)")
    return df


def load_supplement_s2():
    """논문 보충자료 zip 에서 S2 (Tox21 hERG qHTS 활성표) 를 꺼내 캐시."""
    out = CACHE / "supplements" / "S2_tox21_hERG.csv"
    if out.exists():
        df = pd.read_csv(out, encoding="utf-8")
    else:
        zp = CACHE / "supplements" / "EuropePMC_supplementaryFiles.zip"
        if not zp.exists():
            return None
        try:
            z = zipfile.ZipFile(zp)
            inner_name = [n for n in z.namelist() if n.endswith("-s001.zip")]
            if not inner_name:
                return None
            inner = zipfile.ZipFile(io.BytesIO(z.read(inner_name[0])))
            cand = [n for n in inner.namelist() if n.lower().endswith(".csv") and "B" in n]
            if not cand:
                return None
            raw = inner.read(cand[0])
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
            df.columns = ["CASRN", "SMILES", "PREFERRED_NAME", "neglogIC50", "Aff"][:len(df.columns)]
            df.to_csv(out, index=False, encoding="utf-8")
            log(f"  보충자료 S2 추출: {out.name} ({len(df)}행)")
        except Exception as exc:  # noqa: BLE001
            log(f"  ! 보충자료 S2 파싱 실패: {exc}")
            return None
    if "neglogIC50" not in df.columns:
        df.columns = ["CASRN", "SMILES", "PREFERRED_NAME", "neglogIC50", "Aff"][:len(df.columns)]
    df["neglogIC50"] = pd.to_numeric(df["neglogIC50"], errors="coerce")
    return df


def _build_key_index(df: pd.DataFrame, key_col: str) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, k in enumerate(df[key_col].tolist()):
        if k:
            idx.setdefault(k, []).append(i)
    return idx


def _cached_external_keys(uniq: list[str]) -> dict[str, dict]:
    """외부 SMILES 구조 키를 캐시 (재실행 시 재계산 방지)."""
    p = CACHE / "external_smiles_keys.csv"
    have: dict[str, dict] = {}
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        for r in df.to_dict("records"):
            have[str(r["smiles"])] = r
    todo = [s for s in uniq if s not in have]
    if todo:
        log(f"    구조 키 신규 계산 {len(todo)}개 (캐시 {len(have)}개)")
        rows = []
        for i, s in enumerate(todo):
            k = mol_keys(s) or {}
            rows.append({
                "smiles": s,
                "canonical_isomeric": k.get("canonical_isomeric", ""),
                "inchikey": k.get("inchikey", ""),
                "connectivity_key": k.get("connectivity_key", ""),
                "norm_inchikey": k.get("norm_inchikey", ""),
                "has_salt": bool(k.get("has_salt", False)),
            })
            if i % 5000 == 0 and i:
                log(f"      {i}/{len(todo)}")
        new = pd.DataFrame(rows)
        if p.exists():
            new = pd.concat([pd.read_csv(p, low_memory=False), new], ignore_index=True)
        new.to_csv(p, index=False)
        have = {str(r["smiles"]): r for r in new.to_dict("records")}
    return have


def _cached_external_fps(uniq: list[str]) -> dict:
    """외부 SMILES Morgan fingerprint 캐시 (pickle)."""
    import pickle
    from rdkit import DataStructs
    p = CACHE / "external_smiles_fps.pkl"
    have: dict[str, object] = {}
    if p.exists():
        try:
            with open(p, "rb") as f:
                have = pickle.load(f)
        except Exception:  # noqa: BLE001
            have = {}
    todo = [s for s in uniq if s not in have]
    if todo:
        log(f"    fingerprint 신규 계산 {len(todo)}개 (캐시 {len(have)}개)")
        for s in todo:
            have[s] = morgan_fp(s)
        with open(p, "wb") as f:
            pickle.dump(have, f, protocol=3)
    return have


def stage_match() -> None:
    log("== stage match ==")
    ds = pd.read_csv(OUT / "dataset_keys.csv", low_memory=False)
    ext = build_external_records()

    # --- 외부 레코드 구조 키 계산 (SMILES 고유값만)
    uniq = sorted({s for s in ext["record_smiles"].tolist() if isinstance(s, str) and s})
    log(f"  외부 고유 SMILES {len(uniq)}개 → 구조 키 계산")
    keymap = _cached_external_keys(uniq)
    for c in ["inchikey", "connectivity_key", "parent_inchikey", "norm_inchikey"]:
        ext[c] = [keymap.get(s, {}).get(c, "") if isinstance(s, str) else "" for s in ext["record_smiles"]]
    ext["has_salt"] = [bool(keymap.get(s, {}).get("has_salt")) if isinstance(s, str) else False
                       for s in ext["record_smiles"]]
    ext.to_csv(CACHE / "external_records_keys.csv", index=False)

    idx_ik = _build_key_index(ext, "inchikey")
    idx_conn = _build_key_index(ext, "connectivity_key")
    idx_norm = _build_key_index(ext, "norm_inchikey")

    # --- 유사도용 fingerprint (미매칭 화합물에만 사용, 고유 SMILES 당 1회 계산)
    from rdkit import DataStructs
    fp_map = _cached_external_fps(uniq)
    ext_fps, ext_fp_i = [], []
    for i, s in enumerate(ext["record_smiles"].tolist()):
        fp = fp_map.get(s)
        if fp is not None:
            ext_fps.append(fp)
            ext_fp_i.append(i)
    log(f"  외부 fingerprint {len(ext_fps)}개 준비")

    matches: list[dict] = []
    for row in ds.itertuples(index=False):
        rec_idxs, mtype, sim = [], None, np.nan
        if row.inchikey and row.inchikey in idx_ik:
            rec_idxs, mtype = idx_ik[row.inchikey], "exact_stereo_match"
        elif row.connectivity_key and row.connectivity_key in idx_conn:
            rec_idxs, mtype = idx_conn[row.connectivity_key], "exact_connectivity_match"
        elif row.norm_inchikey and row.norm_inchikey in idx_norm:
            rec_idxs, mtype = idx_norm[row.norm_inchikey], "normalized_structure_match"
        else:
            fp = morgan_fp(row.original_smiles)
            if fp is not None and ext_fps:
                sims = DataStructs.BulkTanimotoSimilarity(fp, ext_fps)
                j = int(np.argmax(sims))
                if sims[j] >= 0.90:
                    rec_idxs, mtype, sim = [ext_fp_i[j]], "similarity_only", float(sims[j])
        if mtype is None:
            matches.append({"dataset_row_index": row.dataset_row_index, "match_type": "no_match",
                            "similarity": np.nan, "external_row": None})
        else:
            for i in rec_idxs:
                matches.append({"dataset_row_index": row.dataset_row_index, "match_type": mtype,
                                "similarity": sim, "external_row": i})

    mdf = pd.DataFrame([m for m in matches if m["external_row"] is not None])
    nomatch = [m["dataset_row_index"] for m in matches if m["external_row"] is None]
    log(f"  매칭: {mdf['dataset_row_index'].nunique()}행 / 미매칭 {len(nomatch)}행")

    # --- 결과 테이블 작성
    ext2 = ext.reset_index(drop=True)
    rows = []
    for m in mdf.itertuples(index=False):
        d = ds.iloc[int(ds.index[ds["dataset_row_index"] == m.dataset_row_index][0])]
        e = ext2.iloc[m.external_row]
        cls = {th: _classify_record(e["standard_type"], e["standard_relation"], e["standard_value"],
                                    e["standard_units"], e["pchembl_value"], th)
               for th in THRESHOLDS_UM}
        c10 = cls[PRIMARY_THRESHOLD_UM]
        c10s = _classify_record(e["standard_type"], e["standard_relation"], e["standard_value"],
                                e["standard_units"], e["pchembl_value"], PRIMARY_THRESHOLD_UM, strict=True)
        rows.append({
            "dataset_row_index": int(d["dataset_row_index"]),
            "original_smiles": d["original_smiles"],
            "canonical_smiles": d["canonical_isomeric"],
            "dataset_label": int(d["dataset_label"]),
            "compound_name": e["compound_name"],
            "external_database": e["source"],
            "external_compound_id": e["external_compound_id"],
            "assay_id": e["assay_id"],
            "assay_type": e["assay_type"],
            "assay_group": e["assay_group"],
            "assay_description": e["assay_description"],
            "measured_value": e["standard_value"],
            "measured_unit": e["standard_units"],
            "measured_value_um": c10["value_um"],
            "measure_family": c10["measure_family"],
            "censored": c10["censored"] if c10["censored"] in (">", "<") else "",
            "reported_activity": e.get("reported_activity", "") if isinstance(e, pd.Series) else "",
            "pchembl_value": e["pchembl_value"],
            "threshold_used": f"{PRIMARY_THRESHOLD_UM:g} uM",
            "derived_herg_class": c10["derived_class"],
            "derived_class_1um": cls[1.0]["derived_class"],
            "derived_class_3um": cls[3.0]["derived_class"],
            "derived_class_30um": cls[30.0]["derived_class"],
            "derived_class_100um": cls[100.0]["derived_class"],
            "derived_class_10um_strict": c10s["derived_class"],
            "match_type": m.match_type,
            "similarity": m.similarity,
            "salt_flag_dataset": bool(d["has_salt"]),
            "salt_flag_external": bool(e["has_salt"]),
            "stereo_flag_dataset": bool(d["has_stereo"]),
            "dataset_inchikey": d["inchikey"],
            "external_inchikey": e["inchikey"],
            "document_id": e["document_id"],
            "document_year": e["document_year"],
            "source_doi_or_url": e["source_url"],
            "notes": "",
        })
    out = pd.DataFrame(rows)
    out["label_agreement"] = np.where(
        (out["dataset_label"] == LABEL_TOXIC) & (out["derived_herg_class"] == "blocker"), "agree",
        np.where((out["dataset_label"] == LABEL_NONTOXIC) & (out["derived_herg_class"] == "non_blocker"),
                 "agree",
                 np.where(out["derived_herg_class"].isin(["blocker", "non_blocker"]), "disagree", "not_comparable")))
    out.to_csv(OUT / "matched_compounds.csv", index=False)
    log(f"  matched_compounds.csv: {len(out)}행 (분자-레코드 쌍)")

    # --- 제외 사례 (숨기지 않고 전부 기록)
    ex = []
    for r in nomatch:
        ex.append({"dataset_row_index": r, "reason": "no_structure_match_in_external_data",
                   "detail": "InChIKey/연결성/정규화 모두 불일치, ECFP Tanimoto < 0.90"})
    for m in mdf[mdf["match_type"] == "similarity_only"].itertuples(index=False):
        ex.append({"dataset_row_index": m.dataset_row_index, "reason": "similarity_only_excluded_from_primary",
                   "detail": f"ECFP Tanimoto={m.similarity:.3f} (동일 화합물로 확정 불가)"})
    pd.DataFrame(ex).to_csv(OUT / "excluded_cases.csv", index=False)

    # --- 매칭 통계 요약
    summary = {
        "dataset_rows": int(len(ds)),
        "dataset_rows_matched": int(mdf["dataset_row_index"].nunique()),
        "dataset_rows_no_match": int(len(nomatch)),
        "match_type_counts_rows": {k: int(v) for k, v in mdf["match_type"].value_counts().items()},
        "match_type_counts_compounds": {
            k: int(v) for k, v in
            mdf.groupby("dataset_row_index")["match_type"].first().value_counts().items()},
        "external_records_total": int(len(ext2)),
        "matched_record_pairs": int(len(mdf)),
        "sources": {k: int(v) for k, v in out["external_database"].value_counts().items()},
    }
    with open(OUT / "match_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- 3. stats

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - half), min(1.0, c + half))


def _metrics(y_true_pos: np.ndarray, y_pred_pos: np.ndarray) -> dict:
    tp = int(np.sum((y_true_pos == 1) & (y_pred_pos == 1)))
    fn = int(np.sum((y_true_pos == 1) & (y_pred_pos == 0)))
    fp = int(np.sum((y_true_pos == 0) & (y_pred_pos == 1)))
    tn = int(np.sum((y_true_pos == 0) & (y_pred_pos == 0)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    bal = np.nanmean([sens, spec])
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "n": tp + tn + fp + fn,
            "sensitivity": sens, "specificity": spec, "precision": prec,
            "accuracy": acc, "balanced_accuracy": bal}


def _aggregate_compound(g: pd.DataFrame, class_col: str) -> str:
    """화합물 단위 집계: 상충 assay는 ambiguous 로 보존 (강제 축약 금지)."""
    vals = [v for v in g[class_col].tolist() if v in ("blocker", "non_blocker")]
    if not vals:
        return "unknown"
    if len(set(vals)) > 1:
        return "ambiguous"
    return vals[0]


def cluster_bootstrap_ci(groups: list[tuple[np.ndarray, np.ndarray]], n_boot: int = 2000,
                         seed: int = 0) -> dict:
    """클러스터(스캐폴드) 단위 부트스트랩 CI."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    if n < 3:
        return {"ba_lo": np.nan, "ba_hi": np.nan, "acc_lo": np.nan, "acc_hi": np.nan}
    bas, accs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        yt = np.concatenate([groups[i][0] for i in pick])
        yp = np.concatenate([groups[i][1] for i in pick])
        m = _metrics(yt, yp)
        bas.append(m["balanced_accuracy"])
        accs.append(m["accuracy"])
    bas = np.array(bas, dtype=float)
    accs = np.array(accs, dtype=float)
    return {"ba_lo": float(np.nanpercentile(bas, 2.5)), "ba_hi": float(np.nanpercentile(bas, 97.5)),
            "acc_lo": float(np.nanpercentile(accs, 2.5)), "acc_hi": float(np.nanpercentile(accs, 97.5))}


def permutation_test(y_true_pos: np.ndarray, y_pred_pos: np.ndarray,
                     n_perm: int = 10000, seed: int = 0) -> dict:
    """라벨 순열검정: 일치율/balanced accuracy가 우연과 구분되는지."""
    rng = np.random.default_rng(seed)
    obs = _metrics(y_true_pos, y_pred_pos)
    obs_ba = obs["balanced_accuracy"]
    obs_acc = obs["accuracy"]
    ba_null, acc_null = [], []
    y = y_true_pos.copy()
    for _ in range(n_perm):
        rng.shuffle(y)
        m = _metrics(y, y_pred_pos)
        ba_null.append(m["balanced_accuracy"])
        acc_null.append(m["accuracy"])
    ba_null = np.array(ba_null, dtype=float)
    acc_null = np.array(acc_null, dtype=float)
    return {
        "observed_balanced_accuracy": obs_ba,
        "observed_accuracy": obs_acc,
        "perm_ba_mean": float(np.nanmean(ba_null)),
        "perm_ba_sd": float(np.nanstd(ba_null)),
        "p_balanced_accuracy": float((np.sum(ba_null >= obs_ba) + 1) / (n_perm + 1)),
        "p_accuracy": float((np.sum(acc_null >= obs_acc) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def _class_columns(g_pot: pd.DataFrame, g_pct: pd.DataFrame) -> dict:
    """potency 계열 레코드와 %inhibition 계열 레코드를 분리해 클래스 산출.

    지시문 요구: percent inhibition 단독 측정은 IC50 와 동일하게 변환하지 않는다.
    따라서 주 분석용 클래스(class_*um)는 potency 계열만 사용하고,
    %inhibition 기반 클래스는 class_pct_* 로 별도 보존한다.
    """
    out = {
        "n_records": len(g_pot) + len(g_pct),
        "n_records_potency": len(g_pot),
        "n_records_percent_inhibition": len(g_pct),
        "class_10um": _aggregate_compound(g_pot, "derived_herg_class") if len(g_pot) else "unknown",
        "class_1um": _aggregate_compound(g_pot, "derived_class_1um") if len(g_pot) else "unknown",
        "class_3um": _aggregate_compound(g_pot, "derived_class_3um") if len(g_pot) else "unknown",
        "class_30um": _aggregate_compound(g_pot, "derived_class_30um") if len(g_pot) else "unknown",
        "class_100um": _aggregate_compound(g_pot, "derived_class_100um") if len(g_pot) else "unknown",
        "class_pct_10um": _aggregate_compound(g_pct, "derived_herg_class") if len(g_pct) else "unknown",
        "class_10um_strict": _aggregate_compound(g_pot, "derived_class_10um_strict") if len(g_pot) else "unknown",
        "min_value_um": g_pot["measured_value_um"].min() if len(g_pot) else np.nan,
        "median_value_um": g_pot["measured_value_um"].median() if len(g_pot) else np.nan,
        "max_value_um": g_pot["measured_value_um"].max() if len(g_pot) else np.nan,
        "any_censored": bool(g_pot["censored"].isin([">", "<"]).any()) if len(g_pot) else False,
        "n_assays": pd.concat([g_pot["assay_id"], g_pct["assay_id"]]).nunique() if len(g_pot) + len(g_pct) else 0,
        "assay_groups": "|".join(sorted(set(pd.concat([g_pot["assay_group"], g_pct["assay_group"]])))),
    }
    return out


def compound_level_table(matched: pd.DataFrame):
    """dataset compound 단위 집계 테이블 (source 별 + 통합)."""
    prim = matched[matched["match_type"] != "similarity_only"].copy()
    prim["_is_pot"] = prim["measure_family"] == "potency"
    prim["_is_pct"] = prim["measure_family"] == "percent_inhibition"

    def _one(keys, g, keynames):
        g_pot = g[g["_is_pot"]]
        g_pct = g[g["_is_pct"]]
        rec = dict(zip(keynames, keys))
        rec.update(_class_columns(g_pot, g_pct))
        return rec

    rows = []
    grp_cols = ["dataset_row_index", "original_smiles", "canonical_smiles", "dataset_label",
                "external_database"]
    for keys, g in prim.groupby(grp_cols, dropna=False):
        rec = _one(keys, g, grp_cols)
        rec["match_type"] = g["match_type"].iloc[0]
        rec["external_ids"] = ",".join(sorted(set(g["external_compound_id"].astype(str))))
        rows.append(rec)
    out = pd.DataFrame(rows)

    # 통합(모든 source) 화합물 단위
    rows_all = []
    grp_all = ["dataset_row_index", "original_smiles", "canonical_smiles", "dataset_label"]
    for keys, g in prim.groupby(grp_all, dropna=False):
        rec = _one(keys, g, grp_all)
        rec["n_source"] = g["external_database"].nunique()
        rec["sources"] = "|".join(sorted(set(g["external_database"])))
        rec["match_type"] = g["match_type"].iloc[0]
        rec["external_ids"] = ",".join(sorted(set(g["external_compound_id"].astype(str))))
        rows_all.append(rec)
    return out, pd.DataFrame(rows_all)


def stage_stats() -> None:
    log("== stage stats ==")
    matched = pd.read_csv(OUT / "matched_compounds.csv", low_memory=False)
    ds = pd.read_csv(OUT / "dataset_keys.csv", low_memory=False)
    per_source, per_compound = compound_level_table(matched)
    per_source.to_csv(OUT / "compound_level_by_source.csv", index=False)
    per_compound.to_csv(OUT / "compound_level.csv", index=False)

    metrics_rows = []

    def evaluate(df: pd.DataFrame, class_col: str, scope: str, unit: str,
                 subset_col: str | None = None, seeds=None) -> None:
        d = df[df[class_col].isin(["blocker", "non_blocker"])].copy()
        if d.empty:
            return
        y_true = np.where(d["dataset_label"] == LABEL_TOXIC, 1, 0)
        y_pred = np.where(d[class_col] == "blocker", 1, 0)
        m = _metrics(y_true, y_pred)
        lo_ba, hi_ba = wilson_ci(int(round(m["balanced_accuracy"] * m["n"])), m["n"])
        # 일치율(전체)
        agree = int(np.sum(y_true == y_pred))
        a_lo, a_hi = wilson_ci(agree, m["n"])
        # 라벨별 일치율
        tox = d[d["dataset_label"] == LABEL_TOXIC]
        non = d[d["dataset_label"] == LABEL_NONTOXIC]
        rows = [{
            "scope": scope, "unit": unit, "class_rule": class_col,
            "n": m["n"], "TP": m["TP"], "FN": m["FN"], "FP": m["FP"], "TN": m["TN"],
            "agreement": agree / m["n"], "agreement_lo95": a_lo, "agreement_hi95": a_hi,
            "disagreement": 1 - agree / m["n"], "n_clear_compounds": m["n"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "precision": m["precision"], "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "bal_acc_lo95": lo_ba, "bal_acc_hi95": hi_ba,
            "n_label0": len(tox), "n_label1": len(non),
            "agree_label0_rate": (tox[class_col] == "blocker").mean() if len(tox) else np.nan,
            "agree_label1_rate": (non[class_col] == "non_blocker").mean() if len(non) else np.nan,
        }]
        if unit == "compound":
            if "canonical_smiles" in d.columns:
                smis = d["canonical_smiles"]
            elif "original_smiles" in d.columns:
                smis = d["original_smiles"]
            else:
                smis = d["dataset_row_index"].astype(str)
            scaff = [murcko_scaffold(s) or f"__mol_{i}" for i, s in enumerate(smis)]
            groups: dict[str, list[int]] = {}
            for i, s in enumerate(scaff):
                groups.setdefault(s, []).append(i)
            gs = [(y_true[ix], y_pred[ix]) for ix in groups.values()]
            ci = cluster_bootstrap_ci(gs)
            rows[0].update({"cluster_ba_lo95": ci["ba_lo"], "cluster_ba_hi95": ci["ba_hi"],
                            "n_clusters": len(groups)})
            perm = permutation_test(y_true, y_pred)
            rows[0].update({f"perm_{k}": v for k, v in perm.items()})
        metrics_rows.extend(rows)

    # (a) 행(레코드) 단위 — similarity_only 는 주 분석에서 제외
    mvalid = matched[matched["match_type"] != "similarity_only"]
    for th, col in [(1.0, "derived_class_1um"), (3.0, "derived_class_3um"),
                    (10.0, "derived_herg_class"), (30.0, "derived_class_30um"),
                    (100.0, "derived_class_100um")]:
        for src in ["ALL"] + sorted(mvalid["external_database"].unique()):
            sub = mvalid if src == "ALL" else mvalid[mvalid["external_database"] == src]
            evaluate(sub, col, f"{src}|threshold={th:g}uM", "record")

    # (b) 화합물 단위 (source 별 + 통합, 임계값별)
    for th, col in [(1.0, "class_1um"), (3.0, "class_3um"), (10.0, "class_10um"),
                    (30.0, "class_30um"), (100.0, "class_100um")]:
        for src in ["ALL"] + sorted(per_source["external_database"].unique()):
            sub = per_source if src == "ALL" else per_source[per_source["external_database"] == src]
            evaluate(sub, col, f"{src}|threshold={th:g}uM", "compound")
    for th, col in [(10.0, "class_10um")]:
        evaluate(per_compound, col, f"merged_sources|threshold={th:g}uM", "compound")

    # (b-2) percent inhibition 단독 측정만 있는 화합물: IC50 와 동일 변환하지 않고 별도 보고
    pct_only = per_compound[per_compound["class_10um"] == "unknown"].copy()
    evaluate(pct_only, "class_pct_10um", "percent_inhibition_only|rule=>=50%", "compound")

    # (b-3) 경계 규칙 민감도: 블록커 판정을 `IC50 < 10 uM` (strict) 로 계산
    evaluate(mvalid, "derived_class_10um_strict", "ALL|strict<10uM", "record")
    for src in sorted(per_source["external_database"].unique()):
        evaluate(per_source[per_source["external_database"] == src], "class_10um_strict",
                 f"{src}|strict<10uM", "compound")
    evaluate(per_compound, "class_10um_strict", "merged_sources|strict<10uM", "compound")

    # (c) ChEMBL 을 제외한 독립 소스만 사용: potency 우선, 없으면 %inhibition >= 50%
    ind = per_source[per_source["external_database"] != "ChEMBL"].copy()
    ind["class_ind"] = np.where(ind["class_10um"] != "unknown", ind["class_10um"],
                                ind["class_pct_10um"])
    ind_merged = ind.groupby(["dataset_row_index", "dataset_label"], as_index=False).agg(
        cls=("class_ind", lambda s: _aggregate_compound(pd.DataFrame({"c": list(s)}), "c")))
    ind_merged = ind_merged.rename(columns={"cls": "class_ind"})
    evaluate(ind_merged, "class_ind", "independent_sources_only|combined_rule", "compound")

    # (b-4) 경계값(정확히 10 uM) 화합물의 라벨 분포
    pot_exact_all = mvalid[mvalid["measure_family"] == "potency"].copy()
    pot_exact_all["v"] = pd.to_numeric(pot_exact_all["measured_value_um"], errors="coerce")
    at_boundary = pot_exact_all[np.isclose(pot_exact_all["v"], PRIMARY_THRESHOLD_UM, rtol=0, atol=1e-9)]
    bnd = at_boundary.groupby("dataset_row_index")["dataset_label"].first().value_counts()
    bnd_df = pd.DataFrame({
        "boundary_value_um": [PRIMARY_THRESHOLD_UM],
        "n_compounds_with_a_10uM_measurement": [int(bnd.sum())],
        "n_label0_toxic": [int(bnd.get(LABEL_TOXIC, 0))],
        "n_label1_nontoxic": [int(bnd.get(LABEL_NONTOXIC, 0))],
    })
    bnd_df.to_csv(OUT / "boundary_analysis.csv", index=False)

    # assay 그룹(측정 방식)별 — 서로 다른 assay를 합치지 않기 위함
    prim = matched[matched["match_type"] != "similarity_only"]
    for ag, g in prim.groupby("assay_group"):
        if len(g) < 20:
            continue
        ev = g.groupby(["dataset_row_index", "dataset_label"]).apply(
            lambda x: _aggregate_compound(x, "derived_herg_class"), include_groups=False).reset_index()
        ev.columns = ["dataset_row_index", "dataset_label", "cls"]
        evaluate(ev, "cls", f"assay_group={ag}", "compound")

    # --- 라벨링 규칙 역추적: 어떤 집계/임계값이 대상 라벨을 재현하는가
    rev_rows = []
    che = matched[(matched["external_database"] == "ChEMBL") &
                  (matched["match_type"] != "similarity_only")].copy()
    che["potency_um"] = pd.to_numeric(che["measured_value_um"], errors="coerce")
    che["pchembl"] = pd.to_numeric(che["pchembl_value"], errors="coerce")
    pot = che[che["measure_family"] == "potency"].copy()
    pot_exact = pot[pot["censored"].isna() | (pot["censored"] == "")]
    for agg_name, fn in [("min", "min"), ("median", "median"), ("max", "max")]:
        agg = pot_exact.groupby("dataset_row_index")["potency_um"].agg(fn)
        lab = ds.set_index("dataset_row_index")["dataset_label"]
        d = pd.DataFrame({"v": agg, "label": lab.reindex(agg.index)}).dropna()
        for th in THRESHOLDS_UM:
            for strict in (False, True):
                y_true = np.where(d["label"] == LABEL_TOXIC, 1, 0)
                y_pred = np.where(d["v"] < th if strict else d["v"] <= th, 1, 0)
                m = _metrics(y_true, y_pred)
                rev_rows.append({
                    "aggregation": f"ChEMBL_exact_{agg_name}_potency_uM"
                                   + ("_strict_lt" if strict else "_le"),
                    "threshold_um": th, "n_compounds": m["n"],
                    "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                    "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
    aggp = che.dropna(subset=["pchembl"]).groupby("dataset_row_index")["pchembl"].max()
    d = pd.DataFrame({"v": aggp, "label": lab.reindex(aggp.index)}).dropna()
    for th, lbl in [(5.0, "pchembl>=5 (~<=10uM)")]:
        y_true = np.where(d["label"] == LABEL_TOXIC, 1, 0)
        y_pred = np.where(d["v"] >= th, 1, 0)
        m = _metrics(y_true, y_pred)
        rev_rows.append({"aggregation": f"ChEMBL_{lbl}", "threshold_um": 10.0, "n_compounds": m["n"],
                         "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                         "sensitivity": m["sensitivity"], "specificity": m["specificity"]})

    # 결합 규칙: (potency 최소값 < 10 uM) 또는 (%inhibition 최대값 >= 50%)
    prim_all = matched[matched["match_type"] != "similarity_only"]
    pp = prim_all[prim_all["measure_family"] == "potency"].copy()
    pp["v"] = pd.to_numeric(pp["measured_value_um"], errors="coerce")
    pmin = pp.dropna(subset=["v"]).groupby("dataset_row_index")["v"].min()
    pc = prim_all[prim_all["measure_family"] == "percent_inhibition"].copy()
    pc["v"] = pd.to_numeric(pc["measured_value"], errors="coerce")
    pmax = pc.dropna(subset=["v"]).groupby("dataset_row_index")["v"].max()
    idx = pmin.index.union(pmax.index)
    pred_blocker = ((pmin.reindex(idx) < PRIMARY_THRESHOLD_UM).fillna(False).astype(bool)
                    | (pmax.reindex(idx) >= 50.0).fillna(False).astype(bool))
    y_true = np.where(lab.reindex(idx).values == LABEL_TOXIC, 1, 0)
    m = _metrics(y_true, pred_blocker.values.astype(int))
    rev_rows.append({"aggregation": "combined_potency_strict_lt10uM_or_pct_ge50",
                     "threshold_um": 10.0, "n_compounds": m["n"],
                     "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
    pd.DataFrame(rev_rows).to_csv(OUT / "labeling_rule_reverse.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(OUT / "metrics.csv", index=False)
    log(f"  metrics.csv: {len(metrics_rows)}행")

    # --- 불일치 사례 + 원인 분석
    conflicts = []
    che_comp = per_compound.copy()
    for r in che_comp.itertuples(index=False):
        cls = getattr(r, "class_10um")
        if cls not in ("blocker", "non_blocker"):
            continue
        expect = "blocker" if r.dataset_label == LABEL_TOXIC else "non_blocker"
        if cls == expect:
            continue
        g = matched[(matched["dataset_row_index"] == r.dataset_row_index) &
                    (matched["match_type"] != "similarity_only")]
        reasons = []
        mv = pd.to_numeric(g["measured_value_um"], errors="coerce")
        if mv.notna().any() and (mv.min() <= 10 * 3) and (mv.max() >= 10 / 3):
            reasons.append("measurement_near_threshold")
        classes = g["derived_herg_class"].tolist()
        if len({c for c in classes if c in ("blocker", "non_blocker")}) > 1:
            reasons.append("conflicting_across_assays")
        if (g["assay_group"].nunique() > 1):
            reasons.append("multiple_assay_types")
        if (g["censored"].isin([">", "<"])).any():
            reasons.append("censored_measurement")
        if (g["measure_family"] == "percent_inhibition").any():
            reasons.append("percent_inhibition_only")
        if (g["salt_flag_dataset"] != g["salt_flag_external"]).any():
            reasons.append("salt_form_difference")
        if (g["stereo_flag_dataset"] == False).all() and (g["match_type"] == "exact_connectivity_match").any():
            reasons.append("stereochemistry_unspecified")
        if not reasons:
            reasons.append("unexplained")
        conflicts.append({
            "scope": "merged_sources", "dataset_row_index": r.dataset_row_index,
            "original_smiles": r.original_smiles, "dataset_label": r.dataset_label,
            "experimental_class": cls, "expected_class": expect,
            "n_records": r.n_records, "min_value_um": r.min_value_um,
            "sources": r.sources, "primary_reason": reasons[0],
            "all_reasons": "|".join(reasons),
            "external_ids": ",".join(sorted({str(x) for x in g["external_compound_id"]})),
            "assay_ids": "|".join(sorted({str(x) for x in g["assay_id"]})[:6]),
            "assay_groups": "|".join(sorted({str(x) for x in g["assay_group"]})),
            "measured_values": "|".join(str(x) for x in list(g["measured_value"].head(6))),
            "units": "|".join(sorted({str(x) for x in g["measured_unit"]})),
        })
    pd.DataFrame(conflicts).to_csv(OUT / "label_conflicts.csv", index=False)
    log(f"  label_conflicts.csv: {len(conflicts)}행")

    # 요약 json
    prim = pd.DataFrame(metrics_rows)
    key = prim[(prim["scope"] == "ALL|threshold=10uM") & (prim["unit"] == "compound")]
    summary = {
        "dataset_rows": int(len(ds)),
        "dataset_unique_compounds": int(ds["original_smiles"].nunique()),
        "primary_metrics": key.to_dict("records"),
    }
    with open(OUT / "stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])


# ---------------------------------------------------------------- 4. endpoints

def fetch_multi_target_activities(mids: list[str], chunk: int = 50) -> Path:
    """매칭된 ChEMBL molecule 의 모든 target activity (종말점 탐색용)."""
    out_dir = CACHE / "multitarget"
    out_dir.mkdir(exist_ok=True)
    for i in range(0, len(mids), chunk):
        part = mids[i:i + chunk]
        h = hashlib.md5(",".join(part).encode()).hexdigest()[:12]
        p = out_dir / f"act_{i:05d}_{h}.jsonl"
        if p.exists() and p.stat().st_size > 0:
            continue
        offset, total = 0, None
        while True:
            try:
                j = http_json(f"{CHM_BASE}/activity.json", {
                    "molecule_chembl_id__in": ",".join(part),
                    "pchembl_value__isnull": "false",
                    "only": "activity_id,molecule_chembl_id,target_chembl_id,standard_type,"
                            "standard_relation,standard_value,standard_units,pchembl_value,"
                            "assay_chembl_id,assay_type",
                    "limit": 1000, "offset": offset, "format": "json"})
            except Exception as exc:  # noqa: BLE001
                log(f"    멀티타깃 조회 실패 chunk {i}: {exc}")
                break
            rows = j.get("activities", [])
            total = j["page_meta"]["total_count"] if total is None else total
            append_jsonl(p, rows)
            offset += 1000
            if offset >= total or not rows:
                break
        if i % (chunk * 10) == 0:
            log(f"    멀티타깃 {i}/{len(mids)}")
    return out_dir


def fetch_target_meta(tids: list[str]) -> dict:
    path = CACHE / "target_meta.json"
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [t for t in tids if t not in known]
    for i in range(0, len(todo), 100):
        part = todo[i:i + 100]
        try:
            j = http_json(f"{CHM_BASE}/target.json", {
                "target_chembl_id__in": ",".join(part), "limit": 1000, "format": "json"})
        except Exception as exc:  # noqa: BLE001
            log(f"    target meta 실패: {exc}")
            continue
        for t in j.get("targets", []):
            known[t["target_chembl_id"]] = {
                "pref_name": t.get("pref_name"), "organism": t.get("organism"),
                "target_type": t.get("target_type")}
    path.write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")
    return known


def stage_endpoints() -> None:
    log("== stage endpoints ==")
    per_compound = pd.read_csv(OUT / "compound_level.csv", low_memory=False)
    matched = pd.read_csv(OUT / "matched_compounds.csv", low_memory=False)
    che = matched[(matched["external_database"] == "ChEMBL") &
                  (matched["match_type"] != "similarity_only")]
    mid_to_row: dict[str, int] = {}
    for r in che.itertuples(index=False):
        for mid in str(r.external_compound_id).split(","):
            mid = mid.strip()
            if mid.startswith("CHEMBL"):
                mid_to_row.setdefault(mid, int(r.dataset_row_index))
    mids = sorted(mid_to_row)
    log(f"  종말점 스캔 대상 화합물 {len(mids)}개")
    d = fetch_multi_target_activities(mids)

    acts = []
    for p in sorted(d.glob("act_*.jsonl")):
        acts.extend(read_jsonl(p))
    log(f"  멀티타깃 activity {len(acts)}건")
    if not acts:
        log("  ! 멀티타깃 activity 없음 → 종말점 스캔 생략")
        return
    adf = pd.DataFrame(acts)
    adf["dataset_row_index"] = adf["molecule_chembl_id"].map(mid_to_row)
    lab = per_compound.set_index("dataset_row_index")["dataset_label"]
    adf["dataset_label"] = adf["dataset_row_index"].map(lab)
    adf = adf.dropna(subset=["dataset_row_index", "dataset_label"])
    adf["pchembl"] = pd.to_numeric(adf["pchembl_value"], errors="coerce")
    adf = adf.dropna(subset=["pchembl"])

    rows = []
    for tid, g in adf.groupby("target_chembl_id"):
        comp = g.groupby("dataset_row_index").agg(
            pchembl_max=("pchembl", "max"), label=("dataset_label", "first"))
        if len(comp) < 30:
            continue
        y_true = np.where(comp["label"].values == LABEL_TOXIC, 1, 0)
        y_active = np.where(comp["pchembl_max"].values >= 5.0, 1, 0)
        m_fwd = _metrics(y_true, y_active)           # active -> toxic 라벨과 일치?
        m_rev = _metrics(y_true, 1 - y_active)       # active -> 비독성 라벨과 일치?
        best = m_fwd if m_fwd["balanced_accuracy"] >= m_rev["balanced_accuracy"] else m_rev
        rows.append({
            "target_chembl_id": tid, "n_compounds": len(comp),
            "n_active_at_10uM": int(y_active.sum()),
            "direction": "active_matches_label0" if best is m_fwd else "active_matches_label1",
            "balanced_accuracy": best["balanced_accuracy"], "accuracy": best["accuracy"],
            "sensitivity": best["sensitivity"], "specificity": best["specificity"],
            "agreement": best["accuracy"],
        })
    scan = pd.DataFrame(rows)
    if scan.empty:
        log("  ! 스캔 가능한 target 없음")
        return
    # 순열 기반 p-value (라벨 셔플)
    rng = np.random.default_rng(0)
    pvals = []
    for r in scan.itertuples(index=False):
        g = adf[adf["target_chembl_id"] == r.target_chembl_id]
        comp = g.groupby("dataset_row_index").agg(pchembl_max=("pchembl", "max"),
                                                  label=("dataset_label", "first"))
        y_true = np.where(comp["label"].values == LABEL_TOXIC, 1, 0)
        y_active = np.where(comp["pchembl_max"].values >= 5.0, 1, 0)
        obs = r.balanced_accuracy
        null = []
        y = y_true.copy()
        for _ in range(1000):
            rng.shuffle(y)
            m = max(_metrics(y, y_active)["balanced_accuracy"],
                    _metrics(y, 1 - y_active)["balanced_accuracy"])
            null.append(m)
        null = np.array(null)
        pvals.append(float((np.sum(null >= obs) + 1) / (len(null) + 1)))
    scan["perm_p_balanced_accuracy"] = pvals
    # Benjamini-Hochberg
    order = np.argsort(scan["perm_p_balanced_accuracy"].values)
    m_test = len(scan)
    q = np.empty(m_test)
    prev = 1.0
    for rank, ix in enumerate(order[::-1]):
        i = m_test - rank
        val = scan["perm_p_balanced_accuracy"].values[ix] * m_test / i
        prev = min(prev, val)
        q[ix] = prev
    scan["bh_q_value"] = q
    meta = fetch_target_meta(scan["target_chembl_id"].tolist())
    scan["target_pref_name"] = scan["target_chembl_id"].map(lambda t: (meta.get(t) or {}).get("pref_name"))
    scan["target_organism"] = scan["target_chembl_id"].map(lambda t: (meta.get(t) or {}).get("organism"))
    scan["target_type"] = scan["target_chembl_id"].map(lambda t: (meta.get(t) or {}).get("target_type"))
    scan["is_herg"] = scan["target_chembl_id"] == CHEMBL_HERG_TARGET
    scan = scan.sort_values("balanced_accuracy", ascending=False)
    scan.to_csv(OUT / "endpoint_scan.csv", index=False)
    log(f"  endpoint_scan.csv: {len(scan)} targets")
    log(scan.head(15).to_string(index=False))


def stage_alt_datasets() -> None:
    """Tox21 12-target 패널 + ClinTox 로 'hERG 이외 종말점'과의 적합도 비교.

    목적: hERG가 지지되더라도 그보다 대상 라벨을 더 잘 설명하는 종말점이 있는지
    독립 공개 데이터로 탐색한다. (사용자 요구: 반증적 종말점 탐색)
    """
    log("== stage alt_datasets ==")
    sources = {
        "Tox21": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
                  CACHE / "tox21.csv.gz"),
        "ClinTox": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
                    CACHE / "clintox.csv.gz"),
    }
    ds = pd.read_csv(OUT / "dataset_keys.csv", low_memory=False)
    ds_map_ik = dict(zip(ds["inchikey"], ds["dataset_row_index"]))
    ds_map_conn = dict(zip(ds["connectivity_key"], ds["dataset_row_index"]))
    ds_label = dict(zip(ds["dataset_row_index"], ds["dataset_label"]))

    rows = []
    records_log = []
    for name, (url, path) in sources.items():
        if not path.exists():
            try:
                txt = _request(url, timeout=180).content
                path.write_bytes(txt)
                log(f"  {name} 다운로드 ({len(txt)} bytes)")
            except Exception as exc:  # noqa: BLE001
                log(f"  ! {name} 다운로드 실패: {exc}")
                records_log.append({"source": name, "url": url, "status": f"ERROR {exc}"})
                continue
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        records_log.append({"source": name, "url": url, "status": "ok", "rows": int(len(df)),
                            "columns": list(df.columns)})
        smi_col = "smiles" if "smiles" in df.columns else df.columns[0]
        uniq = sorted({s for s in df[smi_col].dropna().unique() if isinstance(s, str)})
        keymap = {s: mol_keys(s) or {} for s in uniq}
        df["_ik"] = [keymap.get(s, {}).get("inchikey", "") if isinstance(s, str) else "" for s in df[smi_col]]
        df["_conn"] = [keymap.get(s, {}).get("connectivity_key", "") if isinstance(s, str) else "" for s in df[smi_col]]
        df["_row"] = df["_ik"].map(ds_map_ik)
        df["_row"] = df["_row"].fillna(df["_conn"].map(ds_map_conn))
        m = df.dropna(subset=["_row"])
        log(f"  {name}: {len(df)} 화합물 중 {m['_row'].nunique()}개가 대상 데이터와 구조 매칭")
        end_cols = [c for c in df.columns if not c.startswith("_") and c != smi_col
                    and pd.api.types.is_numeric_dtype(df[c])]
        for col in end_cols:
            sub = m.dropna(subset=[col])
            if sub["_row"].nunique() < 15:
                continue
            comp = sub.groupby("_row")[col].max()
            y_alt = np.where(comp.values >= 0.5, 1, 0)
            if len(set(y_alt)) < 2:
                continue
            y_true = np.array([1 if ds_label[r] == LABEL_TOXIC else 0 for r in comp.index])
            m_fwd = _metrics(y_true, y_alt)            # alt active -> dataset toxic
            m_rev = _metrics(y_true, 1 - y_alt)        # alt active -> dataset non-toxic
            best = m_fwd if m_fwd["balanced_accuracy"] >= m_rev["balanced_accuracy"] else m_rev
            p = permutation_test(y_true, y_alt if best is m_fwd else 1 - y_alt, n_perm=1000)["p_balanced_accuracy"]
            rows.append({
                "dataset": name, "endpoint": col, "n_compounds": int(comp.shape[0]),
                "n_active": int(y_alt.sum()),
                "direction": "alt_active=dataset_label0(toxic)" if best is m_fwd
                             else "alt_active=dataset_label1(non_toxic)",
                "balanced_accuracy": best["balanced_accuracy"], "accuracy": best["accuracy"],
                "sensitivity": best["sensitivity"], "specificity": best["specificity"],
                "perm_p": p,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        log("  ! 대체 종말점 매칭 표본 부족")
    else:
        order = np.argsort(out["perm_p"].values)
        n = len(out)
        q = np.empty(n)
        prev = 1.0
        for rank, ix in enumerate(order[::-1]):
            i = n - rank
            prev = min(prev, out["perm_p"].values[ix] * n / i)
            q[ix] = prev
        out["bh_q_value"] = q
        out = out.sort_values("balanced_accuracy", ascending=False)
        out.to_csv(OUT / "alt_endpoint_scan.csv", index=False)
        log(out.head(12).to_string(index=False))
    (CACHE / "alt_datasets_log.json").write_text(
        json.dumps(records_log, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_props() -> None:
    """보조 구조특성 분석 (§9): 라벨 그룹 간 차이 + 매칭 서브셋에서 실험 클래스와의 관계.

    화학적 특성만으로 종말점을 확정하지 않는다는 전제 하에, hERG 가설과
    일관되는 방향의 보조 증거가 있는지만 확인한다.
    """
    log("== stage props ==")
    from scipy import stats

    ds = pd.read_csv(OUT / "dataset_keys.csv", low_memory=False)
    base = load_dataset()
    ds = ds.merge(base[["dataset_row_index", "MolWt", "clogp", "sa_score", "qed"]],
                  on="dataset_row_index", how="left")

    # RDKit 보조 특성: TPSA, 염기성 아민 개수, HBD/HBA
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    def extras(smi):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            return (np.nan, np.nan, np.nan, np.nan)
        basic = len(m.GetSubstructMatches(Chem.MolFromSmarts(
            "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$(Nc);!$(NS=O);!$(N[N,O])]")))
        return (rdMolDescriptors.CalcTPSA(m), basic,
                rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m))

    ex = [extras(s) for s in ds["canonical_isomeric"]]
    ds["TPSA"] = [e[0] for e in ex]
    ds["n_basic_amine"] = [e[1] for e in ex]
    ds["HBD"] = [e[2] for e in ex]
    ds["HBA"] = [e[3] for e in ex]
    ds["toxic"] = (ds["dataset_label"] == LABEL_TOXIC).astype(int)

    matched = pd.read_csv(OUT / "compound_level.csv", low_memory=False)
    matched = matched[matched["class_10um"].isin(["blocker", "non_blocker"])]
    ds["exp_class"] = ds["dataset_row_index"].map(
        dict(zip(matched["dataset_row_index"], matched["class_10um"])))

    props = ["MolWt", "clogp", "sa_score", "qed", "TPSA", "n_basic_amine", "HBD", "HBA"]
    rows = []
    for p in props:
        a = ds[ds["toxic"] == 1][p].dropna().values   # label=0 (독성)
        b = ds[ds["toxic"] == 0][p].dropna().values   # label=1 (비독성)
        if len(a) < 10 or len(b) < 10:
            continue
        u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank AUC (독성 그룹이 더 큰 값일 확률)
        allv = np.concatenate([a, b])
        r = stats.rankdata(allv)
        auc = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        # 매칭 서브셋에서 실험 클래스와의 관계
        sub = ds[ds["exp_class"].isin(["blocker", "non_blocker"])]
        e1 = sub[sub["exp_class"] == "blocker"][p].dropna().values
        e0 = sub[sub["exp_class"] == "non_blocker"][p].dropna().values
        if len(e1) >= 10 and len(e0) >= 10:
            u2, p2 = stats.mannwhitneyu(e1, e0, alternative="two-sided")
            allv2 = np.concatenate([e1, e0])
            r2 = stats.rankdata(allv2)
            auc2 = (r2[:len(e1)].sum() - len(e1) * (len(e1) + 1) / 2) / (len(e1) * len(e0))
        else:
            p2, auc2 = np.nan, np.nan
        rows.append({
            "property": p,
            "n_toxic_label0": len(a), "n_nontoxic_label1": len(b),
            "mean_label0": float(np.mean(a)), "mean_label1": float(np.mean(b)),
            "median_label0": float(np.median(a)), "median_label1": float(np.median(b)),
            "rank_auc_toxic_higher": float(auc), "mannwhitney_p": float(pval),
            "n_experimental": int(len(e1) + len(e0)),
            "rank_auc_blocker_higher": float(auc2) if np.isfinite(auc2) else np.nan,
            "mannwhitney_p_experimental": float(p2) if np.isfinite(p2) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "property_analysis.csv", index=False)
    ds.to_csv(OUT / "dataset_with_properties.csv", index=False)
    log(out.to_string(index=False))


# ---------------------------------------------------------------- 5. figures

def stage_figures() -> None:
    log("== stage figures ==")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = pd.read_csv(OUT / "metrics.csv")
    per_compound = pd.read_csv(OUT / "compound_level.csv", low_memory=False)
    matched = pd.read_csv(OUT / "matched_compounds.csv", low_memory=False)

    # 1) 혼동행렬 (주 분석)
    prim = metrics[(metrics["scope"] == "ALL|threshold=10uM") & (metrics["unit"] == "compound")]
    if not prim.empty:
        r = prim.iloc[0]
        cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
        fig, ax = plt.subplots(figsize=(4.6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["blocker", "non_blocker"], yticklabels=["label=0", "label=1"])
        ax.set_xlabel("experimental (ChEMBL+PubChem, 10 uM)")
        ax.set_ylabel("dataset label")
        ax.set_title(f"Confusion matrix (n={int(r['n'])})")
        fig.tight_layout()
        fig.savefig(FIG / "fig1_confusion_matrix.png", dpi=160)
        plt.close(fig)

    # 2) 임계값 민감도
    sub = metrics[(metrics["scope"].str.startswith("ALL|threshold")) & (metrics["unit"] == "compound")].copy()
    if not sub.empty:
        sub["th"] = sub["scope"].str.extract(r"threshold=([\d.]+)uM").astype(float)
        sub = sub.sort_values("th")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sub["th"], sub["balanced_accuracy"], "o-", label="balanced accuracy")
        ax.plot(sub["th"], sub["agreement"], "s--", label="agreement")
        ax.plot(sub["th"], sub["sensitivity"], "^:", label="sensitivity (label=0)")
        ax.plot(sub["th"], sub["specificity"], "v:", label="specificity (label=1)")
        ax.set_xscale("log")
        ax.set_xlabel("IC50/AC50 threshold (uM)")
        ax.set_ylabel("metric")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.set_title("Threshold sensitivity (compound level, all sources)")
        fig.tight_layout()
        fig.savefig(FIG / "fig2_threshold_sensitivity.png", dpi=160)
        plt.close(fig)

    # 3) 소스별 일치율
    sub = metrics[(metrics["scope"].str.endswith("threshold=10uM")) & (metrics["unit"] == "compound")].copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["agreement"], width=0.4, label="agreement")
        ax.bar(x + 0.2, sub["balanced_accuracy"], width=0.4, label="balanced accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("|")[0] for s in sub["scope"]], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.legend(fontsize=8)
        ax.set_title("Agreement by source (10 uM, compound level)")
        fig.tight_layout()
        fig.savefig(FIG / "fig3_agreement_by_source.png", dpi=160)
        plt.close(fig)

    # 4) IC50 분포 (label 별)
    pot = matched[(matched["measure_family"] == "potency") &
                  (matched["match_type"] != "similarity_only")].copy()
    pot["v"] = pd.to_numeric(pot["measured_value_um"], errors="coerce")
    pot = pot.dropna(subset=["v"])
    pot = pot[(pot["v"] > 0)]
    if not pot.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bins = np.logspace(np.log10(max(pot["v"].min(), 1e-4)), np.log10(pot["v"].max()), 40)
        for lab, col in [(0, "tab:red"), (1, "tab:blue")]:
            ax.hist(pot[pot["dataset_label"] == lab]["v"], bins=bins, alpha=0.55,
                    label=f"label={lab}", color=col)
        ax.axvline(10, color="k", ls="--", lw=1)
        ax.text(10, ax.get_ylim()[1] * 0.95, " 10 uM", va="top", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("measured potency (uM, record level)")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title("Potency distribution by dataset label")
        fig.tight_layout()
        fig.savefig(FIG / "fig4_potency_distribution.png", dpi=160)
        plt.close(fig)

    # 5) 종말점 스캔
    ep = OUT / "endpoint_scan.csv"
    if ep.exists():
        scan = pd.read_csv(ep)
        top = scan.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 6))
        colors = ["tab:red" if h else "tab:gray" for h in top["is_herg"]]
        ax.barh(range(len(top)), top["balanced_accuracy"], color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f"{n[:38]} ({t})" for n, t in
                            zip(top["target_pref_name"].fillna("NA"), top["target_chembl_id"])],
                           fontsize=7)
        ax.axvline(0.5, color="k", ls=":", lw=1)
        ax.set_xlabel("balanced accuracy (active pChEMBL>=5 vs dataset label)")
        ax.set_title("Endpoint scan: best-fitting targets (red = hERG)")
        fig.tight_layout()
        fig.savefig(FIG / "fig5_endpoint_scan.png", dpi=160)
        plt.close(fig)

    # 6) 매칭 요약
    ms = json.loads((OUT / "match_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(ms["match_type_counts_compounds"].keys())
    vals = [ms["match_type_counts_compounds"][k] for k in keys]
    ax.bar(keys, vals, color="tab:blue")
    ax.set_ylabel("compounds")
    ax.set_title("Match type distribution (compound level)")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_match_summary.png", dpi=160)
    plt.close(fig)
    # 7) 대체 종말점 스캔 (Tox21 / ClinTox)
    alt = OUT / "alt_endpoint_scan.csv"
    if alt.exists():
        a = pd.read_csv(alt)
        if not a.empty:
            a = a.head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(a)), a["balanced_accuracy"], color="tab:green")
            ax.set_yticks(range(len(a)))
            ax.set_yticklabels([f"{d}:{e}" for d, e in zip(a["dataset"], a["endpoint"])], fontsize=7)
            ax.axvline(0.5, color="k", ls=":", lw=1)
            ax.set_xlabel("balanced accuracy (vs dataset label)")
            ax.set_title("Alternative toxicity endpoints (Tox21 / ClinTox)")
            fig.tight_layout()
            fig.savefig(FIG / "fig7_alt_endpoint_scan.png", dpi=160)
            plt.close(fig)

    # 8) 라벨링 규칙 역추적
    rev = OUT / "labeling_rule_reverse.csv"
    if rev.exists():
        r = pd.read_csv(rev)
        pot_only = r[r["aggregation"].str.contains("exact_")]
        if not pot_only.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for agg, g in pot_only.groupby("aggregation"):
                g = g.sort_values("threshold_um")
                ax.plot(g["threshold_um"], g["balanced_accuracy"], "o-", label=agg)
            ax.set_xscale("log")
            ax.set_xlabel("threshold (uM)")
            ax.set_ylabel("balanced accuracy vs dataset label")
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7)
            ax.set_title("Reverse-engineering the labeling rule (ChEMBL hERG)")
            fig.tight_layout()
            fig.savefig(FIG / "fig8_labeling_rule_reverse.png", dpi=160)
            plt.close(fig)
    log("  figures 생성 완료")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="hERG 라벨 검증 파이프라인")
    ap.add_argument("--stage", required=True,
                    choices=["catalog", "fetch", "match", "stats", "endpoints", "alt",
                             "props", "figures", "all"])
    args = ap.parse_args()
    stages = ["catalog", "fetch", "match", "stats", "endpoints", "alt", "props", "figures"] \
        if args.stage == "all" else [args.stage]
    for s in stages:
        {"catalog": stage_catalog, "fetch": stage_fetch, "match": stage_match,
         "stats": stage_stats, "endpoints": stage_endpoints, "alt": stage_alt_datasets,
         "props": stage_props, "figures": stage_figures}[s]()


if __name__ == "__main__":
    main()
