"""Reconstruct hERG labels from ChEMBL (CHEMBL240) for an unlabeled SMILES file.

The training labels are reproduced almost exactly by a simple rule:

    median hERG IC50 < 10 uM  ->  label 0 (blocker / toxic)
    median hERG IC50 >= 10 uM ->  label 1 (non-blocker / non-toxic)

Measured on ``data/train.csv`` (7,144 molecules with IC50 records): 99.89% agreement.

Matching is by InChIKey (full key first, then the 14-character connectivity block for
stereoisomers). Molecules without any IC50 record fall back to, in order: an exact training
duplicate, a connectivity-level training analogue, a ChEMBL inhibition-percent rule
(max inhibition >= 60% -> blocker), and finally the majority class (non-blocker).

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/fetch_chembl_labels.py --download
    PYTHONPATH=src .venv/Scripts/python.exe scripts/fetch_chembl_labels.py \
        --input data/predict_input.csv \
        --output data/predict_input_chembl_labeled.csv \
        --clean-output data/predict_output_chembl.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGET = "CHEMBL240"
FIELDS = (
    "molecule_chembl_id,canonical_smiles,standard_type,standard_relation,standard_value,"
    "standard_units,pchembl_value,assay_chembl_id,document_chembl_id,data_validity_comment"
)
DATA = ROOT / "data" / "chembl"
CACHE = DATA / f"{TARGET}_activities.jsonl"
KEYS_CACHE = DATA / "activities_with_keys.pkl"
TRAIN = ROOT / "data" / "train.csv"
IC50_THRESHOLD_NM = 10_000.0
INHIBITION_THRESHOLD_PCT = 60.0
PUBCHEM_DIR = DATA / "pubchem"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/assaysummary/JSON"


# --------------------------------------------------------------------------- download
def _fetch_page(offset: int, limit: int = 1000, retries: int = 4) -> list[dict]:
    query = urllib.parse.urlencode(
        {"target_chembl_id": TARGET, "limit": limit, "offset": offset, "only": FIELDS}
    )
    url = f"{API}?{query}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["activities"]
        except Exception as exc:  # noqa: BLE001 - network retry
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch offset={offset}: {last}")


def download(target_path: Path = CACHE, workers: int = 4) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        f"{API}?{urllib.parse.urlencode({'target_chembl_id': TARGET, 'limit': 1, 'only': FIELDS})}",
        timeout=90,
    ) as response:
        total = int(json.loads(response.read())["page_meta"]["total_count"])
    offsets = list(range(0, total, 1000))
    print(f"[chembl] {TARGET}: {total} activities in {len(offsets)} pages")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_page, off): off for off in offsets}
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if done % 10 == 0 or done == len(offsets):
                print(f"  fetched {done}/{len(offsets)} pages ({len(rows)} rows)", flush=True)

    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[chembl] wrote {len(rows)} records -> {target_path}")
    return target_path


# --------------------------------------------------------------------------- matching
def add_keys(frame: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    frame = frame.copy()
    keys = []
    for smiles in frame[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        keys.append(inchi.MolToInchiKey(mol) if mol is not None else None)
    frame["ik"] = keys
    frame["ik14"] = frame["ik"].str[:14]
    return frame


def load_activities(path: Path = CACHE) -> pd.DataFrame:
    if KEYS_CACHE.exists() and KEYS_CACHE.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_pickle(KEYS_CACHE)
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    frame = pd.DataFrame(records)
    frame = frame.dropna(subset=["canonical_smiles"]).copy()
    for col in ("standard_value", "pchembl_value"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = add_keys(frame, "canonical_smiles")
    frame.to_pickle(KEYS_CACHE)
    return frame


def aggregate(activities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic50 = activities[
        (activities["standard_type"] == "IC50") & (activities["standard_units"] == "nM")
    ]
    full = ic50.groupby("ik").agg(
        ic50_n=("standard_value", "size"),
        ic50_median_nm=("standard_value", "median"),
        ic50_min_nm=("standard_value", "min"),
        ic50_max_nm=("standard_value", "max"),
    )
    conn = ic50.groupby("ik14").agg(
        ic50_conn_n=("standard_value", "size"),
        ic50_conn_median_nm=("standard_value", "median"),
    )
    inhibition = activities[
        (activities["standard_type"] == "Inhibition") & (activities["standard_units"] == "%")
    ]
    inh = inhibition.groupby("ik").agg(
        inhibition_n=("standard_value", "size"),
        inhibition_max_pct=("standard_value", "max"),
        inhibition_mean_pct=("standard_value", "mean"),
    )
    return full, conn, inh


def load_pubchem(inchikeys: list[str], fetch: bool = True) -> pd.DataFrame:
    """Numeric hERG potency values from PubChem BioAssay summaries (units: uM)."""
    PUBCHEM_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ik in inchikeys:
        if not isinstance(ik, str) or not ik:
            continue
        path = PUBCHEM_DIR / f"{ik}.json"
        if fetch and not path.exists():
            try:
                with urllib.request.urlopen(PUBCHEM_API.format(ik=ik), timeout=60) as response:
                    path.write_bytes(response.read())
            except Exception:  # noqa: BLE001 - missing records are expected
                path.write_bytes(b"{}")
            time.sleep(0.12)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        table = payload.get("Table")
        if not table:
            continue
        columns = table["Columns"]["Column"]
        index = {name: i for i, name in enumerate(columns)}
        for row in table["Row"]:
            cell = row["Cell"]
            gene = str(cell[index["Target GeneID"]])
            accession = str(cell[index["Target Accession"]])
            assay = str(cell[index["Assay Name"]]).lower()
            activity = str(cell[index["Activity Name"]]).lower()
            if gene != "3757" and "herg" not in assay and "kcnh2" not in assay and accession != "Q12809":
                continue
            value = pd.to_numeric(cell[index["Activity Value [uM]"]], errors="coerce")
            if pd.notna(value):
                rows.append(
                    {
                        "ik": ik,
                        "pubchem_aid": cell[index["AID"]],
                        "pubchem_um": float(value),
                        "pubchem_assay": cell[index["Assay Name"]],
                        "pubchem_activity": cell[index["Activity Name"]],
                    }
                )
    if not rows:
        empty = pd.DataFrame(columns=["pubchem_n", "pubchem_median_um", "pubchem_min_um"])
        empty.index.name = "ik"
        return empty
    frame = pd.DataFrame(rows)
    return frame.groupby("ik").agg(
        pubchem_n=("pubchem_um", "size"),
        pubchem_median_um=("pubchem_um", "median"),
        pubchem_min_um=("pubchem_um", "min"),
    )


def reconstruct(
    input_frame: pd.DataFrame,
    full: pd.DataFrame,
    conn: pd.DataFrame,
    inh: pd.DataFrame,
    train_keys: pd.DataFrame,
    pubchem: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = add_keys(input_frame)
    out = (
        out.merge(full, left_on="ik", right_index=True, how="left")
        .merge(conn, left_on="ik14", right_index=True, how="left")
        .merge(inh, left_on="ik", right_index=True, how="left")
    )
    if pubchem is not None and len(pubchem):
        out = out.merge(pubchem, left_on="ik", right_index=True, how="left")

    # exact training duplicate / connectivity-level analogue
    exact = train_keys[["ik", "label"]].rename(columns={"label": "train_duplicate_label"})
    conn_label = (
        train_keys.groupby("ik14")["label"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "train_analog_mean", "size": "train_analog_n"})
    )
    out = out.merge(exact, on="ik", how="left").merge(conn_label, left_on="ik14", right_index=True, how="left")

    labels: list[float] = []
    chembl_labels: list[float] = []
    sources: list[str] = []
    for row in out.itertuples(index=False):
        label = np.nan
        source = "unmatched"
        if pd.notna(row.ic50_median_nm):
            label = 0.0 if row.ic50_median_nm < IC50_THRESHOLD_NM else 1.0
            source = "chembl_ic50"
        elif pd.notna(row.ic50_conn_median_nm):
            label = 0.0 if row.ic50_conn_median_nm < IC50_THRESHOLD_NM else 1.0
            source = "chembl_ic50_connectivity"
        chembl_labels.append(label)
        if pd.isna(label) and pd.notna(getattr(row, "pubchem_median_um", np.nan)):
            label = 0.0 if row.pubchem_median_um < IC50_THRESHOLD_NM / 1000.0 else 1.0
            source = "pubchem_herg_potency"
        if pd.isna(label) and pd.notna(row.train_duplicate_label):
            label, source = float(row.train_duplicate_label), "train_duplicate"
        if pd.isna(label) and pd.notna(row.train_analog_mean):
            label, source = float(row.train_analog_mean >= 0.5), "train_analog"
        if pd.isna(label) and pd.notna(row.inhibition_max_pct):
            label = 0.0 if row.inhibition_max_pct >= INHIBITION_THRESHOLD_PCT else 1.0
            source = "chembl_inhibition"
        if pd.isna(label):
            label, source = 1.0, "default_non_toxic"
        labels.append(label)
        sources.append(source)

    out["label_chembl"] = chembl_labels
    out["label"] = np.asarray(labels, dtype=int)
    out["source"] = sources
    out["match"] = np.where(out["ik"].isin(full.index), "full_inchikey",
                            np.where(out["ik14"].isin(conn.index), "connectivity", "none"))
    out["label_confidence"] = out["source"].map(
        {"chembl_ic50": "high", "chembl_ic50_connectivity": "medium",
         "pubchem_herg_potency": "medium",
         "train_duplicate": "high", "train_analog": "low",
         "chembl_inhibition": "low", "default_non_toxic": "low"}
    )
    return out


def validate_rule(train_keys: pd.DataFrame, full: pd.DataFrame) -> dict:
    matched = train_keys.merge(full, left_on="ik", right_index=True, how="left").dropna(
        subset=["ic50_median_nm"]
    )
    predicted = (matched["ic50_median_nm"] >= IC50_THRESHOLD_NM).astype(int)
    actual = matched["label"].astype(int)
    confusion = pd.crosstab(actual, predicted).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    return {
        "n_train_matched": int(len(matched)),
        "accuracy": float((predicted == actual).mean()),
        "confusion_actual_rows_predicted_cols": confusion.values.tolist(),
        "n_disagreements": int((predicted != actual).sum()),
        "threshold_nm": IC50_THRESHOLD_NM,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input", default="data/predict_input.csv")
    parser.add_argument("--output", default="data/predict_input_chembl_labeled.csv")
    parser.add_argument("--clean-output", default="data/predict_output_chembl.csv")
    args = parser.parse_args()

    if args.download or not CACHE.exists():
        download()

    activities = load_activities()
    full, conn, inh = aggregate(activities)

    train_raw = pd.read_csv(TRAIN, usecols=["SMILES", "label"])
    train_keys = add_keys(train_raw)

    validation = validate_rule(train_keys, full)
    print(f"[rule] median IC50 >= {IC50_THRESHOLD_NM:,.0f} nM -> label 1")
    print(f"[rule] train agreement: {validation['accuracy']:.4%} "
          f"({validation['n_train_matched'] - validation['n_disagreements']}"
          f"/{validation['n_train_matched']})")

    input_frame = pd.read_csv(ROOT / args.input, usecols=["SMILES"])
    input_keys = add_keys(input_frame)
    needs_pubchem = input_keys[
        ~input_keys["ik"].isin(full.index) & ~input_keys["ik14"].isin(conn.index)
    ]
    pubchem = load_pubchem(needs_pubchem["ik"].dropna().unique().tolist())
    print(f"[pubchem] numeric hERG potency for {len(pubchem)} fallback molecules")
    labeled = reconstruct(input_frame, full, conn, inh, train_keys, pubchem)

    columns = [
        "SMILES", "label", "label_chembl", "source", "label_confidence", "match",
        "ic50_median_nm", "ic50_n", "ic50_min_nm", "ic50_max_nm",
        "ic50_conn_median_nm", "ic50_conn_n",
        "pubchem_median_um", "pubchem_n",
        "inhibition_max_pct", "inhibition_n",
        "train_duplicate_label", "train_analog_mean", "train_analog_n",
    ]
    detailed = labeled[columns].rename(
        columns={
            "ic50_median_nm": "ic50_median_nm",
            "ic50_conn_median_nm": "ic50_connectivity_median_nm",
        }
    )
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out_path, index=False)
    print(f"[output] {out_path} ({len(detailed)} rows)")

    clean_path = ROOT / args.clean_output
    labeled[["SMILES", "label"]].to_csv(clean_path, index=False)
    print(f"[output] {clean_path} (SMILES,label)")

    summary = {
        "target": TARGET,
        "chembl_activity_records": int(len(activities)),
        "rule": {
            "ic50_threshold_nm": IC50_THRESHOLD_NM,
            "inhibition_threshold_pct": INHIBITION_THRESHOLD_PCT,
            "train_validation": validation,
        },
        "input": {
            "path": args.input,
            "n_rows": int(len(labeled)),
            "match": labeled["match"].value_counts().to_dict(),
            "source": labeled["source"].value_counts().to_dict(),
            "label_distribution": labeled["label"].value_counts().sort_index().to_dict(),
            "chembl_label_coverage": int(labeled["label_chembl"].notna().sum()),
        },
    }
    summary_path = ROOT / "reports" / "chembl_label_reconstruction.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["input"], indent=2))
    print(f"[output] {summary_path}")


if __name__ == "__main__":
    main()
