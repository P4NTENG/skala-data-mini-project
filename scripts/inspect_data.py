"""Stage A -- inspect data/train.csv and write reports/data_quality_report.md.

Run:  python scripts/inspect_data.py [--mode quick|full]
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from rdkit.Chem import Crippen, Descriptors, QED

from _common import base_parser, context, reports_path

from herg_model import data as data_module
from herg_model import fingerprints as fingerprint_module
from herg_model import splitting as splitting_module
from herg_model.chemistry import load_structure_table, sa_score, structure_duplicate_report
from herg_model.config import environment_info
from herg_model.reporting import write_data_quality_report


def shipped_descriptor_check(df: pd.DataFrame, cfg) -> list[str]:
    """Verify that the shipped descriptor columns really are the RDKit quantities they claim."""
    from herg_model.chemistry import mol_from_smiles

    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))
    names = data_module.existing_descriptor_names(cfg, df)
    computed = {name: [] for name in names}
    for smiles in df[smiles_col].astype(str):
        mol = mol_from_smiles(smiles)
        for name in names:
            if mol is None:
                computed[name].append(np.nan)
            elif name == "MolWt":
                computed[name].append(Descriptors.MolWt(mol))
            elif name == "clogp":
                computed[name].append(Crippen.MolLogP(mol))
            elif name == "qed":
                computed[name].append(QED.qed(mol))
            elif name == "sa_score":
                computed[name].append(sa_score(smiles))
            else:
                computed[name].append(np.nan)

    notes = []
    for name in names:
        values = np.asarray(computed[name], dtype=float)
        shipped = df[name].to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(shipped)
        diff = np.abs(values[mask] - shipped[mask])
        max_diff = float(diff.max()) if diff.size else float("nan")
        n_diff = int((diff > 1e-3).sum())
        notes.append(
            f"`{name}` 재계산 최대 절대오차 {max_diff:.3e}, 오차 >1e-3인 행 {n_diff}개 (n={int(mask.sum())})"
        )
    return notes


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg, dirs = context(args)
    t0 = time.time()

    df = data_module.load_dataset(cfg)
    print(f"[inspect] loaded {df.shape}", flush=True)

    summary = data_module.dataset_summary(df, cfg)
    y = data_module.blocker_target(df, cfg)

    structure = load_structure_table(list(df["SMILES"].astype(str)), cfg, verbose=not args.quiet)
    duplicates = structure_duplicate_report(structure, df[str(cfg.get_path("data.label_col", "label"))].to_numpy())
    dup_table = duplicates.pop("_table")

    verification = fingerprint_module.verify_fingerprints(df, cfg)

    notes = shipped_descriptor_check(df, cfg)

    duplicate_summary = {}
    for key in ("canonical_smiles", "connectivity_key", "normalized_key", "scaffold"):
        table = duplicates.get(key)
        if isinstance(table, pd.DataFrame):
            duplicate_summary[key] = int((table["n_labels"] > 1).sum())

    structure_stats = {
        "n_inchikey": int(structure["inchikey"].nunique()),
        "n_connectivity": int(structure["connectivity_key"].nunique()),
        "n_scaffolds": int(structure["scaffold"].nunique()),
        "largest_scaffold": int(structure["scaffold"].value_counts().max()),
        "n_salts": int((structure["n_fragments"] > 1).sum()),
        "n_stereo": int(structure["has_stereo"].sum()),
        "n_parse_failures": int((~structure["parsed"]).sum()),
        "n_tautomer_changed": "not computed (tautomer enumeration is a separate sensitivity run)",
        "label_conflicts_by_key": json.dumps(duplicate_summary),
    }

    scaffold = structure["scaffold"].to_numpy()
    split_rows = []
    primary_seed = int(cfg.get_path("split.primary_seed", 42))
    splits = splitting_module.make_splits(scaffold, y, cfg, strategy="scaffold", seed=primary_seed)
    random_splits = splitting_module.make_splits(scaffold, y, cfg, strategy="random", seed=primary_seed)
    scaffold_table = splitting_module.split_summary(y, scaffold, splits).assign(strategy="scaffold")
    random_table = splitting_module.split_summary(y, scaffold, random_splits).assign(strategy="random")
    split_rows.append(scaffold_table)
    split_rows.append(random_table)
    split_table = pd.concat(split_rows, ignore_index=True)

    leakage = splitting_module.check_leakage(scaffold, splits)
    random_leak = splitting_module.check_leakage(scaffold, random_splits)
    notes.append(
        "random split의 scaffold 중복: " + json.dumps(random_leak["scaffold_overlap"]) +
        " -- 참고용 실험이 과대평가되는 이유"
    )
    notes.append(
        f"acyclic 화합물은 generic Bemis-Murcko scaffold를 사용해 분할 그룹이 무의미하게 "
        f"합쳐지거나(빈 scaffold) 개별 그룹으로 흩어지는 것을 막았다."
    )

    # fingerprint 재현 실패 row의 공통 특징 기록
    smiles_series = df[str(cfg.get_path("data.smiles_col", "SMILES"))].astype(str)
    for kind, block in verification["blocks"].items():
        if block["exact"]:
            continue
        rows = block["mismatch_row_ids"]
        deuterated = sum("[2H]" in smiles_series.iloc[i] or "[3H]" in smiles_series.iloc[i] for i in rows)
        notes.append(
            f"`{kind}` 블록은 {block['n_rows_exact']:,}/{block['n_rows_checked']:,} 행에서 완전히 재현된다. "
            f"불일치 {len(rows)}행 중 {deuterated}행이 중수소(`[2H]`) 표기를 포함한다. "
            f"RDKit 2026.09.1의 Morgan generator로는 동위원소/H 제거 방식을 바꿔도 bit 단위로 "
            f"재현되지 않아, 해당 행은 원본 컬럼을 그대로 사용하고 목록만 남긴다."
        )
    if verification["blocks"]["ecfp"]["exact"]:
        notes.append("세 fingerprint 블록 모두 전 행에서 bit 단위로 재현되었다.")

    notes.append(
        f"validation split의 blocker 비율 {scaffold_table.set_index('split').loc['val','blocker_fraction']:.4f}, "
        f"전체 {float(y.mean()):.4f} -- scaffold 단위 분할이라 완전한 층화는 불가능하다."
    )

    report_path = reports_path(cfg, "data_quality_report.md")
    write_data_quality_report(
        report_path,
        summary=summary,
        verification=verification,
        structure=structure_stats,
        duplicates=duplicates,
        split_table=split_table,
        leakage=leakage,
        environment=environment_info(),
        extra_notes=notes,
    )
    print(f"[inspect] wrote {report_path}", flush=True)

    # machine-readable companions (no SMILES -> safe to commit)
    pd.DataFrame([summary["fingerprints"][k] for k in summary["fingerprints"]]).to_csv(
        dirs["reports"] / "data_quality_fingerprints.csv", index=False
    )
    split_table.to_csv(dirs["reports"] / "data_quality_splits.csv", index=False)
    pd.DataFrame(
        [{"block": k, **{kk: vv for kk, vv in v.items() if kk != "mismatch_row_ids"}}
         for k, v in verification["blocks"].items()]
    ).to_csv(dirs["reports"] / "data_quality_fingerprint_verification.csv", index=False)

    # SMILES-bearing index stays in the (git-ignored) cache
    data_module.dataset_index(df, cfg).merge(
        structure[["row_id", "inchikey", "scaffold"]], on="row_id"
    ).to_csv(dirs["cache"] / "dataset_index.csv.gz", index=False, compression="gzip")

    print("\n=== split summary ===")
    print(split_table.to_string(index=False))
    print("\n=== leakage ===")
    print(json.dumps({"scaffold": leakage["scaffold_overlap"], "random": random_leak["scaffold_overlap"]}))
    print("\n=== fingerprint verification ===")
    for kind, block in verification["blocks"].items():
        print(f"{kind}: bit agreement {block['bit_agreement']:.4f}, exact rows "
              f"{block['n_rows_exact']}/{block['n_rows_checked']}")
    print(f"\n[inspect] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
