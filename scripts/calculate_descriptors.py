"""Stage C -- compute RDKit + Mordred descriptors and write reports/descriptor_report.md.

Run:  python scripts/calculate_descriptors.py [--mode quick|full]
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from _common import base_parser, context, reports_path

from herg_model import data as data_module
from herg_model import descriptors as descriptor_module
from herg_model import feature_selection as selection_module
from herg_model import preprocessing as preprocessing_module
from herg_model import splitting as splitting_module
from herg_model.chemistry import load_structure_table
from herg_model.reporting import write_descriptor_report


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg, dirs = context(args)
    smiles_col = str(cfg.get_path("data.smiles_col", "SMILES"))

    df = data_module.load_dataset(cfg)
    smiles = list(df[smiles_col].astype(str))
    y = data_module.blocker_target(df, cfg)

    started = time.time()
    rdkit_frame = descriptor_module.compute_rdkit_descriptors(smiles, cfg, verbose=not args.quiet)
    rdkit_seconds = time.time() - started

    started = time.time()
    mordred_frame, mordred_status = descriptor_module.compute_mordred_descriptors(
        smiles, cfg, verbose=not args.quiet
    )
    mordred_seconds = time.time() - started

    rdkit_names = [c for c in rdkit_frame.columns if c != "SMILES"]
    rdkit_matrix = rdkit_frame[rdkit_names].to_numpy(dtype=float)

    structure = load_structure_table(smiles, cfg, verbose=False)
    splits = splitting_module.make_splits(
        structure["scaffold"].to_numpy(), y, cfg, strategy="scaffold",
        seed=int(cfg.get_path("split.primary_seed", 42)),
    )
    train_idx = splits.train

    # constant / duplicate / missing statistics are computed on the TRAINING split only
    pipeline = preprocessing_module.FeaturePipeline(names=rdkit_names, scale=False)
    pipeline.fit(rdkit_matrix[train_idx])
    quality = preprocessing_module.feature_quality_table(rdkit_matrix[train_idx], rdkit_names)

    groups = descriptor_module.assign_groups(rdkit_names)
    group_table = pd.DataFrame(
        [{"group": name, "n_descriptors": len(cols)} for name, cols in groups.items()]
    ).sort_values("n_descriptors", ascending=False).reset_index(drop=True)
    group_table = pd.concat(
        [group_table, pd.DataFrame([{"group": "rdkit_all", "n_descriptors": len(rdkit_names)}])],
        ignore_index=True,
    )

    correlation = preprocessing_module.high_correlation_pairs(
        rdkit_matrix[train_idx], rdkit_names, threshold=0.95, max_report=2000
    )

    mordred_info = dict(mordred_status)
    mordred_info["seconds"] = round(mordred_seconds, 1)
    if mordred_frame is not None:
        mordred_names = [c for c in mordred_frame.columns if c != "SMILES"]
        mordred_matrix = mordred_frame[mordred_names].to_numpy(dtype=float)
        missing = np.isnan(mordred_matrix[train_idx]).mean(axis=0)
        mordred_info["mean_missing_rate"] = float(missing.mean())
        mordred_info["n_high_missing"] = int(
            (missing > float(cfg.get_path("descriptors.mordred.max_missing_rate", 0.1))).sum()
        )
        mordred_info["n_constant_train"] = int(
            (np.nanstd(mordred_matrix[train_idx], axis=0) <= 1e-12).sum()
        )

    report_path = reports_path(cfg, "descriptor_report.md")
    all_nan_rows = int(np.isnan(rdkit_matrix).all(axis=1).sum()) if rdkit_names else len(smiles)
    write_descriptor_report(
        report_path,
        rdkit_info={
            "n_descriptors": len(rdkit_names),
            "n_success": len(smiles) - all_nan_rows,
            "n_rows": len(smiles),
            "seconds": round(rdkit_seconds, 1),
        },
        mordred_info=mordred_info,
        quality=quality,
        group_table=group_table,
        correlation=correlation,
        dropped=pipeline.dropped,
    )
    print(f"[descriptors] wrote {report_path}", flush=True)

    quality.to_csv(dirs["reports"] / "descriptor_quality.csv", index=False)
    group_table.to_csv(dirs["reports"] / "descriptor_groups.csv", index=False)
    if len(correlation):
        correlation.to_csv(dirs["reports"] / "descriptor_correlation_pairs.csv", index=False)

    # feature-name -> group mapping, used later for grouped importance
    mapping = {
        "ecfp": [c for c in df.columns if c.startswith("ecfp_")],
        "fcfp": [c for c in df.columns if c.startswith("fcfp_")],
        "ptfp": [c for c in df.columns if c.startswith("ptfp_")],
    }
    rows = []
    for group, cols in groups.items():
        rows.extend({"feature": c, "group": group, "source": "rdkit_descriptors"} for c in cols)
    for name in cfg.get_path("data.existing_descriptors", []) or []:
        rows.append({"feature": name, "group": "existing_descriptor", "source": "train.csv"})
    for kind, cols in mapping.items():
        rows.extend({"feature": c, "group": f"fingerprint_{kind}", "source": "train.csv"} for c in cols)
    if mordred_frame is not None:
        rows.extend(
            {"feature": c, "group": "mordred_2d", "source": "mordred_2d"}
            for c in mordred_frame.columns if c != "SMILES"
        )
    pd.DataFrame(rows).to_csv(dirs["reports"] / "feature_groups.csv", index=False)

    summary = {
        "rdkit_descriptors": len(rdkit_names),
        "rdkit_seconds": round(rdkit_seconds, 1),
        "mordred": {k: v for k, v in mordred_info.items() if k != "warnings"},
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"[descriptors] training-split dropped: "
          f"{ {k: (len(v) if isinstance(v, list) else v) for k, v in pipeline.dropped.items()} }")


if __name__ == "__main__":
    main()
