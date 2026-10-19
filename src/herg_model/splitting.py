"""Train / validation / test splitting with structural leakage control.

* ``scaffold``  -- primary strategy, groups by Bemis-Murcko scaffold so that an unseen
  scaffold in val/test means an unseen chemotype (the generalisation test that matters).
* ``random``    -- stratified reference split, reported separately for comparability with
  earlier studies that used random splits.

The scaffold split is deterministic for a given seed (groups are ordered by decreasing size
with a seeded tie-break) and greedily assigned to the split whose target proportion is
furthest from being met, with an explicit class-balance penalty. Ratios are *not* forced:
the realised sizes and class distributions are reported instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config

__all__ = [
    "Splits",
    "scaffold_split",
    "random_split",
    "make_splits",
    "repeated_splits",
    "check_leakage",
    "assert_no_leakage",
    "split_summary",
    "SCAFFOLD_SPLIT_TARGETS",
]

SCAFFOLD_SPLIT_TARGETS = {"train": 0.80, "val": 0.10, "test": 0.10}


@dataclass
class Splits:
    """Positional row indices for each split (aligned with the original CSV order)."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str = "scaffold"
    seed: int = 42
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def sizes(self) -> dict[str, int]:
        return {name: int(len(idx)) for name, idx in self.as_dict().items()}


def scaffold_split(
    scaffolds: Sequence[str],
    labels: np.ndarray,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
    balance_classes: bool = True,
) -> Splits:
    """Greedy, class-aware scaffold split (deterministic for a given ``seed``)."""
    scaffolds = np.asarray(scaffolds, dtype=object)
    labels = np.asarray(labels)
    n = len(scaffolds)

    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(scaffolds):
        groups.setdefault(str(key), []).append(idx)

    rng = np.random.default_rng(seed)
    # order: largest first, seeded random tie-break for reproducibility (never rely on dict order)
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    tie_break = {key: rng.random() for key, _ in order}
    order.sort(key=lambda kv: (-len(kv[1]), tie_break[kv[0]]))

    targets = {"train": 1.0 - val_size - test_size, "val": val_size, "test": test_size}
    counts = {k: 0 for k in targets}
    blocker_counts = {k: 0 for k in targets}
    total_blocker = int(labels.sum())
    global_blocker_frac = total_blocker / n if n else 0.0
    split_order = ["val", "test", "train"]

    assignment = np.empty(n, dtype=object)
    for position, (key, indices) in enumerate(order):
        size = len(indices)
        n_blocker = int(labels[indices].sum())
        best_split, best_cost = None, np.inf
        # rotate the tie-break order so that perfectly tied splits are filled evenly
        rotated = split_order[position % len(split_order):] + split_order[:position % len(split_order)]
        for split_name in rotated:
            # remaining target fraction: fill the most deficient split first
            deficit = targets[split_name] - counts[split_name] / n
            new_count = counts[split_name] + size
            balance_gap = 0.0
            if balance_classes and new_count:
                balance_gap = abs((blocker_counts[split_name] + n_blocker) / new_count - global_blocker_frac)
            # scale the balance term by the group size so it acts as a tie-breaker only
            cost = -deficit + 2.0 * balance_gap * (size / n)
            if cost < best_cost - 1e-12:
                best_split, best_cost = split_name, cost
        assignment[indices] = best_split
        counts[best_split] += size
        blocker_counts[best_split] += n_blocker

    splits = Splits(
        train=np.sort(np.where(assignment == "train")[0]),
        val=np.sort(np.where(assignment == "val")[0]),
        test=np.sort(np.where(assignment == "test")[0]),
        strategy="scaffold",
        seed=seed,
        meta={
            "n_scaffold_groups": len(groups),
            "largest_scaffold_group": int(max(len(v) for v in groups.values())),
            "realised_ratios": {k: counts[k] / n for k in targets},
            "target_ratios": dict(SCAFFOLD_SPLIT_TARGETS),
            "balance_classes": balance_classes,
        },
    )
    return splits


def random_split(
    labels: np.ndarray,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> Splits:
    """Stratified random split (reference experiment only)."""
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        idx, test_size=val_size + test_size, random_state=seed, stratify=labels
    )
    relative_val = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=1.0 - relative_val, random_state=seed, stratify=labels[rest_idx]
    )
    return Splits(
        train=np.sort(train_idx),
        val=np.sort(val_idx),
        test=np.sort(test_idx),
        strategy="random",
        seed=seed,
        meta={"target_ratios": {"train": 1 - val_size - test_size, "val": val_size, "test": test_size}},
    )


def make_splits(
    scaffolds: Sequence[str],
    labels: np.ndarray,
    cfg: Config,
    strategy: str | None = None,
    seed: int | None = None,
) -> Splits:
    strategy = strategy or str(cfg.get_path("split.strategy", "scaffold"))
    seed = int(seed if seed is not None else cfg.get_path("split.primary_seed", 42))
    val_size = float(cfg.get_path("split.val_size", 0.10))
    test_size = float(cfg.get_path("split.test_size", 0.10))
    balance = bool(cfg.get_path("split.balance_classes", True))
    if strategy == "scaffold":
        return scaffold_split(scaffolds, labels, val_size, test_size, seed, balance)
    if strategy == "random":
        return random_split(labels, val_size, test_size, seed)
    raise ValueError(f"unknown split strategy {strategy!r}")


def repeated_splits(
    scaffolds: Sequence[str],
    labels: np.ndarray,
    cfg: Config,
    strategy: str | None = None,
) -> list[Splits]:
    """One split per configured seed (the primary seed comes first)."""
    seeds = list(cfg.get_path("split.seeds", [int(cfg.get_path("split.primary_seed", 42))]))
    primary = int(cfg.get_path("split.primary_seed", 42))
    if primary not in seeds:
        seeds.insert(0, primary)
    return [make_splits(scaffolds, labels, cfg, strategy=strategy, seed=s) for s in seeds]


def check_leakage(scaffolds: Sequence[str], splits: Splits) -> dict:
    """Scaffold / canonical-SMILES overlap between every pair of splits."""
    scaffolds = np.asarray(scaffolds, dtype=object)
    parts = splits.as_dict()
    keys = {name: set(scaffolds[idx]) for name, idx in parts.items()}
    report: dict = {"scaffold_overlap": {}, "leaked_scaffolds": {}, "has_leakage": False}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keys[a] & keys[b]
        report["scaffold_overlap"][f"{a}__{b}"] = len(overlap)
        report["leaked_scaffolds"][f"{a}__{b}"] = sorted(overlap)[:10]
        if overlap:
            report["has_leakage"] = True
    return report


def assert_no_leakage(scaffolds: Sequence[str], splits: Splits) -> None:
    report = check_leakage(scaffolds, splits)
    if report["has_leakage"]:
        raise AssertionError(f"scaffold leakage detected: {report['scaffold_overlap']}")


def split_summary(
    labels: np.ndarray,
    scaffolds: Sequence[str],
    splits: Splits,
) -> pd.DataFrame:
    """Realised sizes, class balance and scaffold counts per split."""
    labels = np.asarray(labels)
    scaffolds = np.asarray(scaffolds, dtype=object)
    rows = []
    for name, idx in splits.as_dict().items():
        y = labels[idx]
        rows.append(
            {
                "split": name,
                "n_rows": int(len(idx)),
                "fraction": float(len(idx) / len(labels)),
                "n_blocker": int(y.sum()),
                "n_non_blocker": int((y == 0).sum()),
                "blocker_fraction": float(y.mean()) if len(y) else float("nan"),
                "n_unique_scaffolds": int(len(set(scaffolds[idx]))),
            }
        )
    return pd.DataFrame(rows)
