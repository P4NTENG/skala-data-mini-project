"""Splitting must not leak scaffolds, and the scaffold split must be deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from herg_model.splitting import (
    check_leakage,
    make_splits,
    random_split,
    scaffold_split,
    split_summary,
)


@pytest.fixture()
def toy():
    rng = np.random.default_rng(7)
    n = 400
    scaffolds = np.array([f"scaf{i // 4}" for i in range(n)])          # 100 groups of 4
    labels = rng.integers(0, 2, size=n)
    return scaffolds, labels


def test_scaffold_split_has_no_leakage(toy):
    scaffolds, labels = toy
    splits = scaffold_split(scaffolds, labels, 0.1, 0.1, seed=1)
    report = check_leakage(scaffolds, splits)
    assert not report["has_leakage"]
    assert sum(report["scaffold_overlap"].values()) == 0


def test_scaffold_split_is_deterministic(toy):
    scaffolds, labels = toy
    a = scaffold_split(scaffolds, labels, 0.1, 0.1, seed=3)
    b = scaffold_split(scaffolds, labels, 0.1, 0.1, seed=3)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)
    assert set(a.train) | set(a.val) | set(a.test) == set(range(len(labels)))


def test_different_seeds_give_different_splits(toy):
    scaffolds, labels = toy
    a = scaffold_split(scaffolds, labels, 0.1, 0.1, seed=1)
    b = scaffold_split(scaffolds, labels, 0.1, 0.1, seed=2)
    assert not np.array_equal(a.val, b.val)


def test_scaffold_split_approximates_requested_ratios(toy):
    scaffolds, labels = toy
    splits = scaffold_split(scaffolds, labels, 0.1, 0.1, seed=5)
    summary = split_summary(labels, scaffolds, splits).set_index("split")
    assert summary.loc["train", "fraction"] == pytest.approx(0.8, abs=0.05)
    assert summary.loc["val", "fraction"] == pytest.approx(0.1, abs=0.05)
    assert summary.loc["test", "fraction"] == pytest.approx(0.1, abs=0.05)


def test_random_split_reuses_scaffolds_across_splits(toy):
    scaffolds, labels = toy
    splits = random_split(labels, 0.1, 0.1, seed=1)
    report = check_leakage(scaffolds, splits)
    assert report["has_leakage"]  # this is exactly why the random split is only a reference


def test_make_splits_respects_config(toy, cfg):
    scaffolds, labels = toy
    splits = make_splits(scaffolds, labels, cfg, strategy="scaffold", seed=11)
    assert splits.strategy == "scaffold"
    assert splits.sizes()["train"] > splits.sizes()["val"]
