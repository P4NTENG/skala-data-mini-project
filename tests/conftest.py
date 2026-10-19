"""Shared test fixtures: make ``herg_model`` importable and locate the dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ROOT / "data" / "train.csv"

requires_data = pytest.mark.skipif(not DATA.exists(), reason="data/train.csv is not available")


@pytest.fixture(scope="session")
def cfg():
    from herg_model.config import load_config

    return load_config()
