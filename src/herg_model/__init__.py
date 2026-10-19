"""hERG blocker prediction scaffold.

Reusable package for the experiments described in ``docs/ml_goal.md``:

* fixed label semantics -- ``label = 0`` is the hERG blocker (positive) class
* reproducible SMILES -> descriptor / fingerprint feature computation
* scaffold-based splitting with leakage checks
* controlled E0..E11 comparison experiments and descriptor-group ablations
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
