"""Train-only preprocessing: inf/NaN handling, constant & duplicate removal, imputation, scaling.

Every fitted quantity (column mask, imputation values, scaler statistics) is learned on the
**training split only** and then applied unchanged to validation / test / new SMILES. The
class is deliberately explicit rather than a scikit-learn ``Pipeline`` so that the exact
feature order and the dropped columns can be reported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FeaturePipeline",
    "feature_quality_table",
    "high_correlation_pairs",
    "DEFAULT_DUPLICATE_TOLERANCE",
]

DEFAULT_DUPLICATE_TOLERANCE = 9  # decimal places used when hashing columns for duplicate detection
CONSTANT_ATOL = 1e-12


@dataclass
class FeaturePipeline:
    """Leak-free column cleaning / imputation / scaling."""

    names: list[str]
    scale: bool = False
    drop_constant: bool = True
    drop_duplicate: bool = True
    duplicate_tolerance: int = DEFAULT_DUPLICATE_TOLERANCE
    missing_rate_limit: float = 1.0  # drop columns whose training missing rate exceeds this

    # fitted state ----------------------------------------------------------------
    keep_idx: np.ndarray | None = None
    kept_names: list[str] = field(default_factory=list)
    fill_values: np.ndarray | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    dropped: dict[str, list] = field(default_factory=lambda: {
        "all_missing": [], "high_missing": [], "constant": [], "duplicate": [], "scaffold_leak": [],
    })
    is_fitted: bool = False

    # -----------------------------------------------------------------------------
    @staticmethod
    def _sanitize(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            X = np.where(np.isfinite(X), X, np.nan)
        return X

    def _column_signature(self, column: np.ndarray, fill: float) -> str:
        filled = np.where(np.isnan(column), fill, column)
        rounded = np.round(filled, self.duplicate_tolerance)
        return hashlib.sha1(np.ascontiguousarray(rounded).tobytes()).hexdigest()

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FeaturePipeline":
        X = self._sanitize(X)
        n_rows, n_cols = X.shape
        if len(self.names) != n_cols:
            raise ValueError(f"names ({len(self.names)}) do not match X.shape[1] ({n_cols})")

        missing_rate = np.isnan(X).mean(axis=0) if n_rows else np.zeros(n_cols)
        keep = np.ones(n_cols, dtype=bool)

        all_missing = missing_rate >= 1.0
        keep &= ~all_missing
        self.dropped["all_missing"] = [self.names[j] for j in np.where(all_missing)[0]]

        if self.missing_rate_limit < 1.0:
            high_missing = (missing_rate > self.missing_rate_limit) & keep
            keep &= ~high_missing
            self.dropped["high_missing"] = [self.names[j] for j in np.where(high_missing)[0]]

        if self.drop_constant:
            const = np.zeros(n_cols, dtype=bool)
            for j in np.where(keep)[0]:
                column = X[:, j]
                finite = column[np.isfinite(column)]
                if finite.size == 0 or (finite.max() - finite.min()) <= CONSTANT_ATOL:
                    const[j] = True
            keep &= ~const
            self.dropped["constant"] = [self.names[j] for j in np.where(const)[0]]

        if self.drop_duplicate:
            seen: dict[str, int] = {}
            dup = np.zeros(n_cols, dtype=bool)
            duplicate_pairs: list[tuple[str, str]] = []
            for j in np.where(keep)[0]:
                fill = np.nanmedian(X[:, j]) if np.isfinite(X[:, j]).any() else 0.0
                signature = self._column_signature(X[:, j], float(fill))
                if signature in seen:
                    dup[j] = True
                    duplicate_pairs.append((self.names[j], self.names[seen[signature]]))
                else:
                    seen[signature] = j
            keep &= ~dup
            self.dropped["duplicate"] = duplicate_pairs

        self.keep_idx = np.where(keep)[0]
        self.kept_names = [self.names[j] for j in self.keep_idx]

        kept = X[:, self.keep_idx]
        with np.errstate(invalid="ignore"):
            self.fill_values = np.nanmedian(kept, axis=0) if n_rows else np.zeros(kept.shape[1])
        self.fill_values = np.where(np.isfinite(self.fill_values), self.fill_values, 0.0)

        filled = np.where(np.isnan(kept), self.fill_values, kept)
        if self.scale:
            self.scaler_mean = filled.mean(axis=0)
            std = filled.std(axis=0, ddof=0)
            self.scaler_scale = np.where(std > CONSTANT_ATOL, std, 1.0)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit")
        X = self._sanitize(X)
        kept = X[:, self.keep_idx]
        filled = np.where(np.isnan(kept), self.fill_values, kept)
        if self.scale and self.scaler_mean is not None:
            filled = (filled - self.scaler_mean) / self.scaler_scale
        return filled

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def summary(self) -> dict:
        return {
            "n_input_features": len(self.names),
            "n_output_features": len(self.kept_names),
            "scaled": self.scale,
            "dropped": {k: (len(v) if isinstance(v, list) else v) for k, v in self.dropped.items()},
            "dropped_detail": self.dropped,
        }


def feature_quality_table(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-feature missing / constant / duplicate diagnostics (descriptor & fingerprint blocks)."""
    X = np.asarray(X, dtype=np.float64)
    n_rows = X.shape[0] if X.ndim == 2 else 0
    rows = []
    for j, name in enumerate(names):
        column = X[:, j]
        finite = column[np.isfinite(column)]
        rows.append(
            {
                "feature": name,
                "missing_rate": float(np.isnan(column).mean()) if n_rows else float("nan"),
                "n_unique": int(len(np.unique(finite))) if finite.size else 0,
                "is_constant": bool(finite.size == 0 or (finite.max() - finite.min()) <= CONSTANT_ATOL),
                "min": float(finite.min()) if finite.size else float("nan"),
                "max": float(finite.max()) if finite.size else float("nan"),
                "mean": float(finite.mean()) if finite.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def high_correlation_pairs(
    X: np.ndarray, names: list[str], threshold: float = 0.95, max_report: int = 500
) -> pd.DataFrame:
    """Descriptor pairs with |Pearson r| above ``threshold`` (computed on the training split)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    filled = X.copy()
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(filled, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(filled), med, filled)
    std = filled.std(axis=0, ddof=0)
    active = std > CONSTANT_ATOL
    filled = filled[:, active]
    active_names = [names[j] for j in np.where(active)[0]]
    if filled.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    corr = np.corrcoef(filled, rowvar=False)
    rows = []
    for a in range(corr.shape[0]):
        for b in range(a + 1, corr.shape[1]):
            value = corr[a, b]
            if np.isfinite(value) and abs(value) >= threshold:
                rows.append({"feature_a": active_names[a], "feature_b": active_names[b], "pearson_r": float(value)})
                if len(rows) >= max_report:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "pearson_r"])
