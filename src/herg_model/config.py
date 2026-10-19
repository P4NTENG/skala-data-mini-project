"""Configuration loading and environment introspection."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

#: repository root (``.../skala-data-mini-project``)
ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = ROOT / "configs" / "experiments.yaml"

#: packages whose exact version matters for reproducibility of the feature matrices
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "rdkit",
    "matplotlib",
    "seaborn",
    "lightgbm",
    "xgboost",
    "mordredcommunity",
    "mordred",
    "optuna",
    "shap",
    "joblib",
    "pyyaml",
)


class Config(dict):
    """Nested ``dict`` with attribute and dotted-path access."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"config has no key {item!r}") from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return ``cfg.get_path("split.test_size")`` with a fallback."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info() -> dict[str, Any]:
    """Machine-readable snapshot of the runtime, embedded in every artifact."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``configs/experiments.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = Config(raw)
    _validate(cfg)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def _validate(cfg: Config) -> None:
    meanings = cfg.get_path("data.label_meanings", {})
    if str(meanings.get("0")) != "blocker" or str(meanings.get("1")) != "non_blocker":
        raise ValueError(
            "label semantics are fixed: label=0 must mean 'blocker' and label=1 'non_blocker'"
        )
    if cfg.get_path("project.positive_class") != "blocker":
        raise ValueError("project.positive_class must be 'blocker' (label == 0)")
    for key in ("data", "paths", "split", "models", "experiment"):
        if key not in cfg:
            raise ValueError(f"config is missing the required section {key!r}")


def resolve(path_like: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def as_config(value) -> Config:
    """Coerce a plain mapping (or ``None``) into a :class:`Config`."""
    if isinstance(value, Config):
        return value
    return Config(value or {})


def ensure_dirs(cfg: Config) -> dict[str, Path]:
    """Create the output directories and return them as absolute paths."""
    dirs = {
        "reports": resolve(cfg.get_path("paths.reports", "reports")),
        "artifacts": resolve(cfg.get_path("paths.artifacts", "artifacts")),
        "cache": resolve(cfg.get_path("paths.cache", ".cache/herg_model")),
        "figures": resolve(cfg.get_path("paths.figures", "reports/figures")),
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def mode(cfg: Config) -> str:
    """Active experiment budget mode (``quick`` or ``full``)."""
    value = str(cfg.get_path("experiment.mode", "quick"))
    if value not in {"quick", "full"}:
        raise ValueError("experiment.mode must be 'quick' or 'full'")
    return value


def budget(cfg: Config, key: str, default: Any = None) -> Any:
    """Read a ``{quick: ..., full: ...}`` mapping for the active mode."""
    node = cfg.get_path(key, default)
    if isinstance(node, dict):
        return node.get(mode(cfg), default)
    return node


def stable_hash(payload: Any) -> str:
    """Deterministic short hash used for cache keys."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def config_signature(cfg: Config, *sections: str) -> str:
    """Hash of the selected config sections (used to key caches)."""
    payload = {section: cfg.get_path(section) for section in sections}
    return stable_hash(payload)
