"""Shared helpers for the ``scripts/*.py`` entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from herg_model.config import Config, ensure_dirs, load_config, mode, resolve  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="path to experiments.yaml")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["quick", "full"],
        help="override experiment.mode (controls the hyper-parameter search budget)",
    )
    parser.add_argument("--quiet", action="store_true", help="reduce console output")
    return parser


def get_config(args) -> Config:
    cfg = load_config(args.config)
    if getattr(args, "mode", None):
        cfg["experiment"]["mode"] = args.mode
    return cfg


def context(args) -> tuple[Config, dict[str, Path]]:
    cfg = get_config(args)
    dirs = ensure_dirs(cfg)
    if not args.quiet:
        print(f"[config] {cfg['_config_path']}  mode={mode(cfg)}", flush=True)
    return cfg, dirs


def reports_path(cfg: Config, name: str) -> Path:
    return resolve(cfg.get_path("paths.reports", "reports")) / name


def artifacts_path(cfg: Config, name: str) -> Path:
    return resolve(cfg.get_path("paths.artifacts", "artifacts")) / name


def figures_dir(cfg: Config) -> Path:
    path = resolve(cfg.get_path("paths.figures", "reports/figures"))
    path.mkdir(parents=True, exist_ok=True)
    return path
