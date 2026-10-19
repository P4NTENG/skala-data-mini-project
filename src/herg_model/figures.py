"""Figure generation. Every plot states that the positive class is the hERG blocker (label = 0)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    "plot_experiment_comparison",
    "plot_roc_pr_curves",
    "plot_ablation",
    "plot_confusion_matrices",
    "plot_feature_importance",
    "plot_group_importance",
    "plot_scaffold_performance",
    "plot_split_comparison",
    "plot_seed_stability",
]

POSITIVE_NOTE = "positive class = hERG blocker (label = 0)"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_experiment_comparison(results: pd.DataFrame, path: str | Path, metric: str = "tuned_balanced_accuracy") -> Path:
    """Validation balanced accuracy per experiment, split by model family."""
    pivot = results.pivot_table(index="experiment_id", columns="model", values=metric, aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index))
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(pivot)), 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(metric.replace("tuned_", ""))
    ax.set_xlabel("experiment")
    ax.set_title(f"Validation {metric.replace('tuned_', '')} by experiment\n({POSITIVE_NOTE})")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.legend(fontsize=8, title="model")
    return _save(fig, path)


def plot_roc_pr_curves(
    curves: Mapping[str, dict],
    path: str | Path,
    title: str = "Validation ROC / PR",
) -> Path:
    """``curves`` maps a label to ``{"fpr","tpr","recall","precision","roc_auc","pr_auc"}``."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, payload in curves.items():
        roc_auc = payload.get("roc_auc")
        pr_auc = payload.get("pr_auc")
        axes[0].plot(payload["fpr"], payload["tpr"], label=f"{label} (AUC={roc_auc:.3f})")
        axes[1].plot(payload["recall"], payload["precision"], label=f"{label} (AP={pr_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xlabel("recall (blocker)")
    axes[1].set_ylabel("precision (blocker)")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{title} -- {POSITIVE_NOTE}")
    return _save(fig, path)


def plot_ablation(ablation: pd.DataFrame, path: str | Path, metric: str = "delta_balanced_accuracy") -> Path:
    """Added effect of each descriptor group relative to the fingerprint baseline."""
    frame = ablation.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(frame))))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in frame[metric]]
    ax.barh(frame["label"], frame[metric], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"{metric} vs baseline (validation, {POSITIVE_NOTE})")
    ax.set_title("Descriptor-group ablation")
    return _save(fig, path)


def plot_confusion_matrices(matrices: Mapping[str, np.ndarray], path: str | Path) -> Path:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, matrix) in zip(axes, matrices.items()):
        matrix = np.asarray(matrix)
        ax.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred non-blocker", "pred blocker"], fontsize=8)
        ax.set_yticks([0, 1], ["true non-blocker", "true blocker"], fontsize=8)
        ax.set_title(label, fontsize=9)
    fig.suptitle(f"Confusion matrices -- {POSITIVE_NOTE}")
    return _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, top_k: int = 25) -> Path:
    frame = importance.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(frame))))
    ax.barh(frame["feature"], frame["importance_mean"], color="#2980b9")
    ax.set_xlabel("permutation importance (drop in balanced accuracy)")
    ax.set_title(f"Top {len(frame)} features\n(predicting {POSITIVE_NOTE})")
    return _save(fig, path)


def plot_group_importance(group_table: pd.DataFrame, path: str | Path) -> Path:
    frame = group_table.sort_values("total_importance")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(frame))))
    ax.barh(frame["group"], frame["total_importance"], color="#8e44ad")
    ax.set_xlabel("summed permutation importance")
    ax.set_title(f"Importance aggregated by feature group\n({POSITIVE_NOTE})")
    return _save(fig, path)


def plot_scaffold_performance(scaffold_table: pd.DataFrame, path: str | Path) -> Path:
    frame = scaffold_table[scaffold_table["n_molecules"] >= 2]
    if frame.empty:
        return Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(frame["error_rate"], bins=20, color="#e67e22")
    axes[0].set_xlabel("error rate within scaffold")
    axes[0].set_ylabel("number of scaffolds")
    axes[0].set_title("Per-scaffold error rate distribution")
    axes[1].scatter(frame["n_molecules"], frame["error_rate"], alpha=0.5, color="#16a085")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("molecules per scaffold")
    axes[1].set_ylabel("error rate")
    axes[1].set_title("Error rate vs scaffold size")
    fig.suptitle(f"Scaffold-level performance -- {POSITIVE_NOTE}")
    return _save(fig, path)


def plot_split_comparison(results: pd.DataFrame, path: str | Path, metric: str = "tuned_balanced_accuracy") -> Path:
    pivot = results.pivot_table(index="split_strategy", columns="model", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric.replace("tuned_", ""))
    ax.set_title(f"Scaffold vs random split\n(validation, {POSITIVE_NOTE})")
    ax.legend(fontsize=8, title="model")
    return _save(fig, path)


def plot_seed_stability(stability: pd.DataFrame, path: str | Path, metric: str = "tuned_balanced_accuracy") -> Path:
    if stability.empty:
        return Path(path)
    frame = stability.copy()
    order = frame.groupby("experiment_id")[metric].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(order)), 5))
    positions = np.arange(len(order))
    for offset, (model, group) in enumerate(frame.groupby("model")):
        means = group.groupby("experiment_id")[metric].mean().reindex(order)
        stds = group.groupby("experiment_id")[metric].std().reindex(order).fillna(0)
        ax.bar(positions + offset * 0.25, means, width=0.25, yerr=stds, capsize=3, label=model)
    ax.set_xticks(positions, order, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("tuned_", ""))
    ax.set_title(f"Across-scaffold-seed stability (mean $\\pm$ sd)\n({POSITIVE_NOTE})")
    ax.legend(fontsize=8, title="model")
    return _save(fig, path)
