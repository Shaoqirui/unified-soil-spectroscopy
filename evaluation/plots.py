from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _axes(ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is not None:
        return ax
    _, axis = plt.subplots(figsize=(5, 5))
    return axis


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, label: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _axes(ax)
    ax.scatter(y_true, y_pred, s=6, alpha=0.6)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="#555555")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(label)
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, label: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _axes(ax)
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, s=6, alpha=0.6)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#555555")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Residual")
    ax.set_title(label)
    return ax


def plot_id_curves(
    steps: Sequence[float],
    insertion: Sequence[float],
    deletion: Sequence[float],
    ci_insert: Optional[Sequence[float]] = None,
    ci_delete: Optional[Sequence[float]] = None,
    metric: str = "R²",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _axes(ax)
    steps = np.asarray(steps, dtype=np.float64)
    ins = np.asarray(insertion, dtype=np.float64)
    dele = np.asarray(deletion, dtype=np.float64)
    ax.plot(steps, ins, color="#2f6690", label="Insertion")
    ax.plot(steps, dele, color="#d96459", label="Deletion")
    if ci_insert is not None:
        ci = np.asarray(ci_insert, dtype=np.float64)
        ax.fill_between(steps, ins - ci, ins + ci, color="#2f6690", alpha=0.2)
    if ci_delete is not None:
        ci = np.asarray(ci_delete, dtype=np.float64)
        ax.fill_between(steps, dele - ci, dele + ci, color="#d96459", alpha=0.2)
    ax.set_xlabel("Fraction of features")
    ax.set_ylabel(metric)
    ax.legend()
    return ax
