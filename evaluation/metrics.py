from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, tasks: Iterable[str]) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for idx, task in enumerate(tasks):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        report[task] = {"R2": r2(yt, yp), "RMSE": rmse(yt, yp)}
    report["mean"] = {
        "R2": float(np.mean([report[t]["R2"] for t in tasks])),
        "RMSE": float(np.mean([report[t]["RMSE"] for t in tasks])),
    }
    return report
