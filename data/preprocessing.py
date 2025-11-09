from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter


def snv(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-12
    return ((x - mu) / sd).astype(np.float32)


def max_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
    return (x / m).astype(np.float32)


def sg_filter(
    x: np.ndarray,
    window: int = 11,
    poly: int = 2,
    deriv: int = 1,
    smooth_poly: Optional[int] = 2,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = savgol_filter(
        x,
        window_length=window,
        polyorder=poly,
        deriv=deriv,
        axis=1,
        mode="interp",
    )
    if smooth_poly is not None:
        x = savgol_filter(
            x,
            window_length=window,
            polyorder=smooth_poly,
            deriv=0,
            axis=1,
            mode="interp",
        )
    return x.astype(np.float32)


def build_vnir_matrix(
    vnir: np.ndarray,
    window: int,
    poly: int,
    deriv: int,
) -> np.ndarray:
    return sg_filter(max_norm(snv(vnir)), window=window, poly=poly, deriv=deriv)


def _candidate_compton_columns(columns: Sequence[str]) -> Dict[str, str]:
    hits = ("compton", "scatter", "scattering", "cs", "comp")
    return {
        str(col): columns[idx]
        for idx, col in enumerate(c.lower() for c in columns)
        if any(tok in col for tok in hits)
    }


def find_compton_column(xrf_df) -> Optional[str]:
    src = list(xrf_df.columns)
    mapping = _candidate_compton_columns(src)
    if not mapping:
        return None
    scores = []
    for alias, original in mapping.items():
        try:
            scores.append((original, float(xrf_df[original].mean())))
        except Exception:
            continue
    if not scores:
        return None
    scores.sort(key=lambda item: -item[1])
    return scores[0][0]


def build_xrf_matrix(xrf_df) -> np.ndarray:
    df = xrf_df.copy()
    if "Field" in df.columns:
        df = df.drop(columns=["Field"])
    col = find_compton_column(df)
    mat = df.to_numpy(dtype=np.float64)
    if col is not None:
        denom = np.clip(df[col].to_numpy(dtype=np.float64).reshape(-1, 1), 1e-9, None)
        return (mat / denom).astype(np.float32)
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / norm).astype(np.float32)


def libs_mask_from_ranges(
    wl: np.ndarray,
    ranges: Iterable[Tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros_like(wl, dtype=bool)
    for left, right in ranges:
        mask |= (wl >= left) & (wl <= right)
    return mask


def build_libs_task_matrix(
    wl: np.ndarray,
    libs_df,
    ranges: Sequence[Tuple[float, float]],
    band_min: float,
    band_max: float,
) -> np.ndarray:
    base = (wl >= band_min) & (wl <= band_max)
    mask = base & libs_mask_from_ranges(wl, ranges)
    return libs_df.to_numpy(dtype=np.float64)[:, mask].astype(np.float32)


def build_libs_full_matrix(
    wl: np.ndarray,
    libs_df,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    mask = (wl >= band_min) & (wl <= band_max)
    return libs_df.to_numpy(dtype=np.float64)[:, mask].astype(np.float32)


def preprocess_libs_full(x_full: np.ndarray) -> np.ndarray:
    return max_norm(snv(x_full))
