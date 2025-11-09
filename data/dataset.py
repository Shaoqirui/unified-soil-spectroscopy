from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    build_libs_full_matrix,
    build_libs_task_matrix,
    build_vnir_matrix,
    build_xrf_matrix,
    preprocess_libs_full,
)


TASKS = ["Clay", "OM", "CEC", "pH", "V", "exP", "exK", "exCa", "exMg"]


def discover_files(base_dir: str, out_dir: str) -> Dict[str, str]:
    def scan(path: str) -> Dict[str, str]:
        found: Dict[str, str] = {}
        for root, _, files in os.walk(path):
            for fn in files:
                if not fn.lower().endswith(".txt"):
                    continue
                fp = os.path.join(root, fn)
                name = fn.lower()
                if "soil" in name and "fertility" in name:
                    found.setdefault("soil", fp)
                elif "vnir" in name:
                    found.setdefault("vnir", fp)
                elif "xrf" in name:
                    found.setdefault("xrf", fp)
                elif "libs" in name:
                    found.setdefault("libs", fp)
        return found

    found = scan(base_dir)
    if all(k in found for k in ("soil", "vnir", "xrf", "libs")):
        return found
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(base_dir):
        if not fn.lower().endswith(".zip"):
            continue
        fp = os.path.join(base_dir, fn)
        try:
            with zipfile.ZipFile(fp, "r") as zf:
                zf.extractall(out_dir)
        except zipfile.BadZipFile:
            continue
    found = scan(out_dir)
    if all(k in found for k in ("soil", "vnir", "xrf", "libs")):
        return found
    raise FileNotFoundError("Missing soil/VNIR/XRF/LIBS txt files.")


def load_soil(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID").sort_index()


def load_vnir(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID").sort_index()


def load_xrf(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID").sort_index()


def load_libs(path: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    head = pd.read_csv(path, sep="\t", header=None, nrows=2)
    sample_ids = head.iloc[0, 1:].astype(int).tolist()
    fields = head.iloc[1, 1:].astype(int).tolist()
    meta = pd.DataFrame({"Field": fields}, index=sample_ids)
    meta.index.name = "ID"
    meta.sort_index(inplace=True)

    spec = pd.read_csv(path, sep="\t", header=None, skiprows=2)
    wavelength = spec.iloc[:, 0].to_numpy(dtype=np.float64)
    intensities = spec.iloc[:, 1:]
    intensities.columns = [int(c) for c in intensities.columns]
    matrix = intensities.T
    matrix.index.name = "ID"
    matrix.sort_index(inplace=True)
    return meta, wavelength, matrix


def kennard_stone(targets: np.ndarray, n_train: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dist = np.linalg.norm(targets[:, None, :] - targets[None, :, :], axis=-1)
    first = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(first[0])]
    remaining = set(range(targets.shape[0])) - set(selected)
    selected.append(int(max(remaining, key=lambda idx: dist[idx, selected[0]])))
    remaining -= set([selected[1]])
    while len(selected) < n_train:
        best = max(remaining, key=lambda idx: np.min(dist[idx, selected]))
        selected.append(int(best))
        remaining.remove(best)
    tr = np.array(selected, dtype=int)
    te = np.array(sorted(remaining), dtype=int)
    rng.shuffle(tr)
    return tr, te


@dataclass
class Standardizers:
    vnir: StandardScaler
    xrf: StandardScaler

    def transform(self, vnir: np.ndarray, xrf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vt = self.vnir.transform(vnir).astype(np.float32)
        xt = self.xrf.transform(xrf).astype(np.float32)
        return vt, xt


def build_modal_arrays(
    soil: pd.DataFrame,
    vnir: pd.DataFrame,
    xrf: pd.DataFrame,
    libs_matrix: pd.DataFrame,
    wavelength: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    ids = soil.index.intersection(vnir.index).intersection(xrf.index).intersection(libs_matrix.index)
    soil = soil.loc[ids]
    vnir = vnir.loc[ids]
    xrf = xrf.loc[ids]
    libs_matrix = libs_matrix.loc[ids]

    vnir_mat = vnir.drop(columns=["Field"]).to_numpy(dtype=np.float64) if "Field" in vnir.columns else vnir.to_numpy(dtype=np.float64)
    Xv = build_vnir_matrix(
        vnir_mat,
        window=cfg["sg_window"],
        poly=cfg["sg_poly"],
        deriv=cfg["sg_deriv"],
    )
    Xx = build_xrf_matrix(xrf)
    Xl_by_task = {
        task: build_libs_task_matrix(
            wavelength,
            libs_matrix,
            cfg["libs_ranges"][task],
            cfg["libs_band"][0],
            cfg["libs_band"][1],
        )
        for task in TASKS
    }
    Xl_full = preprocess_libs_full(
        build_libs_full_matrix(
            wavelength,
            libs_matrix,
            cfg["libs_band"][0],
            cfg["libs_band"][1],
        )
    )
    y = soil[TASKS].to_numpy(dtype=np.float32)
    return Xv, Xx, Xl_by_task, Xl_full.astype(np.float32), y


class SoilDataset(Dataset):
    def __init__(
        self,
        Xv: np.ndarray,
        Xx: np.ndarray,
        Xl: np.ndarray,
        targets: np.ndarray,
        teacher_targets: Optional[np.ndarray] = None,
        branch_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.Xv = Xv.astype(np.float32)
        self.Xx = Xx.astype(np.float32)
        self.Xl = Xl.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.teacher_targets = None if teacher_targets is None else teacher_targets.astype(np.float32)
        self.branch_targets = branch_targets or {}

    def __len__(self) -> int:
        return self.Xv.shape[0]

    def __getitem__(self, idx: int):
        xv = self.Xv[idx]
        xx = self.Xx[idx]
        xl = self.Xl[idx]
        y = self.targets[idx]
        if self.teacher_targets is None:
            return xv, xx, xl, y
        aux = [self.branch_targets[key][idx] for key in ("V", "X", "L")]
        return xv, xx, xl, y, self.teacher_targets[idx], *aux


def normalise_targets(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True) + 1e-8
    return mu, sigma, (train - mu) / sigma, (other - mu) / sigma


def branch_targets_from_dict(branch: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        key: np.column_stack([branch[key][task] for task in TASKS]).astype(np.float32)
        for key in ("V", "X", "L")
    }


def save_manifest(paths: Dict[str, str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(paths, fh, ensure_ascii=False, indent=2)
