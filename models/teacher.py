from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from data.dataset import TASKS
from evaluation.metrics import rmse, r2


def autoscale_fit(
    x_train: np.ndarray,
    x_val: np.ndarray,
    mode: str,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
    if mode == "std":
        scaler = StandardScaler(with_mean=False, with_std=True).fit(x_train)
    else:
        scaler = StandardScaler(with_mean=True, with_std=True).fit(x_train)
    return scaler, scaler.transform(x_train), scaler.transform(x_val)


def fit_pls_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    mode: str,
    n_components: int,
) -> Tuple[PLSRegression, StandardScaler, np.ndarray, np.ndarray]:
    scaler, xs, xv = autoscale_fit(x_train, x_val, mode=mode)
    comp = max(1, min(int(n_components), min(xs.shape[1], xs.shape[0] - 1)))
    pls = PLSRegression(n_components=comp, scale=False)
    pls.fit(xs, y_train)
    return pls, scaler, pls.predict(xs).ravel(), pls.predict(xv).ravel()


def fit_concat_pls(
    x_train_list: Sequence[np.ndarray],
    y_train: np.ndarray,
    x_val_list: Sequence[np.ndarray],
    mode: str,
    n_components: int,
) -> Tuple[PLSRegression, StandardScaler, np.ndarray, np.ndarray]:
    xs = np.hstack(x_train_list)
    xv = np.hstack(x_val_list)
    return fit_pls_model(xs, y_train, xv, mode=mode, n_components=n_components)


def fit_linear_meta(
    preds_train: Sequence[np.ndarray],
    y_train: np.ndarray,
    preds_val: Sequence[np.ndarray],
) -> Tuple[LinearRegression, np.ndarray, np.ndarray]:
    stack_train = np.vstack(preds_train).T
    stack_val = np.vstack(preds_val).T
    lr = LinearRegression()
    lr.fit(stack_train, y_train)
    return lr, lr.predict(stack_train).ravel(), lr.predict(stack_val).ravel()


@dataclass
class TeacherReport:
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    metrics: Dict[str, Dict[str, float]]


def train_teacher_ensemble(
    cfg: Dict[str, Dict],
    x_vnir: np.ndarray,
    x_xrf: np.ndarray,
    libs_by_task: Dict[str, np.ndarray],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> TeacherReport:
    train_pred = np.zeros((len(train_idx), len(TASKS)), dtype=np.float32)
    val_pred = np.zeros((len(val_idx), len(TASKS)), dtype=np.float32)
    records: Dict[str, Dict[str, float]] = {}

    for pos, task in enumerate(TASKS):
        conf = cfg[task]
        method = conf["method"]
        y_train = targets[train_idx, pos].reshape(-1, 1)
        y_val = targets[val_idx, pos].ravel()

        if method == "V+X+L-SF-PLS":
            x_train = [x_vnir[train_idx], x_xrf[train_idx], libs_by_task[task][train_idx]]
            x_val = [x_vnir[val_idx], x_xrf[val_idx], libs_by_task[task][val_idx]]
            _, _, pred_train, pred_val = fit_concat_pls(
                x_train,
                y_train,
                x_val,
                conf["VXL_SF"]["scaler"],
                conf["VXL_SF"]["ncomp"],
            )
        elif method == "VNIR-PLS":
            _, _, pred_train, pred_val = fit_pls_model(
                x_vnir[train_idx],
                y_train,
                x_vnir[val_idx],
                conf["V_PLS"]["scaler"],
                conf["V_PLS"]["ncomp"],
            )
        elif method == "VNIR+XRF-SF-PLS":
            x_train = [x_vnir[train_idx], x_xrf[train_idx]]
            x_val = [x_vnir[val_idx], x_xrf[val_idx]]
            _, _, pred_train, pred_val = fit_concat_pls(
                x_train,
                y_train,
                x_val,
                conf["VX_SF"]["scaler"],
                conf["VX_SF"]["ncomp"],
            )
        elif method == "LIBS-PLS":
            libs = libs_by_task[task]
            _, _, pred_train, pred_val = fit_pls_model(
                libs[train_idx],
                y_train,
                libs[val_idx],
                conf["L_PLS"]["scaler"],
                conf["L_PLS"]["ncomp"],
            )
        elif method == "VNIR+LIBS-GR":
            libs = libs_by_task[task]
            _, _, v_train, v_val = fit_pls_model(
                x_vnir[train_idx],
                y_train,
                x_vnir[val_idx],
                conf["V_PLS"]["scaler"],
                conf["V_PLS"]["ncomp"],
            )
            _, _, l_train, l_val = fit_pls_model(
                libs[train_idx],
                y_train,
                libs[val_idx],
                conf["L_PLS"]["scaler"],
                conf["L_PLS"]["ncomp"],
            )
            _, _, vl_train, vl_val = fit_concat_pls(
                [x_vnir[train_idx], libs[train_idx]],
                y_train,
                [x_vnir[val_idx], libs[val_idx]],
                conf["VL_SF"]["scaler"],
                conf["VL_SF"]["ncomp"],
            )
            _, pred_train, pred_val = fit_linear_meta(
                [v_train, l_train, vl_train],
                y_train.ravel(),
                [v_val, l_val, vl_val],
            )
        else:
            raise ValueError(f"Unknown teacher method {method}")

        train_pred[:, pos] = pred_train
        val_pred[:, pos] = pred_val
        records[task] = {"R2": r2(y_val, pred_val), "RMSE": rmse(y_val, pred_val)}

    return TeacherReport(train_pred, val_pred, records)


def train_single_modal_predictors(
    x_vnir: np.ndarray,
    x_xrf: np.ndarray,
    libs_by_task: Dict[str, np.ndarray],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    pred_train = {"V": {}, "X": {}, "L": {}}
    pred_val = {"V": {}, "X": {}, "L": {}}

    for pos, task in enumerate(TASKS):
        y_train = targets[train_idx, pos].reshape(-1, 1)

        _, _, v_tr, v_te = fit_pls_model(
            x_vnir[train_idx],
            y_train,
            x_vnir[val_idx],
            mode="auto",
            n_components=10,
        )
        _, _, x_tr, x_te = fit_pls_model(
            x_xrf[train_idx],
            y_train,
            x_xrf[val_idx],
            mode="auto",
            n_components=10,
        )
        libs = libs_by_task[task]
        _, _, l_tr, l_te = fit_pls_model(
            libs[train_idx],
            y_train,
            libs[val_idx],
            mode="auto",
            n_components=10,
        )
        pred_train["V"][task] = v_tr
        pred_val["V"][task] = v_te
        pred_train["X"][task] = x_tr
        pred_val["X"][task] = x_te
        pred_train["L"][task] = l_tr
        pred_val["L"][task] = l_te

    return pred_train, pred_val
