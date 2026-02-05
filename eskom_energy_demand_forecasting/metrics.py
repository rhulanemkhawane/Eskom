from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "exclude_zeros",
    epsilon: float = 1e-6,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if method == "exclude_zeros":
        mask = y_true != 0
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    if method == "epsilon":
        denom = np.maximum(np.abs(y_true), epsilon)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    raise ValueError(f"Unknown MAPE method: {method}")


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_method: str, mape_epsilon: float) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": safe_mape(y_true, y_pred, method=mape_method, epsilon=mape_epsilon),
    }
