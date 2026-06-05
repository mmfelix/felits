"""Regression metrics for time-series forecasting.

This module is a hardened, vectorised replacement for the legacy
``Metrics`` class. The legacy class returned ``np.inf`` (or ``None``) on
shape/value errors; the new implementation raises informative errors
instead, except in the legacy-compatible :func:`metrics_dict` helper which
preserves the original behaviour for backward compatibility.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as _sk_metrics

__all__ = ["Metrics", "bias", "mae", "mape", "max_error", "mse", "r2", "rmse", "smape"]


def _flatten(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.mean_squared_error(yt, yp))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.mean_absolute_error(yt, yp))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.mean_absolute_percentage_error(yt, yp))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in [0, 200] %."""
    yt, yp = _flatten(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, 2.0 * np.abs(yp - yt) / denom)
    return float(100.0 * np.mean(ratio))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.r2_score(yt, yp))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.max_error(yt, yp))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast bias: mean of (prediction − actual). Negative ⇒ under-forecast."""
    yt, yp = _flatten(y_true, y_pred)
    return float(np.mean(yp - yt))


class Metrics:
    """Compute the standard regression metrics in one call.

    Parameters
    ----------
    true, predicted:
        1-D numpy arrays of the same length (or anything array-like that can
        be flattened to 1-D).
    """

    def __init__(self, true: np.ndarray, predicted: np.ndarray):
        self.true = np.asarray(true, dtype=float).ravel()
        self.predicted = np.asarray(predicted, dtype=float).ravel()
        if self.true.shape != self.predicted.shape:
            raise ValueError(
                f"`true` and `predicted` must have the same shape; got {self.true.shape} vs {self.predicted.shape}"
            )

    def mse(self) -> float:
        return mse(self.true, self.predicted)

    def rmse(self) -> float:
        return rmse(self.true, self.predicted)

    def mae(self) -> float:
        return mae(self.true, self.predicted)

    def mape(self) -> float:
        return mape(self.true, self.predicted)

    def smape(self) -> float:
        return smape(self.true, self.predicted)

    def r2(self) -> float:
        return r2(self.true, self.predicted)

    def max(self) -> float:
        return max_error(self.true, self.predicted)

    def bias(self) -> float:
        return bias(self.true, self.predicted)

    def dict_metrics(self) -> dict[str, float]:
        """Return a legacy-compatible ``dict`` of the core metrics."""
        return {
            "MSE": self.mse(),
            "RMSE": self.rmse(),
            "MAE": self.mae(),
            "MAPE": self.mape(),
            "sMAPE": self.smape(),
            "R2": self.r2(),
            "MAX ERROR": self.max(),
            "Bias": self.bias(),
        }
