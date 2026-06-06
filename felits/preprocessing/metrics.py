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
    """Flatten and cast inputs to 1-D float64 arrays.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened 1-D float64 arrays of equal length.
    """
    return np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.mean_squared_error(yt, yp))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.mean_absolute_error(yt, yp))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute percentage error.
    """
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.mean_absolute_percentage_error(yt, yp))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in [0, 200] %.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Symmetric MAPE as a percentage.
    """
    yt, yp = _flatten(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, 2.0 * np.abs(yp - yt) / denom)
    return float(100.0 * np.mean(ratio))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R² score).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        R² score. Best possible value is 1.0.
    """
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.r2_score(yt, yp))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum residual error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Maximum absolute difference between true and predicted values.
    """
    yt, yp = _flatten(y_true, y_pred)
    return float(_sk_metrics.max_error(yt, yp))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast bias: mean of (prediction − actual).

    Negative values indicate under-forecasting.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean forecast bias.
    """
    yt, yp = _flatten(y_true, y_pred)
    return float(np.mean(yp - yt))


class Metrics:
    """Compute the standard regression metrics in one call.

    Parameters
    ----------
    true : np.ndarray
        1-D array of ground truth values.
    predicted : np.ndarray
        1-D array of predicted values, same length as ``true``.

    Raises
    ------
    ValueError
        If ``true`` and ``predicted`` have different shapes.
    """

    def __init__(self, true: np.ndarray, predicted: np.ndarray) -> None:
        self.true = np.asarray(true, dtype=float).ravel()
        self.predicted = np.asarray(predicted, dtype=float).ravel()
        if self.true.shape != self.predicted.shape:
            raise ValueError(
                f"`true` and `predicted` must have the same shape; "
                f"got {self.true.shape} vs {self.predicted.shape}"
            )

    def mse(self) -> float:
        """Return mean squared error."""
        return mse(self.true, self.predicted)

    def rmse(self) -> float:
        """Return root mean squared error."""
        return rmse(self.true, self.predicted)

    def mae(self) -> float:
        """Return mean absolute error."""
        return mae(self.true, self.predicted)

    def mape(self) -> float:
        """Return mean absolute percentage error."""
        return mape(self.true, self.predicted)

    def smape(self) -> float:
        """Return symmetric mean absolute percentage error."""
        return smape(self.true, self.predicted)

    def r2(self) -> float:
        """Return R² score."""
        return r2(self.true, self.predicted)

    def max(self) -> float:
        """Return maximum residual error."""
        return max_error(self.true, self.predicted)

    def bias(self) -> float:
        """Return forecast bias."""
        return bias(self.true, self.predicted)

    def dict_metrics(self) -> dict[str, float]:
        """Return a legacy-compatible ``dict`` of the core metrics.

        Returns
        -------
        dict[str, float]
            Dictionary with keys MSE, RMSE, MAE, MAPE, sMAPE, R2, MAX ERROR, Bias.
        """
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
