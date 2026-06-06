"""Base interfaces and utilities for time-series forecasting models."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ForecasterProtocol(Protocol):
    """Protocol defining the minimal interface for a forecaster.

    This protocol ensures that any model passed to utility functions
    implements the standard ``fit`` and ``predict`` methods with
    compatible signatures.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def is_dl_available() -> bool:
    """Check if TensorFlow is available for deep learning models.

    Returns
    -------
    bool
        True if TensorFlow is installed, False otherwise.
    """
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def flatten_time_series(X: np.ndarray) -> np.ndarray:
    """Flatten 3D time-series windows to 2D for tabular model compatibility.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, timesteps, features) or
        (n_samples, features).

    Returns
    -------
    np.ndarray
        Flattened array of shape (n_samples, timesteps * features) if
        input is 3D, or the unchanged 2D array.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 3:
        n, t, f = arr.shape
        return arr.reshape(n, t * f)
    return arr
