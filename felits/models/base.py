"""Base classes and utilities for time-series forecasting models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def is_dl_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _load_dl():
    if not is_dl_available():
        raise ImportError(
            "Deep-learning models require TensorFlow. Install with: pip install 'felits[dl]'"
        )
    from tensorflow.keras import backend as backend
    from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout, Input, Layer
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import plot_model

    return {
        "K": backend,
        "GRU": GRU,
        "LSTM": LSTM,
        "Bidirectional": Bidirectional,
        "Dense": Dense,
        "Dropout": Dropout,
        "Input": Input,
        "Layer": Layer,
        "Model": Model,
        "plot_model": plot_model,
    }


def _build_rnn(model_type: str, num_units: int, with_attention: bool):
    sym = _load_dl()
    LSTM, GRU, Bidirectional = sym["LSTM"], sym["GRU"], sym["Bidirectional"]
    if model_type == "LSTM":
        layer = LSTM(units=num_units, return_sequences=with_attention)
    elif model_type == "GRU":
        layer = GRU(units=num_units, return_sequences=with_attention)
    elif model_type == "BiLSTM":
        layer = Bidirectional(LSTM(units=num_units, return_sequences=with_attention))
    elif model_type == "BiGRU":
        layer = Bidirectional(GRU(units=num_units, return_sequences=with_attention))
    else:
        raise ValueError(f"Unknown RNN type {model_type!r}; choose from LSTM/GRU/BiLSTM/BiGRU")
    return layer


class _SklearnForecaster(BaseEstimator, RegressorMixin):
    """Common sklearn-compatible base for the wrappers in this module."""

    def fit(self, X, y):
        X2d = self._flatten(X)
        y2d = np.asarray(y, dtype=float)
        if y2d.ndim == 2:
            from sklearn.multioutput import MultiOutputRegressor

            self.model_ = MultiOutputRegressor(self._make_base_model())
            self.model_.fit(X2d, y2d)
        else:
            y1d = y2d.ravel()
            self._make_base_model().fit(X2d, y1d)
        return self

    def predict(self, X):
        X2d = self._flatten(X)
        pred = np.asarray(self.model_.predict(X2d))
        if pred.ndim == 1:
            return pred
        return pred

    def get_params(self, deep: bool = True):
        return self.model_.get_params(deep=deep)

    def set_params(self, **params):
        self.model_.set_params(**params)
        return self

    def _make_base_model(self):
        raise NotImplementedError

    @staticmethod
    def _flatten(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 3:
            n, t, f = arr.shape
            return arr.reshape(n, t * f)
        return arr
