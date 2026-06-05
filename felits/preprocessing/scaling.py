"""Scaling and sliding-window splitting for time-series forecasting.

Accepts both ``pandas`` and ``polars`` DataFrames at the public API.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .._compat import to_polars

ScalerType = Literal["minmax", "standard", "robust"]

_SCALER_FACTORIES: dict[str, type] = {
    "minmax": "MinMaxScaler",
    "standard": "StandardScaler",
    "robust": "RobustScaler",
}


class TimeSeriesScaler:
    """Fit one scaler on features and a second on the target column (dual-scaler pattern).

    Accepts ``pd.DataFrame`` or ``pl.DataFrame`` at ``fit`` / ``transform``.
    """

    def __init__(
        self, scaling_type: ScalerType = "minmax", feature_range: tuple[float, float] = (0.0, 1.0)
    ):
        if scaling_type not in _SCALER_FACTORIES:
            raise ValueError(f"Unknown scaling_type={scaling_type!r}")
        self.scaling_type = scaling_type
        self.feature_range = feature_range
        self.scaler_: object | None = None
        self.target_scaler_: object | None = None
        self.feature_names_: list[str] = []
        self.target_name_: str = ""

    def _make_scaler(self):
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        if self.scaling_type == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.scaling_type == "standard":
            return StandardScaler()
        elif self.scaling_type == "robust":
            return RobustScaler()
        raise AssertionError(f"Unknown scaler type {self.scaling_type!r}")

    def fit(self, X, target: str) -> "TimeSeriesScaler":
        pdf = to_polars(X)
        if target not in pdf.columns:
            raise KeyError(f"target={target!r} not found in columns.")
        pdf_np = pdf.to_numpy()
        self.scaler_ = self._make_scaler()
        self.scaler_.fit(pdf_np.astype(float))
        self.target_scaler_ = copy.deepcopy(self.scaler_)
        self.target_scaler_.fit(pdf[target].to_numpy().astype(float).reshape(-1, 1))
        self.feature_names_ = pdf.columns
        self.target_name_ = target
        return self

    def transform(self, X) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Call `fit` before `transform`.")
        pdf = to_polars(X)
        return self.scaler_.transform(pdf.to_numpy().astype(float))

    def fit_transform(self, X, target: str) -> np.ndarray:
        return self.fit(X, target).transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Call `fit` before `inverse_transform`.")
        return self.scaler_.inverse_transform(np.asarray(X, dtype=float))

    def inverse_transform_target(self, y) -> np.ndarray:
        if self.target_scaler_ is None:
            raise RuntimeError("Call `fit` before `inverse_transform_target`.")
        arr = np.asarray(y, dtype=float)
        shape = arr.shape
        flat = arr.reshape(-1, 1)
        out = self.target_scaler_.inverse_transform(flat)
        return out.reshape(shape)


@dataclass
class WindowedSplit:
    """Result of :meth:`SlidingWindowSplitter.split`."""

    X: np.ndarray
    y: np.ndarray
    scaler: Optional[TimeSeriesScaler] = None


class SlidingWindowSplitter:
    """Convert a ``DataFrame`` (polars or pandas) into windows.

    Mirrors the legacy ``input_output_splitter``.
    """

    def __init__(
        self,
        target: str,
        hist_window: int = 24,
        pred_window: int = 24,
        jump: bool = True,
        scaling_type: Optional[ScalerType] = "minmax",
    ):
        if hist_window < 1 or pred_window < 1:
            raise ValueError("`hist_window` and `pred_window` must be >= 1")
        self.target = target
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.jump = jump
        self.scaling_type = scaling_type

    def split(self, df) -> WindowedSplit:
        pdf = to_polars(df)
        if self.target not in pdf.columns:
            raise KeyError(f"target={self.target!r} not found in columns.")
        n = len(pdf)
        if n < self.hist_window + self.pred_window:
            raise ValueError("DataFrame too short for the requested hist/pred windows.")

        scaler: Optional[TimeSeriesScaler] = None
        if self.scaling_type is not None:
            scaler = TimeSeriesScaler(scaling_type=self.scaling_type).fit(pdf, self.target)
            data = scaler.transform(pdf)
        else:
            data = pdf.to_numpy().astype(float)

        target_index = pdf.columns.index(self.target)
        if self.jump:
            step = self.pred_window
            n_samples = (n - self.hist_window) // self.pred_window
        else:
            step = 1
            n_samples = n - self.hist_window - self.pred_window + 1

        X = np.empty((n_samples, self.hist_window, data.shape[1]), dtype="float32")
        y = np.empty((n_samples, self.pred_window), dtype="float32")
        for i in range(n_samples):
            start_in = i * step
            start_out = start_in + self.hist_window
            end_out = start_out + self.pred_window
            X[i] = data[start_in:start_out, :]
            y[i] = data[start_out:end_out, target_index]
        return WindowedSplit(X=X, y=y, scaler=scaler)
