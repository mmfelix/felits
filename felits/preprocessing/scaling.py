"""Scaling and sliding-window splitting for time-series forecasting.

This module provides utilities for scaling time-series data and splitting
it into sliding windows for supervised learning using pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

ScalerType = Literal["minmax", "standard", "robust"]


class TimeSeriesScaler:
    """Fit one scaler on features and a second on the target column (dual-scaler pattern).

    This class applies the same scaling transformation to all features, and
    a separate, independent scaling transformation to the target variable.
    This is useful in time-series forecasting to prevent data leakage from
    the target distribution into the feature scaling.

    Parameters
    ----------
    scaling_type : ScalerType, default="minmax"
        The type of scaling to apply. Must be 'minmax', 'standard', or 'robust'.
    feature_range : tuple[float, float], default=(0.0, 1.0)
        Desired range of transformed data. Only used if `scaling_type` is 'minmax'.

    Attributes
    ----------
    scaler_ : Any
        The fitted scaler for the features.
    target_scaler_ : Any
        The fitted scaler for the target variable.
    feature_names_ : list[str]
        The names of the features used during fitting.
    target_name_ : str
        The name of the target variable used during fitting.
    """

    def __init__(
        self,
        scaling_type: ScalerType = "minmax",
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if scaling_type not in ("minmax", "standard", "robust"):
            raise ValueError(
                f"Unknown scaling_type={scaling_type!r}. Must be 'minmax', 'standard', or 'robust'."
            )
        self.scaling_type = scaling_type
        self.feature_range = feature_range
        self.scaler_: Any = None
        self.target_scaler_: Any = None
        self.feature_names_: list[str] = []
        self.target_name_: str = ""

    def _make_scaler(self) -> Any:
        """Create and return the appropriate scikit-learn scaler instance."""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        if self.scaling_type == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        if self.scaling_type == "standard":
            return StandardScaler()
        if self.scaling_type == "robust":
            return RobustScaler()

        raise AssertionError(f"Unknown scaler type {self.scaling_type!r}")

    def fit(self, X: pd.DataFrame, target: str) -> "TimeSeriesScaler":
        """Fit the scalers to the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing both features and the target column.
        target : str
            The name of the target column to scale separately.

        Returns
        -------
        TimeSeriesScaler
            The fitted scaler instance.

        Raises
        ------
        KeyError
            If the `target` column is not found in `X`.
        """
        if target not in X.columns:
            raise KeyError(f"target={target!r} not found in columns: {list(X.columns)}")

        feature_cols = [col for col in X.columns if col != target]
        self.feature_names_ = feature_cols
        self.target_name_ = target

        self.scaler_ = self._make_scaler()
        if feature_cols:
            self.scaler_.fit(X[feature_cols].to_numpy().astype(float))
        else:
            self.scaler_.fit(X.to_numpy().astype(float))

        self.target_scaler_ = self._make_scaler()
        self.target_scaler_.fit(X[target].to_numpy().astype(float).reshape(-1, 1))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the provided data using the fitted scalers.

        If the input contains the target column (set during ``fit``), both
        features and target are scaled independently and returned in the
        original column order. Otherwise only feature columns are scaled.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        np.ndarray
            Scaled array with the same shape as the input.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet.
        """
        if self.scaler_ is None:
            raise RuntimeError("Call `fit` before `transform`.")

        has_target = self.target_name_ in X.columns

        if has_target and self.target_scaler_ is not None:
            feature_cols = [c for c in X.columns if c != self.target_name_]
            scaled_target = self.target_scaler_.transform(
                X[self.target_name_].to_numpy().astype(float).reshape(-1, 1)
            ).ravel()

            if feature_cols:
                scaled_features = self.scaler_.transform(X[feature_cols].to_numpy().astype(float))
            else:
                return self.scaler_.transform(X.to_numpy().astype(float))

            all_cols = list(X.columns)
            result = np.empty((len(X), len(all_cols)), dtype=float)
            feat_idx = 0
            for i, col in enumerate(all_cols):
                if col == self.target_name_:
                    result[:, i] = scaled_target
                else:
                    result[:, i] = scaled_features[:, feat_idx]
                    feat_idx += 1
            return result

        feature_cols = [col for col in self.feature_names_ if col in X.columns]
        return self.scaler_.transform(X[feature_cols].to_numpy().astype(float))

    def fit_transform(self, X: pd.DataFrame, target: str) -> np.ndarray:
        """Fit the scalers and transform the data in one step.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing both features and the target column.
        target : str
            The name of target column to scale separately.

        Returns
        -------
        np.ndarray
            The scaled feature matrix.
        """
        return self.fit(X, target).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the scaled feature data.

        Parameters
        ----------
        X : np.ndarray
            The scaled data to inverse transform.

        Returns
        -------
        np.ndarray
            The original scale data.

        Raises
        ------
        RuntimeError
            If `fit` has not been called yet.
        """
        if self.scaler_ is None:
            raise RuntimeError("Call `fit` before `inverse_transform`.")
        return self.scaler_.inverse_transform(np.asarray(X, dtype=float))

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform the scaled target data.

        Parameters
        ----------
        y : np.ndarray
            The scaled target data to inverse transform.

        Returns
        -------
        np.ndarray
            The original scale target data.

        Raises
        ------
        RuntimeError
            If `fit` has not been called yet.
        """
        if self.target_scaler_ is None:
            raise RuntimeError("Call `fit` before `inverse_transform_target`.")

        arr = np.asarray(y, dtype=float)
        shape = arr.shape
        flat = arr.reshape(-1, 1)
        out = self.target_scaler_.inverse_transform(flat)
        return out.reshape(shape)


@dataclass
class WindowedSplit:
    """Result of :meth:`SlidingWindowSplitter.split`.

    Attributes
    ----------
    X : np.ndarray
        The 3D feature matrix of shape (n_samples, hist_window, n_features).
    y : np.ndarray
        The 2D target matrix of shape (n_samples, pred_window).
    scaler : TimeSeriesScaler or None
        The fitted scaler instance, if scaling was applied.
    """

    X: np.ndarray
    y: np.ndarray
    scaler: Optional[TimeSeriesScaler] = None


class SlidingWindowSplitter:
    """Convert a DataFrame into sliding windows for supervised learning.

    This class handles the conversion of a 2D time-series DataFrame into
    3D windows (samples, timesteps, features) and corresponding 2D targets
    (samples, pred_window), mimicking the legacy ``input_output_splitter``.

    Parameters
    ----------
    target : str
        The name of the target column to predict.
    hist_window : int, default=24
        The number of historical time steps to include in each window.
    pred_window : int, default=24
        The number of future time steps to predict for each window.
    jump : bool, default=True
        If True, windows do not overlap (step size = `pred_window`).
        If False, windows overlap by 1 time step (step size = 1).
    scaling_type : ScalerType or None, default="minmax"
        The type of scaling to apply. If None, no scaling is applied.
    day_aligned : bool, default=False
        If True, anchor every target to a calendar-day boundary at 00:00:
        each sample is ``(X, y)`` where ``y`` covers the 24 hours of a
        calendar day starting at 00:00 (``X0 = hour 0``) and ``X`` is the
        ``HIST_WINDOW`` hours immediately preceding 00:00 of that day.
        Hours before the first valid 00:00 (i.e. the first 00:00 that has
        at least ``HIST_WINDOW`` preceding rows in ``df``) are left as an
        *offset* and are not used as a target start. With ``day_aligned=True``,
        ``PRED_WINDOW`` must equal 24 (one full calendar day). When True,
        ``jump`` is ignored (one window per calendar day).
    jump : bool, default=True
        If True, windows do not overlap (step size = `pred_window`).
        If False, windows overlap by 1 time step (step size = 1).
    scaling_type : ScalerType or None, default="minmax"
        The type of scaling to apply. If None, no scaling is applied.
    """

    def __init__(
        self,
        target: str,
        hist_window: int = 24,
        pred_window: int = 24,
        jump: bool = True,
        scaling_type: Optional[ScalerType] = "minmax",
        day_aligned: bool = False,
    ) -> None:
        if hist_window < 1 or pred_window < 1:
            raise ValueError("`hist_window` and `pred_window` must be >= 1")
        if day_aligned and pred_window != 24:
            raise ValueError(
                f"`day_aligned=True` requires `pred_window=24` (one calendar day), "
                f"got {pred_window}."
            )

        self.target = target
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.jump = jump
        self.scaling_type = scaling_type
        self.day_aligned = day_aligned

    def split(
        self, df: pd.DataFrame, scaler: Optional[TimeSeriesScaler] = None
    ) -> WindowedSplit:
        """Split the DataFrame into sliding windows.

        Parameters
        ----------
        df : pd.DataFrame
            The input time-series DataFrame.
        scaler : TimeSeriesScaler, optional
            A pre-fitted scaler to use for transforming ``df``. If provided,
            it is also returned in the resulting :class:`WindowedSplit` so
            that downstream splits (e.g. val/test) share the same scaling
            learned on train. If ``None`` (default) and ``scaling_type`` is
            not ``None``, a fresh ``TimeSeriesScaler`` is fit on ``df`` as
            before.

        Returns
        -------
        WindowedSplit
            A dataclass containing the 3D feature matrix `X`, the 2D target
            matrix `y`, and the optional fitted `scaler`.

        Raises
        ------
        KeyError
            If the `target` column is not found in `df`.
        ValueError
            If the DataFrame is too short for the requested window sizes,
            or if a pre-fitted ``scaler`` is inconsistent with ``df``.
        """
        if self.target not in df.columns:
            raise KeyError(f"target={self.target!r} not found in columns: {list(df.columns)}")

        n = len(df)
        if n < self.hist_window + self.pred_window:
            raise ValueError(
                f"DataFrame too short (length {n}) for the requested "
                f"hist_window ({self.hist_window}) and pred_window ({self.pred_window})."
            )

        if scaler is not None and self.scaling_type is None:
            raise ValueError(
                "A pre-fitted scaler was passed but scaling_type=None; "
                "either set scaling_type or pass scaler=None."
            )

        if scaler is not None:
            if scaler.scaler_ is None or scaler.target_scaler_ is None:
                raise ValueError(
                    "The provided scaler is not fitted. Call `scaler.fit(df, target)` first."
                )
            if scaler.target_name_ != self.target:
                raise ValueError(
                    f"Pre-fitted scaler was fit on target={scaler.target_name_!r} "
                    f"but splitter target={self.target!r}."
                )
            expected_feat_cols = [c for c in df.columns if c != self.target]
            if list(scaler.feature_names_) != expected_feat_cols:
                raise ValueError(
                    "Pre-fitted scaler feature_names_ do not match df columns "
                    f"(excluding target). Got {list(scaler.feature_names_)}, "
                    f"expected {expected_feat_cols}."
                )
            data = scaler.transform(df)
        elif self.scaling_type is not None:
            scaler = TimeSeriesScaler(scaling_type=self.scaling_type).fit(df, self.target)
            data = scaler.transform(df)
        else:
            data = df.to_numpy().astype(float)

        target_index = list(df.columns).index(self.target)

        if self.day_aligned:
            return self._split_day_aligned(
                data, target_index, scaler, df.index
            )

        if self.jump:
            step = self.pred_window
            n_samples = (n - self.hist_window) // self.pred_window
        else:
            step = 1
            n_samples = n - self.hist_window - self.pred_window + 1

        X = np.empty((n_samples, self.hist_window, data.shape[1]), dtype=np.float32)
        y = np.empty((n_samples, self.pred_window), dtype=np.float32)

        for i in range(n_samples):
            start_in = i * step
            start_out = start_in + self.hist_window
            end_out = start_out + self.pred_window
            X[i] = data[start_in:start_out, :]
            y[i] = data[start_out:end_out, target_index]

        return WindowedSplit(X=X, y=y, scaler=scaler)

    def _split_day_aligned(
        self,
        data: np.ndarray,
        target_index: int,
        scaler: Optional[TimeSeriesScaler],
        index: pd.DatetimeIndex,
    ) -> WindowedSplit:
        """Anchor every target to a calendar-day boundary at 00:00.

        Each sample is ``(X, y)`` where ``y`` covers the 24 hours of a
        calendar day starting at 00:00 (``X0 = hour 0``) and ``X`` is the
        ``HIST_WINDOW`` hours immediately preceding 00:00 of that day.

        Hours before the first valid 00:00 (the first 00:00 with at least
        ``HIST_WINDOW`` preceding rows in ``index``) form an *offset* and
        are not used as a target start.
        """
        n = len(data)
        hist = self.hist_window
        pred = self.pred_window

        if n < hist + pred:
            raise ValueError(
                f"DataFrame too short (length {n}) for hist_window={hist} and "
                f"pred_window={pred}."
            )
        if len(index) != n:
            raise ValueError(
                f"index length ({len(index)}) does not match data length ({n})."
            )

        # Positional indices whose original timestamp is at 00:00.
        hours = np.asarray(index.hour)
        midnight_mask = hours == 0
        midnight_positions = np.flatnonzero(midnight_mask).astype(np.int64)

        # Keep only midnights that have enough preceding history and enough
        # room after for the full target window.
        valid = midnight_positions[midnight_positions >= hist]
        valid = valid[valid + pred <= n]
        n_samples = len(valid)

        if n_samples == 0:
            raise ValueError(
                f"No valid day-aligned windows: need midnight positions m with "
                f"m >= {hist} and m + {pred} <= {n}, but the DataFrame has length {n}."
            )

        X = np.empty((n_samples, hist, data.shape[1]), dtype=np.float32)
        y = np.empty((n_samples, pred), dtype=np.float32)

        for k, m in enumerate(valid):
            X[k] = data[m - hist:m, :]
            y[k] = data[m:m + pred, target_index]

        return WindowedSplit(X=X, y=y, scaler=scaler)
