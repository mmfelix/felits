"""Signal cleaning and preparation for time series."""

from __future__ import annotations

from .decomposition import DecompositionResult, extract_components, seasonal_adjust, stl_decompose
from .imputation import forward_fill, linear_interpolate, time_aware_interpolate
from .metrics import Metrics, bias, mae, mape, max_error, mse, r2, rmse, smape
from .outliers import HampelFilter, hampel_filter, iqr_outlier_detection, three_sigma_filter
from .scaling import SlidingWindowSplitter, TimeSeriesScaler, WindowedSplit

__all__ = [
    "DecompositionResult",
    "extract_components",
    "seasonal_adjust",
    "stl_decompose",
    "forward_fill",
    "linear_interpolate",
    "time_aware_interpolate",
    "Metrics",
    "bias",
    "mae",
    "mape",
    "max_error",
    "mse",
    "r2",
    "rmse",
    "smape",
    "HampelFilter",
    "hampel_filter",
    "iqr_outlier_detection",
    "three_sigma_filter",
    "SlidingWindowSplitter",
    "TimeSeriesScaler",
    "WindowedSplit",
]
