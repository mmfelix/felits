"""FELITS: Feature Engineering and Large-scale Integration for Time Series.

Top-level package exposing the most commonly used classes and functions
through a flat, ergonomic API::

    from felits import (
        HampelFilter, TimeSeriesScaler, SlidingWindowSplitter, Metrics,
        cyclical_encode, fft_features, tsfresh_extract,
        FeatureSelector, granger_feature_selection, shap_feature_selection,
        XGBoostForecaster, RandomForestForecaster, RNNBasedModel,
        OptunaOptimizer, deep_shap_selector,
    )

Submodules are still importable for advanced use::

    from felits.preprocessing import iqr_outlier_detection
    from felits.feature_selection import mrmr_selection
"""

from __future__ import annotations

from . import data, models, optimization
from .feature_extraction import (
    cyclical_encode,
    extract_all_features,
    fats_extract,
    fft_features,
    lag_features,
    rolling_statistics,
    shift_features,
    spectral_entropy,
    tsfresh_extract,
    wavelet_features,
)
from .feature_selection import (
    FeatureSelector,
    adaptive_lasso_selection,
    elastic_net_selection,
    granger_feature_selection,
    lasso_selection,
    lime_explain_instance,
    mrmr_selection,
    mutual_information_ksg,
    permutation_importance_selection,
    rf_importance_selection,
    select_features,
    shap_feature_selection,
    shap_interaction_selection,
    xgboost_importance_selection,
)
from .optimization import OptunaOptimizer
from .preprocessing import (
    DecompositionResult,
    HampelFilter,
    Metrics,
    SlidingWindowSplitter,
    TimeSeriesScaler,
    forward_fill,
    hampel_filter,
    iqr_outlier_detection,
    linear_interpolate,
    mae,
    mape,
    max_error,
    mse,
    r2,
    rmse,
    seasonal_adjust,
    smape,
    stl_decompose,
    three_sigma_filter,
    time_aware_interpolate,
)
from .xai import deep_shap_selector, explain_forecast

__version__ = "0.3.0"
__author__ = "Félix Morales Mareco"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # data
    "data",
    "models",
    "optimization",
    # preprocessing
    "DecompositionResult",
    "HampelFilter",
    "Metrics",
    "SlidingWindowSplitter",
    "TimeSeriesScaler",
    "forward_fill",
    "hampel_filter",
    "iqr_outlier_detection",
    "linear_interpolate",
    "mae",
    "mape",
    "max_error",
    "mse",
    "r2",
    "rmse",
    "seasonal_adjust",
    "smape",
    "stl_decompose",
    "three_sigma_filter",
    "time_aware_interpolate",
    # feature_extraction
    "cyclical_encode",
    "extract_all_features",
    "fats_extract",
    "fft_features",
    "lag_features",
    "rolling_statistics",
    "shift_features",
    "spectral_entropy",
    "tsfresh_extract",
    "wavelet_features",
    # feature_selection
    "FeatureSelector",
    "adaptive_lasso_selection",
    "elastic_net_selection",
    "granger_feature_selection",
    "lasso_selection",
    "lime_explain_instance",
    "mrmr_selection",
    "mutual_information_ksg",
    "permutation_importance_selection",
    "rf_importance_selection",
    "select_features",
    "shap_feature_selection",
    "shap_interaction_selection",
    "xgboost_importance_selection",
    # optimization
    "OptunaOptimizer",
    # xai
    "deep_shap_selector",
    "explain_forecast",
]

# Re-export the DL models and sklearn forecasters at the top level
from .models import (
    BahdanauAttention,
    ForecasterProtocol,
    LightGBMForecaster,
    LinearForecaster,
    LSTMAttentionForecaster,
    PatchTSTForecaster,
    RandomForestForecaster,
    RNNAttentionModel,
    RNNBasedModel,
    XGBoostForecaster,
    is_dl_available,
)

__all__ += [
    "BahdanauAttention",
    "ForecasterProtocol",
    "LightGBMForecaster",
    "LinearForecaster",
    "LSTMAttentionForecaster",
    "PatchTSTForecaster",
    "RNNAttentionModel",
    "RNNBasedModel",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "is_dl_available",
]
