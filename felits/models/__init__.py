"""Predictive models for time series forecasting.

Two model families are exposed:

- Classical, sklearn-compatible forecasters: :class:`XGBoostForecaster`,
  :class:`RandomForestForecaster`, :class:`LinearForecaster`. These work
  out-of-the-box without TensorFlow.
- Deep-learning models built with ``tf.keras``: :class:`RNNAttentionModel`
  and :class:`RNNBasedModel` (with the :class:`BahdanauAttention` layer).
  They require the optional ``[dl]`` extra.
"""

from __future__ import annotations

from .base import ForecasterProtocol, flatten_time_series, is_dl_available
from .dl import (
    BahdanauAttention,
    LSTMAttentionForecaster,
    PatchTSTForecaster,
    RNNAttentionModel,
    RNNBasedModel,
)
from .sklearn import (
    LightGBMForecaster,
    LinearForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

__all__ = [
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
    "flatten_time_series",
    "is_dl_available",
]
