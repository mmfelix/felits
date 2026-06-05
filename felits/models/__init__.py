"""Predictive models for time series forecasting.

Two model families are exposed:

- Classical, sklearn-compatible forecasters: :class:`XGBoostForecaster`,
  :class:`RandomForestForecaster`, :class:`LinearForecaster`. These work
  out-of-the-box without TensorFlow.
- Deep-learning models built with ``tf.keras``: :class:`RNNBasedModel`
  and :class:`RNNAttentionModel` (with the :class:`BahdanauAttention`
  layer). They require the optional ``[dl]`` extra.
"""

from __future__ import annotations

from .base import _SklearnForecaster, is_dl_available
from .dl import BahdanauAttention, RNNAttentionModel, RNNBasedModel
from .sklearn import LinearForecaster, RandomForestForecaster, XGBoostForecaster

__all__ = [
    "BahdanauAttention",
    "LinearForecaster",
    "RNNAttentionModel",
    "RNNBasedModel",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "_SklearnForecaster",
    "is_dl_available",
]
