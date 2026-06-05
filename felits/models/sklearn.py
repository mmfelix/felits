"""Sklearn-compatible forecasters wrapping XGBoost, RandomForest, and LinearRegression."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base import _SklearnForecaster


class XGBoostForecaster(_SklearnForecaster):
    """XGBoost regressor wrapper for time-series windows."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 0,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: object | None = None

    def _make_base_model(self):
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "XGBoostForecaster requires xgboost. Install with: pip install 'felits[xgb]'"
            ) from exc
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class RandomForestForecaster(_SklearnForecaster):
    """Sklearn :class:`RandomForestRegressor` wrapper for time-series windows."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 0,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: object | None = None

    def _make_base_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class LinearForecaster(_SklearnForecaster):
    """Sklearn :class:`LinearRegression` wrapper for time-series windows."""

    def _make_base_model(self):
        return LinearRegression()
