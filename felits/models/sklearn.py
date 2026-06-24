"""Scikit-learn compatible forecasters for time-series windows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base import flatten_time_series

_CUDA_AVAILABLE: bool | None = None


def _resolve_device(device: str, xgb: Any) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' using a minimal probe with cache."""
    if device != "auto":
        return device
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return "cuda" if _CUDA_AVAILABLE else "cpu"
    try:
        probe = xgb.XGBRegressor(
            n_estimators=1, max_depth=1, tree_method="hist", device="cuda"
        )
        probe.fit(np.zeros((2, 2), dtype=float), np.zeros(2, dtype=float))
        _CUDA_AVAILABLE = True
        return "cuda"
    except Exception:
        _CUDA_AVAILABLE = False
        return "cpu"


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """XGBoost regressor wrapper for time-series windows.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.1
        Boosting learning rate.
    random_state : int, default=0
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors.
    device : str, default="auto"
        Device for training. ``"auto"`` probes for CUDA and falls back to
        ``"cpu"`` if unavailable. ``"cuda"`` / ``"cpu"`` are passed through
        to XGBoost as-is.
    subsample : float, default=0.8
        Subsample ratio of the training instances.
    colsample_bytree : float, default=0.8
        Subsample ratio of columns when constructing each tree.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight (hessian) needed in a child.
    reg_lambda : float, default=1.0
        L2 regularization term on weights.
    reg_alpha : float, default=0.0
        L1 regularization term on weights.
    max_bin : int, default=256
        Maximum number of discrete bins for ``tree_method="hist"``.

    Attributes
    ----------
    _model : Any
        The underlying fitted XGBoost model.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 0,
        n_jobs: int = -1,
        device: str = "auto",
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_bin: int = 256,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.max_bin = max_bin
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostForecaster":
        """Fit the XGBoost model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).
        y : np.ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).

        Returns
        -------
        XGBoostForecaster
            The fitted estimator instance.

        Raises
        ------
        ImportError
            If the ``xgboost`` package is not installed.
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoostForecaster requires xgboost. Install with: pip install 'felits[xgb]'"
            ) from exc

        X_flat = flatten_time_series(X)
        y_flat = np.asarray(y, dtype=float)

        device = _resolve_device(self.device, xgb)

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",  # Best practice for speed and memory
            enable_categorical=True,  # Handles cyclic features natively
            device=device,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            max_bin=self.max_bin,
        )
        self._model.fit(X_flat, y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted XGBoost model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,) or (n_samples, n_targets).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X_flat = flatten_time_series(X)
        return np.asarray(self._model.predict(X_flat))


class RandomForestForecaster(BaseEstimator, RegressorMixin):
    """Scikit-learn RandomForestRegressor wrapper for time-series windows.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    random_state : int, default=0
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors.

    Attributes
    ----------
    _model : RandomForestRegressor
        The underlying fitted scikit-learn model.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestForecaster":
        """Fit the Random Forest model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).
        y : np.ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).

        Returns
        -------
        RandomForestForecaster
            The fitted estimator instance.
        """
        X_flat = flatten_time_series(X)
        y_flat = np.asarray(y, dtype=float)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_flat, y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted Random Forest model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,) or (n_samples, n_targets).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X_flat = flatten_time_series(X)
        return np.asarray(self._model.predict(X_flat))


class LinearForecaster(BaseEstimator, RegressorMixin):
    """Scikit-learn LinearRegression wrapper for time-series windows.

    Attributes
    ----------
    _model : LinearRegression
        The underlying fitted scikit-learn model.
    """

    def __init__(self) -> None:
        self._model: LinearRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearForecaster":
        """Fit the Linear Regression model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).
        y : np.ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).

        Returns
        -------
        LinearForecaster
            The fitted estimator instance.
        """
        X_flat = flatten_time_series(X)
        y_flat = np.asarray(y, dtype=float)

        self._model = LinearRegression()
        self._model.fit(X_flat, y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted Linear Regression model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,) or (n_samples, n_targets).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X_flat = flatten_time_series(X)
        return np.asarray(self._model.predict(X_flat))


class LightGBMForecaster(BaseEstimator, RegressorMixin):
    """LightGBM regressor wrapper for time-series windows.

    LightGBM uses leaf-wise tree growth, which often achieves lower loss
    than depth-wise growth (XGBoost) and is significantly faster and more
    memory-efficient on large datasets.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of boosting rounds.
    max_depth : int, default=-1
        Maximum tree depth. -1 means no limit.
    learning_rate : float, default=0.1
        Boosting learning rate.
    random_state : int, default=0
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means using all processors.

    Attributes
    ----------
    _model : Any
        The underlying fitted LightGBM model.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMForecaster":
        """Fit the LightGBM model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).
        y : np.ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).

        Returns
        -------
        LightGBMForecaster
            The fitted estimator instance.

        Raises
        ------
        ImportError
            If the ``lightgbm`` package is not installed.
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBMForecaster requires lightgbm. Install with: pip install lightgbm"
            ) from exc

        X_flat = flatten_time_series(X)
        y_flat = np.asarray(y, dtype=float)

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_flat, y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted LightGBM model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or
            (n_samples, timesteps, features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,) or (n_samples, n_targets).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X_flat = flatten_time_series(X)
        return np.asarray(self._model.predict(X_flat))
