"""Ensemble / tree-based feature importance.

Wrappers around ``RandomForestRegressor``, ``XGBoost`` (optional) and
``sklearn.inspection.permutation_importance``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

__all__ = [
    "permutation_importance_selection",
    "rf_importance_selection",
    "xgboost_importance_selection",
]


def _to_arrays(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        Xarr = X.to_numpy(dtype=float)
    else:
        Xarr = np.asarray(X, dtype=float)
        cols = [f"x{i}" for i in range(Xarr.shape[1])]
    yarr = np.asarray(y, dtype=float).ravel()
    return Xarr, yarr, cols


def _select_from_scores(
    cols: list[str], scores: np.ndarray, threshold: float
) -> tuple[list[str], dict[str, float]]:
    importance = {c: float(s) for c, s in zip(cols, scores, strict=True)}
    max_s = max(scores) if scores.size and scores.max() > 0 else 1.0
    selected = [c for c, s in importance.items() if s >= threshold * max_s]
    return selected, importance


def rf_importance_selection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    threshold: float = 0.01,
    n_estimators: int = 200,
    random_state: int = 0,
) -> tuple[list[str], dict[str, float]]:
    """Random-forest impurity-based importance."""
    Xarr, yarr, cols = _to_arrays(X, y)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1).fit(
        Xarr, yarr
    )
    return _select_from_scores(cols, rf.feature_importances_, threshold)


def xgboost_importance_selection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    threshold: float = 0.01,
    n_estimators: int = 200,
    random_state: int = 0,
) -> tuple[list[str], dict[str, float]]:
    """XGBoost gain-based importance (requires the optional ``xgboost`` dep)."""
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "xgboost_importance_selection requires xgboost. Install with: pip install 'felits[xgb]'"
        ) from exc
    Xarr, yarr, cols = _to_arrays(X, y)
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, tree_method="hist"
    ).fit(Xarr, yarr)
    return _select_from_scores(cols, model.feature_importances_, threshold)


def permutation_importance_selection(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    threshold: float = 0.01,
    n_repeats: int = 10,
    random_state: int = 0,
    scoring: Callable | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Permutation importance (model-agnostic).

    Parameters
    ----------
    model:
        A fitted estimator with a ``predict`` method.
    X, y:
        Held-out data for the permutation test.
    threshold:
        Relative threshold against the max importance (0–1).
    """
    Xarr, yarr, cols = _to_arrays(X, y)
    r = permutation_importance(
        model, Xarr, yarr, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    return _select_from_scores(cols, r.importances_mean, threshold)
