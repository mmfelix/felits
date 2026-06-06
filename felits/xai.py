"""Explainable AI utilities for time-series forecasting.

This module provides high-level wrappers for closed-loop SHAP feature
elimination and local explanations of single forecasts using SHAP and LIME.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .feature_selection.xai import lime_explain_instance, shap_feature_selection

__all__ = [
    "ClosedLoopResult",
    "deep_shap_selector",
    "explain_forecast",
    "plot_lime_explanation",
    "plot_shap_summary",
]


@dataclass
class ClosedLoopResult:
    """Result of :func:`deep_shap_selector`.

    Attributes
    ----------
    history : list[list[str]]
        A list of feature lists representing the selected features at each iteration.
    scores : list[float]
        A list of validation scores at each iteration, if a scoring function was provided.
    final_model : Any
        The final fitted model after the closed-loop selection process.
    selected_features : list[str]
        The final list of selected feature names.
    """

    history: list[list[str]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    final_model: Any = None
    selected_features: list[str] = field(default_factory=list)


def deep_shap_selector(
    model_factory: Callable[[list[str]], Any],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    val_X: pd.DataFrame | None = None,
    val_y: pd.Series | np.ndarray | None = None,
    threshold: float = 0.05,
    max_iters: int = 3,
    score_fn: Callable[[Any, np.ndarray, np.ndarray], float] | None = None,
) -> ClosedLoopResult:
    """Closed-loop SHAP feature elimination (research-grade meta-optimizer).

    At each iteration:
    1. Build a model on the current feature subset.
    2. Compute SHAP values.
    3. Drop features whose mean(|SHAP|) is below ``threshold * max``.
    4. Optionally retrain and re-evaluate.

    The procedure is repeated until the selected set stabilises or
    ``max_iters`` is reached.

    Parameters
    ----------
    model_factory : Callable[[list[str]], Any]
        A callable that takes a list of feature names and returns a fitted model.
    X : pd.DataFrame
        The training data used for SHAP evaluation.
    y : pd.Series or np.ndarray
        The target values corresponding to `X`.
    val_X : pd.DataFrame or None, default=None
        Optional validation features for scoring.
    val_y : pd.Series, np.ndarray, or None, default=None
        Optional validation targets for scoring.
    threshold : float, default=0.05
        Minimum relative mean(|SHAP|) for a feature to be kept.
    max_iters : int, default=3
        Maximum number of pruning iterations.
    score_fn : Callable or None, default=None
        A callable `(model, X_val, y_val) -> float` to evaluate the model at
        each iteration. If None and `val_X`/`val_y` are provided, Mean Absolute
        Error (MAE) is used.

    Returns
    -------
    ClosedLoopResult
        An object containing the selection history, scores, final model, and
        the final list of selected features.
    """
    result = ClosedLoopResult()

    if val_X is not None and val_y is not None and score_fn is None:
        from .preprocessing.metrics import mae as _mae

        def _default_score(model: Any, Xv: np.ndarray, yv: np.ndarray) -> float:
            return float(
                _mae(
                    np.asarray(yv, dtype=float).ravel(),
                    np.asarray(model.predict(Xv)).ravel(),
                )
            )

        score_fn = _default_score

    selected = list(X.columns)
    last_model: Any = None

    for _ in range(max_iters):
        Xs = X[selected]
        model = model_factory(selected)
        last_model = model

        if val_X is not None and score_fn is not None and val_y is not None:
            vXs = val_X[selected] if all(c in val_X.columns for c in selected) else val_X
            result.scores.append(float(score_fn(model, vXs.to_numpy(), np.asarray(val_y))))

        sh = shap_feature_selection(
            model, Xs, y=np.asarray(y) if y is not None else None, threshold=threshold, max_iters=1
        )
        result.history.append(list(sh.selected_features))

        if set(sh.selected_features) == set(selected):
            selected = sh.selected_features
            break

        selected = sh.selected_features

    result.selected_features = selected
    result.final_model = last_model
    return result


def explain_forecast(
    model: Any,
    X_background: pd.DataFrame | np.ndarray,
    instance: np.ndarray,
    feature_names: list[str] | None = None,
    num_lime_features: int = 10,
) -> dict[str, Any]:
    """Compute a local SHAP + LIME explanation for a single forecast.

    Parameters
    ----------
    model : Any
        A fitted estimator with a ``predict`` method.
    X_background : pd.DataFrame or np.ndarray
        Background data used to train the SHAP KernelExplainer.
    instance : np.ndarray
        The single instance (1D array) to explain.
    feature_names : list[str] or None, default=None
        Names of the features. If None, inferred from `X_background` or
        generated as "x0", "x1", etc.
    num_lime_features : int, default=10
        Number of features to include in the LIME explanation.

    Returns
    -------
    dict[str, Any]
        A dictionary containing 'shap_values', 'lime_explanation', and
        'feature_names'.

    Raises
    ------
    ImportError
        If the ``shap`` or ``lime`` packages are not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("explain_forecast requires shap. Install with: pip install shap") from exc

    if isinstance(X_background, pd.DataFrame):
        feature_names = list(X_background.columns) if feature_names is None else feature_names
        Xb = X_background.to_numpy(dtype=float)
    else:
        Xb = np.asarray(X_background, dtype=float)
        feature_names = feature_names or [f"x{i}" for i in range(Xb.shape[1])]

    explainer = shap.KernelExplainer(model.predict, Xb[: min(100, len(Xb))])
    sv = explainer.shap_values(instance.reshape(1, -1), check_additivity=False)
    lime_exp = lime_explain_instance(model, Xb, instance, num_features=num_lime_features)

    return {
        "shap_values": np.asarray(sv),
        "lime_explanation": lime_exp,
        "feature_names": feature_names,
    }


def plot_shap_summary(shap_values: np.ndarray, feature_names: list[str], **kwargs: Any) -> Any:
    """Display a SHAP beeswarm/summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        The SHAP values to plot.
    feature_names : list[str]
        The names of the features corresponding to the SHAP values.
    **kwargs : Any
        Additional keyword arguments passed to ``shap.summary_plot``.

    Returns
    -------
    Any
        The matplotlib figure object.

    Raises
    ------
    ImportError
        If ``shap`` or ``matplotlib`` are not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "plot_shap_summary requires shap and matplotlib. "
            "Install with: pip install shap matplotlib"
        ) from exc

    fig = plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False, **kwargs)
    return fig


def plot_lime_explanation(explanation: Any, **kwargs: Any) -> Any:
    """Display a LIME explanation in a notebook or pop-up window.

    Parameters
    ----------
    explanation : Any
        The LIME Explanation object returned by ``lime_explain_instance``.
    **kwargs : Any
        Additional keyword arguments passed to the LIME display method.

    Returns
    -------
    Any
        The result of ``show_in_notebook`` or ``as_pyplot_figure``.
    """
    try:
        return explanation.show_in_notebook(**kwargs)
    except Exception:
        return explanation.as_pyplot_figure()
