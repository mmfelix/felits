"""Explainable AI utilities for time-series forecasting.

The module exposes:

- :func:`deep_shap_selector` — closed-loop SHAP feature elimination, the
  research-grade meta-optimizer that gives FELITS its distinctive edge.
- :func:`explain_forecast` — local SHAP + LIME explanation of a single
  forecast instance.
- :func:`plot_shap_summary` / :func:`plot_lime_explanation` — thin
  wrappers around the standard visualisation helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .feature_selection.xai import shap_feature_selection

__all__ = [
    "ClosedLoopResult",
    "deep_shap_selector",
    "explain_forecast",
    "plot_lime_explanation",
    "plot_shap_summary",
]


@dataclass
class ClosedLoopResult:
    """Result of :func:`deep_shap_selector`."""

    history: list[list[str]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    final_model: object = None
    selected_features: list[str] = field(default_factory=list)


def deep_shap_selector(
    model_factory: Callable[[list[str]], object],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    val_X: pd.DataFrame | None = None,
    val_y: pd.Series | np.ndarray | None = None,
    threshold: float = 0.05,
    max_iters: int = 3,
    score_fn: Callable[[object, np.ndarray, np.ndarray], float] | None = None,
) -> ClosedLoopResult:
    """Closed-loop SHAP feature elimination (research-grade meta-optimizer).

    At each iteration:

    1. Build a model on the current feature subset.
    2. Compute SHAP values.
    3. Drop features whose mean(|SHAP|) is below ``threshold * max``.
    4. Optionally retrain and re-evaluate.

    The procedure is repeated until the selected set stabilises or
    ``max_iters`` is reached.
    """
    result = ClosedLoopResult()
    if val_X is not None and val_y is not None and score_fn is None:
        from .preprocessing.metrics import mae as _mae

        def _default_score(model, Xv, yv):
            return _mae(np.asarray(yv, dtype=float).ravel(), np.asarray(model.predict(Xv)).ravel())

        score_fn = _default_score

    selected = list(X.columns)
    last_model: object | None = None
    for _ in range(max_iters):
        Xs = X[selected]
        model = model_factory(selected)
        last_model = model
        if val_X is not None and score_fn is not None:
            vXs = val_X[selected] if all(c in val_X.columns for c in selected) else val_X
            result.scores.append(float(score_fn(model, vXs, val_y)))
        sh = shap_feature_selection(model, Xs, y=y, threshold=threshold, max_iters=1)
        result.history.append(list(sh.selected_features))
        if set(sh.selected_features) == set(selected):
            selected = sh.selected_features
            break
        selected = sh.selected_features
    result.selected_features = selected
    result.final_model = last_model
    return result


def explain_forecast(
    model,
    X_background: pd.DataFrame | np.ndarray,
    instance: np.ndarray,
    feature_names: list[str] | None = None,
    num_lime_features: int = 10,
) -> dict:
    """Compute a local SHAP + LIME explanation for a single forecast."""
    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("explain_forecast requires shap. Install with: pip install shap") from exc
    from .feature_selection.xai import lime_explain_instance

    if isinstance(X_background, pd.DataFrame):
        feature_names = list(X_background.columns) if feature_names is None else feature_names
        Xb = X_background.to_numpy()
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


def plot_shap_summary(shap_values: np.ndarray, feature_names: list[str], **kwargs):
    """Display a SHAP beeswarm/summary plot."""
    import matplotlib.pyplot as _plt
    import shap  # type: ignore

    fig = _plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False, **kwargs)
    return fig


def plot_lime_explanation(explanation, **kwargs):
    """Display a LIME explanation in a notebook or pop-up window."""
    try:
        return explanation.show_in_notebook(**kwargs)
    except Exception:
        return explanation.as_pyplot_figure()
