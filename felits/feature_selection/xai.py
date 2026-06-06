"""XAI-driven feature selection and explanation utilities.

This module provides closed-loop SHAP meta-optimization for feature
elimination, as well as local explanations for single forecasts using
SHAP and LIME.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ShapResult",
    "lime_explain_instance",
    "shap_feature_selection",
    "shap_interaction_selection",
]


@dataclass
class ShapResult:
    """Result of :func:`shap_feature_selection`.

    Attributes
    ----------
    selected_features : list[str]
        The list of feature names that were retained after pruning.
    importances : dict[str, float]
        A dictionary mapping feature names to their mean absolute SHAP values.
    history : list[list[str]]
        A list of feature lists representing the selected features at each iteration.
    """

    selected_features: list[str]
    importances: dict[str, float]
    history: list[list[str]] = field(default_factory=list)


def _mean_abs_shap(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Aggregate SHAP values per feature.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features) or (n_samples, n_features, n_outputs).
    feature_names : list[str]
        The names of the features corresponding to the SHAP values.

    Returns
    -------
    dict[str, float]
        A dictionary mapping feature names to their mean absolute SHAP values.
    """
    if shap_values.ndim == 3:
        # Multi-output: take the mean over the output dimension.
        shap_values = shap_values.mean(axis=-1)

    mean_abs = np.abs(shap_values).mean(axis=0)
    return {f: float(v) for f, v in zip(feature_names, mean_abs, strict=True)}


def _is_tree_model(model: Any) -> bool:
    """Check if a model is likely a tree-based model based on its class name.

    Parameters
    ----------
    model : Any
        The model instance to check.

    Returns
    -------
    bool
        True if the model class name suggests it is tree-based, False otherwise.
    """
    cls_name = model.__class__.__name__.lower()
    return any(k in cls_name for k in ("forest", "xgb", "lgbm", "gradient", "tree"))


def shap_feature_selection(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | None = None,
    threshold: float = 0.05,
    max_iters: int = 3,
    retrain: Callable[[Any, np.ndarray, np.ndarray], Any] | None = None,
    background: pd.DataFrame | np.ndarray | None = None,
    shap_method: str = "auto",
) -> ShapResult:
    """Closed-loop SHAP feature selection.

    This function implements a closed-loop SHAP meta-optimizer: a model is
    trained on all features, the global SHAP values are aggregated per feature,
    and features whose mean absolute SHAP value is below a threshold are pruned.
    The procedure is repeated until convergence or a maximum number of iterations.

    Parameters
    ----------
    model : Any
        A fitted estimator with a ``predict`` method. Used to compute SHAP
        values for the current feature set.
    X : pd.DataFrame or np.ndarray
        Validation or training data used for SHAP evaluation.
    y : pd.Series, np.ndarray, or None, default=None
        Optional target vector, used for retraining when provided.
    threshold : float, default=0.05
        Minimum relative mean(|SHAP|) for a feature to be kept. Features with
        importance below ``threshold * max(importance)`` are pruned.
    max_iters : int, default=3
        Maximum number of pruning iterations.
    retrain : Callable or None, default=None
        Optional callable ``(X_train, y_train, X_val) -> fitted_model`` used
        to refit the model after each pruning step. If None, the function does
        not retrain and only reports the selected names.
    background : pd.DataFrame, np.ndarray, or None, default=None
        Optional background dataset for SHAP explainers that need one
        (e.g., ``KernelExplainer``).
    shap_method : str, default="auto"
        "auto" picks the explainer based on the model class (``TreeExplainer``
        for tree models, ``KernelExplainer`` fallback). "tree" or "kernel"
        forces a specific explainer.

    Returns
    -------
    ShapResult
        An object containing the selected features, their importances, and the
        history of feature sets at each iteration.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap_feature_selection requires shap. Install with: pip install shap"
        ) from exc

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

    history: list[list[str]] = []
    selected = list(feature_names)
    last_importances: dict[str, float] = {}

    for _ in range(max_iters):
        history.append(list(selected))

        # Select columns based on current feature set
        if isinstance(X, pd.DataFrame):
            Xs = X[selected]
        else:
            indices = [feature_names.index(s) for s in selected]
            Xs = X_arr[:, indices]

        # SHAP's TreeExplainer has a known memory-corruption issue on very
        # small feature subsets (1-2 features) for some sklearn versions.
        # We catch it defensively and bail out with the current selection.
        try:
            if shap_method == "tree" or (shap_method == "auto" and _is_tree_model(model)):
                explainer = shap.TreeExplainer(model)
            else:
                if background is None:
                    bg = shap.kmeans(Xs, 50) if hasattr(shap, "kmeans") else Xs[: min(100, len(Xs))]
                else:
                    bg = background
                explainer = shap.KernelExplainer(model.predict, bg)

            sv = explainer.shap_values(Xs, check_additivity=False)
        except Exception:
            # SHAP failed on this subset; stop pruning and return what we have.
            break

        last_importances = _mean_abs_shap(np.asarray(sv), selected)
        max_imp = max(last_importances.values()) if last_importances else 0.0

        if max_imp <= 0:
            break

        new_selected = [f for f, v in last_importances.items() if v >= threshold * max_imp]

        if set(new_selected) == set(selected):
            selected = new_selected
            break

        selected = new_selected

        if retrain is not None and y is not None:
            y_arr = np.asarray(y, dtype=float)
            model = retrain(Xs, y_arr, Xs)

    return ShapResult(
        selected_features=selected,
        importances=last_importances,
        history=history,
    )


def shap_interaction_selection(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    top_k: int = 20,
) -> list[tuple[str, str, float]]:
    """Return the top-k most important SHAP interaction pairs.

    Requires SHAP's TreeExplainer with ``shap_interaction_values`` (i.e.,
    tree-based models). The result is sorted by mean absolute interaction
    strength in descending order.

    Parameters
    ----------
    model : Any
        A fitted tree-based estimator.
    X : pd.DataFrame or np.ndarray
        Data used to compute SHAP interaction values.
    top_k : int, default=20
        The number of top interaction pairs to return.

    Returns
    -------
    list[tuple[str, str, float]]
        A list of tuples containing (feature_1, feature_2, interaction_strength).

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap_interaction_selection requires shap. Install with: pip install shap"
        ) from exc

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        Xs = X.to_numpy(dtype=float)
    else:
        Xs = np.asarray(X, dtype=float)
        feature_names = [f"x{i}" for i in range(Xs.shape[1])]

    explainer = shap.TreeExplainer(model)
    inter = np.asarray(explainer.shap_interaction_values(Xs))

    if inter.ndim == 4:  # multi-output
        inter = inter.mean(axis=-1)

    n_features = inter.shape[-1]
    pairs: list[tuple[str, str, float]] = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength = float(np.abs(inter[:, i, j]).mean())
            pairs.append((feature_names[i], feature_names[j], strength))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


def lime_explain_instance(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    instance: pd.Series | np.ndarray,
    num_features: int = 10,
    class_names: list[str] | None = None,
) -> Any:
    """Build a LIME explanation for a single instance.

    Parameters
    ----------
    model : Any
        A fitted estimator with a ``predict`` method.
    X_train : pd.DataFrame or np.ndarray
        The training data used to train the LIME explainer.
    instance : pd.Series or np.ndarray
        The single instance to explain.
    num_features : int, default=10
        The number of features to include in the LIME explanation.
    class_names : list[str] or None, default=None
        Names of the classes (for classification tasks).

    Returns
    -------
    Any
        The LIME ``Explanation`` object, which can be visualized or converted
        to a dictionary via ``as_list()``.

    Raises
    ------
    ImportError
        If the ``lime`` package is not installed.
    """
    try:
        import lime.lime_tabular
    except ImportError as exc:
        raise ImportError(
            "lime_explain_instance requires lime. Install with: pip install lime"
        ) from exc

    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
        Xtr = X_train.to_numpy(dtype=float)
    else:
        Xtr = np.asarray(X_train, dtype=float)
        feature_names = [f"x{i}" for i in range(Xtr.shape[1])]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xtr,
        feature_names=feature_names,
        class_names=class_names,
        mode="regression",
    )

    inst = np.asarray(instance, dtype=float).ravel()
    return explainer.explain_instance(inst, model.predict, num_features=num_features)
