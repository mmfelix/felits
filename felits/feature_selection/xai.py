"""XAI-driven feature selection and explanation utilities.

The flagship of this module is :func:`shap_feature_selection`, which
implements the *closed-loop* SHAP meta-optimizer: a model is trained on
all features, the global SHAP values are aggregated per feature, the
features whose mean(|SHAP|) is below a threshold are pruned, and a new
model is retrained. The procedure is repeated until convergence or a
maximum number of iterations, which mirrors the experimental protocol
from the FELITS research article (20.5% MAE / 28.1% variance reduction).

A complementary helper :func:`lime_explain_instance` is provided for
local explanations of single forecasts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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
    """Result of :func:`shap_feature_selection`."""

    selected_features: list[str]
    importances: dict[str, float]
    history: list[list[str]] = field(default_factory=list)


def _mean_abs_shap(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Aggregate SHAP values per feature.

    ``shap_values`` is expected to be a 2-D array of shape
    ``(n_samples, n_features)``. For tree models the values are returned in
    the column order of the training matrix; for kernel explainers the
    same convention holds.
    """
    if shap_values.ndim == 3:
        # Multi-output: take the mean over the output dimension.
        shap_values = shap_values.mean(axis=-1)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {f: float(v) for f, v in zip(feature_names, mean_abs, strict=True)}


def shap_feature_selection(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | None = None,
    threshold: float = 0.05,
    max_iters: int = 3,
    retrain: Callable | None = None,
    background: pd.DataFrame | np.ndarray | None = None,
    shap_method: str = "auto",
) -> ShapResult:
    """Closed-loop SHAP feature selection.

    Parameters
    ----------
    model:
        A fitted estimator with a ``predict`` method. Used to compute SHAP
        values for the current feature set.
    X:
        Validation / training data used for SHAP evaluation.
    y:
        Optional target (used for retraining when provided).
    threshold:
        Minimum relative mean(|SHAP|) for a feature to be kept. Features
        with importance below ``threshold * max(importance)`` are pruned.
    max_iters:
        Maximum number of pruning iterations.
    retrain:
        Optional callable ``(X_train, y_train, X_val) -> fitted_model`` used
        to refit the model after each pruning step. If ``None`` the
        function does not retrain and only reports the selected names.
    background:
        Optional background dataset for SHAP explainers that need one
        (e.g. ``KernelExplainer``).
    shap_method:
        ``"auto"`` picks the explainer based on the model class
        (``TreeExplainer`` for tree models, ``KernelExplainer`` fallback),
        ``"tree"`` or ``"kernel"`` to force a specific explainer.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shap_feature_selection requires shap. Install with: pip install shap"
        ) from exc

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    else:
        Xarr = np.asarray(X, dtype=float)
        feature_names = [f"x{i}" for i in range(Xarr.shape[1])]
    history: list[list[str]] = []
    selected = list(feature_names)
    last_importances: dict[str, float] = {}

    for _ in range(max_iters):
        history.append(list(selected))
        Xs = (
            X[selected]
            if isinstance(X, pd.DataFrame)
            else Xarr[:, [feature_names.index(s) for s in selected]]
        )
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
            yarr = np.asarray(y)
            model = retrain(Xs, yarr, Xs)
    return ShapResult(
        selected_features=selected,
        importances=last_importances,
        history=history,
    )


def _is_tree_model(model) -> bool:
    cls = model.__class__.__name__.lower()
    return any(k in cls for k in ("forest", "xgb", "lgbm", "gradient", "tree"))


def shap_interaction_selection(
    model,
    X: pd.DataFrame | np.ndarray,
    top_k: int = 20,
) -> list[tuple[str, str, float]]:
    """Return the top-k most important SHAP interaction pairs.

    Requires SHAP's TreeExplainer with ``shap_interaction_values`` (i.e.
    tree-based models). The result is sorted by mean absolute interaction
    strength in descending order.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shap_interaction_selection requires shap. Install with: pip install shap"
        ) from exc
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        Xs = X.to_numpy()
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
    model,
    X_train: pd.DataFrame | np.ndarray,
    instance: pd.Series | np.ndarray,
    num_features: int = 10,
    class_names: list[str] | None = None,
) -> object:
    """Build a LIME explanation for a single instance.

    Returns the LIME ``Explanation`` object, which can be turned into a
    visualisation with ``lime.lime_tabular``'s ``show_in_notebook`` or
    converted to a ``dict`` via ``as_list()``.
    """
    try:
        import lime.lime_tabular  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lime_explain_instance requires lime. Install with: pip install lime"
        ) from exc
    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
        Xtr = X_train.to_numpy()
    else:
        Xtr = np.asarray(X_train, dtype=float)
        feature_names = [f"x{i}" for i in range(Xtr.shape[1])]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xtr,
        feature_names=feature_names,
        class_names=class_names,
        mode="regression",
    )
    inst = np.asarray(instance).ravel()
    return explainer.explain_instance(inst, model.predict, num_features=num_features)
