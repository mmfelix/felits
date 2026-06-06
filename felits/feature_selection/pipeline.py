"""High-level feature-selection pipeline.

:class:`FeatureSelector` chains an arbitrary subset of the methods
exposed in :mod:`felits.feature_selection` (causal, information-theoretic,
regularization, ensemble, XAI) and returns the intersection of the
features that survive every selected step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .causal import granger_feature_selection
from .ensemble import rf_importance_selection
from .information import mrmr_selection
from .regularization import adaptive_lasso_selection, lasso_selection
from .xai import shap_feature_selection

__all__ = ["FeatureSelector", "PipelineResult", "select_features"]


@dataclass
class PipelineResult:
    """The output of :meth:`FeatureSelector.run`.

    Attributes
    ----------
    selected_features : list[str]
        Features that survived all pipeline steps.
    method_outputs : dict[str, list[str]]
        Per-step feature lists before intersection.
    final_importances : dict[str, float]
        Aggregated importance scores (populated when available).
    """

    selected_features: list[str]
    method_outputs: dict[str, list[str]] = field(default_factory=dict)
    final_importances: dict[str, float] = field(default_factory=dict)


class FeatureSelector:
    """Composable feature-selection pipeline.

    Parameters
    ----------
    steps : Sequence[tuple[str, dict]]
        Ordered list of ``(method_name, kwargs)`` tuples. Valid method
        names are ``"granger"``, ``"mrmr"``, ``"lasso"``,
        ``"adaptive_lasso"``, ``"rf"``, and ``"shap"``.

    Examples
    --------
    >>> from felits.feature_selection import FeatureSelector
    >>> fs = FeatureSelector(steps=[
    ...     ("granger", {"max_lag": 12}),
    ...     ("mrmr", {"k_features": 30}),
    ...     ("lasso", {"alpha": "auto"}),
    ... ])
    >>> result = fs.run(X, y)
    >>> X_new = X[result.selected_features]
    """

    VALID_STEPS = ("granger", "mrmr", "lasso", "adaptive_lasso", "rf", "shap")

    def __init__(self, steps: Sequence[tuple[str, dict]]) -> None:
        for name, _ in steps:
            if name not in self.VALID_STEPS:
                raise ValueError(f"Unknown step {name!r}; choose from {list(self.VALID_STEPS)}")
        self.steps = steps

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        shap_model: Any = None,
        shap_X: pd.DataFrame | None = None,
    ) -> PipelineResult:
        """Execute the pipeline and return the surviving features.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series or np.ndarray
            Target values.
        shap_model : Any, default=None
            A fitted estimator required when the pipeline includes the
            ``"shap"`` step.
        shap_X : pd.DataFrame or None, default=None
            Background data for the SHAP explainer. If None, ``X`` is used.

        Returns
        -------
        PipelineResult
            Object containing the selected features and per-step outputs.

        Raises
        ------
        ValueError
            If the ``"shap"`` step is included but ``shap_model`` is None.
        """
        candidates: set[str] | None = None
        outputs: dict[str, list[str]] = {}
        for name, kwargs in self.steps:
            chosen = self._run_step(name, X, y, kwargs, shap_model=shap_model, shap_X=shap_X)
            outputs[name] = chosen
            chosen_set = set(chosen)
            candidates = chosen_set if candidates is None else candidates & chosen_set
        selected = sorted(candidates) if candidates else []
        return PipelineResult(
            selected_features=selected,
            method_outputs=outputs,
            final_importances={},
        )

    @staticmethod
    def _run_step(
        name: str,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        kwargs: dict,
        shap_model: Any = None,
        shap_X: pd.DataFrame | None = None,
    ) -> list[str]:
        """Dispatch a single pipeline step to the appropriate selection function.

        Parameters
        ----------
        name : str
            Step name (must be in ``VALID_STEPS``).
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or np.ndarray
            Target values.
        kwargs : dict
            Keyword arguments forwarded to the selection function.
        shap_model : Any, default=None
            Fitted model for the SHAP step.
        shap_X : pd.DataFrame or None, default=None
            Background data for SHAP.

        Returns
        -------
        list[str]
            Selected feature names for this step.
        """
        if name == "granger":
            return granger_feature_selection(X, target=_infer_target_name(X, y, kwargs), **kwargs)
        if name == "mrmr":
            return mrmr_selection(X, y, **kwargs)
        if name == "lasso":
            return lasso_selection(X, y, **kwargs).selected_features
        if name == "adaptive_lasso":
            return adaptive_lasso_selection(X, y, **kwargs).selected_features
        if name == "rf":
            return rf_importance_selection(X, y, **kwargs)[0]
        if name == "shap":
            if shap_model is None:
                raise ValueError("`shap_model` is required for the 'shap' step.")
            result = shap_feature_selection(
                shap_model, shap_X if shap_X is not None else X, y=y, **kwargs
            )
            return result.selected_features
        raise AssertionError(f"Unhandled step {name!r}")


def _infer_target_name(X: pd.DataFrame, y: Any, kwargs: dict) -> str:
    """Extract the target column name from kwargs for the Granger step.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (unused, kept for API consistency).
    y : Any
        Target values (unused, kept for API consistency).
    kwargs : dict
        Must contain a ``"target"`` key.

    Returns
    -------
    str
        The target column name.

    Raises
    ------
    ValueError
        If ``"target"`` is not present in kwargs.
    """
    target = kwargs.get("target")
    if target is None:
        raise ValueError("`granger` step requires a `target` kwarg naming the target column.")
    return target


def select_features(
    df: pd.DataFrame,
    target: str,
    methods: Sequence[str] = ("mrmr", "lasso"),
    **kwargs: Any,
) -> list[str]:
    """Convenience one-liner for the most common feature-selection pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both features and the target column.
    target : str
        Name of the target column.
    methods : Sequence[str], default=("mrmr", "lasso")
        Selection methods to apply. Valid names: ``"granger"``, ``"mrmr"``,
        ``"lasso"``, ``"adaptive_lasso"``, ``"rf"``.
    **kwargs : Any
        Additional keyword arguments forwarded to each method. Supported
        keys include ``max_lag``, ``k_features``, ``alpha``, ``threshold``.

    Returns
    -------
    list[str]
        Intersection of features selected by all methods.

    Raises
    ------
    ValueError
        If no recognized methods are provided.

    Examples
    --------
    >>> selected = select_features(df, target="demand",
    ...     methods=("mrmr", "lasso"),
    ...     k_features=20, alpha="auto")
    """
    steps: list[tuple[str, dict]] = []
    if "granger" in methods:
        steps.append(("granger", {"target": target, "max_lag": kwargs.get("max_lag", 12)}))
    if "mrmr" in methods:
        steps.append(("mrmr", {"k_features": kwargs.get("k_features", 20)}))
    if "lasso" in methods:
        steps.append(("lasso", {"alpha": kwargs.get("alpha", "auto")}))
    if "adaptive_lasso" in methods:
        steps.append(("adaptive_lasso", {"alpha": kwargs.get("alpha", "auto")}))
    if "rf" in methods:
        steps.append(("rf", {"threshold": kwargs.get("threshold", 0.01)}))
    if not steps:
        raise ValueError(f"No recognised methods in {methods!r}.")
    X = df.drop(columns=[target])
    return FeatureSelector(steps=steps).run(X, df[target]).selected_features
