"""High-level feature-selection pipeline.

:class:`FeatureSelector` chains an arbitrary subset of the methods
exposed in :mod:`felits.feature_selection` (causal, information-theoretic,
regularization, ensemble, XAI) and returns the intersection of the
features that survive every selected step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

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
    """The output of :meth:`FeatureSelector.run`."""

    selected_features: list[str]
    method_outputs: dict[str, list[str]] = field(default_factory=dict)
    final_importances: dict[str, float] = field(default_factory=dict)


class FeatureSelector:
    """Composable feature-selection pipeline.

    Example
    -------
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

    def __init__(self, steps: Sequence[tuple[str, dict]]):
        for name, _ in steps:
            if name not in self.VALID_STEPS:
                raise ValueError(f"Unknown step {name!r}; choose from {list(self.VALID_STEPS)}")
        self.steps = steps

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        shap_model=None,
        shap_X: pd.DataFrame | None = None,
    ) -> PipelineResult:
        """Execute the pipeline and return the surviving features.

        Parameters
        ----------
        X, y:
            Training features and target.
        shap_model, shap_X:
            Required when the pipeline includes the ``"shap"`` step:
            ``shap_model`` is a fitted estimator and ``shap_X`` is the
            background data for the SHAP explainer.
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
        shap_model=None,
        shap_X: pd.DataFrame | None = None,
    ) -> list[str]:
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


def _infer_target_name(X: pd.DataFrame, y, kwargs: dict) -> str:
    target = kwargs.get("target")
    if target is None:
        raise ValueError("`granger` step requires a `target` kwarg naming the target column.")
    return target


def select_features(
    df: pd.DataFrame,
    target: str,
    methods: Sequence[str] = ("mrmr", "lasso"),
    **kwargs,
) -> list[str]:
    """Convenience one-liner for the most common pipelines.

    Example
    -------
    >>> selected = select_features(df, target="demand",
    ...     methods=("mrmr", "lasso"),
    ...     max_lag=12, k_features=20, alpha="auto")
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
