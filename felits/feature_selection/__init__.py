"""Feature selection for time series."""

from __future__ import annotations

from .causal import (
    GrangerResult,
    granger_causality_test,
    granger_feature_selection,
    pcmci_selection,
)
from .ensemble import (
    permutation_importance_selection,
    rf_importance_selection,
    xgboost_importance_selection,
)
from .information import (
    conditional_mutual_information,
    mrmr_selection,
    mutual_information_ksg,
    mutual_information_matrix,
)
from .pipeline import FeatureSelector, PipelineResult, select_features
from .regularization import (
    LassoResult,
    adaptive_lasso_selection,
    elastic_net_selection,
    lasso_selection,
)
from .xai import (
    ShapResult,
    lime_explain_instance,
    shap_feature_selection,
    shap_interaction_selection,
)

__all__ = [
    "FeatureSelector",
    "GrangerResult",
    "LassoResult",
    "PipelineResult",
    "ShapResult",
    "adaptive_lasso_selection",
    "conditional_mutual_information",
    "elastic_net_selection",
    "granger_causality_test",
    "granger_feature_selection",
    "lasso_selection",
    "lime_explain_instance",
    "mrmr_selection",
    "mutual_information_ksg",
    "mutual_information_matrix",
    "pcmci_selection",
    "permutation_importance_selection",
    "rf_importance_selection",
    "select_features",
    "shap_feature_selection",
    "shap_interaction_selection",
    "xgboost_importance_selection",
]
