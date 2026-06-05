from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from felits.feature_selection.xai import shap_feature_selection

pytestmark = pytest.mark.xai


def test_shap_feature_selection_returns_list_and_importances() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 6
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    model = RandomForestRegressor(n_estimators=100, random_state=0).fit(df, y)
    result = shap_feature_selection(model, df, threshold=0.1, max_iters=2)
    assert isinstance(result.selected_features, list)
    assert isinstance(result.importances, dict)
    # Important features should be kept.
    assert "f0" in result.selected_features
    # History should record the iterations.
    assert len(result.history) >= 1
