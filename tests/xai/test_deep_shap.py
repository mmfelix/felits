from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from felits.xai import deep_shap_selector

# SHAP 0.44+ has a known memory-corruption bug in TreeExplainer that
# surfaces during interpreter shutdown when SHAP tests run alongside
# other heavy fixtures in the same pytest process. The test passes
# cleanly in isolation, so we mark it as ``xai`` to allow the user
# to opt out via ``-m "not xai"`` if it crashes in their environment.
pytestmark = pytest.mark.xai


def test_deep_shap_selector_runs() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 6
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])

    def factory(cols):
        sub = df[cols]
        return RandomForestRegressor(n_estimators=30, random_state=0).fit(sub, y)

    val_X = df.copy()
    val_y = y + 0.05 * rng.standard_normal(n)
    result = deep_shap_selector(
        factory, df, y, val_X=val_X, val_y=val_y, max_iters=2, threshold=0.1
    )
    assert len(result.selected_features) > 0
    assert "f0" in result.selected_features
    assert len(result.history) >= 1
