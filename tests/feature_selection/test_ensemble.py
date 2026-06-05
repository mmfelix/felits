from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from felits.feature_selection.ensemble import (
    permutation_importance_selection,
    rf_importance_selection,
)


def test_rf_importance_selection_picks_informative_features() -> None:
    rng = np.random.default_rng(0)
    n, p = 300, 10
    X = rng.standard_normal((n, p))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    selected, importances = rf_importance_selection(df, y, threshold=0.1)
    assert "f0" in selected
    assert "f1" in selected
    assert importances["f0"] > 0
    # No feature should be flagged as more important than the max.
    assert max(importances.values()) <= 1.0 + 1e-9


def test_permutation_importance_selection(synthetic_classification) -> None:
    X, y = synthetic_classification
    model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
    selected, importances = permutation_importance_selection(
        model, X, y, threshold=0.1, n_repeats=3, random_state=0
    )
    assert isinstance(selected, list)
    assert isinstance(importances, dict)
