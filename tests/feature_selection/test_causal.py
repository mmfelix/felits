from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from felits.feature_selection.causal import (
    granger_causality_test,
    granger_feature_selection,
)


def test_granger_detects_dependent_series() -> None:
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + 0.5 * x[i - 1] + 0.1 * rng.standard_normal()
    res = granger_causality_test(x, y, max_lag=3, significance=0.05)
    assert any(r.significant for r in res)


def test_granger_rejects_unequal_shapes() -> None:
    with pytest.raises(ValueError):
        granger_causality_test(np.zeros(5), np.zeros(4), max_lag=1)


def test_granger_feature_selection_returns_causal_features() -> None:
    rng = np.random.default_rng(0)
    n = 400
    common = rng.standard_normal(n)
    target = np.zeros(n)
    for i in range(1, n):
        target[i] = 0.5 * target[i - 1] + 0.4 * common[i - 1] + 0.1 * rng.standard_normal()
    noise = rng.standard_normal(n)
    df = pd.DataFrame({"target": target, "cause": common, "noise": noise})
    selected = granger_feature_selection(df, target="target", max_lag=3, significance=0.05)
    assert "cause" in selected
    assert "noise" not in selected
