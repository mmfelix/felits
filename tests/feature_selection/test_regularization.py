from __future__ import annotations

import numpy as np
import pandas as pd

from felits.feature_selection.regularization import (
    adaptive_lasso_selection,
    elastic_net_selection,
    lasso_selection,
)


def test_lasso_selection_drops_noise() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X = rng.standard_normal((n, p))
    y = X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    res = lasso_selection(df, y, alpha=0.1)
    assert "f0" in res.selected_features
    assert "f1" in res.selected_features
    assert "f2" in res.selected_features
    # Many noise features should be dropped.
    assert len(res.selected_features) < p


def test_lasso_auto_alpha() -> None:
    rng = np.random.default_rng(0)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.1 * rng.standard_normal(n)
    res = lasso_selection(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y, alpha="auto", cv=3
    )
    assert res.alpha > 0


def test_adaptive_lasso_returns_coefs() -> None:
    rng = np.random.default_rng(0)
    n, p = 150, 10
    X = rng.standard_normal((n, p))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(n)
    res = adaptive_lasso_selection(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y, alpha=0.05
    )
    assert "f0" in res.selected_features
    assert "f1" in res.selected_features
    assert "weights" in res.extra


def test_elastic_net_selection() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 12
    X = rng.standard_normal((n, p))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(n)
    res = elastic_net_selection(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y, alpha=0.1, l1_ratio=0.7
    )
    assert "f0" in res.selected_features
    assert "f1" in res.selected_features
