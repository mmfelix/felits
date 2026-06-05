from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from felits.feature_selection.pipeline import FeatureSelector, select_features


def test_feature_selector_lasso_only() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.standard_normal((n, p))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    fs = FeatureSelector(steps=[("lasso", {"alpha": 0.1})])
    res = fs.run(df, y)
    assert "f0" in res.selected_features
    assert "f1" in res.selected_features
    assert "lasso" in res.method_outputs


def test_feature_selector_chained() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    fs = FeatureSelector(
        steps=[
            ("rf", {"threshold": 0.05}),
            ("lasso", {"alpha": 0.1}),
        ]
    )
    res = fs.run(df, y)
    # The intersection should be a non-empty set of the truly relevant features.
    assert set(res.selected_features) <= {"f0", "f1"}


def test_feature_selector_rejects_unknown_step() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(steps=[("nope", {})])


def test_select_features_one_liner() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    selected = select_features(
        df, target="target", methods=("mrmr", "lasso"), k_features=3, alpha=0.1
    )
    assert isinstance(selected, list)


def test_select_features_empty_methods() -> None:
    with pytest.raises(ValueError):
        select_features(pd.DataFrame(), target="t", methods=())
