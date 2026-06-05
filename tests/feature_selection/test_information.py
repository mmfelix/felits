from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from felits.feature_selection.information import (
    mrmr_selection,
    mutual_information_ksg,
    mutual_information_matrix,
)


def test_ksg_detects_dependence() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    y = x**2 + 0.1 * rng.standard_normal(n)
    mi = mutual_information_ksg(x, y, k=3)
    assert mi > 0.1  # non-trivial dependence


def test_ksg_zero_for_independent() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    mi = mutual_information_ksg(x, y, k=3)
    assert mi < 0.05


def test_ksg_rejects_unequal() -> None:
    with pytest.raises(ValueError):
        mutual_information_ksg(np.zeros(5), np.zeros(4))


def test_mutual_information_matrix_shape(synthetic_classification) -> None:
    X, _ = synthetic_classification
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    M = mutual_information_matrix(df, k=2)
    assert M.shape == (10, 10)
    # Diagonal should be NaN (skip_diagonal=True by default).
    assert M.isna().values.diagonal().all()
    # Symmetric.
    np.testing.assert_allclose(M.values, M.values.T, equal_nan=True)


def test_mrmr_selection_returns_k_features() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.standard_normal((n, 8))
    y = (X[:, 0] + X[:, 3] - X[:, 5] + 0.1 * rng.standard_normal(n) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    selected = mrmr_selection(df, y, k_features=4, mi_method="sklearn")
    assert len(selected) == 4
    # Important features should be selected.
    assert "f0" in selected
