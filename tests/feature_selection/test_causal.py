from __future__ import annotations

import warnings

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


def test_granger_suppresses_rank_deficient_warning() -> None:
    """The statsmodels 'covariance of constraints does not have full rank'
    ValueWarning is expected when the candidate is a lag/rolling of the
    target (the common case in the feature-engineering pipeline). It must
    not leak into the prepare_data log. Other warnings must still propagate.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = np.cumsum(rng.normal(size=n)) + 10
    # Candidate features that are near-collinear with y's own lags: this is
    # the case that triggers the rank-deficient OLS in statsmodels.
    df = pd.DataFrame(
        {
            "SIN": y,
            "lag1_SIN": np.concatenate([[np.nan], y[:-1]]),
            "rolling_mean_24_SIN": pd.Series(y).rolling(24, min_periods=1).mean().values,
        }
    )
    df = df.iloc[24:].reset_index(drop=True)

    caught: list[str] = []
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        granger_feature_selection(df, target="SIN", max_lag=4, significance=0.05)
        for w in recorded:
            caught.append(f"{w.category.__name__}: {str(w.message)[:100]}")

    rank_deficient = [m for m in caught if "covariance of constraints" in m]
    assert rank_deficient == [], (
        f"Expected the rank-deficient ValueWarning to be suppressed, "
        f"but got: {rank_deficient}"
    )
